"""Importance maps for weighted-MSE RDO, straight from the Hessian of a perceptual metric.

The metric is a function of the samples the codec actually codes, i.e. of the three
4:2:0 planes (Y, Cb, Cr).  Writing d(x, x + e) for the metric between the source x and a
reconstruction x + e, and using d(x, x) = 0 and grad_xhat d(x, x) = 0,

    d(x, x + e) = 1/2 e^T H e + O(|e|^3),      H = Hessian of d w.r.t. xhat at xhat = x.

H is close to diagonally dominant for the usual full-reference metrics, so

    d(x, x + e) ~ 1/2 sum_i H_ii e_i^2,

which is a weighted MSE with per-sample weights w_i = H_ii.  Those are what this module
estimates and what the encoder consumes.

Differences from TIP_Hess_v2/compute_q/compute_Q_pytorch.py and RDO_PERC/generate_weights.py:

  * Rademacher probes instead of Gaussian ones.  For the Bekas/Hutchinson diagonal
    estimator both are unbiased, but with s_i^2 = 1 the self-normalisation is exact and
    the variance is strictly lower.
  * No per-probe percentile clamp.  Clamping s .* Hs before averaging is a *biased*
    operation: the bias does not shrink as the number of probes grows.  Measured on a
    64x64 crop against the exact diagonal (SSIM): relative error 0.013 without the clamp
    at 128 probes vs 0.110 with it, and the clamped version does not improve from 32 to
    128 probes.  Outliers are handled once, on the final averaged map.
  * The Hessian is taken w.r.t. the 4:2:0 planes directly (chroma upsampling is part of
    the differentiated function) rather than w.r.t. an upsampled YUV image whose chroma
    weights are then decimated.
  * Planes the metric does not depend on (e.g. chroma for a luma-only MS-SSIM) get a flat
    map of ones instead of a degenerate all-zero map.
"""

import os
import contextlib
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_STAC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _STAC not in sys.path:
    sys.path.insert(0, _STAC)
if os.path.join(_STAC, 'Common') not in sys.path:
    sys.path.insert(0, os.path.join(_STAC, 'Common'))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

METRICS = ('MS_SSIM', 'MS_SSIM_RGB', 'SSIM', 'LPIPS', 'LPIPS_ALEX', 'DISTS', 'WD')

# Wasserstein Distortion is parameterised by a per-pixel log2(sigma) map; sigma=0 is
# pointwise feature fidelity and larger sigma tolerates texture resynthesis.  A constant
# map is used here, matching the repo's own reference test.
WD_LOG2_SIGMA = 2.0


def set_wd_sigma(value):
    """Set the constant log2(sigma) the WD Hessian is estimated at."""
    global WD_LOG2_SIGMA
    WD_LOG2_SIGMA = float(value)


# --------------------------------------------------------------------------------------
# colour / resampling helpers (differentiable)
# --------------------------------------------------------------------------------------
def upsample_chroma(c, out_hw):
    """4:2:0 chroma plane -> luma grid, bilinear."""
    return F.interpolate(c, size=out_hw, mode='bilinear', align_corners=False)


# The YUVs are written by ffmpeg's yuv420p default, which is limited range: luma occupies 16..235 and
# chroma 16..240.  Expanding them with the full-range matrix therefore yields RGB at ~86 % of the true
# contrast.  HW_YUV_RANGE selects the convention; it is read here so that map generation and every
# evaluator share one setting rather than disagreeing.
YUV_RANGE = os.environ.get('HW_YUV_RANGE', 'full')


def yuv_to_rgb(y, u, v):
    """BT.601 YCbCr (0..255) -> RGB (0..255), full or limited range.  Inputs on the luma grid."""
    if YUV_RANGE == 'limited':
        y = (y - 16.0) * (255.0 / 219.0)
        cb = (u - 128.0) * (255.0 / 224.0)
        cr = (v - 128.0) * (255.0 / 224.0)
    else:
        cb = u - 128.0
        cr = v - 128.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.cat([r, g, b], dim=1)


# --------------------------------------------------------------------------------------
# metric wrappers:  f(ref_planes, planes) -> scalar
# --------------------------------------------------------------------------------------
def _make_metric(name):
    from Common.utils.q_utils import ssim_func, ms_ssim_func, dists_func, compute_LPIPS_rgb

    name = name.upper()
    if name == 'MS_SSIM':
        # msssim_fs is the same computation as pytorch_msssim.ms_ssim with the protocol defaults (agrees to
        # 1e-7) but takes the Gaussian window and the level weights as arguments instead of deriving them
        # from the input's device, which is what lets torch.func trace it for forward-over-reverse HVPs.
        from msssim_fs import ms_ssim_fs, make_win, LEVEL_WEIGHTS
        cache = {}
        def f_ms(ref, cur):
            x = ref[0]
            key = (x.device, x.dtype, x.shape[1])
            if key not in cache:
                cache[key] = (make_win(x.shape[1], x.device, x.dtype),
                              torch.tensor(LEVEL_WEIGHTS, device=x.device, dtype=x.dtype))
            win, w = cache[key]
            return 1.0 - ms_ssim_fs(ref[0], cur[0], win, w)
        return f_ms, ('Y',)   # distance: 1 - MS-SSIM
    if name == 'MS_SSIM_RGB':
        # MS-SSIM on RGB, the convention of Yang & Bajic; used only for the literature-protocol table.
        # Unlike the luma MS-SSIM above this depends on all three planes, so chroma gets real weights.
        from msssim_std import ms_ssim_std
        def f_msrgb(ref, cur):
            hw = ref[0].shape[-2:]
            r = yuv_to_rgb(ref[0], upsample_chroma(ref[1], hw), upsample_chroma(ref[2], hw))
            c = yuv_to_rgb(cur[0], upsample_chroma(cur[1], hw), upsample_chroma(cur[2], hw))
            return 1.0 - ms_ssim_std(r, c)
        return f_msrgb, ('Y', 'Cb', 'Cr')
    if name == 'SSIM':
        return lambda ref, cur: ssim_func(ref[0], cur[0]), ('Y',)

    if name in ('LPIPS', 'DISTS', 'LPIPS_ALEX'):
        def f(ref, cur):
            hw = ref[0].shape[-2:]
            rgb_ref = yuv_to_rgb(ref[0], upsample_chroma(ref[1], hw), upsample_chroma(ref[2], hw))
            rgb_cur = yuv_to_rgb(cur[0], upsample_chroma(cur[1], hw), upsample_chroma(cur[2], hw))
            if name == 'LPIPS':
                return compute_LPIPS_rgb(rgb_ref, rgb_cur)
            if name == 'LPIPS_ALEX':
                return _lpips_net('alex')(2.0 * (rgb_ref / 255.0 - 0.5), 2.0 * (rgb_cur / 255.0 - 0.5))
            # dists_func expects YUV and converts internally; go through RGB/255 instead.
            # q_utils leaves the DISTS model on the CPU, so move it once.
            from Common.utils.q_utils import D as _D
            if next(_D.parameters(), torch.zeros(1)).device != rgb_ref.device:
                _D.to(rgb_ref.device)
            return _D(rgb_ref / 255.0, rgb_cur / 255.0, require_grad=True, batch_average=True)
        return f, ('Y', 'Cb', 'Cr')

    if name == 'WD':
        def f(ref, cur):
            o = _wd_model()
            r = _planes_to_dists_input(*ref)          # RGB in [0, 1], same as DISTS
            c = _planes_to_dists_input(*cur)
            sig = torch.zeros_like(r[:, 0:1]) + WD_LOG2_SIGMA
            return o(c, r, sig)
        return f, ('Y', 'Cb', 'Cr')

    raise ValueError(f'unknown metric {name}')


_WD = [None]


def _wd_model():
    """VGG16 Wasserstein Distortion, loaded once.  Vendored under vendor/ -- see vendor/README."""
    if _WD[0] is None:
        import sys as _sys
        _here = os.path.dirname(os.path.abspath(__file__))
        _v = os.path.join(_here, 'vendor')
        if _v not in _sys.path:
            _sys.path.insert(0, _v)
        from wasserstein_distortion import VGG16WassersteinDistortion
        m = VGG16WassersteinDistortion().to(DEVICE).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _WD[0] = m
    return _WD[0]


def wd_residual(in0, in1, log2_sigma=None):
    """r(in1) with ||r||^2 == WD(in0, in1) exactly.  Inputs are RGB in [0, 1].

    Wasserstein Distortion is already written as a weighted sum of squared differences that
    vanish at in1 == in0 (see WassersteinDistortionFeature.forward in the reference code):
    the pointwise feature difference, and per pyramid level the difference of local means and
    of local standard deviations.  The weights are relu(1 - |log2_sigma - i|) >= 0, so the
    square roots are real and H = 2 J^T J holds exactly at in1 == in0.
    """
    import torch.nn.functional as Fn
    o = _wd_model()
    if log2_sigma is None:
        log2_sigma = torch.zeros_like(in0[:, 0:1]) + WD_LOG2_SIGMA
    fa = o.feature_backbone(in1, num_scales=3)
    fb = o.feature_backbone(in0, num_scales=3)
    wdf = o.wasserstein_distortion_feature
    parts = []
    for fp, fgt in zip(fa, fb):
        ls = Fn.interpolate(log2_sigma, size=fgt.shape[-2:], mode='bilinear', antialias=True)
        lr = (np.log2(log2_sigma.shape[-2] / fgt.shape[-2])
              + np.log2(log2_sigma.shape[-1] / fgt.shape[-1])) / 2.0
        ls = Fn.relu(ls - lr)

        ma, va = wdf.multi_level_stats(fp)
        mb, vb = wdf.multi_level_stats(fgt)
        maps = [(fp - fgt, None)]
        for i in range(wdf.num_levels):
            sa = torch.sqrt(torch.clamp(va[i], min=1e-8))
            sb = torch.sqrt(torch.clamp(vb[i], min=1e-8))
            maps.append((ma[i] - mb[i], sa - sb))
        for i, (d1, d2) in enumerate(maps):
            w = Fn.relu(1 - torch.abs(ls - i))
            if i > 0:
                ls = wdf.lowpass(ls, stride=2)
            # (w * d^2).mean() over the broadcast shape -> scale each residual by w / numel
            n = d1.numel() // d1.shape[1] * d1.shape[1]
            sc = torch.sqrt(w.clamp_min(0) / n)
            parts.append((sc * d1).reshape(-1))
            if d2 is not None:
                parts.append((sc * d2).reshape(-1))
    return torch.cat(parts)


# --------------------------------------------------------------------------------------
# Gauss-Newton estimator (metrics that are a sum of squares of a vanishing residual)
# --------------------------------------------------------------------------------------
_LPIPS_NETS = {}


# Large inputs (>= CKPT_MIN_PIXELS, i.e. the 2K CLIC images): a 2048x1360 image through VGG16 keeps
# ~3 GiB of saved activations per retained graph and the residual itself is ~1.4 GB, which is what
# pushed the CLIC maps past the 11 GiB of the local card.  Above the threshold the vector-Jacobian
# product J^T v is accumulated per residual group (one VGG stage / LPIPS layer at a time): the group's
# parts are rebuilt in a fresh graph for every probe, back-propagated once with the matching slice of
# the SAME probe vector, and released.  Exactly the single backward with the concatenated v
# (linearity); only the float summation order differs.  Below the threshold (Kodak) nothing changes.
CKPT_MIN_PIXELS = int(os.environ.get('HW_CKPT_MIN_PIXELS', 1_500_000))
BLOCK_ACC_CPU_MIN_PIXELS = int(os.environ.get('HW_BLOCK_ACC_CPU_MIN_PIXELS', 1_500_000))


class _LazyParts:
    """Residual parts rebuilt on demand.  sizes[g] = numel of each part of group g (in the order of
    the concatenated residual); build(g) returns those parts as tensors carrying a graph to `cur`."""

    def __init__(self, sizes, build):
        self.sizes, self.build = sizes, build
        self.total = sum(sum(g) for g in sizes)

    def numel(self):
        return self.total


def _chunk_vjp(x, resid):
    return (resid in (lpips_residual, lpips_alex_residual, dists_residual, wd_residual)
            and x.shape[-1] * x.shape[-2] >= CKPT_MIN_PIXELS)


def _lazy_parts(resid, ref, in1):
    if resid is dists_residual:
        return dists_lazy_parts(ref, in1)
    if resid is wd_residual:
        return wd_lazy_parts(ref, in1)
    return lpips_lazy_parts(ref, in1, net='alex' if resid is lpips_alex_residual else 'vgg')


def _vjp_chunked(lazy, cur, v_cpu, scale=1.0):
    """J^T v = sum_g J_g^T v_g over the residual groups of a _LazyParts (see above).  The probe
    stays on the CPU; only the current group's slice is moved."""
    dev = cur[0].device
    gr, off = None, 0
    for g, sizes in enumerate(lazy.sizes):
        ps = lazy.build(g)
        vs = []
        for p, n in zip(ps, sizes):
            vp = v_cpu[off:off + n].to(dev)
            vs.append((vp * scale if scale != 1.0 else vp).view_as(p))
            off += n
        # retain_graph keeps the shared input graph (chroma upsampling, colour conversion) alive
        # for the next group; the group's own graph is freed with `ps`.
        g_ = torch.autograd.grad(ps, cur, grad_outputs=vs, retain_graph=True)
        gr = list(g_) if gr is None else [x.add_(y) for x, y in zip(gr, g_)]
        del ps, vs, g_
    return tuple(gr)


# Probe prefetching (7 Sept): the Gaussian probes are drawn on the CPU with one seeded generator per probe
# index, which on Kodak is 48M floats for LPIPS and 240M for WD -- a serial 0.1-0.7 s per probe before
# each backward.  A worker thread draws block k+1 while the GPU consumes block k.  Same seeds, same
# draws, same arithmetic; enable with HW_PREFETCH=1 (default on once validated).
PREFETCH = os.environ.get('HW_PREFETCH', '0') == '1'     # opt-in until validated
# HW_PROBE_DEVICE=cuda draws the Gaussian probes on the GPU (Philox, one seeded generator per probe index,
# ~2 ms for 48M floats) instead of the CPU (0.4-0.7 s per probe under load): a different random stream from
# the CPU-drawn maps, statistically equivalent (seed-1 rows move Table 1 by <= 0.2 points).  Opt-in.
PROBE_DEVICE = os.environ.get('HW_PROBE_DEVICE', 'cpu')

# Smoothing splits the probe budget into rounds, one fresh noise draw per round, and every round rebuilds the
# forward pass.  For a metric with an expensive forward (WD runs VGG16 at three scales) those rebuilds, not the
# probes, dominate.  ROUND_SIZE is the probes per round: 16 is what every published number here used.
ROUND_SIZE = int(os.environ.get('HW_ROUND_SIZE', 16))

# Hessian-vector products for the Hutchinson metrics.  Reverse-over-reverse (build the gradient graph with
# create_graph=True, back-propagate through it once per probe) costs ~300 ms/probe for MS-SSIM at Kodak size.
# Forward-over-reverse -- a jvp through the gradient function -- computes the same Hv without ever
# materialising the second graph, and vmap batches the probes into fused kernels: measured 302.7 -> 3.0
# ms/probe on an A100 at batch 16, with rank correlation 1.000000 and max relative difference 3e-7 against
# the reverse-over-reverse result.  HW_HVP=rev restores the old path; HW_HVP_BATCH sets the vmap width.
HVP_MODE  = os.environ.get('HW_HVP', 'fwd')
HVP_BATCH = int(os.environ.get('HW_HVP_BATCH', 16))
BH_CKPT_SEC = float(os.environ.get('HW_BH_CKPT_SEC', 300))   # resume-checkpoint interval of compute_block_hutch


def _hvp_fwd_over_rev(f_scalar, point, probe_stack):
    """Hv for every probe in probe_stack (a tuple of [B, ...] tensors, one per active plane).
    Returns a tuple of [B, ...] tensors.  Raises if torch.func cannot trace f_scalar, which is the
    caller's signal to fall back."""
    from torch.func import grad as _fgrad, jvp as _jvp, vmap as _vmap
    # Warm any lazily built constants (MS-SSIM caches its Gaussian window and level weights on first use)
    # OUTSIDE the transform.  Built inside one, they are captured as traced tensors and every later call
    # reuses them out of context, which surfaces as "bad optional access" from the second probe batch on.
    with torch.no_grad():
        f_scalar(*point)
    if len(point) == 1:
        # Single-plane metrics (SSIM, MS-SSIM).  A scalar argnums keeps grad returning a bare tensor,
        # which vmap handles; a 1-tuple does not.  MS-SSIM's five-level pyramid additionally refuses to
        # batch at all -- vmap raises "bad optional access" for any width above 1, while SSIM batches to
        # 16 -- so an unbatched loop is the fallback.  It still takes the jvp-through-grad route, which is
        # ~6x the double backward on its own; only the extra batching gain is lost.
        gfun = _fgrad(f_scalar, argnums=0)
        try:
            out = _vmap(lambda tg: _jvp(gfun, (point[0],), (tg,))[1])(probe_stack[0])
        except RuntimeError:
            out = torch.stack([_jvp(gfun, (point[0],), (tg,))[1] for tg in probe_stack[0]])
        return (out,)
    gfun = _fgrad(f_scalar, argnums=tuple(range(len(point))))
    def one(*tangents):
        return _jvp(gfun, tuple(point), tuple(tangents))[1]
    return _vmap(one)(*probe_stack)


def _draw_probe(m, seed):
    if PROBE_DEVICE == 'cuda' and torch.cuda.is_available():
        g = torch.Generator(device='cuda').manual_seed(int(seed))
        return torch.randn(m, generator=g, device='cuda')
    return torch.randn(m, generator=torch.Generator(device='cpu').manual_seed(int(seed)))


class _ProbePrefetch:
    def __init__(self, m, seed_of, n_total, batch, start=0):
        import queue, threading
        self.m, self.seed_of, self.n, self.batch = m, seed_of, n_total, batch
        self.q = queue.Queue(maxsize=2)
        self.stop = False
        self.t = threading.Thread(target=self._run, args=(start,), daemon=True)
        self.t.start()

    def _draw(self, done, b):
        return torch.stack([_draw_probe(self.m, self.seed_of(done + j)) for j in range(b)])

    def _run(self, done):
        while done < self.n and not self.stop:
            b = min(self.batch, self.n - done)
            self.q.put((done, self._draw(done, b)))
            done += b

    def get(self, done, b):
        """The probe block for indices done .. done+b-1.  If the worker is out of step (the batch was
        halved after an OOM), it is restarted from the next block and this one is drawn inline."""
        try:
            d0, v = self.q.get(timeout=600)
        except Exception:
            d0, v = -1, None
        if d0 == done and v.shape[0] >= b:
            if v.shape[0] != b:
                self._restart(done + b, b)
            return v[:b]
        self._restart(done + b, b)
        return self._draw(done, b)

    def _restart(self, start, batch):
        self.stop = True
        try:
            while True:
                self.q.get_nowait()
        except Exception:
            pass
        self.__init__(self.m, self.seed_of, self.n, batch, start)


def _lpips_net(net='vgg'):
    """The VGG object is q_utils.lpips_obj (shared with the evaluator); Alex (literature
    protocol, 3.6) is built once per process on the same device."""
    if net == 'vgg':
        from Common.utils.q_utils import lpips_obj
        return lpips_obj
    if net not in _LPIPS_NETS:
        import lpips as lpips_pkg
        from Common.utils.q_utils import lpips_obj
        dev = next(lpips_obj.parameters()).device
        _LPIPS_NETS[net] = lpips_pkg.LPIPS(net=net, verbose=False).to(dev).eval()
    return _LPIPS_NETS[net]


def lpips_alex_residual(in0, in1, parts=False):
    return lpips_residual(in0, in1, net='alex', parts=parts)


def lpips_residual(in0, in1, net='vgg', parts=False):
    """r(in1) with ||r||^2 == lpips_obj(in0, in1) exactly.  Inputs in [-1, 1].

    LPIPS sums, over layers, the spatial mean of  sum_c w_lc (f0 - f1)_lc^2  with the
    released w_lc all non-negative, so it is literally ||r||^2 for
    r_{l,c,hw} = sqrt(w_lc / (H_l W_l)) (f0 - f1)_{l,c,hw}.
    """
    import lpips as lpips_pkg
    o = _lpips_net(net)
    outs0 = o.net.forward(o.scaling_layer(in0))
    outs1 = o.net.forward(o.scaling_layer(in1))
    parts_ = []
    for kk in range(o.L):
        d = lpips_pkg.normalize_tensor(outs0[kk]) - lpips_pkg.normalize_tensor(outs1[kk])
        w = o.lins[kk].model[1].weight                     # [1, C, 1, 1]
        hw = d.shape[2] * d.shape[3]
        parts_.append((d * torch.sqrt(w.clamp_min(0) / hw)).reshape(-1))
    return parts_ if parts else torch.cat(parts_)


def lpips_lazy_parts(in0, in1, net='vgg'):
    """Large-input form of lpips_residual: group l = layer l, rebuilt through slices 1..l+1 of the
    network on demand.  The reference features are normalised once and parked on the CPU."""
    import lpips as lpips_pkg
    o = _lpips_net(net)
    slices = [getattr(o.net, f'slice{k}') for k in range(1, o.L + 1)]
    with torch.no_grad():
        outs0 = o.net.forward(o.scaling_layer(in0))
        n0 = [lpips_pkg.normalize_tensor(t).cpu().pin_memory() for t in outs0]
    sizes = [[t.numel()] for t in outs0]
    del outs0

    def build(l):
        h = o.scaling_layer(in1)
        for sl in slices[:l + 1]:
            h = sl(h)
        d = n0[l].to(h.device, non_blocking=True) - lpips_pkg.normalize_tensor(h)
        w = o.lins[l].model[1].weight
        hw = d.shape[2] * d.shape[3]
        return [(d * torch.sqrt(w.clamp_min(0) / hw)).reshape(-1)]
    return _LazyParts(sizes, build)


def dists_residual(in0, in1, parts=False):
    """r(in1) with ||r||^2 == DISTS(in0, in1) exactly.  Inputs are RGB in [0, 1].

    DISTS_pt normalises alpha/beta by their sum, so with S1, S2 the released similarity
    ratios the score is  sum_k alpha_k (1 - S1_k) + beta_k (1 - S2_k), and each term is a
    square:

        1 - S1 = (mu_x - mu_y)^2 / (mu_x^2 + mu_y^2 + c1)
        1 - S2 = E[((x - mu_x) - (y - mu_y))^2] / (var_x + var_y + c2)

    the second because var_x + var_y - 2 cov = E[((x - mu_x) - (y - mu_y))^2].  The released
    alpha and beta are all non-negative, so the square roots are real.  Both vanish at
    in1 == in0, which is what makes H = 2 J^T J exact there.
    """
    from Common.utils.q_utils import D as o
    if next(o.parameters(), torch.zeros(1)).device != in0.device:
        o.to(in0.device)                    # q_utils leaves the model on the CPU
    c1 = c2 = 1e-6
    f0 = o.forward_once(in0)
    f1 = o.forward_once(in1)
    w = o.alpha.sum() + o.beta.sum()
    alpha = torch.split(o.alpha / w, o.chns, dim=1)
    beta = torch.split(o.beta / w, o.chns, dim=1)
    parts_ = []
    for k in range(len(o.chns)):
        # the convolutions may run in fp16, but c1/c2 are 1e-6 and the feature variances
        # underflow there, so every pooled statistic and division is done in fp32
        x, y = f0[k].float(), f1[k].float()
        mx = x.mean([2, 3], keepdim=True)
        my = y.mean([2, 3], keepdim=True)
        parts_.append((torch.sqrt(alpha[k].float().clamp_min(0)) * (mx - my)
                       / torch.sqrt(mx ** 2 + my ** 2 + c1)).reshape(-1))

        vx = ((x - mx) ** 2).mean([2, 3], keepdim=True)
        vy = ((y - my) ** 2).mean([2, 3], keepdim=True)
        hw = x.shape[2] * x.shape[3]
        parts_.append((torch.sqrt(beta[k].float().clamp_min(0) / hw) * ((x - mx) - (y - my))
                       / torch.sqrt(vx + vy + c2)).reshape(-1))
    return parts_ if parts else torch.cat(parts_)


def dists_lazy_parts(in0, in1):
    """Large-input form of dists_residual: group k = the two parts of stage k (k = 0 is the input
    itself), rebuilt through stages 1..k on demand.  Reference statistics are formed once without
    a graph; (x - mu_x) is parked on the CPU and moved in per group.  Same arithmetic as above."""
    from Common.utils.q_utils import D as o
    if next(o.parameters(), torch.zeros(1)).device != in0.device:
        o.to(in0.device)
    c1 = c2 = 1e-6
    w = o.alpha.sum() + o.beta.sum()
    alpha = torch.split(o.alpha / w, o.chns, dim=1)
    beta = torch.split(o.beta / w, o.chns, dim=1)
    stages = [o.stage1, o.stage2, o.stage3, o.stage4, o.stage5]
    with torch.no_grad():
        f0 = o.forward_once(in0)
        mxs, vxs, dxs, sizes = [], [], [], []
        for k in range(len(o.chns)):
            x = f0[k].float(); mx = x.mean([2, 3], keepdim=True); dx = x - mx
            mxs.append(mx); vxs.append((dx ** 2).mean([2, 3], keepdim=True)); dxs.append(dx.cpu().pin_memory())
            sizes.append([mx.numel(), x.numel()])
        del f0, x, dx

    def build(k):
        if k == 0:
            y = in1
        else:
            h = (in1 - o.mean) / o.std
            for st in stages[:k]:
                h = st(h)
            y = h
        y = y.float()
        mx, vx = mxs[k], vxs[k]
        my = y.mean([2, 3], keepdim=True)
        p1 = (torch.sqrt(alpha[k].float().clamp_min(0)) * (mx - my) / torch.sqrt(mx ** 2 + my ** 2 + c1)).reshape(-1)
        vy = ((y - my) ** 2).mean([2, 3], keepdim=True)
        hw = y.shape[2] * y.shape[3]
        p2 = (torch.sqrt(beta[k].float().clamp_min(0) / hw) * (dxs[k].to(y.device, non_blocking=True) - (y - my))
              / torch.sqrt(vx + vy + c2)).reshape(-1)
        return [p1, p2]
    return _LazyParts(sizes, build)


def wd_lazy_parts(in0, in1, log2_sigma=None):
    """Large-input form of wd_residual: group j = backbone feature j (the normalised input at scale 0,
    or VGG slice k at pyramid scale s), rebuilt through the lowpass pyramid and slices 1..k on demand.
    The reference features and their level statistics are formed once without a graph and parked on
    the CPU; the sigma weights (independent of in1) are formed once.  Same arithmetic and part order
    as wd_residual, so J^T v is identical up to float summation order."""
    import torch.nn.functional as Fn
    o = _wd_model()
    bb, wdf = o.feature_backbone, o.wasserstein_distortion_feature
    if log2_sigma is None:
        log2_sigma = torch.zeros_like(in0[:, 0:1]) + WD_LOG2_SIGMA
    refs, scs, sizes = [], [], []
    with torch.no_grad():
        for fgt in bb(in0, num_scales=3):
            ls = Fn.interpolate(log2_sigma, size=fgt.shape[-2:], mode='bilinear', antialias=True)
            lr = (np.log2(log2_sigma.shape[-2] / fgt.shape[-2])
                  + np.log2(log2_sigma.shape[-1] / fgt.shape[-1])) / 2.0
            ls = Fn.relu(ls - lr)
            mb, vb = wdf.multi_level_stats(fgt)
            sb = [torch.sqrt(torch.clamp(v, min=1e-8)) for v in vb]
            nums = [fgt.numel()] + [m.numel() for m in mb]
            sc = []
            for i in range(wdf.num_levels + 1):
                w = Fn.relu(1 - torch.abs(ls - i))
                if i > 0:
                    ls = wdf.lowpass(ls, stride=2)
                sc.append(torch.sqrt(w.clamp_min(0) / nums[i]))
            r = [fgt.cpu().pin_memory()]
            sz = [fgt.numel()]
            for i in range(wdf.num_levels):
                r += [mb[i].cpu().pin_memory(), sb[i].cpu().pin_memory()]
                sz += [mb[i].numel(), sb[i].numel()]
            refs.append(r); scs.append(sc); sizes.append(sz)
            del mb, vb, sb
    K = len(bb.valid_slices)

    def build(j):
        x = (in1 - bb.mean) / bb.std
        if j == 0:
            fp = x
        else:
            s, k = divmod(j - 1, K)
            for _ in range(s):
                x = bb.lowpass(x, stride=2)
            for kk in range(1, k + 2):
                x = getattr(bb, f'slice{kk}')(x)
            fp = x
        dev = fp.device
        ma, va = wdf.multi_level_stats(fp)
        r, sc = refs[j], scs[j]
        parts = [(sc[0] * (fp - r[0].to(dev, non_blocking=True))).reshape(-1)]
        for i in range(wdf.num_levels):
            sa = torch.sqrt(torch.clamp(va[i], min=1e-8))
            parts.append((sc[i + 1] * (ma[i] - r[1 + 2 * i].to(dev, non_blocking=True))).reshape(-1))
            parts.append((sc[i + 1] * (sa - r[2 + 2 * i].to(dev, non_blocking=True))).reshape(-1))
        return parts
    return _LazyParts(sizes, build)


def _planes_to_dists_input(y, u, v):
    hw = y.shape[-2:]
    return yuv_to_rgb(y, upsample_chroma(u, hw), upsample_chroma(v, hw)) / 255.0


def _planes_to_lpips_input(y, u, v):
    hw = y.shape[-2:]
    rgb = yuv_to_rgb(y, upsample_chroma(u, hw), upsample_chroma(v, hw))
    return 2.0 * (rgb / 255.0 - 0.5)


def _gauss_newton_diag(planes, n_probes, seed=0, jitter=0.0, verbose=True, metric='LPIPS',
                       batch=4, amp=False, amp_scale=1024.0, amp_dtype='bf16'):
    """diag(H) = 2 E_v[(J^T v)^2] for a metric written as a vanishing sum of squares.

    At xhat = x the residual vanishes, so H = 2 J^T J exactly and the estimate is
    unbiased, non-negative by construction, and has relative standard deviation
    sqrt(2 / n) per sample regardless of how much off-diagonal mass H carries.  For LPIPS
    that matters: the measured diagonal dominance is 0.004, which is exactly the regime
    where the Hutchinson estimator's variance blows up.
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    _m = metric.upper()
    to_in, resid = ((_planes_to_lpips_input, lpips_residual) if _m == 'LPIPS' else
                    (_planes_to_lpips_input, lpips_alex_residual) if _m == 'LPIPS_ALEX' else
                    (_planes_to_dists_input, wd_residual) if _m == 'WD' else
                    (_planes_to_dists_input, dists_residual))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True      # the shape never changes within a run
    ref = to_in(*planes)
    acc = [torch.zeros_like(p) for p in planes]

    # batch > 1 vmaps the backward over probes (one kernel launch for the group); amp runs the
    # VGG in fp16.  Both leave E[(J^T v)^2] alone, so they change speed and not the estimand.
    # fp16 gradients underflow, so v is scaled by amp_scale and the square divided by its square
    # -- exact in real arithmetic, and it keeps the VJP inside the fp16 range.
    # bf16 by default: the DISTS residual divides by sqrt(var + 1e-6), whose backward factor
    # reaches ~1e8 and overflows fp16 no matter how the probe is scaled.  bf16 keeps fp32's
    # exponent range, so it cannot overflow, and Ampere and later run it on tensor cores.
    dt = torch.bfloat16 if amp_dtype == 'bf16' else torch.float16
    use_amp = amp and torch.cuda.is_available()
    if use_amp and dt is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        dt = torch.float16
    ctx = torch.autocast('cuda', dtype=dt) if use_amp else contextlib.nullcontext()
    sc = amp_scale if (use_amp and dt is torch.float16) else 1.0

    def one_round(point, nsamp, gen, tag=0):
        nonlocal batch
        cur = [p.detach().clone().requires_grad_(True) for p in point]
        xin = to_in(*cur)
        chunked = _chunk_vjp(xin, resid)                  # large inputs: per-group VJPs, batch 1
        if chunked:
            batch = 1
            torch.backends.cudnn.benchmark = False         # benchmark workspaces cost GiB at 2K
        dev = cur[0].device
        with ctx:
            r = _lazy_parts(resid, ref, xin) if chunked else resid(ref, xin)
        m = r.numel()
        done = 0
        seed_of = lambda idx: (seed * 1000003 + tag * 7919 + idx) & 0x7FFFFFFF
        pf = _ProbePrefetch(m, seed_of, nsamp, batch) if PREFETCH else None
        while done < nsamp:
            b = min(batch, nsamp - done)
            # one generator per probe index, so probe t is the same tensor whatever `batch` is:
            # torch.randn(b, m) is NOT the concatenation of b draws of torch.randn(1, m), so
            # drawing the block in one call would silently make the result batch-dependent
            v = pf.get(done, b) if pf else torch.stack([_draw_probe(m, seed_of(done + j)) for j in range(b)])
            if not chunked:
                v = v.to(dev) * sc if sc != 1.0 else v.to(dev)
            elif v.is_cuda:
                v = v.cpu()                       # the lazy 2K path slices the probe from host memory
            s_try = sc
            while True:
                try:
                    if chunked:
                        gr = _vjp_chunked(r, cur, v[0], s_try)
                    elif batch > 1:
                        gr = torch.autograd.grad(r, cur, grad_outputs=v * (s_try / sc),
                                                 retain_graph=True, is_grads_batched=True)
                    else:
                        gr = torch.autograd.grad(r, cur, grad_outputs=v[0] * (s_try / sc),
                                                 retain_graph=True)
                except torch.cuda.OutOfMemoryError:
                    # the residual can be much larger for some metrics (WD is ~5x LPIPS);
                    # fall back to a smaller batch rather than failing the run
                    if batch == 1:
                        raise
                    batch = max(1, batch // 2)
                    torch.cuda.empty_cache()
                    v = v[:batch]
                    b = batch
                    continue
                # too large a scale overflows fp16 on the way back; too small underflows
                # the 1/sqrt(var + 1e-6) factor in the DISTS residual makes the backward
                # ~1e3x, which saturates fp16, so the scale has to be allowed below 1
                if all(torch.isfinite(x).all() for x in gr) or s_try <= 2.0 ** -14:
                    break
                s_try /= 4.0
            for i in range(3):
                g2 = (gr[i].float() ** 2) / (s_try * s_try)
                acc[i] += g2.sum(0) if batch > 1 else g2
            done += b
            if verbose and done % 64 == 0:
                print(f'      vjp {done}/{nsamp}', flush=True)

    if jitter <= 0.0:
        one_round(planes, n_probes, g)
        total = n_probes
    else:
        # average the Gauss-Newton diagonal over points the encoder actually visits
        rounds = max(1, n_probes // 16)
        per = max(1, n_probes // rounds)
        for rd in range(rounds):
            pt = tuple(p + jitter * torch.randn(p.shape, generator=g).to(p.device) for p in planes)
            one_round(pt, per, g, tag=rd + 1)
        total = rounds * per
    return [(2.0 * a / total).squeeze(0).squeeze(0).cpu().numpy().astype(np.float64) for a in acc]


# --------------------------------------------------------------------------------------
# Column-norm estimator
# --------------------------------------------------------------------------------------
def _colnorm_diag(func, planes, active, n_probes, seed=0, verbose=True, jitter=0.0):
    """||H_{:,j}||, the norm of each column of the Hessian, instead of its diagonal entry.

    With w = H v and v zero-mean, identity covariance, E[w w^T] = H^T H, so the element-wise
    second moment of w estimates the squared column norms:

        h_j = sqrt( (1/m) sum_i (H v_i)_j^2 ) = || H_{:,j} ||.

    The Bekas diagonal estimator only sees H_jj, so it is blind to how much pixel j moves the
    *other* pixels; its accuracy degrades with the diagonal dominance of H.  Measured dominance
    here is 1.02 for SSIM but 0.004 for LPIPS, which is the regime this is meant for.  Two
    incidental benefits: the estimate is non-negative by construction, so nothing gets clipped
    to zero, and being a mean of squares its relative variance is ~sqrt(2/m) regardless of how
    much off-diagonal mass H carries.
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    flat_shapes = [p.shape for p in planes]
    idx = [i for i, nm in enumerate(('Y', 'Cb', 'Cr')) if nm in active]

    def f_of_active(*args):
        cur = list(planes)
        for k, i in enumerate(idx):
            cur[i] = args[k]
        return func(planes, tuple(cur))

    acc = [torch.zeros_like(planes[i]) for i in idx]
    rounds = 1 if jitter <= 0 else max(1, n_probes // ROUND_SIZE)
    per = n_probes // rounds
    for rd in range(rounds):
        base = tuple(planes[i].detach().clone() for i in idx)
        if jitter > 0:
            base = tuple(b + jitter * torch.randn(b.shape, generator=g).to(b.device) for b in base)
        for t in range(per):
            probes = tuple((torch.randint(0, 2, flat_shapes[i], generator=g).float() * 2 - 1).to(planes[0].device)
                           for i in idx)
            hv = torch.autograd.functional.hvp(f_of_active, base, probes)[1]
            if not isinstance(hv, tuple):
                hv = (hv,)
            for k in range(len(idx)):
                acc[k] += hv[k].detach() ** 2
            if verbose and (rd * per + t + 1) % 16 == 0:
                print(f'      colnorm probe {rd * per + t + 1}/{rounds * per}', flush=True)
    total = rounds * per

    out = [None, None, None]
    for k, i in enumerate(idx):
        out[i] = torch.sqrt(acc[k] / total).squeeze(0).squeeze(0).cpu().numpy().astype(np.float64)
    return out


# --------------------------------------------------------------------------------------
# Block Hessians (non-diagonal), from the same Gauss-Newton sketch
# --------------------------------------------------------------------------------------
def compute_block_gn(y, u, v, n_probes=256, seed=0, jitter=0.0, blk=8, verbose=True,
                     metric='LPIPS', batch=4):
    """Per-block Hessians H_b for a sum-of-squares metric, at no extra sampling cost.

    H = 2 J^T J at xhat = x, and for u = J^T v with v white, E[u_b u_b^T] = H_b / 2.  The
    diagonal estimator throws away everything but u .* u; keeping the per-block outer products
    costs one batched rank-1 update per probe and no extra passes.

    Returns a list of three arrays [nby, nbx, blk*blk, blk*blk], row-major within a block
    (index = r * blk + c), matching the encoder's traversal.
    """
    def to_t(a):
        return torch.from_numpy(np.ascontiguousarray(a)).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    planes = (to_t(y), to_t(u), to_t(v))
    _m = metric.upper()
    to_in, resid = ((_planes_to_lpips_input, lpips_residual) if _m == 'LPIPS' else
                    (_planes_to_lpips_input, lpips_alex_residual) if _m == 'LPIPS_ALEX' else
                    (_planes_to_dists_input, wd_residual) if _m == 'WD' else
                    (_planes_to_dists_input, dists_residual))
    ref = to_in(*planes)
    g = torch.Generator(device='cpu').manual_seed(seed)

    # 2K inputs: the accumulators (n_tiles x 64 x 64 per plane, ~1 GB) live on the CPU and the per-probe
    # rank-1 update is formed in tile chunks, so the GPU peak stays that of the lazy VJP itself.
    # accumulators on the CPU for large images whatever the map path: on a 44 GB card the WD block map at 2K
    # fits only without the ~1.8 GB of GPU accumulators + einsum temporaries (CARC, 7 Sept)
    big = planes[0].shape[-1] * planes[0].shape[-2] >= BLOCK_ACC_CPU_MIN_PIXELS
    acc = []
    for p in planes:
        hh, ww = p.shape[-2:]
        acc.append(torch.zeros((hh // blk) * (ww // blk), blk * blk, blk * blk,
                               device='cpu' if big else DEVICE, dtype=torch.float32))

    rounds = 1 if jitter <= 0 else max(1, n_probes // ROUND_SIZE)
    per = n_probes // rounds
    for rd in range(rounds):
        pt = planes
        if jitter > 0:
            pt = tuple(p + jitter * torch.randn(p.shape, generator=g).to(p.device) for p in planes)
        cur = [p.detach().clone().requires_grad_(True) for p in pt]
        xin = to_in(*cur)
        chunked = _chunk_vjp(xin, resid)                  # large inputs: per-group VJPs, batch 1
        if chunked:
            torch.backends.cudnn.benchmark = False         # benchmark workspaces cost GiB at 2K
        dev = cur[0].device
        r = _lazy_parts(resid, ref, xin) if chunked else resid(ref, xin)
        m = r.numel()
        done = 0
        bat = 1 if chunked else batch
        seed_of = lambda idx: (seed * 1000003 + rd * 7919 + idx) & 0x7FFFFFFF
        pf = _ProbePrefetch(m, seed_of, per, bat) if PREFETCH else None
        while done < per:
            b = min(bat, per - done)
            try:
                # probe t is seeded by its index, so `batch` cannot change the result.  The
                # draw is inside the try because for WD the probe block alone is gigabytes.
                vv = pf.get(done, b) if pf else torch.stack([_draw_probe(m, seed_of(done + j)) for j in range(b)])
                if chunked:
                    gr = _vjp_chunked(r, cur, vv[0].cpu() if vv.is_cuda else vv[0])
                elif b > 1:
                    vv = vv.to(dev)
                    gr = torch.autograd.grad(r, cur, grad_outputs=vv, retain_graph=True,
                                             is_grads_batched=True)
                else:
                    gr = torch.autograd.grad(r, cur, grad_outputs=vv[0].to(dev), retain_graph=True)
            except torch.cuda.OutOfMemoryError:
                if bat == 1:
                    raise
                bat = max(1, bat // 2)
                torch.cuda.empty_cache()
                continue
            for i in range(3):
                gi = gr[i][:, 0, 0] if b > 1 else gr[i][0, 0].unsqueeze(0)       # [b,hh,ww]
                hh, ww = gi.shape[-2:]
                ub = gi.unfold(1, blk, blk).unfold(2, blk, blk)      # [b,nby,nbx,blk,blk]
                ub = ub.reshape(gi.shape[0], (hh // blk) * (ww // blk), blk * blk)
                if big:
                    for c in range(0, ub.shape[1], 8192):
                        acc[i][c:c + 8192] += torch.einsum('bnp,bnq->npq', ub[:, c:c + 8192], ub[:, c:c + 8192]).cpu()
                else:
                    acc[i] += torch.einsum('bnp,bnq->npq', ub, ub)
            done += b
            if verbose and done % 32 == 0:
                print(f'      block-vjp {rd * per + done}/{rounds * per}', flush=True)

    out = []
    for i, p in enumerate(planes):
        hh, ww = p.shape[-2:]
        M = (2.0 * acc[i] / (rounds * per)).cpu().numpy().astype(np.float32)
        M = 0.5 * (M + np.transpose(M, (0, 2, 1)))                        # symmetrise
        out.append(M.reshape(hh // blk, ww // blk, blk * blk, blk * blk))
    return out


def _make_psd(M, mode='cholesky', iters=12):
    """Repair a symmetric but indefinite block-Hessian estimate, batched over blocks.

    'eigh'      exact projection, U max(L,0) U^T.  Correct and by far the most expensive:
                35.6 s per 768x512 picture at 8x8 tiles.
    NOTE a ridge H + d I was tried and discarded: d must be about 0.094 lambda_max, which
    dumps that much trace into e^T H e and inflated the block distortion by 28-173% against
    the exact projection - far worse than doing nothing (0.86-0.98).  Fast but not usable.
    'shrink'    interpolate towards diag(M) until PSD; adds no trace.
    'none'      leave it alone.  DEFAULT.  On real coding errors the raw form never produced a
                negative block distortion (0 of 24576 at 64 probes) and the encoder clamps the
                scalar anyway; against the exact projection it moves the BD-rate by under 0.1
                points while saving 35.6 s per picture.  There is no guarantee, and at low probe
                counts negatives do appear (84 of 24576 at 16 probes), so revisit it there.
    """
    if mode == 'none':
        return M
    if mode == 'cholesky':
        raise ValueError('the ridge variant was measured to inflate the distortion by 28-173%; '
                         'use eigh, shrink or none')
    if mode == 'eigh':
        w, V = torch.linalg.eigh(M.double())
        return ((V * w.clamp_min(0.0).unsqueeze(1)) @ V.transpose(1, 2)).float()

    if mode == 'shrink':
        # M <- a M + (1-a) diag(M).  diag(M) is PSD, so some a makes the mix PSD, and unlike a
        # ridge this ADDS no trace: it interpolates towards the diagonal estimator instead.
        n = M.shape[-1]
        D = torch.diag_embed(torch.diagonal(M, dim1=1, dim2=2).clamp_min(0))
        lo = torch.zeros(M.shape[0], device=M.device, dtype=M.dtype)
        hi = torch.ones_like(lo)
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            a = mid.view(-1, 1, 1)
            ok = torch.linalg.cholesky_ex(a * M + (1 - a) * D).info == 0
            lo = torch.where(ok, mid, lo)      # feasible -> push a up
            hi = torch.where(ok, hi, mid)
        a = lo.view(-1, 1, 1)
        return a * M + (1 - a) * D

    n = M.shape[-1]
    eye = torch.eye(n, device=M.device, dtype=M.dtype)
    scale = torch.diagonal(M, dim1=1, dim2=2).abs().amax(1).clamp_min(1e-30)

    lo = torch.zeros_like(scale)
    hi = scale.clone()
    # grow hi until every block factorises
    for _ in range(iters):
        bad = torch.linalg.cholesky_ex(M + hi.view(-1, 1, 1) * eye).info > 0
        if not bool(bad.any()):
            break
        hi = torch.where(bad, hi * 4.0, hi)
    # bisect for the smallest ridge that works
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok = torch.linalg.cholesky_ex(M + mid.view(-1, 1, 1) * eye).info == 0
        hi = torch.where(ok, mid, hi)
        lo = torch.where(ok, lo, mid)
    return M + hi.view(-1, 1, 1) * eye


def compute_block_hutch(y, u, v, metric, n_probes=256, seed=0, jitter=0.0, blk=8, verbose=True,
                        psd='none', ckpt_path=None):
    """Per-block Hessians for ANY metric, from the same Hessian-vector products Hutchinson uses.

    For v zero-mean with identity covariance, E[(H v) v^T] = H, so restricting both factors to
    a block gives an unbiased estimate of that block of the Hessian:

        H_bb  =  E[ (H v)_b  v_b^T ].

    Unlike the LPIPS Gauss-Newton route this is not a Gram matrix, so a finite-sample estimate
    is neither symmetric nor PSD.  Both are repaired: symmetrise, then clamp the eigenvalues at
    zero, otherwise the encoder could be handed a negative distortion.

    ckpt_path: every BH_CKPT_SEC seconds the accumulators, probe counter and generator state are
    saved there (atomically), and a later call with the same arguments resumes from that point
    with the same probe sequence, so a run cut by a job time limit loses at most one interval.
    The file is removed once the map is complete.
    """
    def to_t(a):
        return torch.from_numpy(np.ascontiguousarray(a)).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    planes = (to_t(y), to_t(u), to_t(v))
    func, active = _make_metric(metric)
    idx = [i for i, nm in enumerate(('Y', 'Cb', 'Cr')) if nm in active]

    def f_of_active(*args):
        cur = list(planes)
        for k, i in enumerate(idx):
            cur[i] = args[k]
        return func(planes, tuple(cur))

    g = torch.Generator(device='cpu').manual_seed(seed)
    acc = {}
    for i in idx:
        hh, ww = planes[i].shape[-2:]
        acc[i] = torch.zeros((hh // blk) * (ww // blk), blk * blk, blk * blk,
                             device=DEVICE, dtype=torch.float32)

    rounds = 1 if jitter <= 0 else max(1, n_probes // ROUND_SIZE)
    per = n_probes // rounds

    # ---- resume state ----
    start_rd, start_t, base_saved = 0, 0, None
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location='cpu')
            assert ck['n_probes'] == n_probes and ck['seed'] == seed and ck['jitter'] == jitter and ck['blk'] == blk
            for i in idx:
                acc[i].copy_(ck['acc'][i].to(DEVICE))
            g.set_state(ck['gen'])
            start_rd, start_t, base_saved = ck['rd'], ck['t'], ck.get('base')
            print(f'      resuming block-hvp at {start_rd * per + start_t}/{rounds * per} from {ckpt_path}', flush=True)
        except Exception as e:
            print(f'      checkpoint unusable ({type(e).__name__}: {e}); starting over', flush=True)
            start_rd, start_t, base_saved = 0, 0, None
    last_save = time.time()

    def save_ckpt(rd, t, base):
        nonlocal last_save
        tmp = ckpt_path + '.tmp'
        torch.save(dict(acc={i: acc[i].cpu() for i in idx}, gen=g.get_state(), rd=rd, t=t,
                        base=[b.cpu() for b in base] if jitter > 0 else None,
                        n_probes=n_probes, seed=seed, jitter=jitter, blk=blk), tmp)
        os.replace(tmp, ckpt_path)
        last_save = time.time()
        print(f'      ckpt {rd * per + t}/{rounds * per}', flush=True)

    for rd in range(start_rd, rounds):
        if rd == start_rd and base_saved is not None:
            base = tuple(b.to(DEVICE) for b in base_saved)
        else:
            base = tuple(planes[i].detach().clone() for i in idx)
            if jitter > 0:
                base = tuple(b + jitter * torch.randn(b.shape, generator=g).to(b.device) for b in base)
        # as in _hvp_diag: only the second backward depends on the probe, so the first-order
        # graph is built once per round rather than once per probe
        def draw_b():
            return tuple((torch.randint(0, 2, planes[i].shape, generator=g).float() * 2 - 1).to(planes[0].device)
                         for i in idx)
        def accumulate(hv_k, pr_k, i):
            hh, ww = planes[i].shape[-2:]
            nt = (hh // blk) * (ww // blk)
            a = hv_k.reshape(-1, 1, hh, ww).unfold(2, blk, blk).unfold(3, blk, blk).reshape(-1, nt, blk * blk)
            b = pr_k.reshape(-1, 1, hh, ww).unfold(2, blk, blk).unfold(3, blk, blk).reshape(-1, nt, blk * blk)
            acc[i] += torch.einsum('bti,btj->tij', a, b)
        # same probe sequence whichever path runs (see the diagonal loop)
        cur = grads = None
        t = start_t if rd == start_rd else 0
        while t < per:
            nb = min(HVP_BATCH, per - t) if HVP_MODE == 'fwd' else 1
            ps = [draw_b() for _ in range(nb)]
            done = False
            if HVP_MODE == 'fwd':
                try:
                    stack = tuple(torch.stack([q[k] for q in ps]) for k in range(len(idx)))
                    hv = _hvp_fwd_over_rev(lambda *a: f_of_active(*a), tuple(base), stack)
                    for k, i in enumerate(idx):
                        accumulate(hv[k].detach(), stack[k], i)
                    done = True
                except Exception as e:
                    if verbose and t == 0:
                        print(f'      forward-over-reverse unavailable ({type(e).__name__}), double backward',
                              flush=True)
            if not done:
                if grads is None:
                    cur = tuple(b.detach().clone().requires_grad_(True) for b in base)
                    grads = torch.autograd.grad(f_of_active(*cur), cur, create_graph=True)
                for q in ps:
                    hv = torch.autograd.grad(grads, cur, grad_outputs=q, retain_graph=True)
                    for k, i in enumerate(idx):
                        accumulate(hv[k].detach(), q[k], i)
            t += nb
            if verbose and t % 32 == 0:
                print(f'      block-hvp {rd * per + t}/{rounds * per}', flush=True)
            if ckpt_path and t < per and time.time() - last_save >= BH_CKPT_SEC:
                save_ckpt(rd, t, base)
        del grads, cur
    total = rounds * per
    if ckpt_path and os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    out = []
    for i in range(3):
        hh, ww = planes[i].shape[-2:]
        nb = (hh // blk) * (ww // blk)
        if i not in idx:
            # metric does not see this plane: identity blocks leave its RDO untouched
            M = torch.eye(blk * blk, device=DEVICE).expand(nb, blk * blk, blk * blk).clone()
        else:
            M = acc[i] / total
            M = 0.5 * (M + M.transpose(1, 2))
            M = _make_psd(M, psd).float()
        out.append(M.cpu().numpy().reshape(hh // blk, ww // blk, blk * blk, blk * blk))
    return out


BH_MAGIC = 0x31484257     # 'WBH1'


def write_block_dat(path, mats, blk):
    """magic, blk, then per plane: nby, nbx, float32[nby*nbx*blk^2*blk^2] (raw H_b)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(struct.pack('<II', BH_MAGIC, blk))
        for M in mats:
            f.write(struct.pack('<II', M.shape[0], M.shape[1]))
            f.write(np.ascontiguousarray(M, dtype='<f4').tobytes())


# --------------------------------------------------------------------------------------
# Hessian diagonal
# --------------------------------------------------------------------------------------
def _hvp_diag(func, planes, active, n_probes, seed=0, verbose=True, jitter=0.0):
    """Bekas/Hutchinson estimator of diag(H) with Rademacher probes.

    func     : (ref_planes, planes) -> scalar
    planes   : tuple of 3 tensors [1,1,h,w], the point the Hessian is evaluated at
    active   : which planes the probes touch; the others stay exactly zero
    """
    g = torch.Generator(device='cpu').manual_seed(seed)

    flat_shapes = [p.shape for p in planes]
    idx = [i for i, nm in enumerate(('Y', 'Cb', 'Cr')) if nm in active]

    def f_of_active(*args):
        cur = list(planes)
        for k, i in enumerate(idx):
            cur[i] = args[k]
        return func(planes, tuple(cur))

    acc = [torch.zeros_like(planes[i]) for i in idx]
    # jitter > 0 averages the Hessian over points at the operating distortion level instead of
    # taking it at the source, where the second-order model is exact but least representative
    rounds = 1 if jitter <= 0 else max(1, n_probes // ROUND_SIZE)
    per = n_probes // rounds
    for rd in range(rounds):
        base = tuple(planes[i].detach().clone() for i in idx)
        if jitter > 0:
            base = tuple(b + jitter * torch.randn(b.shape, generator=g).to(b.device) for b in base)
        # torch.autograd.functional.hvp redoes the forward and the first backward on every
        # call, but only the second backward depends on the probe.  Building the first-order
        # graph once per round and reusing it halves the cost of the Hutchinson metrics.
        def draw():
            return [(torch.randint(0, 2, flat_shapes[i], generator=g).float() * 2.0 - 1.0).to(planes[0].device)
                    for i in idx]
        # The probes are drawn once per batch and handed to whichever path runs, so the sequence the
        # estimator sees is exactly the one the double-backward path drew: one draw() per probe, in order.
        # A fallback therefore reuses the probes of the batch that failed instead of advancing the stream.
        cur = grads = None
        t = 0
        while t < per:
            nb = min(HVP_BATCH, per - t) if HVP_MODE == 'fwd' else 1
            ps = [draw() for _ in range(nb)]
            done = False
            if HVP_MODE == 'fwd':
                try:
                    stack = tuple(torch.stack([q[k] for q in ps]) for k in range(len(idx)))
                    hv = _hvp_fwd_over_rev(lambda *a: f_of_active(*a), tuple(base), stack)
                    for k in range(len(idx)):
                        acc[k] += (hv[k].detach() * stack[k]).sum(0)
                    done = True
                except Exception as e:
                    if verbose and t == 0:
                        print(f'      forward-over-reverse unavailable ({type(e).__name__}), double backward',
                              flush=True)
            if not done:
                if grads is None:
                    cur = tuple(b.detach().clone().requires_grad_(True) for b in base)
                    grads = torch.autograd.grad(f_of_active(*cur), cur, create_graph=True)
                for q in ps:
                    hv = torch.autograd.grad(grads, cur, grad_outputs=tuple(q), retain_graph=True)
                    for k in range(len(idx)):
                        acc[k] += hv[k].detach() * q[k]
            t += nb
            if verbose and t % 16 == 0:
                print(f'      probe {rd * per + t}/{rounds * per}', flush=True)
        del grads, cur
    n_probes = rounds * per

    out = [None, None, None]
    for k, i in enumerate(idx):
        out[i] = (acc[k] / float(n_probes)).squeeze(0).squeeze(0).cpu().numpy().astype(np.float64)
    return out


def _clip(raw, clip_pct=99.5):
    """raw diag(H) -> non-negative, outliers capped.  None if the metric ignores the plane."""
    if raw is None:
        return None
    w = np.maximum(raw, 0.0)                       # diag(H) >= 0 for a metric minimised at x
    if not np.any(w > 0):
        return None
    hi = np.percentile(w, clip_pct)
    if hi > 0:
        w = np.minimum(w, hi)                      # a handful of huge weights would dominate
    return w


def compute_raw_diag(y, u, v, metric='MS_SSIM', n_probes=64, seed=0, verbose=True,
                     estimator='auto', jitter=0.0, batch=4):
    """The expensive part: an estimate of diag(H), before any normalisation.

    Split out from compute_importance so that the post-processing knobs (clip percentile,
    plane normalisation) can be swept without paying for the Hessian again.
    """
    def to_t(a):
        return torch.from_numpy(np.ascontiguousarray(a)).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    planes = (to_t(y), to_t(u), to_t(v))

    if estimator == 'auto':
        estimator = 'gn' if metric.upper() in ('LPIPS', 'LPIPS_ALEX', 'DISTS', 'WD') else 'hutch'
    if estimator == 'gn':
        if metric.upper() not in ('LPIPS', 'LPIPS_ALEX', 'DISTS', 'WD'):
            raise ValueError('the Gauss-Newton path needs a vanishing sum-of-squares form '
                             '(LPIPS, DISTS, WD)')
        return _gauss_newton_diag(planes, n_probes, seed=seed, jitter=jitter, verbose=verbose,
                                  metric=metric, batch=batch)
    func, active = _make_metric(metric)
    if estimator == 'colnorm':
        return _colnorm_diag(func, planes, active, n_probes, seed=seed, verbose=verbose, jitter=jitter)
    return _hvp_diag(func, planes, active, n_probes, seed=seed, verbose=verbose, jitter=jitter)


def postprocess_raw(raw, shapes, plane_norm='per_plane', clip_pct=99.5):
    """raw diag(H) -> the three weight planes the encoder reads."""
    clipped = [_clip(r, clip_pct) for r in raw]

    scale = 1.0
    if plane_norm == 'global':
        live = [c for c in clipped if c is not None]
        tot = sum(float(c.sum()) for c in live)
        cnt = sum(c.size for c in live)
        scale = cnt / tot if tot > 0 else 1.0

    out = []
    for i in range(3):
        c = clipped[i]
        if c is None:                              # metric does not see this plane -> leave RDO alone
            out.append(np.ones(shapes[i], dtype=np.float32))
        elif plane_norm == 'global':
            out.append((c * scale).astype(np.float32))
        else:
            out.append((c / c.mean()).astype(np.float32))
    return tuple(out)


def compute_importance(y, u, v, metric='MS_SSIM', n_probes=64, seed=0, verbose=True,
                       estimator='auto', plane_norm='per_plane', clip_pct=99.5, jitter=0.0):
    """y: [H,W], u/v: [H/2,W/2] float arrays in 0..255.  Returns (wy, wu, wv).

    estimator   'auto' picks Gauss-Newton for LPIPS and Hutchinson otherwise, 'gn' and
                'hutch' force one.
    plane_norm  'per_plane' scales every plane to mean one, which keeps VTM's tuned
                luma/chroma lambda balance.  'global' uses one scale for all three planes,
                so the metric's own cross-plane balance survives.
    jitter      standard deviation, in sample units, of the noise the Hessian is averaged
                over; 0 evaluates it at the source (Gauss-Newton path only).
    """
    raw = compute_raw_diag(y, u, v, metric, n_probes, seed, verbose, estimator, jitter)
    shapes = (y.shape, u.shape, v.shape)
    return postprocess_raw(raw, shapes, plane_norm, clip_pct)


# --------------------------------------------------------------------------------------
# .dat container (same layout the HM weighted-SSE fork uses)
# --------------------------------------------------------------------------------------
def write_dat(path, wy, wu, wv):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', 1))
        for p in (wy, wu, wv):
            f.write(np.ascontiguousarray(p, dtype='<f4').tobytes())


def read_dat(path, shapes):
    with open(path, 'rb') as f:
        n = struct.unpack('<I', f.read(4))[0]
        planes = []
        for sh in shapes:
            cnt = sh[0] * sh[1]
            planes.append(np.frombuffer(f.read(4 * cnt), dtype='<f4').reshape(sh).copy())
    return n, planes


# --------------------------------------------------------------------------------------
# yuv420p io
# --------------------------------------------------------------------------------------
def read_yuv420(path, width, height, frame=0):
    fsz = width * height * 3 // 2
    with open(path, 'rb') as f:
        f.seek(frame * fsz)
        buf = f.read(fsz)
    y = np.frombuffer(buf[: width * height], dtype=np.uint8).reshape(height, width).astype(np.float64)
    off = width * height
    cw, ch = width // 2, height // 2
    u = np.frombuffer(buf[off: off + cw * ch], dtype=np.uint8).reshape(ch, cw).astype(np.float64)
    off += cw * ch
    v = np.frombuffer(buf[off: off + cw * ch], dtype=np.uint8).reshape(ch, cw).astype(np.float64)
    return y, u, v


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description='Hessian-derived importance map for weighted-MSE RDO')
    ap.add_argument('--yuv', required=True)
    ap.add_argument('--width', type=int, required=True)
    ap.add_argument('--height', type=int, required=True)
    ap.add_argument('--metric', default='MS_SSIM', choices=METRICS)
    ap.add_argument('--probes', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--estimator', default='auto', choices=['auto', 'gn', 'hutch', 'colnorm'])
    ap.add_argument('--plane-norm', default='per_plane', choices=['per_plane', 'global'])
    ap.add_argument('--clip-pct', type=float, default=99.5)
    ap.add_argument('--jitter', type=float, default=0.0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    y, u, v = read_yuv420(a.yuv, a.width, a.height)
    wy, wu, wv = compute_importance(y, u, v, a.metric, a.probes, a.seed,
                                    estimator=a.estimator, plane_norm=a.plane_norm,
                                    clip_pct=a.clip_pct, jitter=a.jitter)
    write_dat(a.out, wy, wu, wv)
    print(f'{a.out}: Y[min {wy.min():.3f} max {wy.max():.3f} mean {wy.mean():.3f}] '
          f'Cb[mean {wu.mean():.3f}] Cr[mean {wv.mean():.3f}]')


# --------------------------------------------------------------------------------------
# RGB-domain diagonal, for codecs that optimise in RGB (C3 / Cool-chic) rather than YUV
# --------------------------------------------------------------------------------------
def compute_rgb_diag(rgb01, metric='WD', n_probes=128, seed=0, jitter=0.0, verbose=False,
                     batch=4):
    """diag(H) of `metric` with respect to RGB pixels.

    rgb01 : [H, W, 3] float array in [0, 1]      ->  returns [H, W, 3] float64

    The VTM path differentiates w.r.t. Y/Cb/Cr because that is what the encoder codes.  C3
    optimises the RGB image directly, so the map it needs is the RGB-domain diagonal; going
    through YUV and back would fold the colour transform into the weights twice.
    """
    m = metric.upper()
    if m not in ('LPIPS', 'DISTS', 'WD'):
        raise ValueError('RGB diagonal is only wired for the Gauss-Newton metrics')
    resid = {'LPIPS': lambda a, b: lpips_residual(2 * a - 1, 2 * b - 1),
             'DISTS': dists_residual,
             'WD': wd_residual}[m]

    x = torch.from_numpy(np.ascontiguousarray(rgb01)).float().permute(2, 0, 1)[None].to(DEVICE)
    ref = x.detach().clone()
    g = torch.Generator(device='cpu').manual_seed(seed)
    acc = torch.zeros_like(x)

    rounds = 1 if jitter <= 0 else max(1, n_probes // ROUND_SIZE)
    per = n_probes // rounds
    for rd in range(rounds):
        pt = ref if jitter <= 0 else ref + jitter * torch.randn(
            ref.shape, generator=g).to(ref.device)
        cur = pt.detach().clone().requires_grad_(True)
        r = resid(ref, cur)
        mres = r.numel()
        done = 0
        b = batch
        while done < per:
            k = min(b, per - done)
            try:
                # the probe block itself can be gigabytes (WD's residual is ~235M entries),
                # so it has to be inside the backoff, not before it
                v = torch.stack([torch.randn(mres, generator=torch.Generator(device='cpu')
                                             .manual_seed((seed * 1000003 + rd * 7919
                                                           + done + j) & 0x7FFFFFFF))
                                 for j in range(k)]).to(r.device)
                if k > 1:
                    gr = torch.autograd.grad(r, cur, grad_outputs=v, retain_graph=True,
                                             is_grads_batched=True)[0]
                    acc += (gr.float() ** 2).sum(0)
                else:
                    gr = torch.autograd.grad(r, cur, grad_outputs=v[0], retain_graph=True)[0]
                    acc += gr.float() ** 2
            except torch.cuda.OutOfMemoryError:
                if b == 1:
                    raise
                b = max(1, b // 2)
                torch.cuda.empty_cache()
                continue
            done += k
            if verbose and done % 32 == 0:
                print(f'      rgb-vjp {done}/{per}', flush=True)
    out = (2.0 * acc / (rounds * per))[0].permute(1, 2, 0)
    return out.detach().cpu().numpy().astype(np.float64)


def compute_rgb_block(rgb01, metric='WD', n_probes=128, seed=0, jitter=0.0, blk=8,
                      verbose=False, batch=4):
    """Per-tile Hessians with respect to RGB pixels, for codecs that optimise in RGB.

    rgb01 : [H, W, 3] in [0, 1]  ->  [3, nby, nbx, blk*blk, blk*blk] float32

    Tiles are taken within each colour channel independently, matching what the VTM path
    does per plane.  Gauss-Newton, so each tile is PSD and symmetric by construction.
    """
    m = metric.upper()
    if m not in ('LPIPS', 'DISTS', 'WD'):
        raise ValueError('the RGB block estimator needs a Gauss-Newton metric')
    resid = {'LPIPS': lambda a, b: lpips_residual(2 * a - 1, 2 * b - 1),
             'DISTS': dists_residual,
             'WD': wd_residual}[m]

    x = torch.from_numpy(np.ascontiguousarray(rgb01)).float().permute(2, 0, 1)[None].to(DEVICE)
    ref = x.detach().clone()
    hh, ww = ref.shape[-2:]
    nb = (hh // blk) * (ww // blk)
    g = torch.Generator(device='cpu').manual_seed(seed)
    acc = torch.zeros(3, nb, blk * blk, blk * blk, device=DEVICE, dtype=torch.float32)

    rounds = 1 if jitter <= 0 else max(1, n_probes // ROUND_SIZE)
    per = n_probes // rounds
    for rd in range(rounds):
        pt = ref if jitter <= 0 else ref + jitter * torch.randn(
            ref.shape, generator=g).to(ref.device)
        cur = pt.detach().clone().requires_grad_(True)
        r = resid(ref, cur)
        mres = r.numel()
        done, bat = 0, batch
        while done < per:
            b = min(bat, per - done)
            try:
                v = torch.stack([torch.randn(mres, generator=torch.Generator(device='cpu')
                                             .manual_seed((seed * 1000003 + rd * 7919
                                                           + done + j) & 0x7FFFFFFF))
                                 for j in range(b)]).to(r.device)
                if b > 1:
                    gr = torch.autograd.grad(r, cur, grad_outputs=v, retain_graph=True,
                                             is_grads_batched=True)[0]          # [b,1,3,H,W]
                    gi = gr[:, 0]
                else:
                    gr = torch.autograd.grad(r, cur, grad_outputs=v[0], retain_graph=True)[0]
                    gi = gr                                                     # [1,3,H,W]
            except torch.cuda.OutOfMemoryError:
                if bat == 1:
                    raise
                bat = max(1, bat // 2)
                torch.cuda.empty_cache()
                continue
            # [b,3,H,W] -> [b,3,nb,blk*blk], then accumulate the per-tile outer products
            ub = gi.unfold(2, blk, blk).unfold(3, blk, blk)
            ub = ub.reshape(gi.shape[0], 3, nb, blk * blk)
            acc += torch.einsum('bcnp,bcnq->cnpq', ub, ub)
            done += b
            if verbose and done % 32 == 0:
                print(f'      rgb-block-vjp {done}/{per}', flush=True)

    out = (2.0 * acc / (rounds * per)).cpu().numpy().astype(np.float32)
    out = 0.5 * (out + np.transpose(out, (0, 1, 3, 2)))
    return out.reshape(3, hh // blk, ww // blk, blk * blk, blk * blk)
