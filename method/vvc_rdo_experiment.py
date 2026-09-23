"""All-intra VVC (VTM-23.8) experiment: plain-SSE RDO vs Hessian-weighted-MSE RDO.

Pipeline per image
    png --ffmpeg--> yuv420p 8 bit
    yuv --hessian_weights--> per-pixel importance map (.dat)
    yuv --VTM--> bitstream + reconstruction, once per QP and per configuration
    reconstruction --> PSNR / SSIM / MS-SSIM / LPIPS / DISTS
    rate-quality points --> BD-rate of the weighted encoder against the anchor

The anchor and the weighted run share one binary and one config; the only difference is
--WeightedRdo / --WeightedRdoFile / --WeightedRdoTau.
"""

import argparse
import contextlib
import fcntl
import json
import os
import tempfile
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STAC = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, STAC, os.path.join(STAC, 'Common')):
    if p not in sys.path:
        sys.path.insert(0, p)

VTM = os.path.join(STAC, 'VTM_WMSE')
ENCODER = os.path.join(VTM, 'bin', 'EncoderAppStatic')
DECODER = os.path.join(VTM, 'bin', 'DecoderAppStatic')
CFG = os.path.join(VTM, 'cfg', 'encoder_intra_vtm.cfg')


# ======================================================================================
# stage 1 - source preparation
# ======================================================================================
def png_to_yuv(png, yuv):
    if os.path.exists(yuv) and os.path.getsize(yuv) > 0:
        return
    os.makedirs(os.path.dirname(yuv), exist_ok=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', png, '-pix_fmt', 'yuv420p',
                    '-f', 'rawvideo', yuv], check=True)


def image_size(png):
    from PIL import Image
    with Image.open(png) as im:
        return im.size  # (w, h)


# ======================================================================================
# stage 3 - encoding
# ======================================================================================
def encode(args):
    (yuv, w, h, qp, out_dir, tag, weights, tau, rdoq, gnorm, extra) = args
    os.makedirs(out_dir, exist_ok=True)
    bs = os.path.join(out_dir, f'{tag}_qp{qp}.bin')
    rec = os.path.join(out_dir, f'{tag}_qp{qp}_rec.yuv')
    log = os.path.join(out_dir, f'{tag}_qp{qp}.log')
    if os.path.exists(bs) and os.path.exists(rec) and os.path.getsize(rec) == w * h * 3 // 2:
        return bs, rec

    cmd = [ENCODER, '-c', CFG, '-i', yuv, '-wdt', str(w), '-hgt', str(h), '-fr', '1', '-f', '1',
           '-q', str(qp), '-b', bs, '-o', rec,
           '--InputBitDepth=8', '--OutputBitDepth=8', '--InputChromaFormat=420']
    if weights:
        opt = '--WeightedRdoBlockFile' if str(weights).endswith('.bh') else '--WeightedRdoFile'
        cmd += ['--WeightedRdo=1', f'{opt}={weights}', f'--WeightedRdoTau={tau}']
        if rdoq:
            cmd += ['--WeightedRdoq=1']
        if gnorm:
            cmd += ['--WeightedRdoGlobalNorm=1']
    cmd += list(extra)

    with open(log, 'w') as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f'encoder failed ({tag} qp{qp}); see {log}')
    return bs, rec


# ======================================================================================
# stage 4 - quality metrics
# ======================================================================================
class Evaluator:
    """Loaded lazily and once per process; the perceptual nets are heavy."""

    def __init__(self):
        import torch
        from Common.utils.q_utils import ssim_func, ms_ssim_func, compute_LPIPS_rgb, D
        self.torch = torch
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ssim = ssim_func
        from msssim_std import ms_ssim_std
        self.msssim = ms_ssim_std           # standard sigma=1.5 MS-SSIM (protocol since 4 Sept 2026)
        self.lpips = compute_LPIPS_rgb
        import lpips as _lp
        self.lpips_alex = _lp.LPIPS(net='alex', verbose=False).to(self.dev).eval()   # literature protocol (3.6)
        self.dists = D.to(self.dev)   # q_utils leaves the DISTS model on the CPU
        # WD scoring (three sigmas, VGG-feature transport) is expensive and no longer part of the
        # paper: opt in with EVAL_WD=1.  Without it the JSONs simply carry no wd* keys.
        self.wd_sigmas = (0.0, 2.0, 4.0) if os.environ.get('EVAL_WD') else ()
        if self.wd_sigmas:
            from hessian_weights import _wd_model
            self.wd = _wd_model()

    def _t(self, a):
        return self.torch.from_numpy(np.ascontiguousarray(a)).float().unsqueeze(0).unsqueeze(0).to(self.dev)

    def _rgb(self, y, u, v):
        import torch.nn.functional as F
        from hessian_weights import yuv_to_rgb
        ty, tu, tv = self._t(y), self._t(u), self._t(v)
        hw = ty.shape[-2:]
        return yuv_to_rgb(ty, F.interpolate(tu, size=hw, mode='bilinear', align_corners=False),
                          F.interpolate(tv, size=hw, mode='bilinear', align_corners=False))

    def __call__(self, ref, rec):
        """ref/rec: (y, u, v) float arrays 0..255."""
        torch = self.torch
        out = {}
        mse_y = float(np.mean((ref[0] - rec[0]) ** 2))
        out['psnr_y'] = 10 * np.log10(255.0 ** 2 / mse_y) if mse_y > 0 else 99.0
        num = ref[0].size * mse_y
        den = ref[0].size
        for i in (1, 2):
            m = float(np.mean((ref[i] - rec[i]) ** 2))
            num += ref[i].size * m
            den += ref[i].size
        out['psnr_yuv'] = 10 * np.log10(255.0 ** 2 / (num / den)) if num > 0 else 99.0

        with torch.no_grad():
            out['ssim'] = 1.0 - float(self.ssim(self._t(ref[0]), self._t(rec[0])))
            out['ms_ssim'] = float(self.msssim(self._t(ref[0]), self._t(rec[0])))      # ms_ssim_std returns the similarity
            r1, r2 = self._rgb(*ref), self._rgb(*rec)
            out['lpips'] = float(self.lpips(r1, r2))
            out['lpips_alex'] = float(self.lpips_alex(2.0 * (r1 / 255.0 - 0.5), 2.0 * (r2 / 255.0 - 0.5)))
            mse_rgb = float(((r1 - r2) ** 2).mean())
            out['psnr_rgb'] = 10 * np.log10(255.0 ** 2 / mse_rgb) if mse_rgb > 0 else 99.0
            out['dists'] = float(self.dists(r1 / 255.0, r2 / 255.0))
            for sg in self.wd_sigmas:
                sig = torch.zeros_like(r1[:, 0:1]) + sg
                out[f'wd{sg:g}'] = float(self.wd(r2 / 255.0, r1 / 255.0, sig))
        return out


# ======================================================================================
# stage 5 - BD-rate
# ======================================================================================
def _prep_curve(r, q):
    q = np.asarray(q, dtype=np.float64)
    r = np.log(np.asarray(r, dtype=np.float64))
    i = np.argsort(q)
    return q[i], r[i]


def _bd_rate(rate_a, q_a, rate_b, q_b):
    """% rate change of curve A relative to curve B at equal quality (negative = A better).

    Piecewise-linear Bjontegaard rate (6 Sept 2026, protocol decision): log-rate is interpolated
    linearly in quality on each curve and the mean log-rate difference is taken over the shared
    quality range.  The cubic-polynomial variant (_bd_rate_cubic) is kept for reference; it is
    ill-conditioned when three of the four points bunch in quality (saturating LPIPS at low rate).
    """
    qa, ra = _prep_curve(rate_a, q_a)
    qb, rb = _prep_curve(rate_b, q_b)
    lo, hi = max(qa.min(), qb.min()), min(qa.max(), qb.max())
    if hi <= lo:
        return float('nan')
    q = np.linspace(lo, hi, 2000)
    d = float(np.mean(np.interp(q, qa, ra) - np.interp(q, qb, rb)))
    return (np.exp(d) - 1.0) * 100.0


def _bd_rate_cubic(rate_a, q_a, rate_b, q_b):
    """The classic cubic-in-quality fit of log-rate (kept for reference; not used by the tables)."""
    qa, ra = _prep_curve(rate_a, q_a)
    qb, rb = _prep_curve(rate_b, q_b)
    lo, hi = max(qa.min(), qb.min()), min(qa.max(), qb.max())
    if hi <= lo:
        return float('nan')
    pa = np.polyint(np.poly1d(np.polyfit(qa, ra, 3)))
    pb = np.polyint(np.poly1d(np.polyfit(qb, rb, 3)))
    d = ((np.polyval(pa, hi) - np.polyval(pa, lo)) - (np.polyval(pb, hi) - np.polyval(pb, lo))) / (hi - lo)
    return (np.exp(d) - 1.0) * 100.0


def to_quality(metric, values):
    """map every metric to a 'higher is better, PSNR-like' scale so BD-rate is comparable."""
    v = np.asarray(values, dtype=np.float64)
    if metric in ('psnr_y', 'psnr_yuv', 'psnr_rgb'):
        return v
    if metric in ('ssim', 'ms_ssim'):
        # the evaluator stores the SIMILARITY s (q_utils' functions return 1 - s and the evaluator
        # subtracts them from 1; ms_ssim_std returns s directly), so the dB scale is -10 log10(1 - s)
        return -10.0 * np.log10(np.clip(1.0 - v, 1e-8, None))
    if metric in ('lpips', 'dists', 'lpips_alex') or metric.startswith('wd'):
        return -10.0 * np.log10(np.clip(v, 1e-8, None))
    raise ValueError(metric)


METRIC_KEYS = ('psnr_y', 'psnr_yuv', 'ssim', 'ms_ssim', 'lpips', 'dists', 'psnr_rgb', 'lpips_alex') \
    + (('wd0', 'wd2', 'wd4') if os.environ.get('EVAL_WD') else ())


# ======================================================================================
# driver
# ======================================================================================
GPU_LOCK = os.path.join(tempfile.gettempdir(), 'vvc_rdo_gpu.lock')


@contextlib.contextmanager
def gpu_lock():
    """Serialise Hessian estimation across concurrently running experiment processes.

    The LPIPS double-backward needs several GiB; two of them on one card OOMs.  The lock is
    held only around the GPU section, so the encode pools of the different runs still overlap.
    Re-entrant within a process (gpu_claim wraps code that takes it again).
    """
    global _LOCK_FH, _LOCK_DEPTH
    if _LOCK_DEPTH == 0:
        _LOCK_FH = open(GPU_LOCK, 'w')
        fcntl.flock(_LOCK_FH, fcntl.LOCK_EX)
    _LOCK_DEPTH += 1
    try:
        yield
    finally:
        _LOCK_DEPTH -= 1
        if _LOCK_DEPTH == 0:
            fcntl.flock(_LOCK_FH, fcntl.LOCK_UN)
            _LOCK_FH.close()
            _LOCK_FH = None


_LOCK_FH, _LOCK_DEPTH = None, 0
CLIC_MAP_PATTERN = 'gen_maps.py --images work/clic41'


@contextlib.contextmanager
def gpu_claim(need_gb, keep_free_gb=1.0, poll=20):
    """Exclusive use of the GPU while the CLIC driver runs (5 Sept).  Takes gpu_lock (the driver's
    scoring waits on it), waits until no CLIC map subprocess exists and need_gb + keep_free_gb is
    free, then holds a blocker tensor so that only keep_free_gb stays free: the driver's next map
    (gen_maps --min-free-gb 3, polled every 120 s) waits until this block exits."""
    import time
    import torch
    # idle CUDA contexts of other waiters (0.1-0.3 GB each) can keep the card just under a large request while
    # we hold the lock: cap the request so that it can always be satisfied on an otherwise idle card (7 Sept 10:00).
    need_gb = min(need_gb, 8.5 - keep_free_gb)
    with gpu_lock():
        blocker = None
        try:
            waited = 0
            # The CLIC driver's map subprocess and its scoring both take gpu_lock, so holding the lock is
            # what serialises us against them; a process-existence test here deadlocked (6 Sept 02:30:
            # their subprocess blocked on our lock while we waited for it to exit).  Wait on memory only.
            while True:
                free = torch.cuda.mem_get_info()[0] / 2 ** 30 if torch.cuda.is_available() else 1e9
                if free >= need_gb + keep_free_gb:
                    break
                if waited == 0:
                    print(f'  gpu_claim: waiting (free {free:.1f} GB, need {need_gb + keep_free_gb:.1f} GB)', flush=True)
                time.sleep(poll); waited += poll
            if torch.cuda.is_available():
                nblk = int(max(0.0, free - need_gb - keep_free_gb) * 2 ** 30)
                if nblk > 0:
                    blocker = torch.empty(nblk, dtype=torch.uint8, device='cuda')
            if waited:
                print(f'  gpu_claim: acquired after {waited // 60} min', flush=True)
            yield
        finally:
            blocker = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _release_gpu():
    """Return the caching allocator's reserved blocks to the driver, so a process that is only
    waiting for encoders does not pin 10 GiB.  Called after the map arrays are written and dropped."""
    try:
        import torch
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def build_weights(job):
    png, yuv, w, h, metric, probes, seed, out, opts, rawdir = job
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    with gpu_lock():                        # importing hessian_weights loads the VGG to the GPU
        from hessian_weights import (compute_raw_diag, postprocess_raw, read_yuv420, write_dat,
                                     set_wd_sigma)
    if 'wd_sigma' in opts:
        set_wd_sigma(opts['wd_sigma'])
    y, u, v = read_yuv420(yuv, w, h)

    if opts['estimator'] == 'hutchblock':
        from hessian_weights import compute_block_hutch, write_block_dat
        with gpu_lock():
            mats = compute_block_hutch(y, u, v, metric, probes, seed, opts['jitter'],
                                       opts.get('blk', 8), verbose=False,
                                       psd=opts.get('psd', 'cholesky'), ckpt_path=out + '.ckpt')
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        write_block_dat(out + '.tmp', mats, opts.get('blk', 8))
        os.replace(out + '.tmp', out)
        del mats
        _release_gpu()
        return out

    if opts['estimator'] == 'gnblock':
        # non-diagonal: one blk x blk Hessian per tile, from the same Gauss-Newton sketch
        from hessian_weights import compute_block_gn, write_block_dat
        with gpu_lock():
            mats = compute_block_gn(y, u, v, probes, seed, opts['jitter'], opts.get('blk', 8),
                                    verbose=False, metric=metric,
                                    batch=opts.get('vjp_batch', 1))
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        write_block_dat(out + '.tmp', mats, opts.get('blk', 8))
        os.replace(out + '.tmp', out)
        del mats
        _release_gpu()
        return out

    # the Hessian estimate is the expensive part and does not depend on clip_pct or
    # plane_norm, so cache it and let those two be swept for free
    raw = None
    if rawdir:
        os.makedirs(rawdir, exist_ok=True)
        rawfn = os.path.join(rawdir, os.path.basename(out).replace('.dat', '.npz'))
        if os.path.exists(rawfn):
            z = np.load(rawfn)
            raw = [z[k] for k in ('y', 'u', 'v')]
            raw = [None if r.size == 0 else r for r in raw]
    if raw is None:
        with gpu_lock():
            raw = compute_raw_diag(y, u, v, metric, probes, seed, False,
                                   opts['estimator'], opts['jitter'],
                                   batch=opts.get('vjp_batch', 1))
            try:
                import torch
                torch.cuda.empty_cache()   # release the cached activations before the lock is dropped
            except Exception:
                pass
            try:
                import torch
                torch.cuda.empty_cache()      # do not hold the cache while another run waits
            except Exception:
                pass
        if rawdir:
            e = np.zeros(0)
            np.savez_compressed(rawfn + '.tmp.npz',
                                **{k: (e if r is None else r) for k, r in zip('yuv', raw)})
            os.replace(rawfn + '.tmp.npz', rawfn)
    wy, wu, wv = postprocess_raw(raw, (y.shape, u.shape, v.shape),
                                 opts['plane_norm'], opts['clip_pct'])
    write_dat(out + '.tmp', wy, wu, wv)      # atomic: another process may be reading this dir
    os.replace(out + '.tmp', out)
    print(f'  weights {os.path.basename(out)}  Y[min {wy.min():.3f} max {wy.max():.3f}] '
          f'Cb[max {wu.max():.3f}] Cr[max {wv.max():.3f}]', flush=True)
    del raw
    _release_gpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default=os.path.join(STAC, 'Common/Images/KODAK/All'))
    ap.add_argument('--limit', type=int, default=0, help='use only the first N images')
    ap.add_argument('--qps', type=int, nargs='+', default=[27, 32, 37, 42])
    ap.add_argument('--metric', default='MS_SSIM')
    ap.add_argument('--wd-sigma', type=float, default=2.0,
                    help='constant log2(sigma) for Wasserstein Distortion (metric WD)')
    ap.add_argument('--taus', type=float, nargs='*', default=[0.5])
    ap.add_argument('--probes', type=int, default=64)
    ap.add_argument('--estimator', default='auto', choices=['auto', 'gn', 'hutch', 'colnorm', 'gnblock', 'hutchblock'])
    ap.add_argument('--plane-norm', default='per_plane', choices=['per_plane', 'global'])
    ap.add_argument('--clip-pct', type=float, default=99.5)
    ap.add_argument('--blk', type=int, default=8, help='tile size for the gnblock estimator')
    ap.add_argument('--psd', default='none', choices=['eigh', 'shrink', 'none'],
                    help='how to repair an indefinite block-Hessian estimate; the exact eigh '
                         'projection costs 35.6 s per picture and was measured to change the '
                         'BD-rate by under 0.1 points, so it is off by default')
    ap.add_argument('--jitter', type=float, default=0.0)
    ap.add_argument('--vjp-batch', type=int, default=4,
                    help='Gauss-Newton probes per backward pass.  Probes are seeded per index, '
                         'so this changes speed and not the result: verified byte-identical '
                         'bitstreams for batch 1 vs 4 on kodim01/kodim05.  1 disables it.')
    ap.add_argument('--weight-tag', default='', help='suffix for the weight-map cache directory')
    ap.add_argument('--seed', type=int, default=0, help='probe seed (use a distinct --weight-tag per seed)')
    ap.add_argument('--encoder-global-norm', action='store_true',
                    help='let the encoder keep the cross-plane balance of the weight file')
    ap.add_argument('--jitter-per-qp', default='',
                    help='per-QP jitter, e.g. "27:3.3,32:5.4,37:8.3,42:11.5"; one map per QP')
    ap.add_argument('--rdoq-taus', type=float, nargs='*', default=[],
                    help='extra configurations that also scale the RDOQ lambda per block')
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--work', default=os.path.join(HERE, 'work'))
    ap.add_argument('--out', default=os.path.join(HERE, 'results'))
    ap.add_argument('--extra', nargs='*', default=[], help='extra encoder options')
    ap.add_argument('--run-tag', default='', help='suffix added to the configuration tags and the JSON name (not to the map cache): '
                    'lets one map series be encoded under several encoder settings, e.g. --extra --WeightedRdoMask=N')
    ap.add_argument('--encoder-bin', default='', help='encoder binary to use instead of VTM_WMSE/bin/EncoderAppStatic')
    ap.add_argument('--drop-recs', action='store_true', help='delete the reconstructions of the weighted configurations after scoring (bitstreams kept)')
    ap.add_argument('--rdo-mask', type=int, default=None, help='WeightedRdoMask of the decision-ablation encoder (VTM_WMSE_abl); appended to the weighted encodes')
    a = ap.parse_args()
    if a.rdo_mask is not None:
        a.extra = list(a.extra) + [f'--WeightedRdoMask={a.rdo_mask}']
    if a.encoder_bin:
        global ENCODER
        ENCODER = os.path.abspath(a.encoder_bin)        # inherited by the forked encode workers

    os.makedirs(a.work, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)

    pngs = sorted(f for f in os.listdir(a.images) if f.lower().endswith('.png'))
    if a.limit:
        pngs = pngs[: a.limit]
    print(f'{len(pngs)} images, QPs {a.qps}, metric {a.metric}, taus {a.taus}', flush=True)

    # ---- stage 1: yuv ----------------------------------------------------------------
    srcs = []
    for f in pngs:
        stem = os.path.splitext(f)[0]
        png = os.path.join(a.images, f)
        w, h = image_size(png)
        yuv = os.path.join(a.work, 'yuv', f'{stem}.yuv')
        png_to_yuv(png, yuv)
        srcs.append((stem, png, yuv, w, h))

    # ---- stage 2: weights (GPU, sequential) -------------------------------------------
    wdir = a.metric + (('_' + a.weight_tag) if a.weight_tag else '')
    jit_qp = {}
    if a.jitter_per_qp:
        for part in a.jitter_per_qp.split(','):
            k, v = part.split(':')
            jit_qp[int(k)] = float(v)

    def weights_for(stem, png, yuv, w, h):
        out = {}
        for qp in a.qps:
            j = jit_qp.get(qp, a.jitter)
            wopts = dict(estimator=a.estimator, plane_norm=a.plane_norm, clip_pct=a.clip_pct,
                         jitter=j, blk=a.blk, psd=a.psd, vjp_batch=a.vjp_batch,
                         wd_sigma=a.wd_sigma)
            ext = '.bh' if a.estimator in ('gnblock', 'hutchblock') else '.dat'
            name = f'{stem}_qp{qp}{ext}' if jit_qp else f'{stem}{ext}'
            fn = os.path.join(a.work, 'weights', wdir, name)
            # the WD Hessian depends on log2(sigma), so it has to be part of the cache key --
            # otherwise a sigma sweep silently reuses the first sigma's diagonals
            sfx = f'_wds{a.wd_sigma:g}' if a.metric.upper() == 'WD' else ''
            rawdir = os.path.join(a.work, 'raw',
                                  f'{a.metric}_{a.estimator}_n{a.probes}_j{j:g}_s{a.seed}{sfx}')
            out[qp] = build_weights((png, yuv, w, h, a.metric, a.probes, a.seed, fn, wopts, rawdir))
        return out

    # ---- stage 3: encodes -------------------------------------------------------------
    sfx = (('_' + a.weight_tag) if a.weight_tag else '') + (('_' + a.run_tag) if a.run_tag else '')
    configs = [('anchor', None, 0.0, False)]
    for tau in a.taus:
        configs.append((f'w{a.metric.lower()}{sfx}_tau{tau:g}', a.metric, tau, False))
    for tau in a.rdoq_taus:
        configs.append((f'w{a.metric.lower()}{sfx}_tau{tau:g}_rdoq', a.metric, tau, True))

    # Weight generation is GPU-bound and encoding is CPU-bound, and submitting to the pool is
    # non-blocking, so the two are interleaved: anchors (which need no weights) go in first,
    # then each image's weighted jobs are submitted as soon as that image's map is ready.
    # Doing all weights before any encode left 16 cores idle for the whole weight stage.
    n_jobs = len(srcs) * len(configs) * len(a.qps)
    print(f'{n_jobs} encodes on {a.jobs} workers, pipelined with weight generation', flush=True)

    t0 = time.time()
    done = 0
    futs = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for stem, png, yuv, w, h in srcs:
            od = os.path.join(a.work, 'enc', stem)
            for tag, met, tau, rdoq in configs:
                if met:
                    continue
                for qp in a.qps:
                    futs.append(ex.submit(encode, (yuv, w, h, qp, od, tag, None, tau, rdoq,
                                                   a.encoder_global_norm, tuple(a.extra))))

        tw = time.time()
        need_weights = any(met for _, met, _, _ in configs)
        for stem, png, yuv, w, h in srcs:
            if not need_weights:
                break
            wm = weights_for(stem, png, yuv, w, h)          # GPU, while the pool keeps encoding
            od = os.path.join(a.work, 'enc', stem)
            for tag, met, tau, rdoq in configs:
                if not met:
                    continue
                for qp in a.qps:
                    futs.append(ex.submit(encode, (yuv, w, h, qp, od, tag, wm[qp], tau, rdoq,
                                                   a.encoder_global_norm, tuple(a.extra))))
        print(f'weights ready in {time.time() - tw:.0f}s  ({wdir})', flush=True)

        for fu in as_completed(futs):
            fu.result()
            done += 1
            if done % 10 == 0 or done == len(futs):
                el = time.time() - t0
                print(f'  {done}/{len(futs)} encodes  ({el:.0f}s, eta {el / done * (len(futs) - done):.0f}s)',
                      flush=True)

    # ---- stage 4: metrics -------------------------------------------------------------
    from hessian_weights import read_yuv420
    # GPU_CLAIM_GB=<gb>: score under gpu_claim (exclusive GPU while the CLIC driver runs, 5 Sept);
    # otherwise the lock alone, and only for large images (a 2K scoring peaks ~10 GiB).
    claim_gb = float(os.environ.get('GPU_CLAIM_GB', '0') or 0)
    stage4_ctx = gpu_claim(claim_gb) if claim_gb > 0 else contextlib.nullcontext()
    results = {}
    with stage4_ctx:
      ev = Evaluator()
      for stem, png, yuv, w, h in srcs:
        ref = read_yuv420(yuv, w, h)
        od = os.path.join(a.work, 'enc', stem)
        results[stem] = {}
        with (gpu_lock() if (w * h > 1_000_000 and claim_gb <= 0) else contextlib.nullcontext()):
            for tag, met, tau, rdoq in configs:
                pts = []
                for qp in a.qps:
                    bs = os.path.join(od, f'{tag}_qp{qp}.bin')
                    rec = os.path.join(od, f'{tag}_qp{qp}_rec.yuv')
                    m = ev(ref, read_yuv420(rec, w, h))
                    m['qp'] = qp
                    m['bpp'] = os.path.getsize(bs) * 8.0 / (w * h)
                    pts.append(m)
                results[stem][tag] = pts
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        if a.drop_recs:
            for tag, met, tau, rdoq in configs:
                if met:
                    for qp in a.qps:
                        try:
                            os.remove(os.path.join(od, f'{tag}_qp{qp}_rec.yuv'))
                        except FileNotFoundError:
                            pass
        print(f'  metrics {stem} done', flush=True)

    # ---- stage 5: BD-rate --------------------------------------------------------------
    summary = {}
    for tag, met, tau, rdoq in configs[1:]:
        summary[tag] = {}
        for key in METRIC_KEYS:
            per_img = []
            for stem, _, _, _, _ in srcs:
                A = results[stem][tag]
                B = results[stem]['anchor']
                bd = _bd_rate([p['bpp'] for p in A], to_quality(key, [p[key] for p in A]),
                              [p['bpp'] for p in B], to_quality(key, [p[key] for p in B]))
                per_img.append(bd)
            summary[tag][key] = {'per_image': per_img, 'mean': float(np.nanmean(per_img))}

    stamp = time.strftime('%Y%m%d_%H%M%S')
    outfn = os.path.join(a.out, f'vvc_rdo_{a.metric.lower()}{sfx}_{stamp}.json')
    with open(outfn, 'w') as f:
        json.dump({'args': vars(a), 'images': [s[0] for s in srcs], 'configs': [c[0] for c in configs],
                   'results': results, 'bd_rate_vs_anchor': summary}, f, indent=1)

    print('\n' + '=' * 78)
    print(f'BD-rate vs plain-SSE anchor (negative = weighted RDO wins), {len(srcs)} images, QPs {a.qps}')
    print('=' * 78)
    hdr = f'{"config":26s}' + ''.join(f'{k:>11s}' for k in METRIC_KEYS)
    print(hdr)
    for tag in summary:
        row = f'{tag:26s}' + ''.join(f'{summary[tag][k]["mean"]:>10.2f}%' for k in METRIC_KEYS)
        print(row)
    print(f'\nwritten to {outfn}')


if __name__ == '__main__':
    main()
