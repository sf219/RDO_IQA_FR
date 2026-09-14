"""How much of each metric's Hessian sits on the diagonal, and how much inside an 8x8 tile.

Empirical support for the block-diagonal approximation.  For a pixel i the exact Hessian row H_{i.} is
obtained from one Hessian-vector product against the canonical basis vector e_i, so nothing here depends on
the stochastic estimator: these are the matrices the estimator is trying to approximate, not its output.
Rows are taken only from tiles far from the crop border, so a metric with a wide receptive field does not
get credit for mass that fell off the edge.  The Hessian is evaluated at zero distortion (cur = ref), the
point the maps are built at; for the sum-of-squares metrics the residual term vanishes there, so the exact
Hessian and the Gauss-Newton form we ship coincide.

Reported per metric, as medians over rows: the share of the row's squared mass on the diagonal, the share
inside the pixel's own 8x8 tile, the classical dominance ratio |H_ii| / sum_{j!=i} |H_ij|, its block
analogue, and the radius containing 95% of the mass.
Writes results/hess_structure/summary.csv.
"""
import argparse, csv, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); os.chdir(HERE)
import hessian_weights as HW
from hessian_weights import read_yuv420, _make_metric, _hvp_fwd_over_rev

BLK = 8

def rows_for(metric, y, u, v, tiles, batch, margin):
    """Exact Hessian rows (luma-luma block) for every pixel of the given tiles, plus the share of each
    row's mass that leaves the luma plane."""
    def to_t(a):
        return torch.from_numpy(np.ascontiguousarray(a)).float().unsqueeze(0).unsqueeze(0).to(HW.DEVICE)
    planes = (to_t(y), to_t(u), to_t(v))
    func, active = _make_metric(metric)
    idx = [i for i, nm in enumerate(('Y', 'Cb', 'Cr')) if nm in active]
    def f_of_active(*args):
        cur = list(planes)
        for k, i in enumerate(idx):
            cur[i] = args[k]
        # the feature metrics return a [1,1,1,1] tensor rather than a scalar (the shipped path for them is
        # VJP-based and never needed one); the batch is a single image, so the sum is that one value
        return func(planes, tuple(cur)).sum()
    point = tuple(planes[i] for i in idx)
    with torch.no_grad():
        assert func(planes, planes).numel() == 1, f'{metric}: expected one value per image'
    H, W = planes[0].shape[-2:]

    coords = [(ty * BLK + r, tx * BLK + c) for (ty, tx) in tiles for r in range(BLK) for c in range(BLK)]
    out, cross = [], []
    s = 0
    while s < len(coords):
        chunk = coords[s:s + batch]
        pr = []
        for k, i in enumerate(idx):
            z = torch.zeros((len(chunk),) + tuple(planes[i].shape), device=HW.DEVICE)
            if i == 0:
                for b, (r, c) in enumerate(chunk):
                    z[b, 0, 0, r, c] = 1.0
            pr.append(z)
        try:
            hv = _hvp_fwd_over_rev(f_of_active, point, tuple(pr))
        except torch.OutOfMemoryError:
            # the feature metrics at this crop size do not fit many probes at once; back off and retry
            del pr; torch.cuda.empty_cache()
            if batch == 1:
                raise
            batch = max(1, batch // 2)
            continue
        ylum = hv[idx.index(0)].reshape(len(chunk), H, W)
        tot_all = (ylum ** 2).sum(dim=(1, 2))
        for k, i in enumerate(idx):
            if i != 0:
                tot_all = tot_all + (hv[k].reshape(len(chunk), -1) ** 2).sum(1)
        cross.append((1.0 - (ylum ** 2).sum(dim=(1, 2)) / tot_all.clamp_min(1e-30)).detach().cpu().numpy())
        out.append(ylum.detach().cpu().numpy())
        s += len(chunk)
        del hv, ylum, pr
    return np.concatenate(out, 0), coords, np.concatenate(cross, 0)


def profile(rows, coords, rmax):
    """Per row, the fraction of squared mass within Chebyshev radius k, k = 0..rmax.  Returned as the
    median over rows, which is what the figure plots."""
    cur = []
    for k, (r, c) in enumerate(coords):
        sq = rows[k] ** 2; tot = sq.sum()
        if tot <= 0:
            continue
        yy, xx = np.ogrid[:sq.shape[0], :sq.shape[1]]
        d = np.maximum(np.abs(yy - r), np.abs(xx - c)).ravel()
        b = np.bincount(d, weights=sq.ravel(), minlength=rmax + 1)[:rmax + 1]
        cur.append(np.cumsum(b) / tot)
    return np.median(np.stack(cur), axis=0)


def stats(rows, coords, cross):
    dg, bk, dom, bdom, r95 = [], [], [], [], []
    for k, (r, c) in enumerate(coords):
        R = rows[k]
        sq = R ** 2; tot = sq.sum()
        if tot <= 0:
            continue
        ty, tx = r // BLK, c // BLK
        tile = sq[ty * BLK:(ty + 1) * BLK, tx * BLK:(tx + 1) * BLK].sum()
        dg.append(sq[r, c] / tot); bk.append(tile / tot)
        a = np.abs(R); off = a.sum() - a[r, c]
        dom.append(a[r, c] / off if off > 0 else np.inf)
        at = np.abs(R[ty * BLK:(ty + 1) * BLK, tx * BLK:(tx + 1) * BLK]).sum()
        bdom.append(at / a.sum())
        # Chebyshev radius holding 95% of the squared mass
        yy, xx = np.ogrid[:R.shape[0], :R.shape[1]]
        d = np.maximum(np.abs(yy - r), np.abs(xx - c))
        order = np.argsort(d.ravel()); cum = np.cumsum(sq.ravel()[order]) / tot
        r95.append(d.ravel()[order][int(np.searchsorted(cum, 0.95))])
    m = lambda a: float(np.median(a))
    return dict(n_rows=len(dg), diag_frac=m(dg), block_frac=m(bk), dom=m(dom), block_dom=m(bdom),
                r95=m(r95), cross_plane=float(np.median(cross)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics', nargs='+', default=['SSIM', 'MS_SSIM', 'LPIPS', 'LPIPS_ALEX', 'DISTS', 'WD'])
    ap.add_argument('--images', nargs='+', default=['kodim01', 'kodim05', 'kodim19'])
    ap.add_argument('--crop', type=int, default=256)
    ap.add_argument('--margin', type=int, default=64, help='keep sampled tiles this far from the crop edge')
    ap.add_argument('--tiles', type=int, default=6)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--out', default='results/hess_structure/summary.csv')
    ap.add_argument('--profile', default='', help='also write median cumulative mass vs radius here')
    ap.add_argument('--rmax', type=int, default=48)
    a = ap.parse_args()

    # crop 0 means the whole image.  That matters beyond truncation: ssim_func average-pools by
    # max(H, W) // 256 before computing SSIM, so a cropped image is passed through a different
    # preprocessing than the one the shipped maps use.  Measuring at native size removes both issues.
    C, M = a.crop, a.margin
    IH, IW = (512, 768) if C <= 0 else (C, C)
    rng = np.random.default_rng(0)
    tiles = [(int(rng.integers(M // BLK, (IH - M) // BLK)), int(rng.integers(M // BLK, (IW - M) // BLK)))
             for _ in range(a.tiles)]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows_csv = []; prof = {}
    for metric in a.metrics:
        acc = []
        for im in a.images:
            y, u, v = read_yuv420(f'work/yuv/{im}.yuv', 768, 512)
            if C > 0:
                y = y[:C, :C]; u = u[:C // 2, :C // 2]; v = v[:C // 2, :C // 2]
            R, coords, cross = rows_for(metric, y, u, v, tiles, a.batch, M)
            s = stats(R, coords, cross); s['image'] = im; s['metric'] = metric
            acc.append(s)
            if a.profile:
                prof.setdefault(metric, []).append(profile(R, coords, a.rmax).tolist())
            print(f"{metric:11s} {im}  diag {s['diag_frac']:.3f}  tile {s['block_frac']:.3f}  "
                  f"dom {s['dom']:.3f}  blockdom {s['block_dom']:.3f}  r95 {s['r95']:.0f}px", flush=True)
        mean = {k: float(np.mean([x[k] for x in acc])) for k in
                ('diag_frac', 'block_frac', 'dom', 'block_dom', 'r95', 'cross_plane')}
        mean['metric'] = metric; mean['image'] = 'MEAN'; mean['n_rows'] = acc[0]['n_rows'] * len(acc)
        rows_csv += acc + [mean]
        print(f"  -> {metric}: diag {mean['diag_frac']:.3f}  tile {mean['block_frac']:.3f}  "
              f"dom {mean['dom']:.3f}  blockdom {mean['block_dom']:.3f}  r95 {mean['r95']:.1f}px\n", flush=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['metric', 'image', 'n_rows', 'diag_frac', 'block_frac', 'dom',
                                           'block_dom', 'r95', 'cross_plane'])
        w.writeheader(); w.writerows(rows_csv)
    if a.profile:
        import json
        os.makedirs(os.path.dirname(a.profile), exist_ok=True)
        json.dump({'rmax': a.rmax, 'blk': BLK, 'crop': a.crop, 'margin': a.margin,
                   'curves': {k: np.mean(np.stack(v), axis=0).tolist() for k, v in prof.items()}},
                  open(a.profile, 'w'))
        print('->', a.profile)
    print('->', a.out)

if __name__ == '__main__':
    main()
