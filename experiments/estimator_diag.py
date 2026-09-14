"""3.0 estimator diagnostics, no encoding.

Reads the tile files (.bh) and diagonal maps (.dat) exactly as shipped to the encoder and
reports, per set / image / plane:
  neg_diag_frac       fraction of negative diagonal entries
  neg_eig_frac        fraction of negative eigenvalues over all 8x8 tiles (tolerance 1e-6 * max |eig| of the tile)
  neg_energy_frac     sum |eig^-| / sum |eig|
  tiles_with_neg_frac fraction of tiles with at least one negative eigenvalue
  diag_dom_mean       mean over tiles of ||diag(M_b)||_F / ||M_b||_F
Planes whose tiles are all identity (chroma of a luma-only metric) are skipped.
Output: results/estimator_diag/summary.csv (per image rows plus MEAN rows per set/plane).
"""
import argparse, csv, glob, os, struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAGIC_BH = 0x31484257

# set name -> (metric, estimator label, m, sigma)
SETS = {
    'SSIM_sbh8m256':    ('SSIM',    'hutchblock', 256, 0.0),
    'MS_SSIM_mbh8m256': ('MS_SSIM', 'hutchblock', 256, 0.0),   # standard MS-SSIM (sigma 1.5) since 4 Sept
    'LPIPS_bh8':    ('LPIPS',   'gnblock',    256, 3.0),
    'DISTS_dgnb8':  ('DISTS',   'gnblock',    256, 0.0),
    'SSIM_sn256':   ('SSIM',    'hutch',      256, 0.0),
    'MS_SSIM_mn256':('MS_SSIM', 'hutch',      256, 0.0),
    'LPIPS_gn':     ('LPIPS',   'gn',         256, 0.0),
    'DISTS_dgn':    ('DISTS',   'gn',         256, 0.0),
    # unprojected Hutchinson block tiles of the feature metrics (Table 2 HVP row, text): negativity is the point
    'LPIPS_hb8m256': ('LPIPS',  'hutchblock', 256, 0.0),
    'DISTS_dhb8m256':('DISTS',  'hutchblock', 256, 0.0),
}


def read_bh(path):
    with open(path, 'rb') as f:
        magic, blk = struct.unpack('<II', f.read(8))
        if magic != MAGIC_BH:
            raise ValueError(f'{path}: not a dense tile file (magic {magic:08x})')
        n = blk * blk
        planes = []
        for _ in range(3):
            nby, nbx = struct.unpack('<II', f.read(8))
            cnt = nby * nbx * n * n
            M = np.frombuffer(f.read(cnt * 4), dtype='<f4').reshape(nby * nbx, n, n) if cnt else np.zeros((0, n, n), 'f4')
            planes.append(M)
    return blk, planes


def read_dat(path, w, h):
    with open(path, 'rb') as f:
        f.read(4)                                     # numFrames
        dims = [(w, h), (w // 2, h // 2), (w // 2, h // 2)]
        return [np.frombuffer(f.read(pw * ph * 4), dtype='<f4').reshape(ph, pw) for pw, ph in dims]


def tile_stats(M):
    M = M.astype(np.float64)
    M = 0.5 * (M + np.transpose(M, (0, 2, 1)))
    d = np.diagonal(M, axis1=1, axis2=2)
    eig = np.linalg.eigvalsh(M)
    tol = 1e-6 * np.abs(eig).max(axis=1, keepdims=True)
    neg = eig < -tol
    return dict(
        n_tiles=M.shape[0],
        neg_diag_frac=float((d < 0).mean()),
        neg_eig_frac=float(neg.mean()),
        neg_energy_frac=float(np.abs(eig[neg]).sum() / max(np.abs(eig).sum(), 1e-30)),
        tiles_with_neg_frac=float(neg.any(axis=1).mean()),
        diag_dom_mean=float(np.mean(np.linalg.norm(d, axis=1) / np.maximum(np.linalg.norm(M.reshape(M.shape[0], -1), axis=1), 1e-30))),
    )


def is_identity_plane(M):
    if M.shape[0] == 0:
        return True
    n = M.shape[1]
    return bool(np.all(np.abs(M - np.eye(n, dtype=M.dtype)) < 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=os.path.join(HERE, 'work', 'weights'))
    ap.add_argument('--sets', nargs='+', default=list(SETS))
    ap.add_argument('--w', type=int, default=768)
    ap.add_argument('--h', type=int, default=512)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'estimator_diag', 'summary.csv'))
    a = ap.parse_args()
    cols = ['set', 'metric', 'estimator', 'm', 'sigma', 'image', 'plane', 'n_tiles', 'neg_diag_frac',
            'neg_eig_frac', 'neg_energy_frac', 'tiles_with_neg_frac', 'diag_dom_mean']
    rows = []
    for s in a.sets:
        metric, est, m, sig = SETS[s]
        d = os.path.join(a.weights, s)
        files = sorted(glob.glob(os.path.join(d, 'kodim*.bh'))) or sorted(glob.glob(os.path.join(d, 'kodim*.dat')))
        acc = {}
        for f in files:
            img = os.path.basename(f).split('.')[0]
            if f.endswith('.bh'):
                _, planes = read_bh(f)
                for c, M in enumerate(planes):
                    if is_identity_plane(M):
                        continue
                    st = tile_stats(M)
                    rows.append(dict(set=s, metric=metric, estimator=est, m=m, sigma=sig, image=img, plane=c, **st))
                    acc.setdefault(c, []).append(st)
            else:
                planes = read_dat(f, a.w, a.h)
                for c, W in enumerate(planes):
                    if np.all(np.abs(W - 1.0) < 1e-6):
                        continue
                    st = dict(n_tiles=W.size, neg_diag_frac=float((W < 0).mean()), neg_eig_frac='',
                              neg_energy_frac='', tiles_with_neg_frac='', diag_dom_mean='')
                    rows.append(dict(set=s, metric=metric, estimator=est, m=m, sigma=sig, image=img, plane=c, **st))
                    acc.setdefault(c, []).append(st)
        for c, lst in acc.items():
            mean = {k: (float(np.mean([x[k] for x in lst])) if lst[0][k] != '' else '') for k in lst[0]}
            mean['n_tiles'] = int(sum(x['n_tiles'] for x in lst))
            rows.append(dict(set=s, metric=metric, estimator=est, m=m, sigma=sig, image=f'MEAN({len(lst)})', plane=c, **mean))
        print(f'{s}: {len(files)} files')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(a.out)
    for r in rows:
        if r['image'].startswith('MEAN'):
            print(f"{r['set']:14s} plane {r['plane']}  negdiag {r['neg_diag_frac']:.4f}  "
                  + (f"negeig {r['neg_eig_frac']:.4f}  negE {r['neg_energy_frac']:.4f}  tiles>0 {r['tiles_with_neg_frac']:.3f}  dom {r['diag_dom_mean']:.3f}" if r['neg_eig_frac'] != '' else ''))


if __name__ == '__main__':
    main()
