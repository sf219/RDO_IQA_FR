"""Figure: the source image next to the block map of each metric.

Six panels in one row: the source, then the per-pixel diagonal of the 8x8 tiles the encoder is handed,
for each of the five metrics, on one shared log scale so the metrics can be read against each other.
These are the shipped maps -- same estimator, probe budget, smoothing and post-processing as the encodes.
    python maps_fig.py --image kodim23
"""
import argparse, os, struct, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); os.chdir(HERE)
from hessian_weights import read_yuv420, yuv_to_rgb, upsample_chroma
import torch
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
W, H = 768, 512
# Table-1 block series, in the order used everywhere else in the paper
SERIES = [('SSIM', 'SSIM_sbh8m256'), ('MS-SSIM', 'MS_SSIM_mbh8m256'), ('LPIPS', 'LPIPS_bh8'),
          ('DISTS', 'DISTS_dgnb8j3'), ('WD', 'WD_gnb8j3')]

def read_bh_diag_y(p):
    with open(p, 'rb') as f:
        magic, blk = struct.unpack('<II', f.read(8)); n = blk * blk
        nby, nbx = struct.unpack('<II', f.read(8))
        M = np.frombuffer(f.read(nby * nbx * n * n * 4), dtype='<f4').reshape(nby, nbx, n, n)
    d = np.diagonal(M, axis1=2, axis2=3).reshape(nby, nbx, blk, blk)
    d = d.transpose(0, 2, 1, 3).reshape(nby * blk, nbx * blk)[:H, :W]
    # the file holds raw tiles; the encoder normalizes each plane to unit mean diagonal before use, so do
    # the same here or the panels are not comparable across metrics
    return d / max(d.mean(), 1e-30)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default='kodim23')
    ap.add_argument('--out', default='paper/figures/maps.png')
    a = ap.parse_args()
    y, u, v = read_yuv420(f'work/yuv/{a.image}.yuv', W, H)
    t = lambda x: torch.from_numpy(x).float()[None, None]
    rgb = yuv_to_rgb(t(y), upsample_chroma(t(u), (H, W)), upsample_chroma(t(v), (H, W)))
    rgb = rgb[0].permute(1, 2, 0).clamp(0, 255).numpy() / 255.0

    maps = [(nm, read_bh_diag_y(f'work/weights/{d}/{a.image}.bh')) for nm, d in SERIES]
    L = [np.log10(np.maximum(m, 1e-2)) for _, m in maps]
    # a few extreme tiles would otherwise set the range and flatten every panel
    allv = np.concatenate([x.ravel() for x in L])
    lo, hi = np.percentile(allv, 1.0), np.percentile(allv, 99.0)

    # two rows of three fits a single column; one row of six only fits full width, which costs a page
    fig, axg = plt.subplots(2, 3, figsize=(3.4, 1.33))
    ax = axg.ravel()
    ax[0].imshow(rgb); ax[0].set_title('Source', fontsize=7, pad=1.8)
    for k, ((nm, _), l) in enumerate(zip(maps, L), 1):
        im = ax[k].imshow(l, cmap='gray', vmin=lo, vmax=hi)
        ax[k].set_title(nm, fontsize=7, pad=1.8)
    for b in ax:
        b.set_xticks([]); b.set_yticks([])
        for sp in b.spines.values():
            sp.set_visible(False)
    fig.subplots_adjust(left=0.004, right=0.90, top=0.90, bottom=0.01, wspace=0.05, hspace=0.30)
    cax = fig.add_axes([0.912, 0.02, 0.018, 0.86])
    cb = fig.colorbar(im, cax=cax)          # no label: the caption says the scale is log10 of the weight
    cb.ax.tick_params(labelsize=6, length=2, pad=1.2)
    os.makedirs('paper/figures', exist_ok=True)
    fig.savefig(a.out, bbox_inches='tight', dpi=320)
    print(a.out)

if __name__ == '__main__':
    main()
