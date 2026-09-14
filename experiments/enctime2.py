"""Encoder-side cost, in two panels: the encode itself, and the importance map that feeds it.

Left  -- encoding time per QP, one bar group per weighting (the map is already on disk).
Right -- time to build the map against probe count, which is the knob that trades the map's
         accuracy against its cost.  The map is built once per image and reused across every
         QP and every tau, so the two panels are not on the same per-encode footing.
"""
import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from enctime import METHODS, times   # noqa: E402

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})

COL = {'SSIM': 'tab:green', 'MS_SSIM': 'tab:purple', 'LPIPS': 'tab:red', 'DISTS': 'tab:orange'}
NAME = {'SSIM': 'SSIM', 'MS_SSIM': 'MS-SSIM', 'LPIPS': 'LPIPS', 'DISTS': 'DISTS'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', default=os.path.join(HERE, 'work', 'timebench'))
    ap.add_argument('--maps', default=os.path.join(HERE, 'bench', 'map_time.json'))
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--qps', type=int, nargs='+', default=[27, 32, 37, 42])
    ap.add_argument('--out', default=os.path.join(HERE, 'fig_dump', 'cost_two_panel.png'))
    a = ap.parse_args()

    images = [f'kodim{i:02d}' for i in range(1, a.n + 1)]
    # shared y: both panels are seconds, so separate scales (or a log right panel) make the
    # map look cheaper than it is next to the encode
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)

    # ---- left: encoding time -------------------------------------------------------------
    present = [(t, l, c) for t, l, c in METHODS if len(times(a.enc, t, a.qps[0], images)) > 0]
    x = np.arange(len(a.qps))
    w = 0.8 / len(present)
    for i, (tag, lab, col) in enumerate(present):
        m = [times(a.enc, tag, q, images).mean() for q in a.qps]
        axL.bar(x + i * w - 0.4 + w / 2, m, w, label=lab, color=col)
    axL.set_xticks(x); axL.set_xticklabels([f'QP {q}' for q in a.qps])
    axL.set_ylabel('Time (s)', fontsize=15)
    axL.set_title('Encoding', fontsize=15)
    axL.legend(fontsize=9, ncol=2, frameon=False)

    # ---- right: map time vs probe count ---------------------------------------------------
    rows = json.load(open(a.maps))
    ns = sorted({r['n'] for r in rows})
    xr = np.arange(len(ns))
    keys = [(m, k) for m in ('SSIM', 'MS_SSIM', 'LPIPS', 'DISTS') for k in ('diag', 'block')
            if any(r['metric'] == m and r['kind'] == k for r in rows)]
    wr = 0.8 / max(len(keys), 1)
    for i, (m, k) in enumerate(keys):
        v = [next((r['sec'] for r in rows if r['metric'] == m and r['kind'] == k
                   and r['n'] == n), np.nan) for n in ns]
        axR.bar(xr + i * wr - 0.4 + wr / 2, v, wr, color=COL[m],
                hatch='' if k == 'diag' else '///', edgecolor='white', linewidth=.4,
                label=f'{NAME[m]}, {k}')
    axR.set_xticks(xr); axR.set_xticklabels([str(n) for n in ns])
    axR.set_xlabel('Probes', fontsize=15)
    axR.set_title('Importance map', fontsize=15)
    axR.legend(fontsize=9, ncol=2, frameon=False)

    for ax in (axL, axR):
        ax.tick_params(labelsize=12)
        ax.grid(alpha=.3, ls='--', axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches='tight', dpi=300)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
