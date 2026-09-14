"""Encoding time per QP, by weighting.

Times come from the VTM logs of the experiment runs, which executed 7-14 encoders at a time.
CPU (user) time is used rather than wall clock, but memory-bandwidth and clock contention still
inflate everything; all configurations ran under comparable load, so read this as a relative
comparison, not as a clean benchmark.
"""

import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})

RE_T = re.compile(r'Total Time:\s*([\d.]+)\s*sec\.\s*\[user\]')

METHODS = [('anchor',                  'VTM',             '0.35'),
           ('wssim_t24_tau0.25_rdoq',  'SSIM diag',       'tab:green'),
           ('wssim_sbh8_tau1_rdoq',    'SSIM block',      'tab:olive'),
           ('wms_ssim_t24_tau0.25_rdoq', 'MS-SSIM diag',  'tab:cyan'),
           ('wms_ssim_mbh8_tau1_rdoq', 'MS-SSIM block',   'tab:purple'),
           ('wlpips_f24_tau0.25_rdoq', 'LPIPS diag',      'tab:blue'),
           ('wlpips_bh8_tau0.5_rdoq',  'LPIPS block',     'tab:red'),
           ('wdists_d24_tau0.25_rdoq', 'DISTS diag',      'tab:brown'),
           ('wdists_dbh8_tau2_rdoq',   'DISTS block',     'tab:orange')]

# The clean sequential benchmark writes a flat directory with short tags instead of the
# per-image subdirectories the experiment driver uses.
FLAT = {'anchor': 'anchor', 'wssim_t24_tau0.25_rdoq': 'ssim_diag',
        'wssim_sbh8_tau1_rdoq': 'ssim_block', 'wms_ssim_t24_tau0.25_rdoq': 'msssim_diag',
        'wms_ssim_mbh8_tau1_rdoq': 'msssim_block', 'wlpips_f24_tau0.25_rdoq': 'lpips_diag',
        'wlpips_bh8_tau0.5_rdoq': 'lpips_block', 'wdists_d24_tau0.25_rdoq': 'dists_diag',
        'wdists_dbh8_tau2_rdoq': 'dists_block'}


def times(encdir, tag, qp, images):
    out = []
    for im in images:
        f = os.path.join(encdir, im, f'{tag}_qp{qp}.log')
        if not os.path.exists(f) and tag in FLAT:
            f = os.path.join(encdir, f'{im}_{FLAT[tag]}_qp{qp}.log')
        if not os.path.exists(f):
            continue
        m = RE_T.search(open(f, errors='ignore').read())
        if m:
            out.append(float(m.group(1)))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', default=os.path.join(HERE, 'work', 'enc'))
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--qps', type=int, nargs='+', default=[27, 32, 37, 42])
    ap.add_argument('--out', default=os.path.join(HERE, 'fig_dump', 'enctime.png'))
    a = ap.parse_args()

    images = [f'kodim{i:02d}' for i in range(1, a.n + 1)]
    present = [(t, l, c) for t, l, c in METHODS if len(times(a.enc, t, a.qps[0], images)) > 0]

    x = np.arange(len(a.qps))
    w = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    print(f'{"method":16s}' + ''.join(f'{("QP"+str(q)):>10s}' for q in a.qps))
    for i, (tag, lab, col) in enumerate(present):
        mu = [times(a.enc, tag, q, images).mean() for q in a.qps]
        sd = [times(a.enc, tag, q, images).std() for q in a.qps]
        ax.bar(x + i * w, mu, w, yerr=sd, capsize=2, color=col, label=lab,
               error_kw=dict(lw=0.8, alpha=.6))
        print(f'{lab:16s}' + ''.join(f'{m:>10.1f}' for m in mu))
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels([str(q) for q in a.qps], fontsize=13)
    ax.set_xlabel('QP', fontsize=15)
    ax.set_ylabel('Encoding time (s)', fontsize=15)
    ax.tick_params(labelsize=12)
    ax.grid(axis='y', alpha=.3, ls='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=10, ncol=2)
    fig.savefig(a.out, bbox_inches='tight', dpi=300)
    print('\nwrote', a.out)


if __name__ == '__main__':
    main()
