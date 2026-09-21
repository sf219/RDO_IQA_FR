"""Figures from results/kodak_json/summary.csv (24 images, CTC QPs):
  tau_sweep.png   4 panels: target-metric BD-rate vs Y-PSNR BD-rate, diag + block curves (alpha labels), PerceptQPA point
  probe_sweep.png 4 panels: target-metric BD-rate vs m at the default tau, diag + block
Works on partial data (missing series are skipped).  python plots/figs.py
"""
import csv, json, os, collections, re
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})   # house style
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(HERE)
QPS = '22,27,32,37'
# The tau-sweep bases must match the ones the tables use, or the figure shows a different system from the
# text: the Gauss-Newton rows moved to the smoothed (sigma=3) maps on 7 Sept.  The last two entries are the
# probe-sweep prefixes, which stay on the unsmoothed m-series (that is the sweep that was run over m).
# WD has no probe-count sweep, so its n-prefixes are empty and it simply contributes no points to the
# probe-sweep panel; it is a full row everywhere else.
MET = {'SSIM': ('wssim_sn256', 'wssim_sbh8m256', 'bd_ssim', 'sn', 'sbh8n'), 'MS-SSIM': ('wms_ssim_mn256', 'wms_ssim_mbh8m256', 'bd_ms_ssim', 'mn', 'mbh8n'),
       'LPIPS': ('wlpips_gnj3', 'wlpips_bh8', 'bd_lpips', 'gnn', 'gnb8n'), 'DISTS': ('wdists_dgnj3', 'wdists_dgnb8j3', 'bd_dists', 'dgnn', 'dgnb8n'),
       'WD': ('wwd_gnj3', 'wwd_gnb8j3', 'bd_wd2', '', '')}
PFX = {'SSIM': 'wssim_', 'MS-SSIM': 'wms_ssim_', 'LPIPS': 'wlpips_', 'DISTS': 'wdists_', 'WD': 'wwd_'}
NPANEL = len(MET)
rows = [r for r in csv.DictReader(open('results/kodak_json/summary.csv')) if r['n_images'] == '24' and r['qps'] == QPS and r['rdoq'] == '1']
qpa = json.load(open('work/qpa_bd_ctc.json'))['bd_rate_vs_anchor'] if os.path.exists('work/qpa_bd_ctc.json') else None
tau_def = json.load(open('work/default_tau.json')) if os.path.exists('work/default_tau.json') else {}
by = collections.defaultdict(dict)
for r in rows:
    by[r['base']][float(r['tau'])] = r
os.makedirs('paper/figures', exist_ok=True)

# --- tau sweep -----------------------------------------------------------------------------
# Drawn at the width it is printed at (\textwidth ~ 7.2 in) so the fonts are not scaled down by the include;
# one legend for the whole strip, above the panels, so it cannot sit on a curve.
fig, ax = plt.subplots(1, NPANEL, figsize=(7.2, 1.05))
for a, (met, (dbase, bbase, key, _, _)) in zip(ax, MET.items()):
    for base, lab, mk in ((dbase, 'D', 'o'), (bbase, 'B', 's')):
        pts = sorted(by[base].items())
        if len(pts) < 2: continue
        x = [float(r['bd_psnr_y']) for _, r in pts]; y = [float(r[key]) for _, r in pts]
        a.plot(x, y, marker=mk, ms=2.8, lw=1.1, label=lab)
    if qpa:
        a.plot(qpa['psnr_y'], qpa[key.replace('bd_', '')], marker='*', ms=7, color='k', ls='none', label='PQA'); a.axvline(qpa['psnr_y'], color='gray', ls='--', lw=0.6)
    a.set_title(met, fontsize=8, pad=2); a.set_xlabel('Y-PSNR BD-rate (%)', fontsize=7, labelpad=1.5); a.grid(alpha=0.3)
    a.tick_params(labelsize=6.5, pad=1.5, length=2)
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)
ax[0].set_ylabel('Target BD-rate (%)', fontsize=7, labelpad=1.5)
# minimal legend inside the first panel (its upper right is empty); the caption expands D, B and PQA
lg = ax[0].legend(loc='upper right', fontsize=6, frameon=True, fancybox=False, handlelength=1.3, handletextpad=0.4,
                  borderpad=0.3, labelspacing=0.2, borderaxespad=0.3)
lg.get_frame().set_edgecolor('0.75'); lg.get_frame().set_linewidth(0.5)
fig.tight_layout(pad=0.3, w_pad=0.5); fig.savefig('paper/figures/tau_sweep.png', dpi=300, bbox_inches='tight'); print('paper/figures/tau_sweep.png')

# --- probe sweep ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, NPANEL, figsize=(13 * NPANEL / 4.0, 3.2)); any_pts = False
for a, (met, (dbase, bbase, key, dpre, bpre)) in zip(ax, MET.items()):
    for kind, pre, full, mk in (('Diagonal', dpre, dbase, 'o'), ('Block', bpre, bbase, 's')):
        t = tau_def.get(full)
        if t is None: continue
        ms, ys = [], []
        for m in (16, 32, 64, 128):
            b = f'{PFX[met]}{pre}{m}'
            if b in by and t in by[b]: ms.append(m); ys.append(float(by[b][t][key]))
        if t in by[full]: ms.append(256); ys.append(float(by[full][t][key]))
        if len(ms) >= 2: a.plot(ms, ys, marker=mk, label=kind); any_pts = True
    a.set_xscale('log', base=2); a.set_title(met); a.set_xlabel('probes m'); a.grid(alpha=0.3)
ax[0].set_ylabel('target-metric BD-rate (%)'); ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig('paper/figures/probe_sweep.png', dpi=160); print('paper/figures/probe_sweep.png', '(partial)' if not any_pts else '')


# --- combined 2x4 figure for the paper (tau sweep on top, probe sweep below) -------------------
fig, axs = plt.subplots(2, NPANEL, figsize=(14 * NPANEL / 4.0, 4.0))
for a, (met, (dbase, bbase, key, _, _)) in zip(axs[0], MET.items()):
    for base, lab, mk in ((dbase, 'Diagonal', 'o'), (bbase, 'Block', 's')):
        pts = sorted(by[base].items())
        if len(pts) < 2: continue
        x = [float(r['bd_psnr_y']) for _, r in pts]; y = [float(r[key]) for _, r in pts]
        a.plot(x, y, marker=mk, label=lab)
    if qpa:
        a.plot(qpa['psnr_y'], qpa[key.replace('bd_', '')], marker='*', ms=11, color='k', ls='none', label='PerceptQPA'); a.axvline(qpa['psnr_y'], color='gray', ls='--', lw=0.8)
    a.set_title(met); a.set_xlabel('Y-PSNR BD-rate (%)'); a.grid(alpha=0.3)
axs[0, 0].set_ylabel('target-metric BD-rate (%)'); axs[0, 0].legend(fontsize=8)
for a, (met, (dbase, bbase, key, dpre, bpre)) in zip(axs[1], MET.items()):
    for kind, pre, full, mk in (('Diagonal', dpre, dbase, 'o'), ('Block', bpre, bbase, 's')):
        t = tau_def.get(full)
        if t is None: continue
        ms, ys = [], []
        for m in (16, 32, 64, 128):
            b = f'{PFX[met]}{pre}{m}'
            if b in by and t in by[b]: ms.append(m); ys.append(float(by[b][t][key]))
        if t in by[full]: ms.append(256); ys.append(float(by[full][t][key]))
        if len(ms) >= 2: a.plot(ms, ys, marker=mk, label=kind)
    a.set_xscale('log', base=2); a.set_xlabel('probes m'); a.grid(alpha=0.3)
axs[1, 0].set_ylabel('target-metric BD-rate (%)')
fig.tight_layout(); fig.savefig('paper/figures/sweeps.png', dpi=170); print('paper/figures/sweeps.png')
