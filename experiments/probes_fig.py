"""Runtime versus accuracy of the block map against the probe budget, at one Y-PSNR cost.

x: time to build the map on an A100 (mean over kodim01/05/19, work/maptime_probes_<img>.json, carc/probes_maptime.sbatch).
y: BD-rate on the metric the map is built for, read at PerceptQPA's Y-PSNR cost (+8.15 %) by interpolating the
   tau sweep of that (metric, m) series -- so every marker sits at the same fidelity loss, unlike the old
   Table 5, which quoted each m at its own cost and tau.
One line per metric, markers m = 16, 32, 64, 128, 256.  The encoder's own cost does not depend on m (Fig. 5).
Series: m < 256 from carc/probes_sweep.sbatch (tags sbh8n<m>, mbh8n<m>, bh8n<m>, dgnb8j3n<m>, gnb8j3n<m>);
m = 256 is the Table-1 block series.  Writes figures/probes.png and results/probes/fig.csv.
"""
import csv, glob, json, os, collections
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
COL = {'SSIM': 'tab:green', 'MS_SSIM': 'tab:purple', 'LPIPS': 'tab:red', 'DISTS': 'tab:orange', 'WD': 'tab:blue'}
NAME = {'SSIM': 'SSIM', 'MS_SSIM': 'MS-SSIM', 'LPIPS': 'LPIPS', 'DISTS': 'DISTS', 'WD': 'WD'}
# metric -> (base prefix of the m<256 series, base of the m=256 series, quality key)
SER = {'SSIM': ('wssim_sbh8n', 'wssim_sbh8m256', 'bd_ssim'), 'MS_SSIM': ('wms_ssim_mbh8n', 'wms_ssim_mbh8m256', 'bd_ms_ssim'),
       'LPIPS': ('wlpips_bh8n', 'wlpips_bh8', 'bd_lpips'), 'DISTS': ('wdists_dgnb8j3n', 'wdists_dgnb8j3', 'bd_dists'),
       'WD': ('wwd_gnb8j3n', 'wwd_gnb8j3', 'bd_wd2')}
MS = [16, 32, 64, 128, 256]
AT = json.load(open('work/qpa_cost.json'))['psnr_y_bd_rate']

rows = [r for r in csv.DictReader(open('results/kodak_json/summary.csv')) if r['n_images'] == '24' and r['qps'] == '22,27,32,37' and r['rdoq'] == '1']
by = collections.defaultdict(dict)
for r in rows:
    by[r['base']][float(r['tau'])] = r

def at_cost(base, key):
    pts = sorted((float(r['bd_psnr_y']), float(r[key])) for r in by.get(base, {}).values() if r.get(key))
    pts = [p for p in pts if all(np.isfinite(p))]
    if len(pts) < 2:
        return None
    x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    return float(np.interp(AT, x, y)), bool(x.min() <= AT <= x.max()), len(pts)

# map time per (metric, m): mean over the timed images
T = collections.defaultdict(list)
for f in glob.glob('work/maptime_probes_*.json'):
    for r in json.load(open(f)):
        if r['kind'] == 'block':
            T[(r['metric'], int(r['n']))].append(float(r['sec']))
tsec = {k: float(np.mean(v)) for k, v in T.items()}
TIMING = 'measured per m (carc/probes_maptime.sbatch)'
if not tsec:
    # fallback until the A100 timing job runs: the Fig.-5 times at m = 256 on the A100, scaled by m/256 --
    # the estimator does one matrix-vector product per probe, and the old bench (bench/map_time.json) is linear in m
    M256 = json.load(open('work/maptime_a100_fwd.json'))
    for met in SER:
        for m in MS:
            tsec[(met, m)] = float(M256[f'{met}|block']) * m / 256.0
    TIMING = 'Fig.-5 A100 times at m=256 scaled by m/256 (timing job pending)'

# explicit placement (no tight_layout: it shrinks the axes to fit the legend inside the figure height, so the plot got
# smaller every time the legend moved down).  Figure fractions: axes 1.25 in tall, 0.86 in below it for ticks, the
# x label, a clear gap and the legend; the paper scales the image to the column width, so only the axes width matters.
fig = plt.figure(figsize=(3.4, 1.45))
ax = fig.add_axes([0.15, 0.51, 0.83, 0.448])   # axes 0.65 in tall; the 0.74 in below it (ticks, label, gap, legend) is unchanged
out = []
for met, (pre, full, key) in SER.items():
    xs, ys, ms = [], [], []
    for m in MS:
        base = full if m == 256 else f'{pre}{m}'
        c = at_cost(base, key); t = tsec.get((met, m))
        out.append(dict(metric=met, m=m, base=base, map_time_s=t if t is not None else '', bd_target=c[0] if c else '',
                        inside=int(c[1]) if c else '', n_tau=c[2] if c else ''))
        if c is None or t is None:
            continue
        xs.append(t); ys.append(c[0]); ms.append(m)
    if xs:
        ax.plot(xs, ys, marker='o', ms=3.5, lw=1.3, color=COL[met], label=NAME[met])
ax.set_xscale('log')
ax.set_xlabel('Hessian estimation time (s)', fontsize=9)
ax.set_ylabel('BD-rate (%)', fontsize=9)
ax.grid(alpha=.3, ls='--', which='both'); ax.set_axisbelow(True); ax.tick_params(labelsize=8)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
hd, lb = ax.get_legend_handles_labels()
lg = fig.legend(hd, lb, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=5, fontsize=7, frameon=True, fancybox=False,
                handlelength=1.4, columnspacing=1.0, borderpad=0.4)
lg.get_frame().set_edgecolor('0.75'); lg.get_frame().set_linewidth(0.6)
os.makedirs('figures', exist_ok=True); os.makedirs('results/probes', exist_ok=True)
fig.savefig('figures/probes.png', bbox_inches='tight', dpi=300)
with open('results/probes/fig.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print(f'at Y-PSNR cost {AT:+.2f} %   [timing: {TIMING}]')
for r in out:
    print(f"  {NAME[r['metric']]:8s} m={r['m']:3d}  time {r['map_time_s'] if r['map_time_s']=='' else round(r['map_time_s'],2)!s:>6}  "
          f"BD {r['bd_target'] if r['bd_target']=='' else round(r['bd_target'],1)!s:>6}{'' if r['inside'] in ('',1) else '*'}  (n_tau={r['n_tau']})")
print('-> figures/probes.png')
