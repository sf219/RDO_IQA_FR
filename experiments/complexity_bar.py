"""Two-panel cost figure in the house style of enctime2.py: encoding time and curvature-map time, both in
seconds on a shared axis so the two are directly comparable (that is the whole point of not normalising).
Encoding times are the mean over the four CTC QPs on the three benched Kodak images; map times are for one
map at m=256 on the same images.  Map times come from work/maptime_uniform.json when the uniform run has
finished, otherwise from the older bench/map_time.json (four metrics, one image, CPU-drawn probes).
Writes figures/complexity_bar.png."""
import csv, json, collections, os, statistics as st
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
COL = {'SSIM': 'tab:green', 'MS_SSIM': 'tab:purple', 'LPIPS': 'tab:red', 'DISTS': 'tab:orange', 'WD': 'tab:blue'}
NAME = {'SSIM': 'SSIM', 'MS_SSIM': 'MS-SSIM', 'LPIPS': 'LPIPS', 'DISTS': 'DISTS', 'WD': 'WD'}
METS = ['SSIM', 'MS_SSIM', 'LPIPS', 'DISTS', 'WD']

R = [r for r in csv.DictReader(open('results/complexity/summary.csv')) if r['kind'] == 'enc' and r['enc_time_s']]
by = collections.defaultdict(dict)
for r in R:
    by[(r['image'], r['qp'])][(r['metric'], r['config'])] = float(r['enc_time_s'])
anchor = st.mean([v[('-', 'anchor')] for v in by.values() if ('-', 'anchor') in v])
def enc(m, k):
    t = [v[(m, k)] for v in by.values() if (m, k) in v]
    return st.mean(t) if t else np.nan

# Map times: prefer the datacentre-GPU run (work/maptime_l40s.json), fall back to the local GTX 1080 Ti
# measurement, then to the pre-GPU-probe numbers.  CARD is printed so the caption can name the hardware.
for fn, CARD in (('work/maptime_a100_fwd.json', 'A100-PCIE-40GB'), ('work/maptime_a100.json', 'A100-PCIE-40GB'), ('work/maptime_l40s.json', 'L40S'),
                 ('work/maptime_uniform.json', 'GTX 1080 Ti')):
    if os.path.exists(fn):
        M = json.load(open(fn)); break
else:
    M, CARD = {}, 'legacy'
def mp(m, k): return M.get(f'{m}|{k}', np.nan)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 1.5), sharey=True)
x = np.arange(len(METS)); w = 0.38
for ax, f, title in ((axL, enc, 'Encoding'), (axR, mp, 'Curvature estimation')):
    for i, k in enumerate(('diag', 'block')):
        ax.bar(x + (i - 0.5) * w, [f(m, k) for m in METS], w,
               color=[COL[m] for m in METS], hatch='' if k == 'diag' else '///',
               edgecolor='white', linewidth=.4)
    ax.set_xticks(x); ax.set_xticklabels([NAME[m] for m in METS], fontsize=12)
    ax.set_title(title, fontsize=15)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=.3, ls='--', axis='y'); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
axL.set_ylabel('Time (s)', fontsize=15)
axL.axhline(anchor, color='k', lw=1.0, ls='--')   # SSE-RDO encode time; named in the caption
h = [plt.Rectangle((0, 0), 1, 1, facecolor='0.6', hatch=hh, edgecolor='white', linewidth=.4) for hh in ('', '///')]
axR.legend(h, ['diagonal', 'block'], fontsize=11, ncol=2, frameon=False, loc='upper right')
fig.tight_layout()
os.makedirs('figures', exist_ok=True)
fig.savefig('figures/complexity_bar.png', bbox_inches='tight', dpi=300)
print(f'map times measured on {CARD}; encode anchor {anchor:.1f} s')
for m in METS:
    print(f'  {NAME[m]:8s} enc {enc(m,"diag"):6.1f}/{enc(m,"block"):6.1f} s   map {mp(m,"diag"):6.1f}/{mp(m,"block"):6.1f} s')
