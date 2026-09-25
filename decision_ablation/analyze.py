"""Decision-level ablation table: which RDO decisions carry the weighted-RDO gain.

Reads summary.csv.  For every (metric, mask) series
w<metric>_<tag>_m<mask> it interpolates the tau sweep (RDOQ rows) to PerceptQPA's Y-PSNR cost
(work/qpa_cost.json, +8.15 %) -- the paper's matched-cost reading -- and reports the BD-rate on the
target metric and on the other four.  Then, per decision k:
  LOO  = BD(255 - 2^k) - BD(mask 255)   : target-metric gain lost when that one decision goes back to SSE
  ONLY = BD(2^k)                        : target-metric gain when only that decision is weighted
Writes work/rdo_ablation/table.md and table.csv.  Rows outside their sweep are marked with * (value = nearest sweep endpoint)..
"""
import csv, json, os, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
SUMMARY, COST, OUT = 'summary.csv', 'qpa_cost.json', '.'

DEC = [(0, 'partition'), (1, 'LFNST index'), (2, 'MTS flag'), (3, 'MTS/TS type'), (4, 'luma mode'),
       (5, 'ISP'), (6, 'chroma'), (7, 'RDOQ lambda')]
SER = {'SSIM': ('wssim_sbh8m256', 'bd_ssim'), 'MS_SSIM': ('wms_ssim_mbh8m256', 'bd_ms_ssim'),
       'LPIPS': ('wlpips_bh8', 'bd_lpips'), 'DISTS': ('wdists_dgnb8j3', 'bd_dists'), 'WD': ('wwd_gnb8j3', 'bd_wd2')}
Q = ['bd_psnr_y', 'bd_ssim', 'bd_ms_ssim', 'bd_lpips', 'bd_dists', 'bd_wd2']
AT = json.load(open(COST))['psnr_y_bd_rate']

rows = [r for r in csv.DictReader(open(SUMMARY))
        if r['rdoq'] == '1' and r['n_images'] == '24' and r['qps'] == '22,27,32,37']
by = collections.defaultdict(dict)
for r in rows:
    by[r['base']][float(r['tau'])] = r

def at_cost(base):
    pts = [(float(r['bd_psnr_y']), r) for r in by.get(base, {}).values() if r.get('bd_psnr_y')]
    if len(pts) < 2:
        return None
    pts.sort(key=lambda t: t[0])
    x = np.array([p[0] for p in pts])
    inside = x.min() <= AT <= x.max()
    out = {}
    for k in Q:
        y = np.array([float(p[1][k]) if p[1].get(k) else np.nan for p in pts])
        out[k] = float(np.interp(AT, x, y)) if np.all(np.isfinite(y)) else float('nan')
    return out, inside, len(pts)

def cell(v, inside=True):
    return '' if v is None or v != v else f'{v:+.1f}' + ('' if inside else '*')

def peak(base, key):
    """Largest saving on the target metric anywhere on the sweep, with the tau and Y-PSNR cost where it occurs;
    weak masks never reach the matched cost, so this is the honest per-mask number."""
    pts = [(float(r['tau']), float(r['bd_psnr_y']), float(r[key])) for r in by.get(base, {}).values() if r.get(key)]
    if not pts:
        return None
    t, c, v = min(pts, key=lambda q: q[2])
    return v, t, c, max(q[1] for q in pts)

md = [f'# Decision ablation, block maps, Kodak, at +{AT:.2f} % Y-PSNR (interpolated along tau)', '']
csvrows = []
for met, (base, key) in SER.items():
    ref = at_cost(base + '_m255') or at_cost(base)        # mask 255 re-run if present, else the paper's series
    md += [f'## {met}  (series `{base}`, target column `{key[3:]}`)', '',
           f'reference, all decisions weighted (mask 255): target {cell(*((ref[0][key], ref[1]) if ref else (None,)))}', '',
           '| decision | leave-one-out mask | target BD (LOO) | gain lost | only-one mask | target BD (ONLY, matched cost) | ONLY peak BD (tau, cost, max cost) | ' + ' | '.join(f'LOO {q[3:]}' for q in Q) + ' |',
           '|---|---|---|---|---|---|---|' + '---|' * len(Q)]
    for k, name in DEC:
        loo = at_cost(f'{base}_m{255 - (1 << k)}'); one = at_cost(f'{base}_m{1 << k}')
        lost = (loo[0][key] - ref[0][key]) if (ref and loo) else None
        pk = peak(f'{base}_m{1 << k}', key)
        pks = f'{pk[0]:+.1f} (tau {pk[1]:g}, {pk[2]:+.1f} %, max {pk[3]:+.1f} %)' if pk else ''
        md.append(f'| {name} | {255 - (1 << k)} | {cell(*((loo[0][key], loo[1]) if loo else (None,)))} | {cell(lost)} | {1 << k} | '
                  f'{cell(*((one[0][key], one[1]) if one else (None,)))} | {pks} | '
                  + ' | '.join(cell(loo[0][q], loo[1]) if loo else '' for q in Q) + ' |')
        csvrows.append(dict(metric=met, decision=name, bit=k, ref_target=ref[0][key] if ref else '',
                            loo_mask=255 - (1 << k), loo_target=loo[0][key] if loo else '', loo_inside=int(loo[1]) if loo else '',
                            gain_lost=lost if lost is not None else '', one_mask=1 << k, one_target=one[0][key] if one else '',
                            one_inside=int(one[1]) if one else '', one_peak=pk[0] if pk else '', one_peak_tau=pk[1] if pk else '',
                            one_peak_cost=pk[2] if pk else '', one_max_cost=pk[3] if pk else '',
                            **{f'loo_{q[3:]}': loo[0][q] if loo else '' for q in Q}, **{f'one_{q[3:]}': one[0][q] if one else '' for q in Q}))
    md.append('')
md += ['Negative BD-rate = saving on that metric versus the SSE anchor. "gain lost" = leave-one-out minus reference '
       '(positive: the decision contributes). `*` = the +8.15 % point lies outside the tau sweep; the value shown is the nearest sweep endpoint (np.interp clamps), not an extrapolation.']
os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, 'table.md'), 'w').write('\n'.join(md) + '\n')
with open(os.path.join(OUT, 'table.csv'), 'w', newline='') as fh:
    if csvrows:
        w = csv.DictWriter(fh, fieldnames=list(csvrows[0].keys())); w.writeheader(); w.writerows(csvrows)
print('\n'.join(md))
