"""Table 2 (paper form): CLIC professional validation, diag/block rows per metric interpolated along the 3-tau
grid to PerceptQPA's CLIC Y-PSNR cost (same np.interp as Table 1), PerceptQPA row on top.
Reads results/clic/summary.csv (clic_run.py --summary); writes tables/table2_clic_paper.tex and \clicN / \clicAt (the Y-PSNR cost, quoted in the caption;
the Y-PSNR column itself is not printed since every row sits at that cost)."""
import csv, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
Q = ['bd_psnr_y', 'bd_ssim', 'bd_ms_ssim', 'bd_lpips', 'bd_dists', 'bd_wd2']
rows = list(csv.DictReader(open('results/clic/summary.csv')))
by = {r['config']: r for r in rows}
n = int(by['qpa']['n_images']); at = float(by['qpa']['bd_psnr_y'])
def f(v): return f'${v:+.1f}$' if v == v else '--'
def grid(prefix):
    pts = sorted((float(c.split('_tau')[1].split('_')[0]), by[c]) for c in by if c.startswith(prefix + '_tau'))
    x = np.array([float(r['bd_psnr_y']) for _, r in pts]); o = np.argsort(x)
    inside = x.min() <= at <= x.max()
    return [float(np.interp(at, x[o], np.array([float(r[k]) if r.get(k) else np.nan for _, r in pts])[o])) for k in Q], inside
open('tables/clic_macros.tex', 'w').write(f'\\providecommand{{\\clicN}}{{{n}}}\n\\providecommand{{\\clicAt}}{{{at:+.1f}}}\n')
lines = ['\\begin{tabular}{llrrrrr}', '\\toprule', 'RDO Target & Map & SSIM & MS-SSIM & LPIPS & DISTS & WD \\\\', '\\midrule',
         '-- & PQA & ' + ' & '.join(f(float(by['qpa'][k]) if by['qpa'].get(k) else float('nan')) for k in Q[1:]) + ' \\\\', '\\midrule']
out = {}
def full(prefix):   # every configuration of the prefix present on all n images
    cs = [c for c in by if c.startswith(prefix + '_')]
    return bool(cs) and all(int(by[c]['n_images']) == n for c in cs)
def pref(base):   # sigma-3 pass rows when they cover all images (unified GN recipe, 7 Sept)
    return base + 'j3' if full(base + 'j3') else base
METR = [('SSIM', 'wssim'), ('MS-SSIM', 'wms_ssim'), ('LPIPS', pref('wlpips')), ('DISTS', pref('wdists'))] + ([('WD', 'wwdj3')] if full('wwdj3') else [])
body = []
for metric, key in METR:
    for kind, tag in (('diag', 'D'), ('block', 'B')):
        v, inside = grid(f'{key}_{kind}'); out[(metric, tag)] = v
        body.append((metric, tag, v, inside))
# best (most negative) per quality column, PerceptQPA included; Y-PSNR is a cost, not a score, so it is skipped
qpa_v = [float(by['qpa'][k]) if by['qpa'].get(k) else float('nan') for k in Q]
best = []
for i in range(len(Q)):
    if i == 0:
        best.append(None); continue
    col = [v[i] for _, _, v, _ in body if v[i] == v[i]]
    if qpa_v[i] == qpa_v[i]:
        col.append(qpa_v[i])
    best.append(min(col) if col else None)
def fb(v, i):
    if v != v:
        return '--'
    t = f'{v:+.1f}'
    return '$\\mathbf{%s}$' % t if best[i] is not None and abs(v - best[i]) < 5e-2 else f'${t}$'
lines[4] = '-- & PQA & ' + ' & '.join(fb(qpa_v[i], i) for i in range(1, len(Q))) + ' \\\\'
for g, (metric, tag, v, inside) in enumerate(body):
    shade = '\\rowcolor{metband}' if (g // 2) % 2 == 0 else ''
    lines.append(f"{shade}{metric if tag == 'D' else ''} & {tag}{'' if inside else '$^*$'} & "
                 + ' & '.join(fb(v[i], i) for i in range(1, len(Q))) + ' \\\\')
lines += ['\\bottomrule', '\\end{tabular}']
open('tables/table2_clic_paper.tex', 'w').write('\n'.join(lines) + '\n')
print(f'clic paper table: n={n} at Y-PSNR {at:+.2f}')
for (m, t), v in out.items(): print(f'  {m:8s} {t}  ' + '  '.join(f'{k[3:]} {x:+.2f}' for k, x in zip(Q, v)))
