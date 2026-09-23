"""Table 2 (ablation), Kodak, 24 images, CTC QPs: the neural-network metrics only, every row read at the same
Y-PSNR cost as Table 1 (PerceptQPA's, +8.1 %) by interpolating along the tau sweep -- so rows are
comparable, unlike the earlier form that quoted each row at its own default alpha and cost.

The HVP row uses Hutchinson tiles projected onto the PSD cone (--psd eigh).  Unprojected, the DISTS tiles carry
42 % of their eigenvalue mass on negative eigenvalues (estimator_diag.py, 24-image mean); unit-mean normalisation then scales them up and the encoder is handed negative distortion,
which is what produced Y-PSNR BD-rates of +170 to +370 % at every alpha.  Those unprojected numbers, the SSIM / MS-SSIM projection check, and the second-seed reproduction (a sentence
in the text rather than a row) are printed for the text.
Writes results/ablation/summary.csv and tables/table3_ablation.tex.
"""
import csv, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
rows = [r for r in csv.DictReader(open('results/kodak_json/summary.csv'))
        if r['n_images'] == '24' and r['qps'] == '22,27,32,37' and r['rdoq'] == '1']
by = {}
for r in rows:
    by.setdefault(r['base'], {})[float(r['tau'])] = r
AT = (json.load(open('work/qpa_cost.json'))['psnr_y_bd_rate'] if os.path.exists('work/qpa_cost.json')
      else json.load(open('work/qpa_bd_ctc.json'))['bd_rate_vs_anchor']['psnr_y'])
COLS = [('LPIPS', 'bd_lpips'), ('DISTS', 'bd_dists'), ('WD', 'bd_wd2')]
ROWS = [('Diagonal GN, $R{=}1$',                 ('wlpips_gn',        'wdists_dgn',        'wwd_gn')),
        ('Diagonal GN, smoothed $\\sigma{=}3$',   ('wlpips_gnj3',      'wdists_dgnj3',      'wwd_gnj3')),
        ('Block GN, $R{=}1$',                     ('wlpips_gnb8',      'wdists_dgnb8',      'wwd_gnb8')),
        ('Block GN, smoothed $\\sigma{=}3$',       ('wlpips_bh8',       'wdists_dgnb8j3',    'wwd_gnb8j3')),
        ('Block HVP', ('wlpips_hb8m256psd','wdists_dhb8m256psd','wwd_hb8m256psd'))]

def at_cost(base, key):
    """(value at AT, inside-grid flag, n_alphas); None when the sweep has fewer than two points."""
    pts = [(float(r['bd_psnr_y']), float(r[key])) for r in by.get(base, {}).values() if r.get(key)]
    pts = [p for p in pts if np.isfinite(p[0]) and np.isfinite(p[1])]
    if len(pts) < 2:
        return None
    x, y = zip(*sorted(pts)); x, y = np.array(x), np.array(y)
    return float(np.interp(AT, x, y)), bool(x.min() <= AT <= x.max()), len(pts)

def fmt(c):
    if c is None: return '--'
    v, inside, _ = c
    return f'${v:+.1f}$' + ('' if inside else '$^{\\dagger}$')

lines = ['\\begin{tabular}{lccc}', '\\toprule', 'Map & LPIPS & DISTS & WD \\\\', '\\midrule']
out = []
allcells = [[at_cost(b, k) for b, (_, k) in zip(bases, COLS)] for _, bases in ROWS]
best = [min((c[0] for c in col if c is not None), default=None) for col in zip(*allcells)]   # best (most negative) per column
def fmtb(c, b):
    t = fmt(c)
    return t.replace(f'${c[0]:+.1f}$', f'$\\mathbf{{{c[0]:+.1f}}}$') if c is not None and b is not None and abs(c[0] - b) < 5e-2 else t
for (label, bases), cells in zip(ROWS, allcells):
    lines.append(f'{label} & ' + ' & '.join(fmtb(c, b) for c, b in zip(cells, best)) + ' \\\\')
    rec = {'row': label, 'at_ypsnr': AT}
    for (name, _), b, c in zip(COLS, bases, cells):
        rec[f'{name.lower()}_base'] = b
        rec[f'{name.lower()}_bd'] = '' if c is None else c[0]
        rec[f'{name.lower()}_inside'] = '' if c is None else int(c[1])
        rec[f'{name.lower()}_n_alpha'] = '' if c is None else c[2]
    out.append(rec)
lines += ['\\bottomrule', '\\end{tabular}']
os.makedirs('results/ablation', exist_ok=True); os.makedirs('tables', exist_ok=True)
open('tables/table3_ablation.tex', 'w').write('\n'.join(lines) + '\n')
with open('results/ablation/summary.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

print(f'Table 2 at Y-PSNR cost {AT:+.2f} % (PerceptQPA)')
for rec in out:
    print(f"  {rec['row']:38s} " + '  '.join(
        f"{n}: {'--':>6s}" if rec[f'{n.lower()}_bd'] == '' else
        f"{n}: {rec[f'{n.lower()}_bd']:+6.1f}{'' if rec[f'{n.lower()}_inside'] else '*'} (n={rec[f'{n.lower()}_n_alpha']})"
        for n, _ in COLS))
print('\nfor the text -- unprojected HVP tiles, tau sweep (Y-PSNR / target):')
for b, k in (('wlpips_hb8m256', 'bd_lpips'), ('wdists_dhb8m256', 'bd_dists'), ('wwd_hb8m256', 'bd_wd2')):
    s = sorted((t, float(r['bd_psnr_y']), float(r[k])) for t, r in by.get(b, {}).items() if r.get(k))
    print(f"  {b:18s} " + ('  '.join(f'a{t:g}: {c:+.0f}/{v:+.1f}' for t, c, v in s) if s else '--'))
print('for the text -- second probe seed vs seed 0, smoothed block GN at the same alpha (Y-PSNR / target):')
for b0, b1, k in (('wlpips_bh8','wlpips_bh8s1','bd_lpips'), ('wdists_dgnb8j3','wdists_dgnb8j3s1','bd_dists')):
    for t, r1 in by.get(b1, {}).items():
        r0 = by.get(b0, {}).get(t)
        if r0: print(f"  {b0:16s} a{t:g}: seed0 {float(r0['bd_psnr_y']):+.1f}/{float(r0[k]):+.1f}  seed1 {float(r1['bd_psnr_y']):+.1f}/{float(r1[k]):+.1f}  diff {abs(float(r1[k])-float(r0[k])):.2f}")
print('for the text -- SSIM / MS-SSIM Hutchinson block, unprojected vs PSD-projected, at their alpha (Y-PSNR / target):')
for lab, sb, mb in (('unprojected', 'wssim_sbh8m256', 'wms_ssim_mbh8m256'), ('PSD-projected', 'wssim_sbh8m256psd', 'wms_ssim_mbh8m256psd')):
    S = by.get(sb, {}); M = by.get(mb, {})
    ts = [t for t in S if t in by.get('wssim_sbh8m256psd', {})] or list(S)[:1]
    tm = [t for t in M if t in by.get('wms_ssim_mbh8m256psd', {})] or list(M)[:1]
    fs = lambda d, t, k: f"a{t:g}: {float(d[t]['bd_psnr_y']):+.1f}/{float(d[t][k]):+.1f}" if t in d else '--'
    print(f"  {lab:14s} SSIM {fs(S, ts[0], 'bd_ssim') if ts else '--'}   MS-SSIM {fs(M, tm[0], 'bd_ms_ssim') if tm else '--'}")
