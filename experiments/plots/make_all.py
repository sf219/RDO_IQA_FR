"""Regenerate every table and figure of the ICASSP 2027 paper from CSVs / result files.

    python plots/make_all.py                 # everything that has data
    python plots/make_all.py --at 6.98       # Table 1 at another common Y-PSNR cost

Outputs: tables/*.tex, figures/*.png, results/table1/summary.csv.  Steps whose inputs are
missing are skipped with a message; nothing is edited by hand.  Series used for Table 1 are in
SERIES below: switch to the regenerated tags (sbh8m256, sn256, ...) once results/vvc_rdo_*.json
carry them on 24 images.
"""
import argparse, csv, json, os, shutil, subprocess, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
os.chdir(HERE)

# Table 1 series: metric -> (diag base tag, block base tag).
SERIES_NEW = {'SSIM': ('wssim_sn256', 'wssim_sbh8m256'), 'MS-SSIM': ('wms_ssim_mn256', 'wms_ssim_mbh8m256'),
              'LPIPS': ('wlpips_gnj3', 'wlpips_bh8'), 'DISTS': ('wdists_dgnj3', 'wdists_dgnb8j3'),
              'WD': ('wwd_gnj3', 'wwd_gnb8j3')}   # regenerated, 24 images, CTC QPs; sigma-3 (unified GN recipe)
SERIES_OLD = {'SSIM': ('wssim_t24', 'wssim_sbh8'), 'MS-SSIM': ('wms_ssim_t24', 'wms_ssim_mbh8'),
              'LPIPS': ('wlpips_f24', 'wlpips_bh8'), 'DISTS': ('wdists_dgnj3', 'wdists_dgnb8j3')}  # development, 12 images, QP 27-42
SERIES = SERIES_NEW
Q = ['bd_psnr_y', 'bd_psnr_yuv', 'bd_ssim', 'bd_ms_ssim', 'bd_lpips', 'bd_dists', 'bd_wd0', 'bd_wd2', 'bd_wd4']
QN = ['Y-PSNR', 'YUV-PSNR', 'SSIM', 'MS-SSIM', 'LPIPS', 'DISTS', 'WD0', 'WD', 'WD4']   # WD = log2 sigma 2


def run(cmd, tag):
    print(f'--- {tag}: {" ".join(cmd)}', flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f'    FAILED ({r.returncode}): {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""}')
    return r.returncode == 0


def copy_fig(src, name):
    if os.path.exists(src):
        os.makedirs('figures', exist_ok=True)
        shutil.copy(src, os.path.join('figures', name)); print(f'    -> figures/{name}')


def table1(at, n_images):
    fn = 'results/kodak_json/summary.csv'
    if not os.path.exists(fn):
        print('table1: no results/kodak_json/summary.csv'); return
    want = '22,27,32,37' if SERIES is SERIES_NEW else '27,32,37,42'
    rows = [r for r in csv.DictReader(open(fn)) if r['rdoq'] == '1' and int(r['n_images']) == n_images and r.get('qps', '') == want]
    by = collections.defaultdict(dict); meta = {}
    for r in rows:
        by[r['base']][float(r['tau'])] = [float(r[q]) if r.get(q) else np.nan for q in Q]
        meta[r['base']] = (r['m'], r['sigma'], r['estimator'])
    out = []; os.makedirs('results/table1', exist_ok=True); os.makedirs('tables', exist_ok=True)
    # every metric gets a shaded band of its two rows, so the D/B pair reads as one block, and the best
    # value of each column (including PerceptQPA, labelled PQA; the caption expands it) is set in bold.  Cells are collected first, then marked.
    lines = ['\\begin{tabular}{llrrrrr}', '\\toprule', 'RDO Target & Map & SSIM & MS-SSIM & LPIPS & DISTS & WD \\\\', '\\midrule']
    body = []
    for met, (dbase, bbase) in SERIES.items():
        for kind, base in (('D', dbase), ('B', bbase)):
            if base not in by or len(by[base]) < 2:
                lines.append(f'{met} & {kind} & \\multicolumn{{5}}{{c}}{{--}} \\\\'); continue
            pts = sorted(by[base].items()); x = np.array([p[1][0] for p in pts]); o = np.argsort(x)
            v = [float(np.interp(at, x[o], np.array([p[1][i] for p in pts])[o])) for i in range(len(Q))]
            inside = x.min() <= at <= x.max()
            out.append(dict(metric=met, kind=kind, base=base, m=meta[base][0], sigma=meta[base][1], estimator=meta[base][2],
                            at_ypsnr=at, inside_sweep=int(inside), **{q: v[i] for i, q in enumerate(Q)}))
            cells = []
            for i, q in enumerate(['bd_ssim', 'bd_ms_ssim', 'bd_lpips', 'bd_dists', 'bd_wd2']):
                val = v[Q.index(q)]
                if not np.isfinite(val):
                    cells.append('--'); continue
                s = f'${val:+.1f}$'
                if QN[Q.index(q)] == met: s = f'$\\mathbf{{{val:+.1f}}}$'
                if not inside: s += '$^\\dagger$'
                cells.append(s)
            body.append((met, kind, v, inside))
    KEYS = ['bd_ssim', 'bd_ms_ssim', 'bd_lpips', 'bd_dists', 'bd_wd2']
    q = json.load(open('work/qpa_bd_ctc.json'))['bd_rate_vs_anchor'] if os.path.exists('work/qpa_bd_ctc.json') else None
    qv = [q.get(k) for k in ('ssim', 'ms_ssim', 'lpips', 'dists', 'wd2')] if q else [None] * 5

    def render(with_qpa):
        # best (most negative) per column over the body and, when shown, the PerceptQPA row
        best = []
        for i in range(len(KEYS)):
            col = [v[Q.index(KEYS[i])] for _, _, v, _ in body]
            col = [c for c in col if np.isfinite(c)]
            if with_qpa and qv[i] is not None:
                col.append(float(qv[i]))
            best.append(min(col) if col else None)

        def cell(val, inside, i):
            if not np.isfinite(val):
                return '--'
            t = f'{val:+.1f}'
            hit = best[i] is not None and abs(val - best[i]) < 5e-2
            return ('$\\mathbf{%s}$' % t if hit else f'${t}$') + ('' if inside else '$^\\dagger$')

        rows_tex = []
        if with_qpa:
            qc = []
            for i in range(len(KEYS)):
                if qv[i] is None:
                    qc.append('--')
                else:
                    t = f'{float(qv[i]):+.1f}'
                    qc.append('$\\mathbf{%s}$' % t if best[i] is not None and abs(float(qv[i]) - best[i]) < 5e-2 else f'${t}$')
            rows_tex += ['--      & PQA & ' + ' & '.join(qc) + ' \\\\', '\\midrule']
        for g, (met, kind, v, inside) in enumerate(body):
            cs = []
            for i in range(len(KEYS)):
                cs.append(cell(v[Q.index(KEYS[i])], inside, i))
            shade = '\\rowcolor{metband}' if (g // 2) % 2 == 0 else ''
            rows_tex.append(f'{shade}{met if kind == "D" else ""} & {kind} & ' + ' & '.join(cs) + ' \\\\')
        return lines + rows_tex + ['\\bottomrule', '\\end{tabular}']

    open('tables/table1_kodak.tex', 'w').write('\n'.join(render(False)) + '\n')
    if q:
        open('tables/table1_paper.tex', 'w').write('\n'.join(render(True)) + '\n')
    with open('results/table1/summary.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(f'table1: {len(out)} rows at Y-PSNR +{at:g}% ({n_images} images) -> tables/table1_kodak.tex')


def complexity_table():
    fn = 'results/complexity/summary.csv'
    if not os.path.exists(fn):
        return
    rows = [r for r in csv.DictReader(open(fn)) if r['kind'] == 'enc']
    tt = collections.defaultdict(list)
    for r in rows:
        tt[(r['metric'], r['config'])].append(float(r['enc_time_s']))
    a = np.mean(tt[('-', 'anchor')])
    lines = ['\\begin{tabular}{lrr}', '\\toprule', 'Metric & diagonal & block \\\\', '\\midrule']
    for met in ('SSIM', 'MS_SSIM', 'LPIPS', 'DISTS'):
        d = np.mean(tt[(met, 'diag')]) / a if (met, 'diag') in tt else np.nan
        b = np.mean(tt[(met, 'block')]) / a if (met, 'block') in tt else np.nan
        lines.append(f'{met.replace("_", "-")} & {d:.2f}$\\times$ & {b:.2f}$\\times$ \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    os.makedirs('tables', exist_ok=True); open('tables/complexity.tex', 'w').write('\n'.join(lines) + '\n')
    print(f'complexity table: anchor {a:.1f} s/encode -> tables/complexity.tex')


def clic_table():
    fn = 'results/clic/summary.csv'
    if not os.path.exists(fn):
        print('clic: no results/clic/summary.csv yet'); return
    rows = list(csv.DictReader(open(fn)))
    lines = ['\\begin{tabular}{lrrrrrr}', '\\toprule', 'Config & $n$ & Y-PSNR & SSIM & MS-SSIM & LPIPS & DISTS \\\\', '\\midrule']
    for r in rows:
        name = r['config'].replace('_', '\\_')
        vals = ' & '.join('$%+.1f$' % float(r[q]) for q in ['bd_psnr_y', 'bd_ssim', 'bd_ms_ssim', 'bd_lpips', 'bd_dists'])
        lines.append(name + ' & ' + r['n_images'] + ' & ' + vals + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    open('tables/table2_clic.tex', 'w').write('\n'.join(lines) + '\n'); print('clic -> tables/table2_clic.tex')


def main():
    ap = argparse.ArgumentParser()
    qpa = json.load(open('work/qpa_cost.json'))['psnr_y_bd_rate'] if os.path.exists('work/qpa_cost.json') else None
    ap.add_argument('--at', type=float, default=qpa if qpa is not None else 6.98, help='common Y-PSNR BD-rate for Table 1 (PerceptQPA cost; default from work/qpa_cost.json)')
    ap.add_argument('--n', type=int, default=24, help='image count of the Table 1 series')
    ap.add_argument('--series', choices=['new', 'old'], default='new', help='regenerated (24 images, CTC) or development (12 images) series')
    a = ap.parse_args()
    global SERIES
    SERIES = SERIES_NEW if a.series == 'new' else SERIES_OLD
    print(f'Table 1: {a.series} series, {a.n} images, at Y-PSNR +{a.at:.2f}%')
    run([PY, 'results_to_csv.py'], 'convert JSON -> CSV')
    if os.path.exists('work/enc/kodim01/qpa_qp22.bin'):
        run([PY, 'qpa_bd.py'], 'PerceptQPA BD-rates (current to_quality)')
    bench = 'work/timebench_v3' if os.path.exists('work/timebench_v3.out') and 'TIMEBENCH_V3_DONE' in open('work/timebench_v3.out').read() else 'work/timebench_v2'
    if os.path.isdir(bench):
        run([PY, 'complexity_csv.py', '--enc', bench], f'complexity CSV ({bench})')
        if run([PY, 'enctime2.py', '--enc', bench, '--n', '3', '--qps'] + (['22', '27', '32', '37'] if bench.endswith('v3') else ['27', '32', '37', '42']) + ['--out', 'fig_dump/cost_two_panel.png'], 'complexity figure'):
            copy_fig('fig_dump/cost_two_panel.png', 'complexity.png')
    complexity_table()
    if not os.path.exists('results/estimator_diag/summary.csv'):
        run([PY, 'estimator_diag.py'], 'estimator diagnostics')
    table1(a.at, a.n)
    run([PY, 'plots/figs.py'], 'tau sweep + probe sweep figures (24 images, CTC, from CSV)')
    if os.path.isdir('results/clic/done') and os.listdir('results/clic/done'):
        run([PY, 'clic_run.py', '--summary'], 'CLIC per-image summary from the done markers')
    clic_table()
    if os.path.exists('results/clic/summary.csv'):
        run([PY, 'clic_table.py'], 'Table 2 paper form')
    if os.path.exists('work/default_tau.json'):
        run([PY, 'lit_table.py'], 'literature-protocol table')
        run([PY, 'ablation_table.py'], 'ablation table')
        run([PY, 'stats_table1.py'], 'Table 1 win counts + bootstrap CIs')
        run([PY, 'paper_numbers.py'], 'paper numbers from CSV')
    print('MAKE_ALL_DONE')


if __name__ == '__main__':
    main()
