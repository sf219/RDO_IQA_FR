"""Table 4: head-to-head with [yang2025bit] under their own protocol.

Every row is one configuration they report, at the RGB-PSNR BD-rate they report for it.  Our columns are
our method read at that same cost, so each line compares two allocations that paid the same price in
fidelity -- the comparison the earlier version of this table could not make, because the rows sat at
different operating points.  Ours uses the map matched to the column's metric, mirroring their own
"opt. SSIM / opt. MS-SSIM / opt. LPIPS" rows.  Anchor, encoder version (VTM-23.0), CTC QPs, chroma
allocation and the RGB metric convention are theirs throughout.
"""
import csv, os, sys
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from litqp_match import THEIRS, families, at_cost

# MS-SSIM-RGB map and LPIPS-Alex map families (weighted RDOQ on).  LIT_MS / LIT_LP select the series shown:
# 'rq_mrgbd'/'rq_alexd' are the diagonal maps, 'rqb_mrgbb'/'rqb_alexb' the block maps (12 Sept grid).
MS, LP = os.environ.get('LIT_MS', 'rqb_mrgbb'), os.environ.get('LIT_LP', 'rqb_alexb')   # paper rows: block maps; LIT_MS=rq_mrgbd LIT_LP=rq_alexd gives the diagonal-map row
MATCH  = 0.98                            # their best configuration's RGB-PSNR cost; ours is read there

def plain(s):
    return s.replace('\\texttt{', '').replace('}', '').replace('\\&', '&').replace("\\'", '')

def main():
    fam, solo = families(os.environ.get('LIT_CSV', 'results/litqp/summary.csv'))
    ms_o = at_cost(fam[MS], MATCH)[0]
    lp_o = at_cost(fam[LP], MATCH)[1]
    q = solo['qpa230']
    # one row per method, each at its own RGB-PSNR cost; ours is read at the cost of the method it is
    # being compared with, so the last two lines are a like-for-like pair
    rows = [('Zero QP map \\cite{yang2025bit}',        -2.75,  -3.46,  -2.42),
            ('\\texttt{PerceptQPA} \\cite{yang2025bit}', 2.85, -11.86, -11.96),
            ('\\texttt{PerceptQPA} (our run)',            q[0],   q[1],   q[2]),
            ("Yang \\& Baji\\'c \\cite{yang2025bit}",   0.98, -11.88, -10.96),
            ('Ours',                                    MATCH,  ms_o,   lp_o)]
    bm = min(r[2] for r in rows); bl = min(r[3] for r in rows)
    def c(v, best):
        return f'$\\mathbf{{{v:+.2f}}}$' if abs(v - best) < 1e-9 else f'${v:+.2f}$'
    L = ['\\begin{tabular}{lrrr}', '\\toprule',
         'Allocation & RGB-PSNR & MS-SSIM & LPIPS \\\\', '\\midrule']
    for i, (lab, cost, m, l) in enumerate(rows):
        if i == len(rows) - 1: L.append('\\midrule')
        L.append(f'{lab} & ${cost:+.2f}$ & {c(m, bm)} & {c(l, bl)} \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    os.makedirs('tables', exist_ok=True)
    open('tables/table4_literature.tex', 'w').write('\n'.join(L) + '\n')
    print(f'{"allocation":32s} {"cost":>6s} {"MS-SSIM":>9s} {"LPIPS":>9s}')
    for lab, cost, m, l in rows:
        print(f'{plain(lab):32s} {cost:+6.2f} {m:+9.2f} {l:+9.2f}')
    print('-> tables/table4_literature.tex')

if __name__ == '__main__':
    main()
