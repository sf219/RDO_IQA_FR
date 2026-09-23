"""Table 3: head-to-head with [yang2025bit] under their own protocol.

Two target-matched pairs: their "opt. MS-SSIM" row and our MS-SSIM-RGB map read at the RGB-PSNR cost they report
for it, then their "opt. LPIPS" row and our LPIPS-Alex map read at its cost.  Each of our rows is ONE map (both
columns from the same sweep), so the pair compares two allocations built for the same metric at the same fidelity
loss.  Anchor, encoder version (VTM-23.0), CTC QPs, chroma allocation and the RGB metric convention are theirs.
"""
import csv, os, sys
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from litqp_match import THEIRS, families, at_cost

# MS-SSIM-RGB map and LPIPS-Alex map families (weighted RDOQ on).  LIT_MS / LIT_LP select the series shown:
# 'rq_mrgbd'/'rq_alexd' are the diagonal maps, 'rqb_mrgbb'/'rqb_alexb' the block maps (12 Sept grid).
MS, LP = os.environ.get('LIT_MS', 'rqb_mrgbb'), os.environ.get('LIT_LP', 'rqb_alexb')   # paper rows: block maps; LIT_MS=rq_mrgbd LIT_LP=rq_alexd gives the diagonal-map row
# their two 'Transfer' rows we pair with (RGB-PSNR cost, MS-SSIM, LPIPS-Alex), as reported in yang2025bit
THEIR_MS = (0.98, -11.88, -10.96)
THEIR_LP = (-0.10, -8.55, -8.33)

def plain(s):
    return s.replace('\\texttt{', '').replace('}', '').replace('\\&', '&').replace("\\'", '')

def main():
    fam, solo = families(os.environ.get('LIT_CSV', 'results/litqp/summary.csv'))
    ms_m, ms_l = at_cost(fam[MS], THEIR_MS[0])     # our MS-SSIM map at their opt.-MS-SSIM cost: both columns
    lp_m, lp_l = at_cost(fam[LP], THEIR_LP[0])     # our LPIPS map at their opt.-LPIPS cost: both columns
    q = solo['qpa230']
    rows = [('Zero QP map \\cite{yang2025bit}',                    -2.75,  -3.46,  -2.42),
            ('\\texttt{PerceptQPA} (reported in \\cite{yang2025bit})', 2.85, -11.86, -11.96),
            ('\\texttt{PerceptQPA} (our run)',                        q[0],   q[1],   q[2]),
            ("Yang \\& Baji\\'c \\cite{yang2025bit}, opt. MS-SSIM", THEIR_MS[0], THEIR_MS[1], THEIR_MS[2]),
            ('Ours, opt. MS-SSIM',                                  THEIR_MS[0], ms_m, ms_l),
            ("Yang \\& Baji\\'c \\cite{yang2025bit}, opt. LPIPS",   THEIR_LP[0], THEIR_LP[1], THEIR_LP[2]),
            ('Ours, opt. LPIPS',                                    THEIR_LP[0], lp_m, lp_l)]
    bp = min(r[1] for r in rows); bm = min(r[2] for r in rows); bl = min(r[3] for r in rows)   # best per column, RGB-PSNR included
    def c(v, best):
        return f'$\\mathbf{{{v:+.2f}}}$' if abs(v - best) < 1e-9 else f'${v:+.2f}$'
    L = ['\\begin{tabular}{lrrr}', '\\toprule',
         'Allocation & RGB-PSNR & MS-SSIM & LPIPS \\\\', '\\midrule']
    for i, (lab, cost, m, l) in enumerate(rows):
        if i == 3: L.append('\\midrule')               # baselines above, the two target-matched pairs below
        band = '\\rowcolor{pairms}' if i in (3, 4) else '\\rowcolor{pairlp}' if i in (5, 6) else ''   # one pastel per pair (colours defined in main.tex)
        L.append(f'{band}{lab} & {c(cost, bp)} & {c(m, bm)} & {c(l, bl)} \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    os.makedirs('tables', exist_ok=True)
    open('tables/table4_literature.tex', 'w').write('\n'.join(L) + '\n')
    print(f'{"allocation":32s} {"cost":>6s} {"MS-SSIM":>9s} {"LPIPS":>9s}')
    for lab, cost, m, l in rows:
        print(f'{plain(lab):32s} {cost:+6.2f} {m:+9.2f} {l:+9.2f}')
    print('-> tables/table4_literature.tex')

if __name__ == '__main__':
    main()
