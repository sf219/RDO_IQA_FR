"""Table 1 of the paper as one page-wide table: Kodak (left) and CLIC (right) side by side.

Merges the two generated tables row by row -- tables/table1_paper.tex (plots/make_all.py) and
tables/table2_clic_paper.tex (clic_table.py) have the same rows (PQA, then D/B per metric) and the same five
metric columns; bold marks stay as each writer set them (best per column within its dataset).
Each half gets a Y-PSNR column: every row is read at PQA's Y-PSNR cost on its set (work/qpa_cost.json for Kodak,
\clicAt for CLIC), so the column is that cost, as in the old CLIC table.  tabular* stretched to \textwidth.
Writes tables/table_kodak_clic.tex.
"""
import json
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
TAB = sys.argv[1] if len(sys.argv) > 1 else 'tables'      # directory holding the two input tables (e.g. the Overleaf clone's)

def rows(path):
    out = []
    for line in open(path):
        line = line.strip()
        if not line.endswith('\\\\') or line.startswith('\\toprule') or '&' not in line:
            continue
        cells = [c.strip() for c in line[:-2].split('&')]
        out.append(cells)
    return out

K = rows(os.path.join(TAB, 'table1_paper.tex')); C = rows(os.path.join(TAB, 'table2_clic_paper.tex'))
assert len(K) == len(C), (len(K), len(C))
KCOST = '$%+.1f$' % json.load(open('work/qpa_cost.json'))['psnr_y_bd_rate']
CCOST = '$%s$' % re.search(r'\\clicAt\}\{([^}]+)\}', open(os.path.join(TAB, 'clic_macros.tex')).read()).group(1)   # literal: the table* is input before clic_macros
lines = ['\\begin{tabular}{llrrrrrrrrrrrr}', '\\toprule',
         ' & & \\multicolumn{6}{c}{Kodak} & \\multicolumn{6}{c}{CLIC} \\\\',
         '\\cmidrule(lr){3-8}\\cmidrule(lr){9-14}']
for k, c in zip(K, C):
    lab = lambda x: re.sub(r'\\rowcolor\{metband\}', '', x).strip()
    head = k[1].strip() == 'Map'
    assert head or (lab(k[0]) == lab(c[0]) and k[1] == c[1]), (k[:2], c[:2])   # same row labels
    assert len(k) == 7 and len(c) == 7, (len(k), len(c))
    kc, cc = ('Y-PSNR', 'Y-PSNR') if head else (KCOST, CCOST)
    lines.append(' & '.join(k[:2] + [kc] + k[2:] + [cc] + c[2:]) + ' \\\\')
    if k[0].strip() in ('RDO Target', 'Target') or k[1].strip() == 'PQA':
        lines.append('\\midrule')
lines += ['\\bottomrule', '\\end{tabular}']
open(os.path.join(TAB, 'table_kodak_clic.tex'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
