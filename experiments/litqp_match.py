"""Read every tau sweep in results/litqp/summary.csv at a common RGB-PSNR cost.

Each family (a map + a QP mechanism) is a curve parameterised by tau: increasing tau trades perceptual
BD-rate back for RGB-PSNR.  RGB-PSNR BD-rate is monotone in tau, so the curve is a function of the cost and
we can linearly interpolate every family to the same cost -- the cost [yang2025bit] reports for each of
their rows -- instead of comparing rows that sit at different operating points.
Targets are their Table II RGB-PSNR BD-rates; reference values are the rest of that row.
"""
import csv, os, sys, collections
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# [yang2025bit] Table II (Kodak), transcribed: (group, label, RGB-PSNR, MS-SSIM, LPIPS-Alex).
# Their best configuration is their "opt. MS-SSIM" row, which is simultaneously their strongest MS-SSIM
# (-11.88) and their strongest LPIPS (-10.96); their opt.-SSIM and opt.-LPIPS rows are dominated by it and
# are not carried into the table.  ALL_THEIRS keeps the full transcription for checking against the paper.
ALL_THEIRS = [('Zero QP map',              -2.75,  -3.46,  -2.42),
              ('PerceptQPA',                2.85, -11.86, -11.96),
              ('Distillation opt. SSIM',    1.80, -10.11, -11.98),
              ('Transfer opt. SSIM',       -0.20, -10.05,  -8.80),
              ('Distillation opt. MS-SSIM', 2.52, -12.74, -13.30),
              ('Transfer opt. MS-SSIM',     0.98, -11.88, -10.96),
              ('Distillation opt. LPIPS',   0.65, -10.28, -11.00),
              ('Transfer opt. LPIPS',      -0.10,  -8.55,  -8.33)]

THEIRS = [(0, '\\texttt{PerceptQPA}',      2.85, -11.86, -11.96),
          (0, "Yang \\& Baji\\'c",        0.98, -11.88, -10.96)]

def families(path=None):
    path = path or (sys.argv[1] if len(sys.argv) > 1 else 'results/litqp/summary.csv')
    fam = collections.defaultdict(list)
    solo = {}
    for r in csv.DictReader(open(path)):
        c = r['config']
        row = (float(r['bd_psnr_rgb']), float(r['bd_ms_ssim_rgb']), float(r['bd_lpips_alex']))
        if '_tau' in c:
            fam[c.split('_tau')[0]].append((float(c.split('_tau')[1]),) + row)
        else:
            solo[c] = row
    return fam, solo

def at_cost(pts, target):
    """Linear interpolation of a family's two metrics at a given RGB-PSNR BD-rate."""
    p = sorted(pts, key=lambda t: t[1])          # sort by cost
    x = [t[1] for t in p]
    if not (x[0] <= target <= x[-1]):
        return None
    return (float(np.interp(target, x, [t[2] for t in p])),
            float(np.interp(target, x, [t[3] for t in p])))

def main():
    fam, solo = families()
    for k, v in sorted(solo.items()):
        print(f'{k:22s}  RGB-PSNR {v[0]:+6.2f}   MS-SSIM {v[1]:+7.2f}   LPIPS-Alex {v[2]:+7.2f}')
    print()
    names = sorted(fam)
    for _g, label, cost, ms_t, lp_t in THEIRS:
        print(f'--- at RGB-PSNR BD-rate {cost:+.2f}%  ({label.replace(chr(92)+chr(92)+chr(92)+chr(92), "")})')
        print(f'    {"[yang2025bit]":26s} MS-SSIM {ms_t:+7.2f}   LPIPS-Alex {lp_t:+7.2f}')
        for n in names:
            r = at_cost(fam[n], cost)
            if r is None:
                lo = min(t[1] for t in fam[n]); hi = max(t[1] for t in fam[n])
                print(f'    {n:26s} out of range [{lo:+.2f},{hi:+.2f}]')
            else:
                print(f'    {n:26s} MS-SSIM {r[0]:+7.2f}   LPIPS-Alex {r[1]:+7.2f}'
                      f'   ({r[0]-ms_t:+.2f} / {r[1]-lp_t:+.2f})')
        print()

if __name__ == '__main__':
    main()
