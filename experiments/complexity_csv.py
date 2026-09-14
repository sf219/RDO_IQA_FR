"""Emit results/complexity/summary.csv (plan Section 5) from work/timebench/*.log and bench/map_time.json.
Rows: enc  image,metric,config,qp,enc_time_s   (config in {anchor, diag, block}; anchor metric='-')
      map  image,metric,config,m,map_time_s,corr
Re-run after re-timing the timebench with the exact float kernel."""
import argparse, csv, glob, json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--enc', default=os.path.join(HERE, 'work', 'timebench'))
ap.add_argument('--maps', default=os.path.join(HERE, 'bench', 'map_time.json'))
ap.add_argument('--out', default=os.path.join(HERE, 'results', 'complexity', 'summary.csv'))
a = ap.parse_args()
MET = {'ssim': 'SSIM', 'msssim': 'MS_SSIM', 'lpips': 'LPIPS', 'dists': 'DISTS', 'wd': 'WD'}
rows = []
for f in sorted(glob.glob(os.path.join(a.enc, '*.log'))):
    m = re.match(r'(kodim\d+)_(.+?)_qp(\d+)\.log', os.path.basename(f))
    t = re.findall(r'Total Time:\s+([\d.]+)', open(f).read())
    if not (m and t):
        continue
    img, cfg, qp = m.groups()
    met, kind = ('-', 'anchor') if cfg == 'anchor' else (MET[cfg.split('_')[0]], cfg.split('_')[1])
    rows.append(dict(kind='enc', image=img, metric=met, config=kind, qp=qp, m='', enc_time_s=t[-1], map_time_s='', corr=''))
for r in json.load(open(a.maps)):
    rows.append(dict(kind='map', image='kodim01', metric=r['metric'], config=r['kind'], qp='', m=r['n'],
                     enc_time_s='', map_time_s=f"{r['sec']:.2f}", corr=f"{r['corr']:.3f}"))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f'{a.out}: {sum(r["kind"]=="enc" for r in rows)} enc rows, {sum(r["kind"]=="map" for r in rows)} map rows')
