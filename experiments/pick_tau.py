"""Default tau per Table-1 config = grid point whose 24-image Y-PSNR BD-rate is nearest PerceptQPA's cost,
with PerceptQPA's cost computed here from work/enc/<img>/{qpa,anchor}_qp<QP>.bin/_rec.yuv at the protocol QPs.
Writes work/default_tau.json {config tag: tau} and work/qpa_cost.json.  Run after results_to_csv.py."""
import csv, json, os, collections, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from vvc_rdo_experiment import _bd_rate
from hessian_weights import read_yuv420
QPS = [22, 27, 32, 37]; W, H = 768, 512
IMGS = [f'kodim{i:02d}' for i in range(1, 25)]
TAGS = ['wssim_sbh8m256', 'wssim_sn256', 'wdists_dgn', 'wdists_dgnb8', 'wlpips_gn', 'wlpips_gnb8', 'wms_ssim_mn256', 'wms_ssim_mbh8m256',
        'wwd_gn', 'wwd_gnb8', 'wwd_gnj3', 'wwd_gnb8j3',   # WD rows, log2 sigma 2; sigma-3 maps from 7 Sept
        'wlpips_gnj3', 'wlpips_bh8',   # LPIPS rows = sigma-3 smoothed maps (7 Sept, user decision)
        'wdists_dgnj3', 'wdists_dgnb8j3']   # DISTS rows = sigma-3 (unified GN recipe)

def psnr_y(ref, rec):
    m = float(np.mean((ref[0] - rec[0]) ** 2)); return 10 * np.log10(255.0 ** 2 / m) if m > 0 else 99.0

per = []
for im in IMGS:
    ref = read_yuv420(os.path.join(HERE, 'work', 'yuv', f'{im}.yuv'), W, H); d = os.path.join(HERE, 'work', 'enc', im); cur = {}
    for tag in ('qpa', 'anchor'):
        pts = []
        for q in QPS:
            bs = os.path.join(d, f'{tag}_qp{q}.bin'); rec = os.path.join(d, f'{tag}_qp{q}_rec.yuv')
            if not (os.path.exists(bs) and os.path.exists(rec)):
                sys.exit(f'missing {bs}')
            pts.append((os.path.getsize(bs) * 8.0 / (W * H), psnr_y(ref, read_yuv420(rec, W, H))))
        cur[tag] = pts
    per.append(_bd_rate([p[0] for p in cur['qpa']], [p[1] for p in cur['qpa']], [p[0] for p in cur['anchor']], [p[1] for p in cur['anchor']]))
QPA = float(np.nanmean(per)); print(f'PerceptQPA Y-PSNR BD-rate vs anchor, {len(IMGS)} images, QPs {QPS}: {QPA:+.2f} %')
json.dump({'qps': QPS, 'n_images': len(IMGS), 'psnr_y_bd_rate': QPA, 'per_image': per}, open(os.path.join(HERE, 'work', 'qpa_cost.json'), 'w'), indent=1)
rows = [r for r in csv.DictReader(open(os.path.join(HERE, 'results/kodak_json/summary.csv'))) if r['rdoq'] == '1' and r['n_images'] == '24' and r.get('qps', '') == ','.join(map(str, QPS))]
by = collections.defaultdict(dict)
for r in rows:
    by[r['base']][float(r['tau'])] = float(r['bd_psnr_y'])
out = {}
for t in TAGS:
    if by[t]:
        out[t] = min(by[t], key=lambda k: abs(by[t][k] - QPA))
        print(f'{t:20s} ' + '  '.join(f'tau{k:g}:{v:+.2f}' for k, v in sorted(by[t].items())) + f'  -> {out[t]:g}')
missing = [t for t in TAGS if t not in out]
if missing:
    sys.exit(f'default_tau.json NOT written: {len(missing)} Table-1 series missing on 24 images at {QPS}: {missing}')
json.dump(out, open(os.path.join(HERE, 'work', 'default_tau.json'), 'w'), indent=1); print('work/default_tau.json written')
