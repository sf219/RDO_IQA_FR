"""3.1 statistics: win counts and bootstrap CIs for Table 1 (24 Kodak images, CTC QPs).

Per-image BD-rates come from results/kodak_json/bd_per_image.csv (default tau per config from
work/default_tau.json) and, for PerceptQPA, from work/qpa_bd_ctc.json.  For each metric:
  wins_BD   images where block < diag on the target metric (lower BD-rate = better)
  wins_BQ   images where block < PerceptQPA
  CI95 of mean(block) - mean(diag) and mean(block) - mean(PerceptQPA), 1000 image resamples
Writes results/table1/stats.csv.  Rule 6 of PLAN.md.
"""
import csv, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
QPS = '22,27,32,37'; N_BOOT = 1000; SEED = 0
CFG = {'SSIM': ('wssim_sn256', 'wssim_sbh8m256', 'ssim'), 'MS-SSIM': ('wms_ssim_mn256', 'wms_ssim_mbh8m256', 'ms_ssim'),
       'LPIPS': ('wlpips_gnj3', 'wlpips_bh8', 'lpips'), 'DISTS': ('wdists_dgnj3', 'wdists_dgnb8j3', 'dists'),
       'WD': ('wwd_gnj3', 'wwd_gnb8j3', 'wd2')}   # sigma-3 (unified GN recipe)
tau = json.load(open(os.path.join(HERE, 'work', 'default_tau.json')))
qpa = json.load(open(os.path.join(HERE, 'work', 'qpa_bd_ctc.json')))['per_image']
rows = [r for r in csv.DictReader(open(os.path.join(HERE, 'results/kodak_json/bd_per_image.csv'))) if r['qps'] == QPS]
imgs = [f'kodim{i:02d}' for i in range(1, 25)]

def per_image(base, key):
    t = tau.get(base); d = {r['image']: float(r['bd_rate']) for r in rows if r['base'] == base and float(r['tau']) == t and r['quality'] == key and r['config'].endswith('rdoq')}
    missing = [im for im in imgs if im not in d]
    if missing:
        return None
    return np.array([d[im] for im in imgs])

rng = np.random.default_rng(SEED); out = []
for met, (dbase, bbase, key) in CFG.items():
    D, B = per_image(dbase, key), per_image(bbase, key)
    Qv = np.array(qpa[key]) if key in qpa else None
    if D is None or B is None or Qv is None or dbase not in tau or bbase not in tau:
        print(f'{met:8s} pending'); continue
    idx = rng.integers(0, len(imgs), size=(N_BOOT, len(imgs)))
    bd = (B[idx] - D[idx]).mean(axis=1); bq = (B[idx] - Qv[idx]).mean(axis=1)
    r = dict(metric=met, tau_diag=tau[dbase], tau_block=tau[bbase], mean_diag=D.mean(), mean_block=B.mean(), mean_qpa=Qv.mean(),
             wins_block_vs_diag=int((B < D).sum()), wins_block_vs_qpa=int((B < Qv).sum()), n=len(imgs),
             diff_BD=B.mean() - D.mean(), ci95_BD_lo=np.percentile(bd, 2.5), ci95_BD_hi=np.percentile(bd, 97.5),
             diff_BQ=B.mean() - Qv.mean(), ci95_BQ_lo=np.percentile(bq, 2.5), ci95_BQ_hi=np.percentile(bq, 97.5))
    out.append(r)
    print(f"{met:8s} tau D/B {r['tau_diag']:g}/{r['tau_block']:g}  D {r['mean_diag']:+6.2f}  B {r['mean_block']:+6.2f}  QPA {r['mean_qpa']:+6.2f} | "
          f"B<D {r['wins_block_vs_diag']}/24  B-D {r['diff_BD']:+5.2f} [{r['ci95_BD_lo']:+.2f},{r['ci95_BD_hi']:+.2f}] | "
          f"B<QPA {r['wins_block_vs_qpa']}/24  B-QPA {r['diff_BQ']:+5.2f} [{r['ci95_BQ_lo']:+.2f},{r['ci95_BQ_hi']:+.2f}]")
os.makedirs(os.path.join(HERE, 'results/table1'), exist_ok=True)
with open(os.path.join(HERE, 'results/table1/stats.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print('results/table1/stats.csv')
