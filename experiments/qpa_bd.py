"""PerceptQPA BD-rates vs the anchor on 24 Kodak images at the protocol QPs, every evaluator key, with the current
to_quality.  Writes work/qpa_bd_ctc.json (per-image + mean) used by stats_table1.py, lit_table.py and make_all.py."""
import sys, os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); os.chdir(HERE)
import vvc_rdo_experiment as X
from hessian_weights import read_yuv420
QPS = [22, 27, 32, 37]; W, H = 768, 512; IMGS = [f'kodim{i:02d}' for i in range(1, 25)]
KEYS = list(X.METRIC_KEYS)   # every evaluator key (wd0/wd2/wd4 when EVAL_WD=1)
# cache: skip the GPU pass when the JSON exists and already carries every key the evaluator would write
_cached = json.load(open('work/qpa_bd_ctc.json')) if os.path.exists('work/qpa_bd_ctc.json') else None
if _cached and not os.environ.get('QPA_FORCE') and not os.path.exists('work/qpa_points.json') and all(k in _cached['bd_rate_vs_anchor'] for k in X.METRIC_KEYS):
    print('qpa_bd: cached (work/qpa_bd_ctc.json has every metric key)'); sys.exit(0)
# stored points (work/qpa_points.json) avoid the GPU pass when they carry every key; otherwise score and store them
PTS = 'work/qpa_points.json'
_pts = json.load(open(PTS)) if os.path.exists(PTS) else None
if _pts and all(k in p for im in _pts.values() for pts in im.values() for p in pts for k in X.METRIC_KEYS):
    per = {k: [] for k in KEYS}
    for im in IMGS:
        cur = _pts[im]
        for k in KEYS:
            per[k].append(X._bd_rate([p['bpp'] for p in cur['qpa']], X.to_quality(k, [p[k] for p in cur['qpa']]), [p['bpp'] for p in cur['anchor']], X.to_quality(k, [p[k] for p in cur['anchor']])))
    print('qpa_bd: from stored points')
else:
  with X.gpu_claim(2.0):                    # exclusive GPU while the CLIC driver runs (5 Sept)
    ev = X.Evaluator(); per = {k: [] for k in KEYS}; _store = {}
    for im in IMGS:
          ref = read_yuv420(f'work/yuv/{im}.yuv', W, H); d = f'work/enc/{im}'; cur = {}
          for tag in ('qpa', 'anchor'):
              pts = []
              for q in QPS:
                  m = ev(ref, read_yuv420(f'{d}/{tag}_qp{q}_rec.yuv', W, H)); m['bpp'] = os.path.getsize(f'{d}/{tag}_qp{q}.bin') * 8.0 / (W * H); pts.append(m)
              cur[tag] = pts
          _store[im] = cur
          for k in KEYS:
              per[k].append(X._bd_rate([p['bpp'] for p in cur['qpa']], X.to_quality(k, [p[k] for p in cur['qpa']]), [p['bpp'] for p in cur['anchor']], X.to_quality(k, [p[k] for p in cur['anchor']])))
    json.dump(_store, open(PTS, 'w'))
out = {k: float(np.nanmean(v)) for k, v in per.items()}
json.dump({'qps': QPS, 'n_images': 24, 'bd_rate_vs_anchor': out, 'per_image': per}, open('work/qpa_bd_ctc.json', 'w'), indent=1)
json.dump(out, open('work/qpa_bd_ctc_full.json', 'w'), indent=1)
print('PerceptQPA: ' + '  '.join(f'{k} {v:+.2f}' for k, v in out.items()))
