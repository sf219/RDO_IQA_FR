"""Day-one converter: results/vvc_rdo_*.json -> CSV in the Section-3 schema.

Writes to results/kodak_json/:
  rows.csv          one row per (image, qp, config):
                    image, qp, metric, config, base, estimator, kind, tau, m, sigma, R, seed, rdoq,
                    bpp, ypsnr, ssim, msssim, lpips, dists, enc_time_s, map_time_s, source
  bd_per_image.csv  one row per (image, config, quality key): BD-rate vs anchor from the JSON
  summary.csv       one row per (source, config): mean BD-rates, n_images, and the map arguments

Timing fields are empty for these runs (the JSONs never recorded them); they come from
results/complexity/summary.csv only.  R is not recorded either: the plan's rule R = m/16 when
sigma > 0 else 1 is applied and flagged in the 'R_rule' column of summary.csv.
"""
import argparse, csv, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
QKEYS = ['psnr_y', 'psnr_yuv', 'ssim', 'ms_ssim', 'lpips', 'dists', 'psnr_rgb', 'lpips_alex', 'wd0', 'wd2', 'wd4']


def parse_tag(tag):
    base, _, rest = tag.partition('_tau')
    tau = rest.split('_')[0] if rest else ''
    return base, tau, rest.endswith('rdoq')


def estimator_of(args, metric):
    e = args.get('estimator')
    if e in (None, 'auto'):
        e = 'hutch' if metric in ('ssim', 'ms_ssim') else 'gn'
        if args.get('estimator') is None:
            e = 'legacy_' + e
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'kodak_json'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, bd_rows, summ = [], [], []
    seen_anchor = set()
    for f in sorted(glob.glob(os.path.join(a.results, 'vvc_rdo_*.json')), key=os.path.getmtime):
        d = json.load(open(f))
        if not isinstance(d, dict) or 'results' not in d:
            continue
        args = d.get('args', {})
        metric = (args.get('metric') or os.path.basename(f).split('_')[2]).lower()
        src = os.path.basename(f)[:-5]
        m = args.get('probes'); sig = args.get('jitter') or 0.0
        qps = ','.join(str(q) for q in (args.get('qps') or []))
        R = (m // 16 if (m and sig) else 1)
        est = estimator_of(args, metric)
        for img, cfgs in d['results'].items():
            for tag, pts in cfgs.items():
                if tag == 'anchor':
                    base, tau, rdoq, e_, kind = 'anchor', '', False, '-', 'anchor'
                else:
                    base, tau, rdoq = parse_tag(tag); e_ = est; kind = 'block' if 'block' in est else 'diag'
                for p in pts:
                    if tag == 'anchor':
                        key = (img, p['qp'])
                        if key in seen_anchor:
                            continue
                        seen_anchor.add(key)
                    rows.append(dict(image=img, qp=p['qp'], metric=metric if tag != 'anchor' else '-', config=tag, base=base,
                                     estimator=e_, kind=kind, tau=tau, m=m if tag != 'anchor' else '', sigma=sig if tag != 'anchor' else '',
                                     R=R if tag != 'anchor' else '', seed=0, rdoq=int(rdoq), bpp=p['bpp'], ypsnr=p['psnr_y'],
                                     ssim=p['ssim'], msssim=p['ms_ssim'], lpips=p['lpips'], dists=p['dists'],
                                     enc_time_s='', map_time_s='', qps=qps, source=src))
        imgs = list(d['results'])
        # recompute BD-rates from the per-QP points with the current to_quality (the JSON's own
        # bd_rate_vs_anchor fields were written with the code of the time and may be stale)
        from vvc_rdo_experiment import _bd_rate, to_quality
        bdr = {}
        for tag in d['results'][imgs[0]]:
            if tag == 'anchor':
                continue
            bdr[tag] = {}
            for k in QKEYS:
                per_img = []
                for im in imgs:
                    pts, anc = d['results'][im][tag], d['results'][im]['anchor']
                    if not all(k in p_ for p_ in pts + anc):
                        per_img = []; break
                    per_img.append(_bd_rate([p_['bpp'] for p_ in pts], to_quality(k, [p_[k] for p_ in pts]), [p_['bpp'] for p_ in anc], to_quality(k, [p_[k] for p_ in anc])))
                if per_img:
                    bdr[tag][k] = {'per_image': per_img, 'mean': float(np.nanmean(per_img))}
        for tag, per in bdr.items():
            base, tau, rdoq = parse_tag(tag)
            s = dict(source=src, config=tag, base=base, metric=metric, estimator=est,
                     kind='block' if 'block' in est else 'diag', tau=tau, m=m, sigma=sig, R=R, R_rule='m/16 if sigma>0 else 1',
                     psd=args.get('psd'), clip_pct=args.get('clip_pct'), plane_norm=args.get('plane_norm'), rdoq=int(rdoq), n_images=len(imgs), qps=qps)
            for k in QKEYS:
                v = per.get(k)
                if not v:
                    s[f'bd_{k}'] = ''
                    continue
                s[f'bd_{k}'] = v['mean']
                for img, x in zip(imgs, v['per_image']):
                    bd_rows.append(dict(source=src, config=tag, base=base, metric=metric, tau=tau, m=m, sigma=sig, qps=qps, image=img, quality=k, bd_rate=x))
            summ.append(s)

    def dump(name, lst):
        if not lst:
            return
        with open(os.path.join(a.out, name), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(lst[0].keys())); w.writeheader(); w.writerows(lst)
        print(f'{name}: {len(lst)} rows')
    dump('rows.csv', rows); dump('bd_per_image.csv', bd_rows); dump('summary.csv', summ)


if __name__ == '__main__':
    main()
