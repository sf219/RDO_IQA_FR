"""3.5 CLIC pipeline, per image: maps (GPU) -> anchor, PerceptQPA, diag and block per metric at the
Kodak default tau -> score -> delete the tile files.  Resumable: encodes are cached by
vvc_rdo_experiment.encode, maps by existence, scored images by results/clic/done/<stem>.json.

  python clic_run.py --images work/clic20 --jobs 14 --vjp-batch 2
  python clic_run.py --summary            # aggregate done/*.json -> rows.csv, summary.csv
"""
import argparse, csv, glob, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vvc_rdo_experiment as X

QPS = [22, 27, 32, 37]
# metric -> (diag estimator, diag taus, block estimator, block taus): the Kodak regeneration grid
# (config.yaml: regen_tau_grid), so CLIC can be interpolated to PerceptQPA's CLIC cost exactly as Kodak.
CONFIGS = {
    'SSIM':    ('hutch', [0.25, 0.5, 1.0], 'hutchblock', [1.0, 2.0, 4.0]),
    'MS_SSIM': ('hutch', [0.25, 0.5, 1.0], 'hutchblock', [0.5, 1.0, 2.0]),
    'LPIPS':   ('gn',    [0.25, 0.5, 1.0], 'gnblock',    [0.5, 1.0, 2.0]),
    'DISTS':   ('gn',    [0.5, 1.0, 2.0],  'gnblock',    [2.0, 4.0, 6.0]),
    'WD':      ('gn',    [0.0625, 0.125, 0.25], 'gnblock', [1.0, 2.0, 4.0]),   # log2 sigma 2; Kodak defaults 0.125 / 2
}
QKEYS = ['psnr_y', 'psnr_yuv', 'ssim', 'ms_ssim', 'lpips', 'dists', 'wd0', 'wd2', 'wd4']   # wd*: present only after rescore_wd.py


TAG_SUFFIX = ''   # e.g. 'j3' for the sigma-3 pass: tags become wlpipsj3_diag_tau..., markers go to --done-dir


def tag_of(metric, kind, tau):
    return f'w{metric.lower()}{TAG_SUFFIX}_{kind}_tau{tau:g}_rdoq'


def score_image(stem, w, h, yuv, od, tags):
    from hessian_weights import read_yuv420
    ref = read_yuv420(yuv, w, h)
    out = {}
    with X.gpu_lock():                      # scoring a 2K image through VGG/DISTS/Alex peaks at ~10 GiB
        ev = X.Evaluator()
        for tag in tags:
            pts = []
            for qp in QPS:
                bs = os.path.join(od, f'{tag}_qp{qp}.bin'); rec = os.path.join(od, f'{tag}_qp{qp}_rec.yuv')
                m = ev(ref, read_yuv420(rec, w, h)); m['qp'] = qp; m['bpp'] = os.path.getsize(bs) * 8.0 / (w * h)
                pts.append(m)
            out[tag] = pts
        del ev
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass
    return out


def run(a):
    pngs = sorted(f for f in os.listdir(a.images) if f.lower().endswith('.png'))[a.start:]
    if a.limit:
        pngs = pngs[: a.limit]
    global TAG_SUFFIX; TAG_SUFFIX = a.tag_suffix
    if a.diag_taus or a.block_taus:
        for m in a.metrics:
            dest, dtaus, best, btaus = CONFIGS[m]; CONFIGS[m] = (dest, a.diag_taus or dtaus, best, a.block_taus or btaus)
    done_dir = os.path.join(a.out, a.done_dir); os.makedirs(done_dir, exist_ok=True)
    metrics = a.metrics
    for i, f in enumerate(pngs):
        stem = os.path.splitext(f)[0]
        marker = os.path.join(done_dir, stem + '.json')
        if os.path.exists(marker):
            print(f'[{i + 1}/{len(pngs)}] {stem}: done', flush=True); continue
        png = os.path.join(a.images, f); w, h = X.image_size(png)
        yuv = os.path.join(a.work, 'yuv', f'{stem}.yuv'); X.png_to_yuv(png, yuv)
        od = os.path.join(a.work, 'enc', stem)
        t0 = time.time(); futs = []; tags = ['anchor', 'qpa']; tiles = []
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for qp in QPS:
                futs.append(ex.submit(X.encode, (yuv, w, h, qp, od, 'anchor', None, 0.0, False, False, ())))
                futs.append(ex.submit(X.encode, (yuv, w, h, qp, od, 'qpa', None, 0.0, False, False, ('--PerceptQPA=1',))))
            for metric in metrics:
                dest, dtaus, best, btaus = CONFIGS[metric]
                for kind, est, taus in (('diag', dest, dtaus), ('block', best, btaus)):
                    ext = '.bh' if est.endswith('block') else '.dat'
                    fn = os.path.join(a.work, 'weights', f'{metric}_clic{TAG_SUFFIX}_{kind}', stem + ext)
                    if not (os.path.exists(fn) and os.path.getsize(fn) > 0):
                        # one subprocess per map: a 2K map through VGG holds ~10 GiB that the in-process
                        # generator does not give back; the subprocess exits and the memory is freed.
                        tm = time.time()
                        import subprocess
                        r = subprocess.run([sys.executable, os.path.join(HERE, 'gen_maps.py'), '--images', a.images,
                                            '--start', str(a.start + i), '--limit', '1', '--metric', metric, '--estimator', est,
                                            '--probes', str(a.probes), '--vjp-batch', str(a.vjp_batch), '--jitter', str(a.jitter), '--weight-tag', f'clic{TAG_SUFFIX}_{kind}',
                                            '--work', a.work], capture_output=True, text=True)
                        if r.returncode != 0 or not os.path.exists(fn):
                            sys.exit(f'map {metric} {kind} for {stem} failed:\n' + r.stderr[-2000:])
                        print(f'    map {metric} {kind}: {time.time() - tm:.0f}s', flush=True)
                    if ext == '.bh':
                        tiles.append(fn)
                    for tau in taus:
                        tag = tag_of(metric, kind, tau); tags.append(tag)
                        for qp in QPS:
                            futs.append(ex.submit(X.encode, (yuv, w, h, qp, od, tag, fn, tau, True, False, ())))
            for k, fu in enumerate(as_completed(futs)):
                fu.result()
            print(f'    {len(futs)} encodes done ({time.time() - t0:.0f}s)', flush=True)
        res = score_image(stem, w, h, yuv, od, tags)
        json.dump(dict(image=stem, w=w, h=h, results=res, configs=CONFIGS, probes=a.probes), open(marker, 'w'))
        if not a.keep_tiles:
            for fn in tiles:
                os.remove(fn)
        print(f'[{i + 1}/{len(pngs)}] {stem} {w}x{h}: {time.time() - t0:.0f}s', flush=True)
    print('CLIC_RUN_DONE', flush=True)


def summary(a):
    rows, bd = [], {}
    # main markers plus any smoothed-pass markers (done_j3, done_j3dw, done_j6; CARC 7 Sept): each configuration's BD-rate is taken
    # against the anchor of its own marker; anchor/qpa rows come from the main markers only.
    # done_ssimwd (CARC, 12 Sept) re-ran the SSIM and MS-SSIM configurations with WD scored; it goes first so
    # those rows come wholly from one run rather than mixing its WD with the older run's other keys
    files = sorted(glob.glob(os.path.join(a.out, 'done_ssimwd', '*.json'))) + sorted(glob.glob(os.path.join(a.out, 'done_ms0125', '*.json'))) + sorted(glob.glob(os.path.join(a.out, 'done', '*.json'))) + sorted(glob.glob(os.path.join(a.out, 'done_j3', '*.json'))) + sorted(glob.glob(os.path.join(a.out, 'done_j3dw', '*.json'))) + sorted(glob.glob(os.path.join(a.out, 'done_j6', '*.json'))) + sorted(glob.glob(os.path.join(a.out, 'done_j6x', '*.json'))) + sorted(glob.glob(os.path.join(a.out, 'done_j3db1', '*.json')))
    # The qpa row is re-encoded in every pass, so it appears in several marker directories.  Take each
    # (config, key, image) once, first pass wins: the main markers supply the keys they scored and the
    # later ones (which ran with EVAL_WD=1) add only the keys the main pass lacks, e.g. wd0/wd2/wd4.
    seen = set()
    for f in files:
        d = json.load(open(f)); stem = d['image']; res = d['results']
        anc = res['anchor']; aux = os.path.basename(os.path.dirname(f)) != 'done'
        for tag, pts in res.items():
            if aux and tag == 'anchor':
                continue
            if not (aux and tag == 'qpa'):
                for p in pts:
                    rows.append(dict(image=stem, qp=p['qp'], config=tag, bpp=p['bpp'], ypsnr=p['psnr_y'], ssim=p['ssim'],
                                     msssim=p['ms_ssim'], lpips=p['lpips'], dists=p['dists']))
            if tag == 'anchor':
                continue
            for k in QKEYS:
                if not all(k in p for p in pts + anc) or (tag, k, stem) in seen:
                    continue
                seen.add((tag, k, stem))
                v = X._bd_rate([p['bpp'] for p in pts], X.to_quality(k, [p[k] for p in pts]),
                               [p['bpp'] for p in anc], X.to_quality(k, [p[k] for p in anc]))
                bd.setdefault((tag, k), []).append(v)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'rows.csv'), 'w', newline='') as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wtr.writeheader(); wtr.writerows(rows)
    tags = sorted({t for t, _ in bd})
    with open(os.path.join(a.out, 'summary.csv'), 'w', newline='') as fh:
        wtr = csv.writer(fh); wtr.writerow(['config', 'n_images'] + [f'bd_{k}' for k in QKEYS])
        for t in tags:
            cell = lambda k: (f'{np.nanmean(bd[(t, k)]):.3f}' if (t, k) in bd else '')
            wtr.writerow([t, len(bd[(t, QKEYS[0])])] + [cell(k) for k in QKEYS])
            print(f'{t:28s} n={len(bd[(t, QKEYS[0])]):2d} ' + ' '.join(f'{k}:{np.nanmean(bd[(t, k)]):+7.2f}' for k in QKEYS if (t, k) in bd))
    print(f'{len(files)} images -> {a.out}/rows.csv, summary.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default=os.path.join(HERE, 'work', 'clic20'))
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--metrics', nargs='+', default=['SSIM', 'MS_SSIM', 'LPIPS', 'DISTS'])
    ap.add_argument('--probes', type=int, default=256)
    ap.add_argument('--vjp-batch', type=int, default=2)
    ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--keep-tiles', action='store_true')
    ap.add_argument('--work', default=os.path.join(HERE, 'work', 'clic'))
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'clic'))
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--jitter', type=float, default=0.0, help='metric smoothing sigma for the maps (sigma-3 recipe: 3)')
    ap.add_argument('--tag-suffix', default='', help="config-tag suffix, e.g. j3 -> wlpipsj3_diag_tau...")
    ap.add_argument('--done-dir', default='done', help='marker directory under --out')
    ap.add_argument('--diag-taus', type=float, nargs='+', default=None, help='override the diag tau grid of the selected metrics')
    ap.add_argument('--block-taus', type=float, nargs='+', default=None, help='override the block tau grid of the selected metrics')
    a = ap.parse_args()
    summary(a) if a.summary else run(a)


if __name__ == '__main__':
    main()
