"""Maps only (no encoding): generate curvature maps with the same code path and cache layout as
vvc_rdo_experiment.py, so a later encode run with the same --metric/--estimator/--probes/--jitter/
--weight-tag finds them and skips generation.

  python gen_maps.py --metric SSIM --estimator hutchblock --probes 256 --weight-tag sbh8m256
  python gen_maps.py --metric LPIPS --estimator gn --probes 256 --images <dir> --limit 20

Writes work/yuv/<stem>.yuv (ffmpeg, BT.601 default) and work/weights/<METRIC>_<tag>/<stem>.{dat|bh}.
"""
import argparse, contextlib, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
STAC = os.path.dirname(os.path.dirname(HERE))
import vvc_rdo_experiment as X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default=os.path.join(STAC, 'Common/Images/KODAK/All'))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--start', type=int, default=0, help='skip the first N images (for splitting across GPUs)')
    ap.add_argument('--shard', default='', help='J/K: after --start/--limit, keep images with index %% K == J')
    ap.add_argument('--reverse', action='store_true', help='process the images in descending name order')
    ap.add_argument('--metric', default='SSIM')
    ap.add_argument('--probes', type=int, default=256)
    ap.add_argument('--estimator', default='auto', choices=['auto', 'gn', 'hutch', 'colnorm', 'gnblock', 'hutchblock'])
    ap.add_argument('--plane-norm', default='per_plane', choices=['per_plane', 'global'])
    ap.add_argument('--clip-pct', type=float, default=99.5)
    ap.add_argument('--blk', type=int, default=8)
    ap.add_argument('--psd', default='none', choices=['eigh', 'shrink', 'none'])
    ap.add_argument('--jitter', type=float, default=0.0)
    ap.add_argument('--vjp-batch', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--wd-sigma', type=float, default=2.0)
    ap.add_argument('--weight-tag', default='')
    ap.add_argument('--work', default=os.path.join(HERE, 'work'))
    ap.add_argument('--claim-gb', type=float, default=0.0,
                    help='hold the GPU exclusively for each map via gpu_claim (needs this many GB); 0 = off')
    ap.add_argument('--min-free-gb', type=float, default=3.0,
                    help='wait until this much GPU memory is free before each map (0 = never wait)')
    a = ap.parse_args()

    pngs = sorted(f for f in os.listdir(a.images) if f.lower().endswith('.png'))
    pngs = pngs[a.start:]
    if a.limit:
        pngs = pngs[: a.limit]
    if a.shard:
        j, k = (int(x) for x in a.shard.split('/'))
        pngs = [f for n, f in enumerate(pngs) if n % k == j]
    if a.reverse:
        pngs = pngs[::-1]
    wdir = a.metric + (('_' + a.weight_tag) if a.weight_tag else '')
    ext = '.bh' if a.estimator in ('gnblock', 'hutchblock') else '.dat'
    wopts = dict(estimator=a.estimator, plane_norm=a.plane_norm, clip_pct=a.clip_pct, jitter=a.jitter,
                 blk=a.blk, psd=a.psd, vjp_batch=a.vjp_batch, wd_sigma=a.wd_sigma)
    rawdir = os.path.join(a.work, 'raw', f'{a.metric}_{a.estimator}_n{a.probes}_j{a.jitter:g}_s{a.seed}')
    os.makedirs(os.path.join(a.work, 'weights', wdir), exist_ok=True)
    print(f'{len(pngs)} images -> work/weights/{wdir}  ({a.estimator}, m={a.probes}, jitter={a.jitter}, seed={a.seed})', flush=True)
    for i, f in enumerate(pngs):
        stem = os.path.splitext(f)[0]
        png = os.path.join(a.images, f)
        w, h = X.image_size(png)
        yuv = os.path.join(a.work, 'yuv', f'{stem}.yuv')
        X.png_to_yuv(png, yuv)
        fn = os.path.join(a.work, 'weights', wdir, stem + ext)
        if os.path.exists(fn) and os.path.getsize(fn) > 0:
            print(f'  [{i + 1}/{len(pngs)}] {stem}: exists', flush=True); continue
        t = time.time()
        # Another GPU job (the CLIC driver at batch 1 holds ~10.5 of 11 GB) makes every map OOM.
        # Wait for room instead of crashing: this serialises map generation behind it automatically.
        if a.min_free_gb > 0:
            import torch
            waited = 0
            while torch.cuda.is_available() and torch.cuda.mem_get_info()[0] / 2**30 < a.min_free_gb:
                if waited == 0:
                    print(f'  waiting for {a.min_free_gb:g} GB of free GPU memory ...', flush=True)
                time.sleep(120); waited += 120
            if waited:
                print(f'  GPU free after {waited // 60} min', flush=True)
        with (X.gpu_claim(a.claim_gb) if a.claim_gb > 0 else contextlib.nullcontext()):
            X.build_weights((png, yuv, w, h, a.metric, a.probes, a.seed, fn, wopts, rawdir))
        if a.claim_gb > 0:
            time.sleep(5)          # let a process blocked on the lock (the CLIC driver) take its turn
        print(f'  [{i + 1}/{len(pngs)}] {stem} {w}x{h}: {time.time() - t:.0f}s', flush=True)
    print('GEN_MAPS_DONE', flush=True)


if __name__ == '__main__':
    main()
