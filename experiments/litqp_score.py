"""BD-rate for the literature-protocol QP-adaptation comparison (work/enc230).

These encodes sit outside the main pipeline: a different encoder build (VTM-23.0, their baseline version),
a different anchor (plain VTM-23.0 rather than our SSE anchor), and PerceptQPA left enabled so its chroma bit
allocation is inherited exactly as in [yang2025bit].  So they get their own scoring pass rather than going
through results_to_csv.

Metrics follow their protocol: RGB-PSNR as the cost axis, MS-SSIM computed on RGB (not luma, as everywhere
else in the paper), and LPIPS-Alex.  Writes results/litqp/summary.csv and prints the table.
"""
import csv, os, sys, glob, collections
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); os.chdir(HERE)
import vvc_rdo_experiment as X
from hessian_weights import read_yuv420, yuv_to_rgb, upsample_chroma
from pytorch_msssim import ms_ssim
import lpips as lpips_pkg

QPS = [22, 27, 32, 37]; W, H = 768, 512
IMGS = [f'kodim{i:02d}' for i in range(1, 25)]
# LIT_IMGS restricts the image set, so a partially-finished grid can be read on the subset that is
# complete for every tag rather than averaging different tags over different images.
if os.environ.get('LIT_IMGS'):
    IMGS = [f'kodim{int(i):02d}' for i in os.environ['LIT_IMGS'].split(',')]
ENC = os.environ.get('LIT_ENC', 'work/enc230')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
_alex = lpips_pkg.LPIPS(net='alex').to(DEV).eval()

# LIT_CS selects the YCbCr->RGB convention: '601' (BT.601 full range, the pipeline default) or
# '601lim' (BT.601 with limited-range scaling, the usual CTC convention).  It changes every row of the
# table, so the anchor and every configuration must be scored under the same setting.
CS = os.environ.get('LIT_CS', '601')

def to_rgb(p):
    y, u, v = [torch.from_numpy(a).float()[None, None].to(DEV) for a in p]
    u, v = upsample_chroma(u, (H, W)), upsample_chroma(v, (H, W))
    if CS == '601lim':
        y = (y - 16.0) * (255.0 / 219.0)
        cb = (v * 0 + u - 128.0) * (255.0 / 224.0)
        cr = (v - 128.0) * (255.0 / 224.0)
        r = y + 1.402 * cr
        g = y - 0.344136 * cb - 0.714136 * cr
        b = y + 1.772 * cb
        return torch.cat([r, g, b], 1).clamp(0, 255)
    return yuv_to_rgb(y, u, v).clamp(0, 255)

def score(ref_rgb, rec_rgb, ref_y=None, rec_y=None):
    with torch.no_grad():
        rgb_mse = torch.mean((ref_rgb - rec_rgb) ** 2).item()
        # luma PSNR too, so the RGB-PSNR cost axis of this protocol can be read against the Y-PSNR axis of the main one
        y_mse = float(np.mean((ref_y.astype(np.float64) - rec_y.astype(np.float64)) ** 2)) if ref_y is not None else None
        return dict(
            psnr_rgb = 10.0 * np.log10(255.0 ** 2 / max(rgb_mse, 1e-12)),
            psnr_y = (10.0 * np.log10(255.0 ** 2 / max(y_mse, 1e-12))) if y_mse is not None else float('nan'),
            ms_ssim_rgb = float(ms_ssim(ref_rgb, rec_rgb, data_range=255, size_average=True)),
            lpips_alex = float(_alex(ref_rgb / 127.5 - 1.0, rec_rgb / 127.5 - 1.0).mean()),
        )

def tags_present():
    t = set()
    for f in glob.glob(f'{ENC}/{IMGS[0]}/*_qp{QPS[0]}.bin'):
        t.add(os.path.basename(f)[:-len(f'_qp{QPS[0]}.bin')])
    return sorted(t)

def main():
    tags = tags_present()
    if 'anchor230' not in tags:
        print('no anchor230 encodes yet'); return
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for im in IMGS:
        ref_p = read_yuv420(f'work/yuv/{im}.yuv', W, H); ref = to_rgb(ref_p)
        pts = {}
        for tag in tags:
            rows = []
            ok = True
            for q in QPS:
                bs, rec = f'{ENC}/{im}/{tag}_qp{q}.bin', f'{ENC}/{im}/{tag}_qp{q}_rec.yuv'
                if not (os.path.exists(bs) and os.path.getsize(rec) == W * H * 3 // 2):
                    ok = False; break
                rec_p = read_yuv420(rec, W, H); m = score(ref, to_rgb(rec_p), ref_p[0], rec_p[0])
                m['bpp'] = os.path.getsize(bs) * 8.0 / (W * H)
                rows.append(m)
            if ok: pts[tag] = rows
        if 'anchor230' not in pts: continue
        a = pts['anchor230']
        for tag, rows in pts.items():
            if tag == 'anchor230': continue
            for k in ('psnr_rgb', 'psnr_y', 'ms_ssim_rgb', 'lpips_alex'):
                key = 'ms_ssim' if k == 'ms_ssim_rgb' else ('lpips' if k == 'lpips_alex' else 'psnr_y')
                per[tag][k].append(X._bd_rate([p['bpp'] for p in rows], X.to_quality(key, [p[k] for p in rows]),
                                              [p['bpp'] for p in a],    X.to_quality(key, [p[k] for p in a])))
        print(f'  {im} done', flush=True)
    # output path follows the encode directory, so scoring a control (e.g. the CTU-128 baselines)
    # cannot overwrite the main table
    out = 'results/litqp/summary.csv' if (ENC.endswith('enc230') and not os.environ.get('LIT_IMGS')
                                          and CS == '601') \
          else f"results/litqp/{os.environ.get('LIT_OUT', os.path.basename(ENC))}.csv"
    os.makedirs('results/litqp', exist_ok=True)
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['config', 'n_images', 'bd_psnr_rgb', 'bd_ms_ssim_rgb', 'bd_lpips_alex', 'bd_psnr_y'])
        print(f'\n{"config":26s} {"n":>3} {"RGB-PSNR":>10} {"MS-SSIM(RGB)":>13} {"LPIPS-Alex":>11} {"Y-PSNR":>8}')
        for tag in sorted(per):
            r = [float(np.nanmean(per[tag][k])) for k in ('psnr_rgb', 'ms_ssim_rgb', 'lpips_alex', 'psnr_y')]
            n = len(per[tag]['psnr_rgb'])
            w.writerow([tag, n] + [f'{x:.3f}' for x in r])
            print(f'{tag:26s} {n:3d} {r[0]:+10.2f} {r[1]:+13.2f} {r[2]:+11.2f} {r[3]:+8.2f}')
    print(f'\n-> {out}')

if __name__ == '__main__':
    main()
