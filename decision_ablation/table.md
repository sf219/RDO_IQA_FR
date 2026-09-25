# Decision ablation, block maps, Kodak, at +8.15 % Y-PSNR (interpolated along tau)

## SSIM  (series `wssim_sbh8m256`, target column `ssim`)

reference, all decisions weighted (mask 255): target 

| decision | leave-one-out mask | target BD (LOO) | gain lost | only-one mask | target BD (ONLY, matched cost) | ONLY peak BD (tau, cost, max cost) | LOO psnr_y | LOO ssim | LOO ms_ssim | LOO lpips | LOO dists | LOO wd2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| partition | 254 |  |  | 1 |  |  |  |  |  |  |  |  |
| LFNST index | 253 |  |  | 2 |  |  |  |  |  |  |  |  |
| MTS flag | 251 |  |  | 4 |  |  |  |  |  |  |  |  |
| MTS/TS type | 247 |  |  | 8 |  |  |  |  |  |  |  |  |
| luma mode | 239 |  |  | 16 |  |  |  |  |  |  |  |  |
| ISP | 223 |  |  | 32 |  |  |  |  |  |  |  |  |
| chroma | 191 |  |  | 64 |  |  |  |  |  |  |  |  |
| RDOQ lambda | 127 |  |  | 128 |  |  |  |  |  |  |  |  |

## MS_SSIM  (series `wms_ssim_mbh8m256`, target column `ms_ssim`)

reference, all decisions weighted (mask 255): target 

| decision | leave-one-out mask | target BD (LOO) | gain lost | only-one mask | target BD (ONLY, matched cost) | ONLY peak BD (tau, cost, max cost) | LOO psnr_y | LOO ssim | LOO ms_ssim | LOO lpips | LOO dists | LOO wd2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| partition | 254 |  |  | 1 |  |  |  |  |  |  |  |  |
| LFNST index | 253 |  |  | 2 |  |  |  |  |  |  |  |  |
| MTS flag | 251 |  |  | 4 |  |  |  |  |  |  |  |  |
| MTS/TS type | 247 |  |  | 8 |  |  |  |  |  |  |  |  |
| luma mode | 239 |  |  | 16 |  |  |  |  |  |  |  |  |
| ISP | 223 |  |  | 32 |  |  |  |  |  |  |  |  |
| chroma | 191 |  |  | 64 |  |  |  |  |  |  |  |  |
| RDOQ lambda | 127 |  |  | 128 |  |  |  |  |  |  |  |  |

## LPIPS  (series `wlpips_bh8`, target column `lpips`)

reference, all decisions weighted (mask 255): target -15.5

| decision | leave-one-out mask | target BD (LOO) | gain lost | only-one mask | target BD (ONLY, matched cost) | ONLY peak BD (tau, cost, max cost) | LOO psnr_y | LOO ssim | LOO ms_ssim | LOO lpips | LOO dists | LOO wd2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| partition | 254 | -9.2 | +6.3 | 1 | -6.6* | -7.4 (tau 0.5, +3.3 %, max +6.3 %) | +8.1 | -4.5 | -3.7 | -9.2 | -6.0 | -7.0 |
| LFNST index | 253 | -14.8 | +0.7 | 2 | +0.2* | -1.3 (tau 1, +0.4 %, max +2.9 %) | +8.1 | -6.0 | -5.2 | -14.8 | -10.6 | -11.6 |
| MTS flag | 251 | -14.9 | +0.6 | 4 | -0.8* | -0.9 (tau 0.125, +0.5 %, max +0.6 %) | +8.1 | -6.0 | -5.3 | -14.9 | -10.9 | -11.9 |
| MTS/TS type | 247 | -14.9 | +0.6 | 8 | +0.0* | -0.2 (tau 2, +0.4 %, max +1.0 %) | +8.1 | -5.6 | -4.9 | -14.9 | -11.5 | -11.4 |
| luma mode | 239 | -14.4 | +1.1 | 16 | -1.4* | -1.8 (tau 0.0625, +1.2 %, max +1.3 %) | +8.1 | -5.5 | -4.9 | -14.4 | -9.4 | -11.4 |
| ISP | 223 | -15.1 | +0.4 | 32 | +0.9* | -0.2 (tau 2, +0.2 %, max +2.1 %) | +8.1 | -5.9 | -5.1 | -15.1 | -11.7 | -11.6 |
| chroma | 191 | -15.0 | +0.5 | 64 | -0.5* | -0.7 (tau 2, +0.0 %, max +0.1 %) | +8.1 | -6.2 | -5.5 | -15.0 | -10.4 | -10.7 |
| RDOQ lambda | 127 | -12.0 | +3.5 | 128 | +0.4 | -1.2 (tau 1, +0.8 %, max +8.5 %) | +8.1 | -2.8 | -2.4 | -12.0 | -8.3 | -9.0 |

## DISTS  (series `wdists_dgnb8j3`, target column `dists`)

reference, all decisions weighted (mask 255): target 

| decision | leave-one-out mask | target BD (LOO) | gain lost | only-one mask | target BD (ONLY, matched cost) | ONLY peak BD (tau, cost, max cost) | LOO psnr_y | LOO ssim | LOO ms_ssim | LOO lpips | LOO dists | LOO wd2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| partition | 254 |  |  | 1 |  |  |  |  |  |  |  |  |
| LFNST index | 253 |  |  | 2 |  |  |  |  |  |  |  |  |
| MTS flag | 251 |  |  | 4 |  |  |  |  |  |  |  |  |
| MTS/TS type | 247 |  |  | 8 |  |  |  |  |  |  |  |  |
| luma mode | 239 |  |  | 16 |  |  |  |  |  |  |  |  |
| ISP | 223 |  |  | 32 |  |  |  |  |  |  |  |  |
| chroma | 191 |  |  | 64 |  |  |  |  |  |  |  |  |
| RDOQ lambda | 127 |  |  | 128 |  |  |  |  |  |  |  |  |

## WD  (series `wwd_gnb8j3`, target column `wd2`)

reference, all decisions weighted (mask 255): target 

| decision | leave-one-out mask | target BD (LOO) | gain lost | only-one mask | target BD (ONLY, matched cost) | ONLY peak BD (tau, cost, max cost) | LOO psnr_y | LOO ssim | LOO ms_ssim | LOO lpips | LOO dists | LOO wd2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| partition | 254 |  |  | 1 |  |  |  |  |  |  |  |  |
| LFNST index | 253 |  |  | 2 |  |  |  |  |  |  |  |  |
| MTS flag | 251 |  |  | 4 |  |  |  |  |  |  |  |  |
| MTS/TS type | 247 |  |  | 8 |  |  |  |  |  |  |  |  |
| luma mode | 239 |  |  | 16 |  |  |  |  |  |  |  |  |
| ISP | 223 |  |  | 32 |  |  |  |  |  |  |  |  |
| chroma | 191 |  |  | 64 |  |  |  |  |  |  |  |  |
| RDOQ lambda | 127 |  |  | 128 |  |  |  |  |  |  |  |  |

Negative BD-rate = saving on that metric versus the SSE anchor. "gain lost" = leave-one-out minus reference (positive: the decision contributes). `*` = the +8.15 % point lies outside the tau sweep; the value shown is the nearest sweep endpoint (np.interp clamps), not an extrapolation.
