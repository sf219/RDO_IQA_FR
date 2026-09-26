# Which RDO decisions carry the gain? (side study, not in the paper)

This folder is separate from the reproduction of the paper (`method/`, `experiments/`, `vtm/`, `REPRODUCE.md`);
nothing here is used by any table or figure. It records an ongoing local study: the paper's encoder replaces the
SSE by the IDQD in every rate-distortion decision, and this study switches the weighted distortion on or off per
decision to see which decisions produce the BD-rate gain on the target metric.

**Status:** in progress. Complete: LPIPS, DISTS. Pending: WD, SSIM, MS-SSIM (one run per (metric, mask) at about 90 min).

## Encoder

`vtm_decision_mask.patch` applies on top of `vtm/vtm-23.8_weighted_rdo.patch` and adds `--WeightedRdoMask=N`.
Every SSE call computes both the weighted and the plain distortion; each decision point re-expresses the RD costs
it compares with the distortion its mask bit selects, so a decision can be switched without changing what the other
decisions see. Bits:

| bit | value | decision | where |
|---|---|---|---|
| 0 | 1 | partition: split vs no split, and among split types | `EncCu::xCheckBestMode` when a split candidate is involved |
| 1 | 2 | LFNST index at CU level | `EncCu::xCheckBestMode`, intra candidates differing in `lfnstIdx` |
| 2 | 4 | MTS flag at CU level | same, candidates differing in `mtsFlag` |
| 3 | 8 | MTS type / transform skip per TU | `IntraSearch::xRecurIntraCodingLumaQT`, transform-candidate loop |
| 4 | 16 | intra luma mode (regular, MIP, MRL, BDPCM) | `IntraSearch::estIntraPredLumaQT`, candidate vs best |
| 5 | 32 | ISP candidates vs the rest, and ISP sub-partition costs | `estIntraPredLumaQT` when either side is ISP; `xIntraCodingLumaISP` |
| 6 | 64 | chroma mode, chroma transform, joint Cb-Cr | `estIntraPredChromaQT`, `xRecurIntraChromaCodingQT` |
| 7 | 128 | RDOQ lambda scaling by the block's mean weight | `IntraSearch::xIntraCodingTUBlock` |

Mask 255 is the paper's encoder, mask 0 is plain SSE everywhere. The SATD pre-selection is never weighted, as in the
paper. Early-termination heuristics that read stored costs see the cost of the decision that produced them. Requires
`EncDbOpt=0`. Validation: mask 255 gives bitstreams identical to the paper's encoder (with and without RDOQ), mask 0
identical to the anchor, and the anchors re-encoded by this binary are identical to the paper's anchors.

```bash
cd VTM_WMSE_abl && patch -p1 < ../vtm/vtm-23.8_weighted_rdo.patch && patch -p1 < ../decision_ablation/vtm_decision_mask.patch
```

## Protocol

Kodak, 24 images, QP 22 27 32 37, the paper's block maps (`sbh8m256`, `mbh8m256`, `bh8`, `dgnb8j3`, `gnb8j3`),
RDOQ variant, tau in {0.5, 1, 2, 4}, extended with tau 0.25, 0.125 (and 0.0625, 0.03125) when a mask does not reach
the matched cost. Masks per metric: 255 (reference, same maps), leave-one-out 255 - 2^k, only-one 2^k. One driver
run per (metric, mask):

```bash
python method/vvc_rdo_experiment.py --metric LPIPS --estimator gnblock --probes 256 --jitter 3 --weight-tag bh8 \
    --qps 22 27 32 37 --taus --rdoq-taus 0.5 1 2 4 --jobs 16 --encoder-bin <VTM_WMSE_abl>/bin/EncoderAppStatic \
    --run-tag m254 --rdo-mask 254 --drop-recs
```

Reading: every series is interpolated along tau to PerceptQPA's Y-PSNR cost (+8.15 % BD-rate, `qpa_cost.json`), the
paper's matched-cost reading. Leave-one-out "gain lost" = BD(255 - 2^k) - BD(255). Only-one masks rarely reach the
matched cost; the table gives their value at the nearest sweep endpoint (marked `*`) and their best value on the
sweep with the tau and Y-PSNR cost where it occurs.

## Files

```
README.md                 this file, with the current table appended
table.md, table.csv       the table (python analyze.py regenerates both from summary.csv)
summary.csv               one row per (mask, tau): BD-rates of every run of this study
runs/                     per-run JSONs (per-image, per-QP scores) of every run of this study
analyze.py                builds the table
qpa_cost.json             the matched cost
vtm_decision_mask.patch   the encoder change (VTM_WMSE -> VTM_WMSE_abl)
```

## Current table

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

reference, all decisions weighted (mask 255): target -17.3

| decision | leave-one-out mask | target BD (LOO) | gain lost | only-one mask | target BD (ONLY, matched cost) | ONLY peak BD (tau, cost, max cost) | LOO psnr_y | LOO ssim | LOO ms_ssim | LOO lpips | LOO dists | LOO wd2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| partition | 254 | -11.8 | +5.5 | 1 | -8.0 | -8.4 (tau 4, +2.5 %, max +11.3 %) | +8.1 | +5.7 | +5.3 | -0.8 | -11.8 | -1.3 |
| LFNST index | 253 | -16.3 | +1.0 | 2 | -0.7* | -2.4 (tau 0.25, +2.0 %, max +4.1 %) | +8.1 | +5.1 | +4.8 | -3.8 | -16.3 | -4.7 |
| MTS flag | 251 | -17.0 | +0.3 | 4 | -1.6* | -1.6 (tau 0.03125, +0.9 %, max +0.9 %) | +8.1 | +5.2 | +4.8 | -4.0 | -17.0 | -5.1 |
| MTS/TS type | 247 | -17.9 | -0.6 | 8 | -0.1* | -1.0 (tau 0.25, +1.1 %, max +1.3 %) | +8.1 | +5.0 | +4.8 | -4.3 | -17.9 | -5.2 |
| luma mode | 239 | -16.6 | +0.7 | 16 | -3.5* | -3.5 (tau 0.03125, +2.0 %, max +2.0 %) | +8.1 | +5.1 | +4.8 | -3.5 | -16.6 | -4.8 |
| ISP | 223 | -17.0 | +0.4 | 32 | +0.7* | -0.7 (tau 4, +0.2 %, max +2.1 %) | +8.1 | +5.0 | +4.7 | -4.0 | -17.0 | -5.0 |
| chroma | 191 | -15.4 | +1.9 | 64 | -1.2* | -1.8 (tau 2, +0.2 %, max +0.4 %) | +8.1 | +4.8 | +4.5 | -3.6 | -15.4 | -3.5 |
| RDOQ lambda | 127 | -14.9 | +2.4 | 128 | +1.7 | -1.6 (tau 2, +0.5 %, max +12.7 %) | +8.1 | +5.3 | +5.0 | -3.4 | -14.9 | -4.5 |

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
