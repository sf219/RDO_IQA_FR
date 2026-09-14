# Reproducing the numbers in the paper

Every table, figure and number in the text maps to a command below. Paths are relative to the repository root;
`method/` holds the estimator and the encode/score driver, `experiments/` the table and figure scripts with the
summaries they read, `vtm/` the encoder patches. Series are named by a *tag*; the tag is the suffix of the map
directory `work/weights/<METRIC>_<tag>/` and of the `base` column of the CSV summaries.

## 0. Environment

* Python 3.9 with the packages in `requirements.txt`; `ffmpeg` on the path (PNG to YUV 4:2:0).
* Run the scripts with `method/` on the Python path: `export PYTHONPATH=$PWD/method`.
* Encoders. Main protocol: VTM-23.8 plus `vtm/vtm-23.8_weighted_rdo.patch`. Protocol of Yang and Bajic (Table 4):
  VTM-23.0 plus `vtm/vtm-23.0_qpmap.patch`. Build:
  ```bash
  git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git -b VTM-23.8 VTM_WMSE
  cd VTM_WMSE && patch -p1 < ../vtm/vtm-23.8_weighted_rdo.patch
  mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
  ```
  The driver expects `bin/EncoderAppStatic` and `cfg/encoder_intra_vtm.cfg` of that tree; the paths are set at the
  top of `method/vvc_rdo_experiment.py` and of the shell scripts.
* Encoder options added by the patches: `WeightedRdo` (on/off), `WeightedRdoFile` (diagonal map, `.dat`),
  `WeightedRdoBlockFile` (8x8 tiles, `.bh`), `WeightedRdoTau` (the tau of the paper), `WeightedRdoq` (scale the RDOQ
  Lagrangian by 1/w per transform unit; on in every result), `WeightedRdoGlobalNorm`; VTM-23.0 patch only:
  `WeightedRdoQpMap` / `WeightedRdoQpMapClip` (per-CTU delta-QP = -3 log2 w, clipped to +-4, on top of `PerceptQPA`).
* Environment variables read by `hessian_weights.py` (defaults are what the paper used): `HW_HVP=fwd`
  (forward-over-reverse Hessian-vector products), `HW_HVP_BATCH=16`, `HW_ROUND_SIZE=16` (R = m/16 rounds when
  smoothing), `HW_CKPT_MIN_PIXELS` (activation-checkpointed path for large images), `HW_YUV_RANGE=full` (Sec. 5),
  `HW_BH_CKPT_SEC=300` (resume checkpoints of the block Hutchinson estimator). `EVAL_WD=1` scores Wasserstein
  Distortion (required for every WD column). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is recommended.

## 1. Data

* Kodak: the 24 images, all in landscape orientation (768x512). The driver converts each PNG to
  `work/yuv/<stem>.yuv` with `ffmpeg -pix_fmt yuv420p` (limited-range BT.601 luma, 16..235).
* CLIC 2022 professional validation set: 41 images, cropped to multiples of 8.

## 2. Curvature maps

`method/gen_maps.py` builds one map per image and series (tau-independent; tau is an encoder flag) into
`work/weights/<METRIC>_<tag>/<stem>.dat` (diagonal) or `.bh` (8x8 blocks). The driver of Sec. 3 builds missing maps
itself with the same code.

| Metric | Diagonal (D) | Block (B) |
|---|---|---|
| SSIM | `--estimator hutch --probes 256 --weight-tag sn256` | `--estimator hutchblock --probes 256 --weight-tag sbh8m256` |
| MS-SSIM (luma) | `--estimator hutch --probes 256 --weight-tag mn256` | `--estimator hutchblock --probes 256 --weight-tag mbh8m256` |
| LPIPS (VGG) | `--estimator gn --probes 256 --jitter 3 --weight-tag gnj3` | `--estimator gnblock --probes 256 --jitter 3 --weight-tag bh8` |
| DISTS | `--estimator gn --probes 256 --jitter 3 --weight-tag dgnj3` | `--estimator gnblock --probes 256 --jitter 3 --weight-tag dgnb8j3` |
| WD (log2 sigma = 2) | `--estimator gn --probes 256 --jitter 3 --wd-sigma 2 --weight-tag gnj3` | `--estimator gnblock --probes 256 --jitter 3 --wd-sigma 2 --vjp-batch 1 --weight-tag gnb8j3` |
| LPIPS-Alex (Table 4) | `--metric LPIPS_ALEX --estimator gn --probes 256 --jitter 3 --weight-tag alexd` | `--metric LPIPS_ALEX --estimator gnblock --probes 256 --jitter 3 --weight-tag alexb8` |
| MS-SSIM on RGB (Table 4) | `--metric MS_SSIM_RGB --estimator hutch --probes 256 --weight-tag mrgbd` | `--metric MS_SSIM_RGB --estimator hutchblock --probes 256 --weight-tag mrgbb8` |

Estimators: `hutch` / `hutchblock` = Hutchinson diagonal / block (Eq. 4 of the paper), `gn` / `gnblock` =
Gauss--Newton diagonal / block (Eq. 5). `--jitter 3` is the smoothing sigma (8-bit units) with R = m/16 rounds;
`--psd eigh` projects block tiles onto the PSD cone (the "block Hutchinson (HVP), PSD" row of Table 2:
`--estimator hutchblock --probes 256 --psd eigh` with tags `hb8m256psd` for LPIPS and WD, `dhb8m256psd` for DISTS).
Post-processing: clip at the 99.5th percentile (`--clip-pct 99.5`); per-plane unit-mean normalization is done by the
encoder (`WeightMap`); luma-only metrics get identity chroma. `--seed 0` everywhere (`s1` tags: seed 1).
`--shard J/K` and `--reverse` split a series over several jobs; the block Hutchinson estimator writes a resume
checkpoint (`<map>.bh.ckpt`) every `HW_BH_CKPT_SEC` seconds and continues from it with the same probe sequence.

## 3. Encodes and scoring (main protocol, VTM-23.8)

`method/vvc_rdo_experiment.py` does maps (skipped if cached), encodes, scoring and BD-rates, and writes
`results/vvc_rdo_<metric>_<tag>_<date>.json` (per image and configuration: bpp, Y-PSNR, SSIM, MS-SSIM, LPIPS, DISTS, WD).

```bash
python method/vvc_rdo_experiment.py --metric LPIPS --estimator gnblock --probes 256 --jitter 3 --weight-tag bh8 \
    --qps 22 27 32 37 --taus 0.25 0.5 1 2 3 4 6 --rdoq-taus 0.25 0.5 1 2 3 4 6 --jobs 12
```

`--rdoq-taus` are the tau values encoded with `--WeightedRdoq=1`; every configuration in the paper has it on (the
configuration tags end in `_rdoq`). Anchor encodes (`anchor_qp<QP>`) come from the same driver. Tau grids:

| Series (`base` in `experiments/results/kodak_json/summary.csv`) | tau |
|---|---|
| Table 1: `wssim_sn256`, `wssim_sbh8m256`, `wms_ssim_mn256`, `wms_ssim_mbh8m256`, `wlpips_gnj3`, `wlpips_bh8`, `wdists_dgnj3`, `wdists_dgnb8j3`, `wwd_gnb8j3` | 0.25 0.5 1 2 3 4 6 |
| `wwd_gnj3` (WD diagonal, Table 1) | 0.03125 0.0625 0.125 0.25 0.5 1 2 3 4 6 |
| Table 2 unsmoothed rows: `wlpips_gn`, `wlpips_gnb8`, `wdists_dgn`, `wdists_dgnb8`, `wwd_gnb8` (`wwd_gn` adds 0.125) | 0.25 0.5 1 2 3 4 6 |
| Table 2 PSD HVP row `*psd`; probe sweep of Fig. 6 `*n<m>` | 0.25 0.5 1 2 4 |
| Table 4 (VTM-23.0 grids): `qpmap_*`, `rq_*`, `rqb_*` | 0.25 0.5 1 2 4 (`qpmap_*` also 0) |

MS-SSIM is scored with the standard 5-level MS-SSIM (`method/msssim_std.py`, sigma = 1.5 window); the WD column
is `wd2` (log2 sigma = 2). `PerceptQPA` baseline: encodes with `--PerceptQPA=1`, BD-rates by
`experiments/qpa_bd.py` (`work/qpa_bd_ctc.json`); `experiments/pick_tau.py` writes its Y-PSNR cost to
`work/qpa_cost.json` (+8.15 % on Kodak). Aggregation: `experiments/results_to_csv.py --results method/results`
writes `experiments/results/kodak_json/summary.csv` (BD-rate: piecewise-linear in log-rate over the shared quality range).

## 4. Tables and figures

All scripts live in `experiments/` and read the CSV summaries under `experiments/results/`.

| Item | Command | Notes |
|---|---|---|
| Table 1 (Kodak) | `python plots/make_all.py` -> `tables/table1_paper.tex`; `python stats_table1.py` (win counts, bootstrap CIs); `python paper_numbers.py` (abstract and Main-result numbers) | rows interpolated along tau to `PerceptQPA`'s Y-PSNR cost; series in `plots/make_all.py:SERIES`; `PerceptQPA` is labelled PQA |
| Table 2 (ablation) | `python ablation_table.py` -> `tables/table3_ablation.tex` | series per row in `ablation_table.py:ROWS`, all read at the same cost; the console output also prints the unprojected HVP, second-seed and PSD-check numbers quoted in the text |
| Table 3 (CLIC) | `python clic_run.py --summary --out results/clic` then `python clic_table.py` -> `tables/table2_clic_paper.tex`, `tables/clic_macros.tex` | per-image JSONs come from `clic_run.py` (maps, anchor, `PerceptQPA`, diagonal and block rows, scoring); three tau per row, interpolated to `PerceptQPA`'s Y-PSNR cost on CLIC (+9.7 %); the Y-PSNR column is not printed since every row sits at that cost |
| Table 4 (Yang and Bajic) | maps: `litqp_maps.sh` (diagonal), Sec. 2 block rows; encodes: `TAG=rqb EXTRA=--WeightedRdoq=1 MAPS=$'alexb LPIPS_ALEX_alexb8 bh\nmrgbb MS_SSIM_RGB_mrgbb8 bh' ./litqp_variant2.sh`; scoring: `python litqp_score.py` -> `results/litqp/summary.csv`; table: `python lit_table4.py` -> `tables/table4_literature.tex` | VTM-23.0, CTU 64 (`--CTUSize=64 --MaxBTLumaISlice=64 --MaxBTChromaISlice=32 --MaxBTNonISlice=64 --MaxTTLumaISlice=32 --MaxTTChromaISlice=32`), `--PerceptQPA=1 --WeightedRdo=1 --WeightedRdoQpMap=1`, MS-SSIM and PSNR on RGB, LPIPS-Alex; our row read along tau at the RGB-PSNR cost of their best configuration (+0.98 %) |
| Fig. 3 (maps) | `python maps_fig.py --image kodim23` -> `figures/maps.png` | block maps of the five metrics (Sec. 2 tags), per-plane unit-mean normalized as the encoder does, log10 scale, 1-99 % colour limits |
| Fig. 4 (tau sweep) | `python plots/figs.py` -> `figures/tau_sweep.png` | `plots/figs.py:MET` |
| Fig. 5 (complexity) | encoder times: `./timebench_v2.sh` (strictly sequential, one encoder at a time, kodim01/02/03 x {anchor, D, B per metric} x 4 QPs) then `python complexity_csv.py --enc work/timebench_v2`; map times: `work/maptime_a100_fwd.json` (A100, m = 256, forward-over-reverse HVP, same three images); figure: `python complexity_bar.py` -> `figures/complexity_bar.png` | encoder: Intel Xeon E5-2667 v4, 3.20 GHz, single-threaded |
| Fig. 6 (probe budget) | `python probes_fig.py` -> `figures/probes.png`, `results/probes/fig.csv` | block series `w<metric>_<tag>n<m>` for m = 16, 32, 64, 128 (tags `sbh8n<m>`, `mbh8n<m>`, `bh8n<m>`, `dgnb8j3n<m>`, `gnb8j3n<m>`, tau 0.25-4) plus the Table 1 block series at m = 256, every point read at `PerceptQPA`'s Y-PSNR cost; time axis = Fig. 5 A100 time at m = 256 scaled by m/256 |
| Block-diagonal energy (Sec. 4 text) | `python hess_structure.py --metrics SSIM MS_SSIM --crop 0 --margin 192 --tiles 6 --batch 4 --out results/hess_structure/summary_full.csv` and `python hess_structure.py --metrics LPIPS DISTS WD --crop 512 --margin 192 --tiles 6 --batch 4 --out results/hess_structure/summary_c512.csv` | exact Hessian rows on kodim01/05/19; SSIM/MS-SSIM at native size, the feature metrics on 512x512 crops; MEAN rows |
| Negative eigen-energy of HVP tiles (Sec. 4 text) | `python estimator_diag.py --sets DISTS_dhb8m256 LPIPS_hb8m256` -> `results/estimator_diag/summary.csv` | `neg_energy_frac`, plane Y, 24-image mean |

## 5. Conventions that affect numbers

* **YCbCr to RGB.** The YUVs are limited range; `hessian_weights.yuv_to_rgb` uses the full-range matrix
  (`HW_YUV_RANGE=full`). Every result, maps and scoring alike, uses this one convention.
* **BD-rate** is piecewise-linear in log-rate over the shared quality range, not the cubic fit.
* **Matched cost.** Tables 1-3 and Fig. 6 read every row at `PerceptQPA`'s Y-PSNR BD-rate by linear interpolation
  along the tau sweep, inside the sweep only; Table 4 at the RGB-PSNR cost Yang and Bajic report.
* **Weighted RDOQ** (`--WeightedRdoq=1`) is on in every result of the paper.
* **Hutchinson block tiles** are indefinite for the neural-network metrics (42 % of the eigenvalue mass negative for
  DISTS); the paper's HVP row uses `--psd eigh`.

## 6. Numbers in the text

| Paper element | Source |
|---|---|
| Abstract and introduction ranges (16-24 %, up to 12 % over `PerceptQPA`, 10-30 % runtime), Main-result numbers | `python paper_numbers.py` (reads `results/table1/summary.csv`, `results/table1/stats.csv`, `work/qpa_bd_ctc.json`, `results/complexity/summary.csv`) |
| Bootstrap 95 % intervals (Main result) | `python stats_table1.py` -> `results/table1/stats.csv` |
| `PerceptQPA` cost +8.1 % (Kodak), +9.7 % (CLIC) | `work/qpa_cost.json` (`pick_tau.py`); `tables/clic_macros.tex` (`clic_table.py`) |
| Block-diagonal energy 99 / 99 / 63 / 15 / 0.3 %, diagonal 11 / 42 / 18 / 4 / 0.02 % | `results/hess_structure/summary_full.csv` (SSIM, MS-SSIM) and `summary_c512.csv` (LPIPS, DISTS, WD), MEAN rows |
| Ablation text: sigma = 6 and 10 worse; Gauss--Newton over block Hutchinson at the same m | `python ablation_table.py` console output (`wlpips_bh8j6`, `wlpips_bh8j10` sweeps; HVP rows) |
| Complexity: block map 1.10 to 1.25x the anchor; maps 1-10 s per image | `complexity_bar.py` (mean block encode time over images and QPs divided by the anchor mean; per metric SSIM 1.12, MS-SSIM 1.16, LPIPS 1.22, DISTS 1.10, WD 1.25); `work/maptime_a100_fwd.json` |
| Number of probes: DISTS and WD move by about one point from m = 64 to 256, SSIM and MS-SSIM keep improving, m = 64 keeps at least 83 % of the gain at a quarter of the time | `python probes_fig.py` console output, `results/probes/fig.csv` (`bd_target`, `map_time_s`) |
