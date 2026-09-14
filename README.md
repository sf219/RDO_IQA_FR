# Rate-distortion optimization for full-reference image quality metrics via stochastic curvature estimates

Code, VTM encoder patches and result summaries for

> S. Fernández-Menduiña, E. Pavez, and A. Ortega, "Rate-distortion optimization for full-reference image quality
> metrics via stochastic curvature estimates," submitted to IEEE ICASSP 2027.

```bibtex
@inproceedings{fernandez2027rdo,
  title     = {Rate-Distortion Optimization for Full-Reference Image Quality Metrics via Stochastic Curvature Estimates},
  author    = {Fern\'andez-Mendui\~na, Samuel and Pavez, Eduardo and Ortega, Antonio},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2027},
  note      = {submitted}
}
```

The SSE inside the rate-distortion optimization of a VVC encoder is replaced by an input-dependent quadratic
distortion (IDQD) whose weight matrix is the Hessian of a full-reference metric (SSIM, MS-SSIM, LPIPS, DISTS,
Wasserstein Distortion) at the source image. The Hessian is estimated from Hessian-vector products, restricted to
its diagonal or to 8x8 diagonal blocks, and passed to the encoder as a map. The decoder is unchanged.

## Layout

```
method/        the method
  hessian_weights.py      curvature estimators (Hutchinson and Gauss-Newton, diagonal and block), smoothing, PSD projection, map I/O
  gen_maps.py             builds the maps of a dataset
  vvc_rdo_experiment.py   maps -> VTM encodes over a tau grid -> scoring (Y-PSNR, SSIM, MS-SSIM, LPIPS, DISTS, WD) -> BD-rates
  Common/utils/q_utils.py metric wrappers (SSIM, MS-SSIM, LPIPS, DISTS); msssim_std.py, msssim_fs.py: MS-SSIM variants
  vendor/                 Wasserstein Distortion, reference implementation (Apache-2.0)
vtm/           encoder patches
  vtm-23.8_weighted_rdo.patch   weighted SSE in every RDO decision + RDOQ multiplier (main protocol)
  vtm-23.0_qpmap.patch          the same, plus a CTU-level delta-QP map on top of PerceptQPA (Table 3 protocol)
experiments/   one script per table and figure, the CSV summaries they read, and their outputs
  results/     summaries      tables/  the paper's tables      figures/  the paper's figures
REPRODUCE.md   every number in the paper, the command that produced it, and the conventions
```

| paper item | script in `experiments/` |
|---|---|
| Table 1 (Kodak + CLIC), Fig. 4 | `plots/make_all.py`, `clic_run.py`, `clic_table.py`, `combined_table.py`, `plots/figs.py` |
| Table 2 | `ablation_table.py` |
| Table 3 | `litqp_variant2.sh`, `litqp_score.py`, `lit_table4.py` |
| Fig. 3 | `maps_fig.py` |
| Fig. 5 | `timebench_v2.sh`, `complexity_csv.py`, `complexity_bar.py` |
| Fig. 6 | `probes_fig.py` |
| numbers in the text | `hess_structure.py`, `estimator_diag.py`, `stats_table1.py`, `paper_numbers.py` |

## Setup

```bash
pip install -r requirements.txt          # Python 3.9; ffmpeg must be on the path
export PYTHONPATH=$PWD/method

git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git -b VTM-23.8 VTM_WMSE
cd VTM_WMSE && patch -p1 < ../vtm/vtm-23.8_weighted_rdo.patch
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
```

The same with `-b VTM-23.0` and `vtm/vtm-23.0_qpmap.patch` for the Table 3 protocol. Both patches add
`source/Lib/CommonLib/WeightMap.{h,cpp}` and touch the encoder only; bitstreams decode with the stock decoder.
Encoder options: `--WeightedRdo`, `--WeightedRdoFile` (diagonal map), `--WeightedRdoBlockFile` (8x8 tiles),
`--WeightedRdoTau`, `--WeightedRdoq`.

## Running

```bash
# maps for one series (LPIPS, smoothed Gauss-Newton block map, 256 probes)
python method/gen_maps.py --metric LPIPS --estimator gnblock --probes 256 --jitter 3 --weight-tag bh8

# the full series: maps (cached), encodes at four QPs over a tau grid, scoring, BD-rates -> results/vvc_rdo_lpips_bh8_<date>.json
python method/vvc_rdo_experiment.py --metric LPIPS --estimator gnblock --probes 256 --jitter 3 --weight-tag bh8 \
    --qps 22 27 32 37 --taus 0.25 0.5 1 2 3 4 6 --rdoq-taus 0.25 0.5 1 2 3 4 6 --jobs 12

# tables and figures from the summaries
cd experiments && python results_to_csv.py --results ../method/results && python plots/make_all.py
```

Datasets: Kodak (24 images) and the CLIC 2022 professional validation set; see `REPRODUCE.md`. Set `EVAL_WD=1`
to score Wasserstein Distortion. Paths to the encoder binaries and to the datasets are set at the top of
`method/vvc_rdo_experiment.py` and of the two shell scripts.

## License

MIT (`LICENSE`). The patches under `vtm/` modify the JVET VTM reference software and are distributed under its
BSD-3-Clause license. `method/vendor/wasserstein_distortion/` is Apache-2.0.
