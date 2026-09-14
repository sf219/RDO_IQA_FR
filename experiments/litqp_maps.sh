#!/bin/bash
# Diagonal maps for the literature-protocol QP-adaptation comparison.  The QP-map mode averages per-pixel
# weights over each 64x64 block, so it needs the diagonal form; the block tiles we kept are not usable here.
cd /home/samustac/WORK/USC/STAC/TIP_Hess_v2/week_7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HW_PROBE_DEVICE=cuda GPU_CLAIM_GB=3
PY=/home/samustac/anaconda3/envs/tip_torch/bin/python
echo "=== LPIPS-Alex diagonal $(date +%H:%M) ==="
$PY gen_maps.py --metric LPIPS_ALEX --estimator gn --probes 256 --jitter 3 --weight-tag alexd \
  --claim-gb 4 --min-free-gb 2 2>&1 | grep -E "^\s+\[|Error|Traceback"
echo "=== MS-SSIM-RGB diagonal $(date +%H:%M) ==="
$PY gen_maps.py --metric MS_SSIM_RGB --estimator hutch --probes 256 --weight-tag mrgbd \
  --claim-gb 4 --min-free-gb 2 2>&1 | grep -E "^\s+\[|Error|Traceback"
for d in LPIPS_ALEX_alexd MS_SSIM_RGB_mrgbd; do echo "$d: $(ls work/weights/$d/*.dat 2>/dev/null | wc -l)/24"; done
touch work/LITQP_MAPS_DONE; echo LITQP_MAPS_DONE
