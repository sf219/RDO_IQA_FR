#!/bin/bash
# Sequential encoding-time benchmark (plan 3.7), re-timed with the exact symmetric-float block
# kernel + diagonal-plane path (2026-09-03).  Strictly one encoder at a time, nothing else on
# the machine.  Output: work/timebench_v2/<img>_<cfg>_qp<QP>.log  ->  complexity_csv.py --enc.
# Same 3 images x 9 configs x 4 QPs as work/timebench (old double kernel, 2026-08-31).
set -u
cd "$(dirname "$0")"
BIN=${BIN:-/home/samustac/WORK/USC/STAC/VTM_WMSE/bin/EncoderAppStatic}
CFG=/home/samustac/WORK/USC/STAC/VTM_WMSE/cfg/encoder_intra_vtm.cfg
OUT=work/timebench_v2
COMMON="--InputBitDepth=8 --OutputBitDepth=8 --InputChromaFormat=420"
W="--WeightedRdo=1 --WeightedRdoTau=1 --WeightedRdoq=1"
mkdir -p $OUT
TMP=/tmp/claude-1003/-home-samustac-WORK-USC-STAC/c74f9fef-c31a-4533-be06-4ec23c1152ae/scratchpad/tb
$BIN --help 2>&1 | head -1 > /dev/null
for IM in kodim01 kodim02 kodim03; do
  for QP in 27 32 37 42; do
    for SPEC in \
      "anchor" \
      "ssim_diag    $W --WeightedRdoFile=work/weights/SSIM_sn256/$IM.dat" \
      "ssim_block   $W --WeightedRdoBlockFile=work/weights/SSIM_sbh8/$IM.bh" \
      "msssim_diag  $W --WeightedRdoFile=work/weights/MS_SSIM_mn256/$IM.dat" \
      "msssim_block $W --WeightedRdoBlockFile=work/weights/MS_SSIM_mbh8/$IM.bh" \
      "lpips_diag   $W --WeightedRdoFile=work/weights/LPIPS_gn/$IM.dat" \
      "lpips_block  $W --WeightedRdoBlockFile=work/weights/LPIPS_bh8/$IM.bh" \
      "dists_diag   $W --WeightedRdoFile=work/weights/DISTS_dgn/$IM.dat" \
      "dists_block  $W --WeightedRdoBlockFile=work/weights/DISTS_dgnb8/$IM.bh"
    do
      read -r TAG REST <<< "$SPEC"
      LOG=$OUT/${IM}_${TAG}_qp${QP}.log
      [ -s "$LOG" ] && grep -q 'Total Time' "$LOG" && continue
      $BIN -c $CFG -i work/yuv/$IM.yuv -wdt 768 -hgt 512 -fr 1 -f 1 -q $QP \
           -b $TMP/tbv2.bin -o /dev/null $COMMON ${REST:-} > "$LOG" 2>&1
      echo "$IM $TAG qp$QP $(grep -o 'Total Time:[^[]*' "$LOG")"
    done
  done
done
rm -f $TMP/tbv2.bin
echo TIMEBENCH_V2_DONE
