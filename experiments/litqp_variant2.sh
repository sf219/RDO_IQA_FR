#!/bin/bash
# litqp_variant.sh with the map type selectable: MAPS lists "<label> <weights dir> <ext>" triples, ext dat
# (diagonal, --WeightedRdoFile) or bh (block tiles, --WeightedRdoBlockFile).  Everything else as in
# litqp_variant.sh: their encoder version, anchor, CTU geometry, PerceptQPA chroma, our luma QP map.
set -u
TAG=${TAG:?set TAG}; EXTRA=${EXTRA:-}; MAPS=${MAPS:?set MAPS}; TAUS=${TAUS:-0.25 0.5 1 2 4}
cd /home/samustac/WORK/USC/STAC/TIP_Hess_v2/week_7
B=${B:-/home/samustac/WORK/USC/STAC/VTM_230/bin/EncoderAppStatic}   # override to test another build
C=/home/samustac/WORK/USC/STAC/VTM_WMSE/cfg/encoder_intra_vtm.cfg
COM="--InputBitDepth=8 --OutputBitDepth=8 --InputChromaFormat=420 --CTUSize=64 --MaxBTLumaISlice=64 --MaxBTChromaISlice=32 --MaxBTNonISlice=64 --MaxTTLumaISlice=32 --MaxTTChromaISlice=32"
run_one() {
  read -r im qp tag rest <<< "$1"
  local d=work/enc230/$im; mkdir -p $d
  local bs=$d/${tag}_qp${qp}.bin rec=$d/${tag}_qp${qp}_rec.yuv
  [ -s "$bs" ] && [ "$(stat -c%s "$rec" 2>/dev/null || echo 0)" -eq $((768*512*3/2)) ] && return 0
  $B -c $C -i work/yuv/$im.yuv -wdt 768 -hgt 512 -fr 1 -f 1 -q $qp -b $bs -o $rec $COM $rest > $d/${tag}_qp${qp}.log 2>&1
}
export -f run_one; export B C COM
echo "$MAPS" | while read -r L D E; do [ -n "$L" ] || continue; until [ "$(ls work/weights/$D/*.$E 2>/dev/null | wc -l)" -ge 24 ]; do sleep 30; done; done
J=${JOBS:-work/${TAG}_jobs.txt}; : > $J      # JOBS/DONE/FILL overrides let a second run extend a TAG that is still running
for im in $(seq -f 'kodim%02g' 1 24); do for qp in 22 27 32 37; do for T in $TAUS; do
  echo "$MAPS" | while read -r L D E; do [ -n "$L" ] || continue
    opt=$([ "$E" = bh ] && echo WeightedRdoBlockFile || echo WeightedRdoFile)
    echo "$im $qp ${TAG}_${L}_tau$T --PerceptQPA=1 --WeightedRdo=1 --WeightedRdoQpMap=1 --WeightedRdoQpMapClip=4 --WeightedRdoTau=$T --$opt=work/weights/$D/$im.$E $EXTRA" >> $J
  done
done; done; done
echo "=== $TAG: $(wc -l < $J) encodes, maps: $(echo "$MAPS" | tr '\n' ';'), extra='$EXTRA', start $(date +%H:%M) ==="
xargs -a $J -d '\n' -P ${P:-12} -I{} bash -c 'run_one "{}"'
for p in 1 2; do [ "${FILL:-1}" = 1 ] || break; n=$(find work/enc230 -name "${TAG}_*.bin" -size 0 | wc -l); [ "$n" -eq 0 ] && break
  find work/enc230 -name "${TAG}_*.bin" -size 0 | while read -r b; do rm -f "$b" "${b%.bin}_rec.yuv"; done; xargs -a $J -d '\n' -P ${P:-12} -I{} bash -c 'run_one "{}"'; done
echo "done: $(ls work/enc230/*/${TAG}_*.bin 2>/dev/null | wc -l), empty $(find work/enc230 -name "${TAG}_*.bin" -size 0 | wc -l)"; touch ${DONE:-work/${TAG}_DONE}; echo ${DONE:-${TAG}_DONE}
