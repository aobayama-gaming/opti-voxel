#!/usr/bin/env bash
# Integer-frame end-to-end check: the same bench run at several world origins must give the same
# scene, because the current SDF is written in metres around the origin (SDFMetreBridge).
#
#   tools/origin_parity.sh [--voxel=M] [-- bench args...]
#   e.g. tools/origin_parity.sh -- --mode=direct --validate=64 --audit=1
#        tools/origin_parity.sh --voxel=0.05 -- --mode=direct --rig=1
#
# Runs (no build, load scenario, 60 steady frames unless overridden): origin 0, origin 0 again (noise floor), ~1e15 m ("far"), and an
# origin where the GPU low words wrap ("wrap", 7 * 2^32 voxels). All origins are multiples of the root
# chunk span, so nothing is snapped. Compares chunk/triangle/quad counts exactly and the loaded /
# steady_end screenshots against the noise floor. Report: bench/results/<timestamp>-parity.md
# Exit code: 0 parity holds, 1 a run failed, 2 parity broken. Delete with the exact noise (the
# terrain then depends on the absolute position).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VOXEL="0.5"
BENCH_ARGS=()
SEEN_DASH=0
for a in "$@"; do
    if [[ $SEEN_DASH == 1 ]]; then
        BENCH_ARGS+=("$a")
    else
        case "$a" in
            --voxel=*) VOXEL="${a#--voxel=}" ;;
            --) SEEN_DASH=1 ;;
            *) BENCH_ARGS+=("$a") ;;
        esac
    fi
done

# Origins in voxels: multiples of 7 * 2^32 (= 2^16 root spans of 458752 voxels for chunk_size 32,
# skirt 2, 15 LODs). "far" is about 1e15 m whatever the voxel size.
read -r FAR WRAP <<< "$(python3 -c "
w = 7 * 2**32
k = round(1e15 / float('$VOXEL') / w)
print(k * w, w)")"
declare -A ORIGINS=(
    [zero]="0,0,0"
    [zero2]="0,0,0"
    [far]="$FAR,-$WRAP,-$FAR"
    [wrap]="$WRAP,$WRAP,-$WRAP"
)
LIST=""
for label in zero zero2 far wrap; do
    echo ">> $label: origin ${ORIGINS[$label]}"
    "$ROOT/tools/bench.sh" --no-build --tag="parity-$label" --scenario=load --steady=60 --voxel="$VOXEL" "${BENCH_ARGS[@]}" \
        --origin="${ORIGINS[$label]}" --shot=loaded:loaded.png --shot=steady_end:steady.png > /dev/null 2>&1
    CODE=$?
    DIR="$(cat "$ROOT/bench/results/LAST")"
    if [[ $CODE -ne 0 ]]; then
        echo ">> run $label failed (exit $CODE): $DIR"
        exit 1
    fi
    LIST="$LIST $label=$DIR"
done

OUT="$ROOT/bench/results/$(date +%Y%m%d-%H%M%S)-parity.md"
python3 "$ROOT/tools/origin_parity.py" "$OUT" $LIST
