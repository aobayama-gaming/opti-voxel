#!/usr/bin/env bash
# Integer-frame end-to-end check: the same world point seen from different world origins must give the
# same meshes. The terrain is a function of the absolute position (exact noise), so every chunk meshed
# in two runs that look at the same place must have bit-identical page contents.
#
#   tools/origin_parity.sh [--voxel=M] [-- bench args...]
#   e.g. tools/origin_parity.sh -- --mode=direct --validate=64 --audit=1
#        tools/origin_parity.sh --voxel=0.05 -- --mode=direct
#
# Runs (no build, load scenario, 60 steady frames unless overridden, page audit always on):
#   zero, zero2   origin 0 twice: determinism (identical digests and counts) and the screenshot noise floor
#   near          origin one root span away, camera on the same world point as zero
#   far_a, far_b  camera at ~1e15 m, origins on it and one root span away
#   wrap_a, wrap_b camera at 2^48 voxels on x and z (voxel and noise-cell low words wrap there for every
#                 octave up to 2^16 voxels), origins on each side of it
# Pairs are compared on the per-chunk mesh digests of the page audit (bench chunks.json): the same
# non-empty chunks, each with an identical digest. A camera one root span
# away from the origin (229 km at 0.5 m) jitters at the cm level in float, so pair screenshots and counts
# are reported, not gated. Report: bench/results/<timestamp>-parity.md
# Exit code: 0 parity holds, 1 a run failed, 2 parity broken.
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

# Origins and cameras in voxels. R = the root chunk span (2^15 ids of 14 voxels, chunk_size 32,
# skirt 2, 15 LODs); every origin is a multiple of it, so nothing is snapped. "far" is about 1e15 m
# whatever the voxel size. y stays 0: the terrain's datum is at absolute y ~ 300 m.
read -r R FAR WRAPX WRAPZ <<< "$(python3 -c "
r = 2**15 * 14
k = round(1e15 / float('$VOXEL') / r)
w = 2**48
print(r, k * r, (w // r) * r, (-w // r) * r)")"
W=$((1 << 48))
declare -A ORIGIN CAMERA
ORIGIN[zero]="0,0,0";                              CAMERA[zero]="0,0,0"
ORIGIN[zero2]="0,0,0";                             CAMERA[zero2]="0,0,0"
ORIGIN[near]="$R,0,-$R";                           CAMERA[near]="0,0,0"
ORIGIN[far_a]="$FAR,0,-$FAR";                      CAMERA[far_a]="$FAR,0,-$FAR"
ORIGIN[far_b]="$((FAR + R)),0,$((-FAR + R))";      CAMERA[far_b]="$FAR,0,-$FAR"
ORIGIN[wrap_a]="$WRAPX,0,$WRAPZ";                  CAMERA[wrap_a]="$W,0,-$W"
ORIGIN[wrap_b]="$((WRAPX + R)),0,$((WRAPZ + R))";  CAMERA[wrap_b]="$W,0,-$W"
LIST=""
for label in zero zero2 near far_a far_b wrap_a wrap_b; do
    echo ">> $label: origin ${ORIGIN[$label]} camera ${CAMERA[$label]}"
    "$ROOT/tools/bench.sh" --no-build --tag="parity-$label" --scenario=load --steady=60 --voxel="$VOXEL" "${BENCH_ARGS[@]}" \
        --audit=1 --origin="${ORIGIN[$label]}" --camera_voxel="${CAMERA[$label]}" \
        --shot=loaded:loaded.png --shot=steady_end:steady.png > /dev/null 2>&1
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
