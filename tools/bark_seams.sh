#!/usr/bin/env bash
# Bark seam test: loads the 5 cm forest at a few close-up poses (the camera of marks where seams were seen, and the
# preset's), takes the page shader's bark probe views (bench.gd --probe) and measures how much the bark pattern
# jumps across chunk borders against how much it changes inside a chunk (tools/bark_seams.py). Needs a window and an
# unlocked session; one build, then ~20 s per pose.
#
#   tools/bark_seams.sh [--no-build] [--only=a,b] [-- extra bench args]
#
# Report: bench/results/<date>-bark-seams/report.md, with <pose>_seams.png (seam pairs coloured by their jump).
# Exit code: 0 pass, 1 a seam class has too many visible jumps, 2 a bench run failed.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD=1
ONLY=""
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build) BUILD=0 ;;
        --only=*) ONLY="${1#--only=}" ;;
        --) shift; EXTRA=("$@"); break ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done

FOREST=(--mode=direct --trees=4000 --tree_area=316 --voxel=0.05 --lod_error=2 --steady=20)
# name: camera args (Godot metres, degrees)
POSES=(
    "trunk_seam:--cam_pos=-65.77,1246.92,-96.75 --pitch=-11.5 --yaw=55.6"
    "trunks_a:--cam_pos=-83.89,1243.73,-167.91 --pitch=-10.9 --yaw=5.4"
    "trunks_b:--cam_pos=-174.50,1245.05,-247.56 --pitch=-33.6 --yaw=65.9"
    "trunks_c:--cam_pos=-69.47,1251.45,-74.12 --pitch=-15.6 --yaw=94.5"
    "trunks_d:--cam_pos=-158.47,1248.68,-86.34 --pitch=-15.8 --yaw=143.9"
    "preset:--cam_pos=-52,320,-95 --above_ground=5 --pitch=-5 --yaw=-90"
)

OUT="$ROOT/bench/results/$(date +%Y%m%d-%H%M%S)-bark-seams"
mkdir -p "$OUT"
if [[ $BUILD == 1 ]]; then
    (cd "$ROOT" && scons platform=linux target=template_debug -j"$(nproc)" >"$OUT/build.log" 2>&1) || {
        echo "build failed: $OUT/build.log" >&2
        exit 2
    }
fi

PREFIXES=()
for pose in "${POSES[@]}"; do
    name="${pose%%:*}"
    if [[ -n "$ONLY" && ",$ONLY," != *",$name,"* ]]; then
        continue
    fi
    read -r -a cam <<<"${pose#*:}"
    echo ">> $name"
    if ! BENCH_RESOLUTION="${BENCH_RESOLUTION:-1920x1080}" "$ROOT/tools/bench.sh" --no-build --tag="bark-$name" \
        "${FOREST[@]}" "${cam[@]}" --probe="$OUT/$name" "${EXTRA[@]}" >"$OUT/$name.log" 2>&1; then
        echo "bench failed for $name: $OUT/$name.log" >&2
        exit 2
    fi
    PREFIXES+=("$OUT/$name")
done

python3 "$ROOT/tools/bark_seams.py" --overlay --md "$OUT/report.md" "${PREFIXES[@]}"
status=$?
echo "report: $OUT/report.md"
exit $status
