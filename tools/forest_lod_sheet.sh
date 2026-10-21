#!/usr/bin/env bash
# The forest seen from 20 m to 13 km up, shaded and in the LOD view: one contact sheet to judge the hand-off from
# the trees' parts to their moment blobs (nothing should thin out on the way up).
#
#   tools/forest_lod_sheet.sh [--no-build] [--pitch=DEG] [--altitudes=M,M,...] [-- extra bench args]
#
# Two bench runs (--scenario=rise, --view=0 then --view=2) on --preset=forest5cm, the camera over the forest's
# middle, a screenshot per altitude. Writes bench/results/<timestamp>-forest-lod/: sheet.png (rows: altitudes,
# columns: shaded | LOD) and report.md (per step: the chunks loaded, the instance lists' longest per LOD, the
# lists truncated during the step). Windowed (a GPU run). Exit code: non-zero if a run failed.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD=1
PITCH=-90
ALTITUDES="20,50,100,200,400,800,1600,3200,6400,13000"
EXTRA=()
SEEN_DASH=0
for a in "$@"; do
    if [[ $SEEN_DASH == 1 ]]; then
        EXTRA+=("$a")
    else
        case "$a" in
            --no-build) BUILD=0 ;;
            --pitch=*) PITCH="${a#--pitch=}" ;;
            --altitudes=*) ALTITUDES="${a#--altitudes=}" ;;
            --) SEEN_DASH=1 ;;
            *) echo "usage: $0 [--no-build] [--pitch=DEG] [--altitudes=M,M,...] [-- bench args]"; exit 1 ;;
        esac
    fi
done

OUT="$ROOT/bench/results/$(date +%Y%m%d-%H%M%S)-forest-lod"
mkdir -p "$OUT"
FAILED=0
# The forest square is centred on (SCENE_X, SCENE_Z) = (-50, -150) m.
COMMON=(--mode=direct --preset=forest5cm --scenario=rise --cam_pos=-50,320,-150 --pitch="$PITCH" --yaw=0
    --altitudes="$ALTITUDES" --steady=1 --max_frames=60000)
for view in 0 2; do
    build=()
    [[ $BUILD == 1 && $view == 0 ]] || build=(--no-build)
    echo ">> view $view"
    if ! "$ROOT/tools/bench.sh" "${build[@]}" --tag="forest-lod-v$view" "${COMMON[@]}" --view=$view \
        --rise_shots=shots "${EXTRA[@]}"; then
        FAILED=1
    fi
    run="$(cat "$ROOT/bench/results/LAST")"
    mkdir -p "$OUT/v$view"
    cp "$run"/shots/*.png "$OUT/v$view/" 2> /dev/null
    cp "$run/stats.json" "$OUT/v$view-stats.json" 2> /dev/null
done

IFS=',' read -r -a ALTS <<< "$ALTITUDES"
TILES=()
for i in "${!ALTS[@]}"; do
    for view in 0 2; do
        f="$OUT/v$view/rise_${i}_${ALTS[$i]%.*}m.png"
        [[ -f "$f" ]] && TILES+=("$f") || TILES+=("null:")
    done
done
montage "${TILES[@]}" -tile 2x -geometry 640x360+2+2 -background '#222' "$OUT/sheet.png"

python3 - "$OUT" "$ALTITUDES" << 'EOF'
import json, sys
out, alts = sys.argv[1], sys.argv[2].split(",")
try:
    stats = json.load(open(f"{out}/v0-stats.json"))["phases"]
except (OSError, KeyError, ValueError):
    sys.exit(0)
lines = ["# Forest LOD sheet", "", "| step | altitude m | loaded | wall s | truncated | longest list per LOD |", "|---|---|---|---|---|---|"]
prev = 0
for i, a in enumerate(alts):
    p = stats.get(f"rise_{i}")
    if not p:
        continue
    t = p["terrain"].get("table", {})
    trunc = int(t.get("list_truncated", 0))
    longest = " ".join(f"{l}:{int(v['max'])}" for l, v in sorted(t.get("list_len_per_lod", {}).items(), key=lambda kv: int(kv[0])))
    lines.append(f"| {i} | {a} | {p['terrain']['chunks']['loaded']} | {p['wall_ms'] / 1000:.1f} | {trunc - prev} | {longest} |")
    prev = trunc
open(f"{out}/report.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
echo ">> $OUT/sheet.png"
exit $FAILED
