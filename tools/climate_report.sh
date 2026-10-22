#!/usr/bin/env bash
# Climate report (ClimateReport): biome and field maps of a climate table (src/climate.cpp describes the format).
# Pure CPU, headless (no GPU, no window).
#
#   tools/climate_report.sh [--no-build] [--climate=PATH] [--voxel=M] [--lods=N] [--extent_km=KM] [--size=N] [--surface=0|1] [--center=X,Z] [--lattice=M]
#
# Outputs land in bench/results/<timestamp>-climate/: report.md, biomes_*.png, ground_height.png, ground_shaded.png,
# ground_normals.png, field_*.png, heights.f32 + heights.json (tools/landforms/compare.py), godot.log.
# Exit code: 0 ok, 1 build failed, 2 godot crashed/timed out, 3 the table was refused.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GODOT="${GODOT:-$HOME/.local/bin/godot}"
BUILD=1
ARGS=()
for a in "$@"; do
    case "$a" in
        --no-build) BUILD=0 ;;
        --climate=*|--voxel=*|--lods=*|--extent_km=*|--size=*|--surface=*|--center=*|--lattice=*) ARGS+=("$a") ;;
        *) echo "usage: $0 [--no-build] [--climate=PATH] [--voxel=M] [--lods=N] [--extent_km=KM] [--size=N] [--surface=0|1] [--center=X,Z] [--lattice=M]"; exit 1 ;;
    esac
done

RUN_DIR="$ROOT/bench/results/$(date +%Y%m%d-%H%M%S)-climate"
mkdir -p "$RUN_DIR"

if [[ $BUILD == 1 ]]; then
    echo ">> building"
    if ! (cd "$ROOT" && scons platform=linux target=template_debug -j"$(nproc)" > "$RUN_DIR/build.log" 2>&1); then
        grep -E "error|Error" "$RUN_DIR/build.log" | head -30
        echo ">> BUILD FAILED (full log: $RUN_DIR/build.log)"
        exit 1
    fi
fi

ulimit -c 0
timeout "${CLIMATE_TIMEOUT:-600}" "$GODOT" --headless --path "$ROOT/project" res://debug/climate_report.tscn -- \
    --out="$RUN_DIR" "${ARGS[@]}" > "$RUN_DIR/godot.log" 2>&1
CODE=$?
if grep -q '^CLIMATE: .*"error":' "$RUN_DIR/godot.log"; then
    grep '^CLIMATE: ' "$RUN_DIR/godot.log"
    echo ">> the climate table was refused (log: $RUN_DIR/godot.log)"
    exit 3
fi
if [[ ! -f "$RUN_DIR/report.md" ]] || ! grep -q '^CLIMATE: .*"ok":true' "$RUN_DIR/godot.log"; then
    tail -20 "$RUN_DIR/godot.log"
    echo ">> no report: godot crashed or timed out (exit $CODE, log: $RUN_DIR/godot.log)"
    exit 2
fi
cat "$RUN_DIR/report.md"
echo ">> $RUN_DIR"
