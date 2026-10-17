#!/usr/bin/env bash
# Hermite cascade report (CascadeReport): fidelity, cost per chunk and PNG slices of the CPU reference.
# Pure CPU, headless (no GPU, no window).
#
#   tools/cascade_report.sh [--no-build] [--scale=N]
#
# Outputs land in bench/results/<timestamp>-cascade/: report.md, slice_*.png, godot.log.
# Exit code: 0 ok, 1 build failed, 2 godot crashed/timed out.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GODOT="${GODOT:-$HOME/.local/bin/godot}"
BUILD=1
SCALE=1
for a in "$@"; do
    case "$a" in
        --no-build) BUILD=0 ;;
        --scale=*) SCALE="${a#--scale=}" ;;
        *) echo "usage: $0 [--no-build] [--scale=N]"; exit 1 ;;
    esac
done

RUN_DIR="$ROOT/bench/results/$(date +%Y%m%d-%H%M%S)-cascade"
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
timeout "${CASCADE_TIMEOUT:-1200}" "$GODOT" --headless --path "$ROOT/project" res://debug/cascade_report.tscn -- \
    --out="$RUN_DIR" --scale="$SCALE" > "$RUN_DIR/godot.log" 2>&1
CODE=$?
if [[ ! -f "$RUN_DIR/report.md" ]] || ! grep -q '^CASCADE: .*"ok":true' "$RUN_DIR/godot.log"; then
    tail -20 "$RUN_DIR/godot.log"
    echo ">> no report: godot crashed or timed out (exit $CODE, log: $RUN_DIR/godot.log)"
    exit 2
fi
cat "$RUN_DIR/report.md"
echo ">> $RUN_DIR"
