#!/usr/bin/env bash
# Integer-frame harness: build (unless --no-build), run res://debug/frame_spike.tscn, check the result.
#
#   tools/frame_test.sh [--no-build] [--voxel=<metres>]
#
# Outputs land in bench/results/<timestamp>-frame/ : frame.json, godot.log.
# Exit code: 0 ok, 1 build failed, 2 godot crashed/timed out, 3 errors in log, 4 a check failed.
# Needs a GPU and an unlocked session (it opens a window).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GODOT="${GODOT:-$HOME/.local/bin/godot}"
BUILD=1
PASS=()
for a in "$@"; do
    case "$a" in
        --no-build) BUILD=0 ;;
        *) PASS+=("$a") ;;
    esac
done

RUN_DIR="$ROOT/bench/results/$(date +%Y%m%d-%H%M%S)-frame"
mkdir -p "$RUN_DIR"

if [[ $BUILD == 1 ]]; then
    echo ">> building"
    if ! (cd "$ROOT" && scons platform=linux target=template_debug -j"$(nproc)" > "$RUN_DIR/build.log" 2>&1); then
        grep -E "error|Error" "$RUN_DIR/build.log" | head -30
        echo ">> BUILD FAILED (full log: $RUN_DIR/build.log)"
        exit 1
    fi
fi

echo ">> running godot (${PASS[*]:-})"
timeout "${FRAME_TIMEOUT:-300}" "$GODOT" --path "$ROOT/adapter/godot/project" --resolution 640x360 res://debug/frame_spike.tscn -- \
    "${PASS[@]}" --out="$RUN_DIR/frame.json" > "$RUN_DIR/godot.log" 2>&1
CODE=$?

# Same known engine exit abort as tools/bench.sh (thread_model=2), accepted once the run finished.
if [[ $CODE -eq 134 ]] && grep -q "BENCH: done" "$RUN_DIR/godot.log" \
    && grep -q "This function (finalize) can only be called from the render thread" "$RUN_DIR/godot.log"; then
    CODE=0
fi
# The scene quits with 1 when a check failed; the report says which.
[[ $CODE -eq 1 ]] && grep -q "BENCH: done" "$RUN_DIR/godot.log" && CODE=0
ERRORS=$(grep -E -A1 "^(ERROR|SCRIPT ERROR|USER ERROR)|SHADER COMPILATION FAILED" "$RUN_DIR/godot.log" \
    | grep -vE "finalize|rendering_device.cpp:7470|^--$" | grep -cE "^(ERROR|SCRIPT ERROR|USER ERROR)|SHADER COMPILATION FAILED")

OK=0
if [[ -f "$RUN_DIR/frame.json" ]]; then
    python3 - "$RUN_DIR/frame.json" <<'PY' && OK=1
import json, sys
r = json.load(open(sys.argv[1]))
keys = ["ok", "origins", "cases", "probe_sets", "arch_programs", "samples", "worldpos_mismatch", "anchor_mismatch", "pairs",
        "pair_cpu_mismatch", "pair_gpu_mismatch", "continuity_steps", "continuity_violations",
        "cross_origin_points", "cross_origin_mismatch", "cpu_checks_failed", "inexact_evaluations",
        "inexact_mismatch", "relative_points", "relative_mismatch", "probe_mismatch", "gpu_device_check", "error"]
for k in keys:
    if k in r:
        print(f"   {k:24} {r[k]}")
for e in r.get("examples", []):
    print("   example:", e)
sys.exit(0 if r.get("ok") else 1)
PY
fi

echo ">> log: $RUN_DIR/godot.log  (errors=$ERRORS exit=$CODE ok=$OK)"
[[ $ERRORS -gt 0 ]] && grep -E -A1 "^(ERROR|SCRIPT ERROR|USER ERROR)|SHADER COMPILATION FAILED" "$RUN_DIR/godot.log" \
    | grep -vE "finalize|rendering_device.cpp:7470" | head -20
[[ $CODE -ne 0 ]] && exit 2
[[ $ERRORS -gt 0 ]] && exit 3
[[ $OK -ne 1 ]] && exit 4
exit 0
