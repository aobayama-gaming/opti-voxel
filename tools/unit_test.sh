#!/usr/bin/env bash
# C++ unit tests (src/tests/, doctest) run inside headless Godot: no GPU, a few seconds.
#
#   tools/unit_test.sh [--no-build] [filter]
#
# filter: doctest wildcard on test case names, e.g. "ChunkMath*" or "*parent*".
# Outputs land in bench/results/<timestamp>-unit/godot.log.
# Exit code: 0 all passed, 1 build failed, 2 godot crashed/timed out, 4 a test failed.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GODOT="${GODOT:-$HOME/.local/bin/godot}"
BUILD=1
FILTER=""
for a in "$@"; do
    case "$a" in
        --no-build) BUILD=0 ;;
        -*) echo "usage: $0 [--no-build] [filter]"; exit 1 ;;
        *) FILTER="$a" ;;
    esac
done

RUN_DIR="$ROOT/bench/results/$(date +%Y%m%d-%H%M%S)-unit"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/godot.log"

if [[ $BUILD == 1 ]]; then
    echo ">> building"
    if ! (cd "$ROOT" && scons platform=linux target=template_debug -j"$(nproc)" > "$RUN_DIR/build.log" 2>&1); then
        grep -E "error|Error" "$RUN_DIR/build.log" | head -30
        echo ">> BUILD FAILED (full log: $RUN_DIR/build.log)"
        exit 1
    fi
fi

timeout "${UNIT_TIMEOUT:-120}" "$GODOT" --headless --path "$ROOT/adapter/godot/project" res://debug/unit_tests.tscn -- \
    --filter="$FILTER" > "$LOG" 2>&1
CODE=$?

# doctest's report: failures with their expressions, then the totals.
sed -n '/^\[doctest\] doctest version/,/^\[doctest\] Status:/p' "$LOG" | grep -vE '^\[doctest\] (doctest version|run with)'
REPORT="$(grep -m1 '^UNIT: ' "$LOG" | cut -c7-)"
if [[ -z $REPORT ]]; then
    tail -20 "$LOG"
    echo ">> no UNIT report: godot crashed or timed out (exit $CODE, log: $LOG)"
    exit 2
fi
echo ">> $REPORT  (log: $LOG)"
[[ $REPORT == *'"ok":true'* ]] || exit 4
exit 0
