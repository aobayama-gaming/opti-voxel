#!/usr/bin/env bash
# Weather report (WeatherReport, docs/weather.md "Report"): a world file's weather run on the CPU alone and mapped at game
# minutes after its spin-up. Pure CPU, headless (no GPU, no window).
#
#   tools/weather_report.sh [--no-build] [--world=PATH] [--minutes=0,1,10] [--width=512]
#
# Outputs land in bench/results/<timestamp>-weather/: report.md, cover_<min>.png, wind_<min>.png, godot.log.
# Exit code: 0 ok, 1 build failed, 2 godot crashed/timed out, 3 the world has no weather or was refused.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GODOT="${GODOT:-$HOME/.local/bin/godot}"
BUILD=1
ARGS=()
for a in "$@"; do
    case "$a" in
        --no-build) BUILD=0 ;;
        --world=*|--minutes=*|--width=*) ARGS+=("$a") ;;
        *) echo "usage: $0 [--no-build] [--world=PATH] [--minutes=0,1,10] [--width=512]"; exit 1 ;;
    esac
done

RUN_DIR="$ROOT/bench/results/$(date +%Y%m%d-%H%M%S)-weather"
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
timeout "${WEATHER_TIMEOUT:-900}" "$GODOT" --headless --path "$ROOT/adapter/godot/project" res://debug/weather_report.tscn -- \
    --out="$RUN_DIR" "${ARGS[@]}" > "$RUN_DIR/godot.log" 2>&1
CODE=$?
if grep -q '^WEATHER: .*"error":' "$RUN_DIR/godot.log"; then
    grep '^WEATHER: ' "$RUN_DIR/godot.log" | cut -c1-400
    echo ">> no weather (log: $RUN_DIR/godot.log)"
    exit 3
fi
if [[ ! -f "$RUN_DIR/report.md" ]] || ! grep -q '^WEATHER: .*"ok":true' "$RUN_DIR/godot.log"; then
    tail -20 "$RUN_DIR/godot.log"
    echo ">> no report: godot crashed or timed out (exit $CODE, log: $RUN_DIR/godot.log)"
    exit 2
fi
cat "$RUN_DIR/report.md"
echo ">> $RUN_DIR"
