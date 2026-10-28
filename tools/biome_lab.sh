#!/usr/bin/env bash
# Biome lab: a local web page to edit the climate tables, preview them (climate_report.sh) and fly them (the player).
#
#   tools/biome_lab.sh [--no-build] [--port=8765] [--no-open]
#
# Builds the extension once (the previews then run --no-build), then serves http://127.0.0.1:PORT/ until Ctrl+C.
# Details: docs/climate.md, "Biome lab".
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LAB="$ROOT/bench/results/biome-lab"
BUILD=1
ARGS=()
for a in "$@"; do
    case "$a" in
        --no-build) BUILD=0 ;;
        --port=*|--no-open) ARGS+=("$a") ;;
        *) echo "usage: $0 [--no-build] [--port=8765] [--no-open]"; exit 1 ;;
    esac
done

if [[ $BUILD == 1 ]]; then
    echo ">> building"
    mkdir -p "$LAB"
    if ! (cd "$ROOT" && scons platform=linux target=template_debug -j"$(nproc)" > "$LAB/build.log" 2>&1); then
        grep -E "error|Error" "$LAB/build.log" | head -30
        echo ">> BUILD FAILED (full log: $LAB/build.log)"
        exit 1
    fi
fi
exec python3 "$ROOT/tools/biome_lab/server.py" "${ARGS[@]}"
