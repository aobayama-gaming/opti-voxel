#!/usr/bin/env bash
# Biome mixer (adapter/godot/project/debug/biome_mixer/): edit a world's biomes as node graphs, drawn beside them by the
# engine (ClimateReportNode.preview: the game's climate and cascade, sampled on the CPU). Needs the extension built.
#
#   tools/biome_mixer.sh [--biome=NAME] [--world] [--set=ID.KEY=V] [--shot=PATH]   a window
#   tools/biome_mixer.sh --check                                                  headless self-test
#
# --check prints the MIXER: line and exits 0 ok, 2 godot crashed / timed out, 4 not ok (a script error too: the check
# builds every node's UI). Worlds save to adapter/godot/project/climate/worlds/; the last session autosaves (Godot's
# user:// folder).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GODOT="${GODOT:-$HOME/.local/bin/godot}"
SCENE=res://debug/biome_mixer/biome_mixer.tscn

if [[ " $* " == *" --check "* ]]; then
    OUT="$(timeout 120 "$GODOT" --headless --path "$ROOT/adapter/godot/project" "$SCENE" -- --check 2>&1)"
    CODE=$?
    LINE="$(grep '^MIXER: ' <<< "$OUT")"
    if [[ -z "$LINE" ]]; then
        echo "$OUT" | tail -30
        echo ">> no MIXER report (crash or timeout)"
        exit 2
    fi
    echo "$LINE"
    grep -E "SCRIPT ERROR|ERROR:" <<< "$OUT" | head -20
    # The scene quits 0 when its report is ok; a script error fails it whatever the report says.
    [[ $CODE == 0 ]] && ! grep -q "SCRIPT ERROR" <<< "$OUT" || exit 4
    exit 0
fi

exec "$GODOT" --path "$ROOT/adapter/godot/project" --render-thread safe "$SCENE" -- "$@"
