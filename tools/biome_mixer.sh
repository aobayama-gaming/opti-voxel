#!/usr/bin/env bash
# Biome mixer (adapter/godot/project/debug/biome_mixer/): build one biome from the ground's octaves in a node graph,
# raymarched live beside it. GDScript and one shader: no build needed, the extension isn't used.
#
#   tools/biome_mixer.sh [--open=PATH] [--import=BIOME] [--set=ID.PARAM=V] [--shot=PATH]   a window
#   tools/biome_mixer.sh --check                                                          headless self-test
#
# --check prints the MIXER: line and exits 0 ok, 2 godot crashed / timed out, 4 not ok. Biomes save to
# adapter/godot/project/climate/biomes/<name>.biome.json; the last session autosaves (Godot's user:// folder).
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
    # The scene quits 0 when its report is ok.
    [[ $CODE == 0 ]] || exit 4
    exit 0
fi

exec "$GODOT" --path "$ROOT/adapter/godot/project" --render-thread safe "$SCENE" -- "$@"
