#!/usr/bin/env bash
# The core's engine boundary, enforced. The core files (src/core_files.txt, also the Dawn adapter's source list) (the engine-free vocabulary the
# liboptivoxel plan builds on) must compile on their own with no godot include path: the same core
# compiles into other engines and into a web build. Each one is compiled alone (`-fsyntax-only`, no
# godot-cpp `-I`), so a godot header reached through an include, a godot macro or an unqualified godot
# name fails here, not only a spelled `godot::`.
#
# The pending files have had their own lines converted (no godot include, no `godot::`) but still reach
# godot through the header named next to them: they are grep-checked so the conversion holds, and move
# to core_files in the change that cuts that chain.
#
#   tools/check_core_purity.sh
#
# Exit code: 0 the core is clean, 1 a core file needs godot. Grow core_files, never shrink it.

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CXX="${CXX:-g++}"

mapfile -t core_files < <(grep -v '^#' src/core_files.txt | grep .)

# file -> the header it still reaches godot through (empty: every converted file is in core_files)
pending_files=(
)

FAILED=0
grep_clean() {
    local f="$1"
    if [ ! -f "$f" ]; then
        echo "core file missing: $f (moved or deleted? update the list in the same commit)"
        return 1
    fi
    local hits
    hits=$(grep -nE '#include[[:space:]]*[<"]godot|\bgodot::[A-Za-z_]|using[[:space:]]+namespace[[:space:]]+godot' "$f" 2>/dev/null)
    if [ -n "$hits" ]; then
        echo "core file $f mentions godot code:"
        echo "$hits"
        return 1
    fi
}

for f in "${core_files[@]}"; do
    grep_clean "$f" || { FAILED=1; continue; }
    if ! out=$("$CXX" -std=c++17 -fsyntax-only -Isrc -x c++ "$f" 2>&1); then
        echo "core file $f does not compile without godot:"
        echo "$out" | grep -m3 -E 'error|In file included from'
        FAILED=1
    fi
done

for entry in "${pending_files[@]}"; do
    grep_clean "${entry%% *}" || FAILED=1
done

if [ "$FAILED" -ne 0 ]; then
    echo "core purity check FAILED"
    exit 1
fi
echo "core purity: ${#core_files[@]} files compile without godot, ${#pending_files[@]} pending (own lines clean)"
