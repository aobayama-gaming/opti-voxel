#!/usr/bin/env bash
# The core's engine boundary, enforced. The core files below (the engine-free vocabulary the
# liboptivoxel plan builds on) must not include godot or name its types: the same core compiles into
# other engines and into a web build. Comments may explain history in words ("godot's Vector3i"), so
# the gate greps includes and code references (`godot::x`), not prose.
#
#   tools/check_core_purity.sh
#
# Exit code: 0 the core is clean, 1 a core file mentions godot code. Add a file to the list only in
# the commit that de-Godots it; grow the list, never shrink it.

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

core_files=(
    src/ov/types.hpp
    src/ov/format.hpp
    src/ov/containers.hpp
    src/ov/murmur3.hpp
    src/ov/on_error.hpp
    src/ov/bytes.hpp
    src/world_coord.hpp
    src/chunk_math.hpp
    src/chunk_math.cpp
    src/page_format.hpp
    src/lod_structures.hpp
    src/chunk_memory.hpp
    src/chunk_memory.cpp
    src/instance_table.hpp
    src/instance_table.cpp
    src/cascade.hpp
    src/cascade.cpp
    src/archetype_ir.hpp
    src/archetype_ir.cpp
    src/gpu_cost_model.hpp
)

FAILED=0
for f in "${core_files[@]}"; do
    if [ ! -f "$f" ]; then
        echo "core file missing: $f (moved or deleted? update the list in the same commit)"
        FAILED=1
        continue
    fi
    HITS=$(grep -nE '#include[[:space:]]*<godot|\bgodot::[A-Za-z_]' "$f" 2>/dev/null)
    if [ -n "$HITS" ]; then
        echo "core file $f mentions godot code:"
        echo "$HITS"
        FAILED=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "core purity check FAILED"
    exit 1
fi
echo "core purity: ${#core_files[@]} files clean"
