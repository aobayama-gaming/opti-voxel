#!/usr/bin/env bash
# Builds the C++ reference (oracle) from the unchanged src/ files plus the godot shim.
# Produces <out>/oracle_f32 (real float build) and <out>/oracle_f64 (every float promoted to double).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
out="${1:-$here/build}"

mkdir -p "$out/src"
for f in CPU_mesher.cpp CPU_mesher.hpp chunk_math.cpp chunk_math.hpp noise.cpp noise.hpp sdf_base.hpp sdf_dummy.hpp; do
	cp "$repo/src/$f" "$out/src/"
done
# Stubs must sit next to the copied sources: quoted includes resolve there first.
cp "$here"/stubs/*.hpp "$out/src/"

srcs=("$here/oracle.cpp" "$out/src/CPU_mesher.cpp" "$out/src/chunk_math.cpp" "$out/src/noise.cpp")
flags=(-std=c++17 -O2 -w -I"$here/shim" -I"$out/src" -include "$here/shim/godot_shim.hpp")

g++ "${flags[@]}" "${srcs[@]}" -o "$out/oracle_f32"
g++ "${flags[@]}" -DOPTI_DOUBLE "${srcs[@]}" -o "$out/oracle_f64"
echo "built $out/oracle_f32 $out/oracle_f64"
