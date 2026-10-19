#!/usr/bin/env bash
# Validates the Luau port against the real C++ code (built from src/ by cpp_oracle/build.sh).
#   tools/validate.sh [luau_binary]
# Needs g++, python3 and the Luau CLI (https://github.com/luau-lang/luau/releases, `luau` on PATH
# or passed as $1). Exits non-zero on any mismatch.
#
# f64 oracle: C++ with float promoted to double -> must match: same triangles (winding included)
#             and same vertices (to 1e-4, i.e. double rounding only).
# f32 oracle: the real float build -> same vertex and triangle counts. Positions are reported but not
#             gated: CPUMesher's edge normals use a 1e-4 central difference at |coords| ~ 300, which is
#             near float32 resolution, and the QEF amplifies that noise at coarse LODs (up to ~1 unit
#             at LOD 4). The Luau port computes in doubles, so it is the less noisy of the two.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
luau="${1:-luau}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

"$here/cpp_oracle/build.sh" "$work/oracle" >/dev/null

echo "== SDF (20000 points)"
"$luau" --codegen "$here/luau/sdf_points.luau" >"$work/sdf_luau.txt"
fail=0
for b in f64 f32; do
	cut -d' ' -f1-3 "$work/sdf_luau.txt" | "$work/oracle/oracle_$b" sdf >"$work/sdf_$b.txt"
	res="$(python3 "$here/compare.py" sdf "$work/sdf_luau.txt" "$work/sdf_$b.txt")"
	echo "$b $res"
	python3 -c "import json,sys; r=json.loads(sys.argv[1]); sys.exit(0 if r['sign_flips']==0 and r['max']<(1e-5 if sys.argv[2]=='f64' else 1e-3) else 1)" "$res" "$b" || fail=1
done

echo "== Edit focus (Smooth Terrain occupancy unchanged)"
"$luau" --codegen "$here/luau/edit_focus.luau" || fail=1

# Surface, caves, deep, empty, negative coordinates, LOD 1..5 (LOD >= 5 takes the mass-point path).
chunks=(
	"1 43 1" "1 41 1" "1 39 1" "1 37 1" "-3 43 -5" "-21 41 13" "9 45 -7" "1 47 1"
	"2 42 2" "-6 42 10" "4 44 4" "-12 36 20" "8 40 8" "16 48 16" "32 32 32"
)
echo "== Meshes"
for c in "${chunks[@]}"; do
	read -r cx cy cz <<<"$c"
	"$luau" --codegen "$here/luau/mesh_dump.luau" -a "$cx" "$cy" "$cz" >"$work/m_luau.txt"
	for b in f64 f32; do
		"$work/oracle/oracle_$b" mesh "$cx" "$cy" "$cz" >"$work/m_$b.txt"
		res="$(python3 "$here/compare.py" mesh "$work/m_luau.txt" "$work/m_$b.txt" 1e-4)"
		ok="$(python3 -c "
import json,sys; r=json.loads(sys.argv[1])
good = r['luau_vertices'] == r['oracle_vertices'] and r['luau_triangles'] == r['oracle_triangles']
if sys.argv[2] == 'f64':
    good = good and (r['one_to_one'] or r['luau_vertices'] == 0)
    good = good and r['triangles_matched'] == r['oracle_triangles'] and r['vertices_over_tol'] == 0
print('ok' if good else 'FAIL')" "$res" "$b")"
		printf '%-12s %s %-4s %s\n' "($cx,$cy,$cz)" "$b" "$ok" "$res"
		[ "$ok" = ok ] || fail=1
	done
done

if [ "$fail" -ne 0 ]; then
	echo "VALIDATION FAILED"
	exit 1
fi
echo "VALIDATION OK"
