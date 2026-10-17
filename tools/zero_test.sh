#!/usr/bin/env bash
# Exact-zero SDF test: the mesher must not lose or flip a quad where a sample is exactly 0.0.
#
#   tools/zero_test.sh [-- extra bench args]
#
# Two direct load runs with the debug zero pillar (SDFDummy): its flat top lies exactly on a lattice
# row (SDF exactly 0.0 there at LODs 0-6), the second run moves the top up 1 cm (no exact zero, same
# signs everywhere). Both must mesh the same number of triangles, and look the same from above (a
# flipped top is culled and shows as a hole). Also runs the page audit and the direct/legacy validator.
# Report: bench/results/<timestamp>-zero.md. Exit code: 0 pass, 1 fail. No build, no cooldown.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXTRA=()
[[ "${1:-}" == "--" ]] && { shift; EXTRA=("$@"); }

# Above the pillar (top at 416 m, x -52.75, z -110), looking down at it.
CAM=(--cam_pos=-52.75,430,-96 --pitch=-45)
declare -A DIRS
FAILED=0
for mode in 1 2; do
    "$ROOT/tools/bench.sh" --no-build --tag="zero-$mode" --mode=direct --scenario=load --steady=10 \
        --zero_pillar=$mode "${CAM[@]}" --validate=64 --audit=1 --shot=loaded:loaded.png "${EXTRA[@]}" > /dev/null 2>&1
    code=$?
    DIRS[$mode]=$(cat "$ROOT/bench/results/LAST")
    if [[ $code -ne 0 ]]; then
        echo ">> zero_pillar=$mode: bench failed (exit $code), run dir: ${DIRS[$mode]}"
        FAILED=1
    fi
done

OUT="$ROOT/bench/results/$(date +%Y%m%d-%H%M%S)-zero.md"
python3 - "${DIRS[1]}" "${DIRS[2]}" "$OUT" "$FAILED" <<'PY'
import json, sys
import numpy as np
from PIL import Image
exact, offset, out, failed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

def tri(d):
    s = json.load(open(f"{d}/stats.json"))
    return s["final_terrain"]["chunks"]["live_triangles"]

a = np.asarray(Image.open(f"{exact}/loaded.png").convert("RGB")).astype(int)
b = np.asarray(Image.open(f"{offset}/loaded.png").convert("RGB")).astype(int)
diff = np.abs(a - b).max(axis=2)
share = float((diff > 3).mean())
t_exact, t_offset = tri(exact), tri(offset)
# The 1 cm shift moves the top's vertices slightly: allow shading noise, not a missing/culled face
# (the top covers ~10% of the frame).
ok = not failed and t_exact == t_offset and share < 0.003
lines = [
    f"# Zero test: {'PASS' if ok else 'FAIL'}", "",
    "| run | live triangles |", "|---|---|",
    f"| exact top | {t_exact} |", f"| top +1 cm | {t_offset} |", "",
    f"Screenshot: {share * 100:.3f}% of pixels differ by more than 3 (limit 0.3%).", "",
    f"Runs: `{exact}`, `{offset}`",
]
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines[:8]))
print(f">> report: {out}")
sys.exit(0 if ok else 1)
PY
