#!/usr/bin/env bash
# GPU cascade harness: build (unless --no-build), run res://debug/cascade_spike.tscn, check the result.
#
#   tools/cascade_test.sh [--no-build] [--lods=25] [--voxel=0.5] [--check=0|1] [--points=0|1] [--table=0|1] [--timing=0|1] [--runs=10]
#                         [--sets=8,4]
#
# check: the cascade passes (cascade_glsl.hpp) vs Cascade::run, every channel of every level and the
# sdf_sample, chunks of every LOD near 0, +-2^60 and across 2^32 boundaries (G1), and the LOD relation on
# the GPU outputs (G3). points: the discoverer's point items (CascadePasses::create_points) vs Cascade::point +
# SDFCascade::sdf, 5,120 corners per LOD near 0, +-2^60, across 2^32-voxel boundaries and lattice index 2^32
# (D1 of the GPU discoverer). table: a cascade terrain with 600 scattered instances, the meshers' samples with the
# chunks' instance lists vs SDFCascade::sample (G9: the fold onto the cascade, bit for bit). timing: GPU ms of the passes for single-LOD batches of 128 chunks at LOD 0, 4, ...
# (G6), both channel sets (G7). GPU clocks are sampled during the run (nvidia-smi).
# Outputs land in bench/results/<timestamp>-cascade/ : cascade.json, cascade.md, godot.log, gpu.csv.
# Exit code: 0 ok, 1 build failed, 2 godot crashed/timed out, 3 errors in log, 4 a check failed.
# Needs a GPU and an unlocked session (it opens a window).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GODOT="${GODOT:-$HOME/.local/bin/godot}"
BUILD=1
PASS=()
for a in "$@"; do
    case "$a" in
        --no-build) BUILD=0 ;;
        *) PASS+=("$a") ;;
    esac
done

RUN_DIR="$ROOT/bench/results/$(date +%Y%m%d-%H%M%S)-cascade"
mkdir -p "$RUN_DIR"

if [[ $BUILD == 1 ]]; then
    echo ">> building"
    if ! (cd "$ROOT" && scons platform=linux target=template_debug -j"$(nproc)" > "$RUN_DIR/build.log" 2>&1); then
        grep -E "error|Error" "$RUN_DIR/build.log" | head -30
        echo ">> BUILD FAILED (full log: $RUN_DIR/build.log)"
        exit 1
    fi
fi

TELEMETRY_PID=""
if command -v nvidia-smi > /dev/null; then
    nvidia-smi --query-gpu=timestamp,clocks.gr,temperature.gpu,power.draw --format=csv,noheader,nounits -lms 500 > "$RUN_DIR/gpu.csv" 2> /dev/null &
    TELEMETRY_PID=$!
fi

echo ">> running godot (${PASS[*]:-})"
timeout "${CASCADE_TIMEOUT:-900}" "$GODOT" --path "$ROOT/project" --resolution 640x360 res://debug/cascade_spike.tscn -- \
    "${PASS[@]}" --out="$RUN_DIR/cascade.json" > "$RUN_DIR/godot.log" 2>&1
CODE=$?
[[ -n $TELEMETRY_PID ]] && kill "$TELEMETRY_PID" 2> /dev/null

# Same known engine exit abort as tools/bench.sh (thread_model=2), accepted once the run finished.
if [[ $CODE -eq 134 ]] && grep -q "BENCH: done" "$RUN_DIR/godot.log" \
    && grep -q "This function (finalize) can only be called from the render thread" "$RUN_DIR/godot.log"; then
    CODE=0
fi
# The scene quits with 1 when a check failed; the report says which.
[[ $CODE -eq 1 ]] && grep -q "BENCH: done" "$RUN_DIR/godot.log" && CODE=0
ERRORS=$(grep -E -A1 "^(ERROR|SCRIPT ERROR|USER ERROR)|SHADER COMPILATION FAILED" "$RUN_DIR/godot.log" \
    | grep -vE "finalize|rendering_device.cpp:7470|^--$" | grep -cE "^(ERROR|SCRIPT ERROR|USER ERROR)|SHADER COMPILATION FAILED")

OK=0
if [[ -f "$RUN_DIR/cascade.json" ]]; then
    python3 - "$RUN_DIR/cascade.json" "$RUN_DIR/gpu.csv" "$RUN_DIR/cascade.md" <<'PY' && OK=1
import json, os, statistics, sys
r = json.load(open(sys.argv[1]))
lines = []
clocks = []
if os.path.exists(sys.argv[2]):
    for row in open(sys.argv[2]):
        parts = [p.strip() for p in row.split(",")]
        try:
            clocks.append((float(parts[1]), float(parts[2])))
        except (IndexError, ValueError):
            pass
if clocks:
    busy = [c for c, _ in clocks if c > 400] or [c for c, _ in clocks]
    lines.append(f"GPU clock median {statistics.median(busy):.0f} MHz (samples > 400 MHz: {len(busy)}/{len(clocks)}), "
                 f"temperature max {max(t for _, t in clocks):.0f} C")
    lines.append("")
check = r.get("check", {})
if check:
    lines += ["| set | chunks | min chunks / LOD | levels | channel values | differing | samples | differing | G3 points | differing | ok |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, d in check.items():
        lines.append(f"| {s} ch | {d['chunks']} | {d['min_chunks_per_lod']} | {d['levels']} | {d['values']:,} | {d['differing']} | "
                     f"{d['samples']:,} | {d['samples_differing']} | {d['relation_points']:,} | {d['relation_differing']} | {d['ok']} |")
    lines.append("")
pts = r.get("points", {})
if pts:
    lines += ["| set | point items | per LOD | differing | scratch MB | ok |", "|---|---|---|---|---|---|"]
    for s, d in pts.items():
        lines.append(f"| {s} ch | {d['points']:,} | {d['points_per_lod']:,} | {d['differing']} | {d['scratch_mb']:.2f} | {d['ok']} |")
    lines.append("")
tab = r.get("table", {})
if tab:
    lines += ["| G9 instance table | records | chunks | samples | folded | differing | ok |", "|---|---|---|---|---|---|---|",
              f"| 4 ch, LODs 0-6 | {tab['records']} | {tab['chunks']} | {tab['samples']:,} | {tab['samples_folded']:,} | {tab['samples_differing']} | {tab['ok']} |", ""]
t = r.get("timing", {})
rows = t.get("rows", [])
if rows:
    n = t.get("chunks_per_batch", 128)
    by = {(x["set"], x["lod"]): x for x in rows}
    sets = sorted({x["set"] for x in rows}, reverse=True)
    lods = sorted({x["lod"] for x in rows})
    lines.append(f"GPU time of the cascade passes, single-LOD batches of {n} chunks, median of {t.get('runs')} runs "
                 f"(GPU timestamps; wall clock of submit + sync in brackets). Scratch: " +
                 ", ".join(f"{s} ch {t.get('scratch_mb_' + s, 0):.1f} MB" for s in sets) + f" for {n} chunks.")
    lines.append("")
    head = "| LOD | levels |" + "".join(f" {s} ch ms / batch | {s} ch us / chunk |" for s in sets)
    if len(sets) == 2:
        head += f" {sets[0]} / {sets[1]} |"
    lines += [head, "|" + "---|" * (head.count("|") - 1)]
    for lod in lods:
        cells = f"| {lod} | {by[(sets[0], lod)]['levels']} |"
        vals = []
        for s in sets:
            x = by[(s, lod)]
            g = x["gpu_ms_median"] if x["gpu_runs"] else x["wall_ms_median"]
            vals.append(g)
            cells += f" {g:.3f} ({x['wall_ms_median']:.2f}) | {1000 * g / n:.1f} |"
        if len(sets) == 2:
            cells += f" {vals[0] / vals[1]:.3f} |"
        lines.append(cells)
    lines.append("")
    for s in sets:
        g0 = by[(s, lods[0])]
        gl = by[(s, lods[-1])]
        a = g0["gpu_ms_median"] if g0["gpu_runs"] else g0["wall_ms_median"]
        b = gl["gpu_ms_median"] if gl["gpu_runs"] else gl["wall_ms_median"]
        lines.append(f"G6 ({s} ch): LOD {lods[0]} / LOD {lods[-1]} = {a / b:.3f} (pass <= 1.15)")
    if len(sets) == 2:
        tot = [sum((by[(s, l)]["gpu_ms_median"] if by[(s, l)]["gpu_runs"] else by[(s, l)]["wall_ms_median"]) for l in lods) for s in sets]
        lines.append(f"G7: {sets[0]} ch / {sets[1]} ch over all LODs = {tot[0] / tot[1]:.3f} (keep 8 unless > 1.15)")
    lines.append("")
    lines.append("Per level (last run, ms, j = top ... 0):")
    for x in rows:
        lines.append(f"- {x['set']} ch LOD {x['lod']}: " + ", ".join(f"{v:.3f}" for v in x["gpu_ms_per_level_last"]))
text = "\n".join(lines)
print(text)
open(sys.argv[3], "w").write(text + "\n")
for s, d in list(check.items()) + list(pts.items()):
    for e in d.get("examples", [])[:8]:
        print(f"   example ({s} ch):", e)
if "error" in r:
    print("   error:", r["error"])
sys.exit(0 if r.get("ok") else 1)
PY
fi

echo ">> log: $RUN_DIR/godot.log  (errors=$ERRORS exit=$CODE ok=$OK)"
[[ $ERRORS -gt 0 ]] && grep -E -A1 "^(ERROR|SCRIPT ERROR|USER ERROR)|SHADER COMPILATION FAILED" "$RUN_DIR/godot.log" \
    | grep -vE "finalize|rendering_device.cpp:7470" | head -20
[[ $CODE -ne 0 ]] && exit 2
[[ $ERRORS -gt 0 ]] && exit 3
[[ $OK -ne 1 ]] && exit 4
exit 0
