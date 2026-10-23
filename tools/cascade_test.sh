#!/usr/bin/env bash
# GPU cascade harness: build (unless --no-build), run res://debug/cascade_spike.tscn, check the result.
#
#   tools/cascade_test.sh [--no-build] [--lods=25] [--voxel=0.5] [--check=0|1] [--points=0|1] [--table=0|1] [--biomes=0|1] [--timing=0|1] [--runs=10]
#                         [--sets=8,4,4c]
#
# check: the cascade passes (cascade_glsl.hpp) vs Cascade::run, every channel of every level and the
# sdf_sample, chunks of every LOD near 0, +-2^60 and across 2^32 boundaries (G1), and the LOD relation on
# the GPU outputs (G3). points: point items (CascadePasses::create_points, the discoverer's corners and the site walk's
# readings) vs SDFCascade::ground_unit, ground_field and evaluate_cull, one mixed-LOD batch of 5,120 voxels per LOD (0-8,
# 12, 16, 24) near 0, +-2^60, across 2^32-voxel boundaries and lattice index 2^32, any lattice index (D1), and their GPU ms
# per batch of 1k-64k points at LOD 0 and 5 (D2). table: a cascade terrain with 600 scattered instances, the meshers' samples with the
# chunks' instance lists vs SDFCascade::sample_sdf + skirt (G9: the fold onto the cascade, bit for bit). biomes: the example climate's per-sample ground material (the biome pick at
# level 0) vs SDFCascade::ground_material, every LOD (G10, on 4c and every other climate set asked). timing: GPU ms of the passes for single-LOD batches of 128 chunks at LOD 0, 4, ...
# (G6), both channel sets (G7: 8 vs 4), each set against the plain 4 channels. Sets: 8, 4, 4c (LINEAR4 + the example
# climate at 5 cm, its fields carried and modulating the ground), 4f (its fields without multipliers), 4p (the plain climate: the
# example before shaped bands), 4s (the shaped test table: every shaped-band operation), 4d (fieldless_test.json: a
# modulation with no field), 4r (region_test.json: two landforms behind regions, 11 field groups), 4w (world.json: the eight landforms, 17 groups;
# run it with --lods=25); default 8,4,4c.
# GPU clocks are sampled during the run (nvidia-smi).
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
pts = {k: v for k, v in r.get("points", {}).items() if k != "timing"}
if pts:
    lines += ["| D1 point items (mixed LODs) | points | per LOD | differing | batches | scratch MB | ok |", "|---|---|---|---|---|---|---|"]
    for s, d in pts.items():
        lines.append(f"| {s} ch | {d['points']:,} | {d['points_per_lod']:,} | {d['differing']} | {d['batches']} | {d['scratch_mb']:.1f} | {d['ok']} |")
    lines.append("")
pt = r.get("points", {}).get("timing", {})
if pt.get("rows"):
    lines += [f"D2 point items, GPU ms per single-LOD batch (median of {pt.get('runs')} runs):", "",
              "| set | LOD | points | GPU ms | us / point |", "|---|---|---|---|---|"]
    for x in pt["rows"]:
        lines.append(f"| {x['set']} ch | {x['lod']} | {x['points']:,} | {x['gpu_ms_median']:.3f} | {x['us_per_point']:.3f} |")
    lines.append("")
tab = r.get("table", {})
if tab:
    lines += ["| G9 instance table | records | chunks | samples | folded | differing | ok |", "|---|---|---|---|---|---|---|",
              f"| 4 ch, LODs 0-6 | {tab['records']} | {tab['chunks']} | {tab['samples']:,} | {tab['samples_folded']:,} | {tab['samples_differing']} | {tab['ok']} |", ""]
bio = r.get("biomes", {})
if bio:
    lines += ["| G10 biome materials | chunks | samples | differing | per biome | ok |", "|---|---|---|---|---|---|"]
    for s, d in bio.items():
        if "error" in d:
            lines.append(f"| {s}, every LOD | - | - | - | {d['error']} | {d['ok']} |")
            continue
        per = ", ".join(f"{k} {v:,}" for k, v in d.get("samples_per_biome", {}).items())
        lines.append(f"| {s}, every LOD | {d['chunks']} | {d['samples']:,} | {d['samples_differing']} | {per} | {d['ok']} |")
    lines.append("")
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
    # Every other set against the plain 4 channels (without them, against the last set).
    base = "4" if "4" in sets else sets[-1]
    ratios = [s for s in sets if s != base]
    head += "".join(f" {s} / {base} |" for s in ratios)
    lines += [head, "|" + "---|" * (head.count("|") - 1)]
    for lod in lods:
        cells = f"| {lod} | {by[(sets[0], lod)]['levels']} |"
        vals = []
        for s in sets:
            x = by[(s, lod)]
            g = x["gpu_ms_median"] if x["gpu_runs"] else x["wall_ms_median"]
            vals.append(g)
            cells += f" {g:.3f} ({x['wall_ms_median']:.2f}) | {1000 * g / n:.1f} |"
        for s in ratios:
            cells += f" {vals[sets.index(s)] / vals[sets.index(base)]:.3f} |"
        lines.append(cells)
    lines.append("")
    for s in sets:
        g0 = by[(s, lods[0])]
        gl = by[(s, lods[-1])]
        a = g0["gpu_ms_median"] if g0["gpu_runs"] else g0["wall_ms_median"]
        b = gl["gpu_ms_median"] if gl["gpu_runs"] else gl["wall_ms_median"]
        lines.append(f"G6 ({s} ch): LOD {lods[0]} / LOD {lods[-1]} = {a / b:.3f} (pass <= 1.15)")
    def total(s):
        return sum((by[(s, l)]["gpu_ms_median"] if by[(s, l)]["gpu_runs"] else by[(s, l)]["wall_ms_median"]) for l in lods)
    if "8" in sets and "4" in sets:
        lines.append(f"G7: 8 ch / 4 ch over all LODs = {total('8') / total('4'):.3f} (keep 8 unless > 1.15)")
    for s in ratios:
        if s not in ("8", "4") and "4" in sets:
            lines.append(f"{s} ch / 4 ch over all LODs = {total(s) / total('4'):.3f} (the climate's cost)")
    lines.append("")
    lines.append("Per level (last run, ms, j = top ... 0):")
    for x in rows:
        lines.append(f"- {x['set']} ch LOD {x['lod']}: " + ", ".join(f"{v:.3f}" for v in x["gpu_ms_per_level_last"]))
text = "\n".join(lines)
print(text)
open(sys.argv[3], "w").write(text + "\n")
for s, d in list(check.items()) + list(pts.items()) + [("biomes " + k, v) for k, v in bio.items()]:
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
