#!/usr/bin/env bash
# Build the extension, run the scripted bench scene, check the log for problems, print a summary.
#
#   tools/bench.sh [--no-build] [--tag=name] [bench.gd args...]
#   e.g. tools/bench.sh --mode=legacy --scenario=load --shot=loaded:loaded.png
#
# Relative --shot paths are written into the run directory. Outputs land in
# bench/results/<timestamp>-<tag>/ : stats.json, godot.log, screenshots.
# Exit code: 0 ok, 1 build failed, 2 godot crashed/timed out, 3 errors found in log,
# 4 a requested --validate / --audit reported ok=false.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GODOT="${GODOT:-$HOME/.local/bin/godot}"
BUILD=1
TAG=""
PASS=()
for a in "$@"; do
    case "$a" in
        --no-build) BUILD=0 ;;
        --tag=*) TAG="${a#--tag=}" ;;
        *) PASS+=("$a") ;;
    esac
done

MODE="legacy"
for a in "${PASS[@]}"; do [[ "$a" == --mode=* ]] && MODE="${a#--mode=}"; done
RUN_DIR="$ROOT/bench/results/$(date +%Y%m%d-%H%M%S)-${TAG:-$MODE}"
mkdir -p "$RUN_DIR"

# Make relative screenshot paths land in the run dir.
ARGS=()
for a in "${PASS[@]}"; do
    if [[ "$a" == --shot=* ]]; then
        spec="${a#--shot=}"; when="${spec%%:*}"; path="${spec#*:}"
        [[ "$path" != /* ]] && path="$RUN_DIR/$path"
        ARGS+=("--shot=$when:$path")
    else
        ARGS+=("$a")
    fi
done
ARGS+=("--out=$RUN_DIR/stats.json")

if [[ $BUILD == 1 ]]; then
    echo ">> building"
    if ! (cd "$ROOT" && scons platform=linux target=template_debug -j"$(nproc)" > "$RUN_DIR/build.log" 2>&1); then
        grep -E "error|Error" "$RUN_DIR/build.log" | head -30
        echo ">> BUILD FAILED (full log: $RUN_DIR/build.log)"
        exit 1
    fi
fi

# Laptop GPUs throttle hard (power cap / thermals). Wait for a cool GPU so runs start from the same
# state, and record clocks during the run so throttled runs are visible in the results.
HAVE_SMI=0
command -v nvidia-smi > /dev/null && HAVE_SMI=1
if [[ $HAVE_SMI == 1 ]]; then
    for i in $(seq 1 "${BENCH_COOLDOWN_MAX_S:-120}"); do
        T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
        [[ "$T" -le "${BENCH_COOL_C:-65}" ]] && break
        [[ $i == 1 ]] && echo ">> cooling down (GPU at ${T}C)"
        sleep 1
    done
    nvidia-smi --query-gpu=clocks.gr,temperature.gpu,power.draw,clocks_throttle_reasons.active \
        --format=csv,noheader,nounits -lms 500 > "$RUN_DIR/gpu_telemetry.csv" 2>/dev/null &
    SMI_PID=$!
fi
echo "$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unknown)" > "$RUN_DIR/platform_profile.txt"

echo ">> running godot (${ARGS[*]})"
timeout "${BENCH_TIMEOUT:-600}" "$GODOT" --path "$ROOT/project" --resolution 1280x720 \
    res://debug/bench.tscn -- "${ARGS[@]}" > "$RUN_DIR/godot.log" 2>&1
CODE=$?
[[ $HAVE_SMI == 1 ]] && kill "$SMI_PID" 2>/dev/null

# Fold telemetry into stats.json (clock median/min, max temperature, share of throttled samples).
python3 - "$RUN_DIR" <<'PY'
import json, os, statistics, sys
run = sys.argv[1]
p, t = os.path.join(run, "stats.json"), os.path.join(run, "gpu_telemetry.csv")
if not os.path.exists(p):
    sys.exit(0)
d = json.load(open(p))
tel = {"platform_profile": open(os.path.join(run, "platform_profile.txt")).read().strip()}
rows = []
if os.path.exists(t):
    for line in open(t):
        parts = [x.strip() for x in line.split(",")]
        try:
            rows.append((float(parts[0]), float(parts[1]), float(parts[2]), int(parts[3], 16)))
        except (ValueError, IndexError):
            pass
if rows:
    clocks = [r[0] for r in rows]
    # 0x4 = SW power cap, 0x20/0x40 = thermal slowdown, 0x8 = HW slowdown
    throttled = [r for r in rows if r[3] & (0x4 | 0x8 | 0x20 | 0x40)]
    tel.update({"gpu_clock_mhz_median": statistics.median(clocks), "gpu_clock_mhz_min": min(clocks),
                "gpu_temp_c_max": max(r[1] for r in rows), "gpu_power_w_median": statistics.median(r[2] for r in rows),
                "throttled_share": len(throttled) / len(rows), "samples": len(rows)})
d["telemetry"] = tel
json.dump(d, open(p, "w"), indent=2)
PY

# Known engine bug: with thread_model=2 (separate render thread, experimental) Godot 4.6.3 aborts in
# RenderingDevice::finalize on every exit, even for an empty scene. Accept it only if the bench finished.
KNOWN_EXIT_ABORT=0
if [[ $CODE -eq 134 ]] && grep -q "BENCH: done" "$RUN_DIR/godot.log" \
    && grep -q "This function (finalize) can only be called from the render thread" "$RUN_DIR/godot.log"; then
    KNOWN_EXIT_ABORT=1
    CODE=0
fi
ERRORS=$(grep -E -A1 "^(ERROR|SCRIPT ERROR|USER ERROR)|SHADER COMPILATION FAILED" "$RUN_DIR/godot.log" \
    | grep -vE "finalize|rendering_device.cpp:7470|^--$" | grep -cE "^(ERROR|SCRIPT ERROR|USER ERROR)|SHADER COMPILATION FAILED")
LEAKS=$(grep -cE "Leaked|leaked|Pages in use" "$RUN_DIR/godot.log")
TEARDOWN=$(grep -oE "chunk instances created=[0-9]+ freed=[0-9]+" "$RUN_DIR/godot.log" | tail -1)
if [[ -n "$TEARDOWN" ]]; then
    CREATED=$(echo "$TEARDOWN" | sed -E 's/.*created=([0-9]+).*/\1/')
    FREED=$(echo "$TEARDOWN" | sed -E 's/.*freed=([0-9]+).*/\1/')
    [[ "$CREATED" != "$FREED" ]] && { echo ">> CHUNK RID LEAK: $TEARDOWN"; ERRORS=$((ERRORS + 1)); }
fi

python3 - "$RUN_DIR/stats.json" <<'PY'
import json, sys, os
p = sys.argv[1]
if not os.path.exists(p):
    print(">> no stats.json produced"); sys.exit(0)
d = json.load(open(p))
print(f">> mode={d['args'].get('mode','legacy')} timed_out={d['timed_out']} frames={d['total_frames']}")
tel = d.get("telemetry", {})
if tel:
    print(f">> GPU: profile={tel.get('platform_profile')} clock median={tel.get('gpu_clock_mhz_median')} min={tel.get('gpu_clock_mhz_min')} MHz"
          f" temp max={tel.get('gpu_temp_c_max')}C power={tel.get('gpu_power_w_median')}W throttled={tel.get('throttled_share', 0):.0%}")
for name, ph in d["phases"].items():
    f, g, t = ph["frame_ms"], ph["render_gpu_ms"], ph["terrain"]
    ch = t.get("chunks", {})
    print(f"   [{name:9}] frames={ph['frames']:5} wall={ph['wall_ms']:8.0f}ms  frame p50={f['p50']:.2f} p99={f['p99']:.2f} max={f['max']:.2f}"
          f"  gpu p50={g['p50']:.2f}  loaded={ch.get('loaded')} pending={ch.get('pending')} tris={ch.get('live_triangles')} drawn_tris={ch.get('drawn_triangles')}"
          f"  rss={ph['memory']['rss_mb']:.0f}MB vram={ph['memory']['video_mem_mb']:.0f}MB")
v = d.get("validation") or {}
if v:
    print(f">> VALIDATION ok={v.get('ok')} chunks_exact={v.get('chunks_exact')}/{v.get('chunks')} matched={v.get('matched_tris')}/{v.get('legacy_tris')}"
          f" unexplained={v.get('unexplained_anomalies')} legacy_unmatched={v.get('legacy_unmatched_tris')} direct_unmatched={v.get('direct_unmatched_tris')} flipped={v.get('winding_flipped_tris')}"
          f" max_pos={v.get('max_position_error_cells', 0):.4f} cells normal max={v.get('max_normal_error_deg', 0):.2f} mean={v.get('mean_normal_error_deg', 0):.3f} deg >5deg={v.get('normals_over_5deg')}")
    if v.get("error"):
        print(">> VALIDATION ERROR:", v["error"])
a = d.get("audit") or {}
if a:
    print(f">> AUDIT ok={a.get('ok')} live={a.get('live')} dead_zero={a.get('dead_zero')} stale={a.get('stale')}"
          f" partial={a.get('partial')} broken={a.get('broken')} bad_header={a.get('bad_header')}")
wo = d.get("final_terrain", {}).get("world_origin")
if wo:
    print(f">> world origin voxel={wo.get('voxel')} chunk={wo.get('chunk')}")
ft = d["final_terrain"]
print(f">> frames_until_loaded={ft.get('frames_until_loaded')} ms_until_loaded={ft.get('ms_until_loaded'):.0f}")
m = ft.get("mesher", {})
for k, v in m.items():
    if isinstance(v, dict) and v.get("count"):
        print(f"   mesher.{k:16} n={v['count']:5} avg={v['avg_ms']:8.2f}ms max={v['max_ms']:8.2f}ms total={v['total_ms']:9.0f}ms")
    elif not isinstance(v, dict):
        print(f"   mesher.{k:16} {v}")
for k, v in ft.get("process", {}).items():
    print(f"   process.{k:17} avg={v['avg_ms']:.3f}ms max={v['max_ms']:.2f}ms")
PY

echo ">> log: $RUN_DIR/godot.log  (errors=$ERRORS leak-lines=$LEAKS exit=$CODE known-engine-exit-abort=$KNOWN_EXIT_ABORT ${TEARDOWN:-})"
if [[ $ERRORS -gt 0 || $LEAKS -gt 0 ]]; then
    grep -E -A1 "^(ERROR|SCRIPT ERROR|USER ERROR)|SHADER COMPILATION FAILED|Leaked|leaked|Pages in use" "$RUN_DIR/godot.log" \
        | grep -vE "finalize|rendering_device.cpp:7470" | sort | uniq -c | sort -rn | head -20
fi
[[ $LEAKS -gt 0 ]] && ERRORS=$((ERRORS + 1))
echo "$RUN_DIR" > "$ROOT/bench/results/LAST"
[[ $CODE -ne 0 ]] && exit 2
[[ $ERRORS -gt 0 ]] && exit 3
# A validation or audit that ran must have passed (empty dicts: not requested).
CHECKS_OK=$(python3 -c "
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print(1); sys.exit()
v, a = d.get('validation') or {}, d.get('audit') or {}
print(int((not v or bool(v.get('ok'))) and (not a or bool(a.get('ok')))))" "$RUN_DIR/stats.json")
[[ "$CHECKS_OK" != "1" ]] && { echo ">> VALIDATION/AUDIT FAILED"; exit 4; }
exit 0
