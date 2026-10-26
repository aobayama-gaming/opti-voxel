#!/usr/bin/env bash
# CPU sampling of a bench run: where each thread's time goes (main thread, render thread, worker pool).
#
#   tools/cpu_profile.sh [--no-build] [--sampler=perf|eu-stack|auto] -- <bench.sh args>
#   e.g. tools/cpu_profile.sh -- --mode=direct --preset=forest5cm_1m --scenario=fly --profile=1
#
# Builds with symbols (debug_symbols=yes: unstripped, -fno-omit-frame-pointer in src/; the next plain build compiles
# everything again), runs tools/bench.sh --no-build with the args, attaches the sampler to the Godot process once it
# exists and writes, into the bench's run dir:
#   perf:     perf.data (perf record -F 499 -g; PERF_CALLGRAPH=dwarf for --call-graph dwarf) and the perf report
#             tables (perf_report_*.txt): on-CPU samples. perf_event_paranoid 2 is enough (an owned process, user space).
#             Sampling 20 threads at 499 Hz slows a 1M load by ~40% (73 -> 106 s): read its frame times as such.
#   eu-stack: stacks.txt, every thread's stack every 0.5 s (EU_STACK_PERIOD): wall-clock samples, the waiting ones
#             counted apart. Each sample stops the process for a moment: its frame times are not the bench's.
#   cpu.md:   tools/cpu_report.py <run>: top self symbols of the main thread and of the other threads, top inclusive.
# auto (the default) takes perf when installed, else eu-stack. Godot's own frames are unnamed (stripped binary).
# Temporary files go under TMPDIR until the run dir exists. Exit code: bench.sh's (1 also for a build failure here).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GODOT="${GODOT:-$HOME/.local/bin/godot}"
BUILD=1
SAMPLER=auto
BENCH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build) BUILD=0 ;;
        --sampler=*) SAMPLER="${1#--sampler=}" ;;
        --) shift; BENCH_ARGS=("$@"); break ;;
        *) echo "usage: $0 [--no-build] [--sampler=perf|eu-stack|auto] -- <bench.sh args>"; exit 1 ;;
    esac
    shift
done
if [[ $SAMPLER == auto ]]; then
    SAMPLER=$(command -v perf > /dev/null && echo perf || echo eu-stack)
fi
command -v "$SAMPLER" > /dev/null || { echo ">> sampler $SAMPLER not found"; exit 1; }

if [[ $BUILD == 1 ]]; then
    echo ">> building with symbols"
    if ! (cd "$ROOT" && scons platform=linux target=template_debug debug_symbols=yes -j"$(nproc)" > "${TMPDIR:-/tmp}/cpu_profile_build.log" 2>&1); then
        grep -E "error|Error" "${TMPDIR:-/tmp}/cpu_profile_build.log" | head -30
        echo ">> BUILD FAILED (full log: ${TMPDIR:-/tmp}/cpu_profile_build.log)"
        exit 1
    fi
fi
LIB=$(ls "$ROOT"/adapter/godot/project/bin/linux/*.so 2>/dev/null | head -1)
if [[ -z "$LIB" ]] || [[ $(nm -C "$LIB" 2> /dev/null | grep -c " [Tt] ") == 0 ]]; then # not grep -q: pipefail would see nm's SIGPIPE
    echo ">> warning: $LIB has no symbols (built without debug_symbols=yes?): the extension's frames stay unnamed"
fi

TMPD=$(mktemp -d "${TMPDIR:-/tmp}/cpu_profile.XXXXXX")
"$ROOT/tools/bench.sh" --no-build "${BENCH_ARGS[@]}" &
BENCH_PID=$!

# The Godot process: the newest one running this project's bench scene, skipping the `timeout` wrapping it
# (GODOT may be a wrapper script that execs the engine, so the exe is not compared to it).
PROJECT="${BENCH_ROOT:-$ROOT}/adapter/godot/project"
TIMEOUT_EXE=$(readlink -f "$(command -v timeout)")
PID=""
for _ in $(seq 1 60); do
    for p in $(pgrep -f "bench.tscn" | sort -rn); do
        exe=$(readlink -f "/proc/$p/exe" 2> /dev/null)
        if [[ -n "$exe" && "$exe" != "$TIMEOUT_EXE" ]] && tr '\0' ' ' < "/proc/$p/cmdline" | grep -qF -- "$PROJECT"; then
            PID=$p
            break 2
        fi
    done
    kill -0 "$BENCH_PID" 2> /dev/null || break
    sleep 0.5
done

SAMPLER_PID=""
if [[ -z "$PID" ]]; then
    echo ">> no Godot process found within 30 s: running without sampling"
elif [[ $SAMPLER == perf ]]; then
    echo ">> perf record on pid $PID"
    CALLGRAPH=()
    [[ "${PERF_CALLGRAPH:-fp}" == dwarf ]] && CALLGRAPH=(--call-graph dwarf) || CALLGRAPH=(-g)
    perf record -F 499 "${CALLGRAPH[@]}" -p "$PID" -o "$TMPD/perf.data" > "$TMPD/perf.log" 2>&1 &
    SAMPLER_PID=$!
else
    echo ">> eu-stack on pid $PID every ${EU_STACK_PERIOD:-0.5} s"
    (
        while kill -0 "$PID" 2> /dev/null; do
            echo "=== SAMPLE $(date +%s.%N)"
            for t in /proc/"$PID"/task/*; do
                echo "COMM ${t##*/} $(cat "$t/comm" 2> /dev/null)"
            done
            eu-stack -m -p "$PID" 2> /dev/null
            sleep "${EU_STACK_PERIOD:-0.5}"
        done
    ) > "$TMPD/stacks.txt" &
    SAMPLER_PID=$!
fi

wait "$BENCH_PID"
CODE=$?
if [[ -n "$SAMPLER_PID" ]]; then
    kill -INT "$SAMPLER_PID" 2> /dev/null # perf stops by itself when the process exits; this covers a stuck one
    wait "$SAMPLER_PID" 2> /dev/null
fi

RUN=$(cat "$ROOT/bench/results/LAST" 2> /dev/null)
if [[ -z "$RUN" || ! -d "$RUN" ]]; then
    echo ">> no run dir (bench/results/LAST): samples left in $TMPD"
    exit "$CODE"
fi
mv "$TMPD"/* "$RUN"/ 2> /dev/null
rmdir "$TMPD" 2> /dev/null
if [[ -f "$RUN/perf.data" ]]; then
    # pid sorts by thread (shown pid:tid); cpu_report.py streams perf script itself (~1 GB of text on a 1M run).
    perf report -i "$RUN/perf.data" --no-children --sort comm,pid,sym --stdio -g none > "$RUN/perf_report_threads.txt" 2> /dev/null
    perf report -i "$RUN/perf.data" --no-children --sort sym --stdio -g none > "$RUN/perf_report_self.txt" 2> /dev/null
    perf report -i "$RUN/perf.data" --children --sort sym --stdio -g none > "$RUN/perf_report_inclusive.txt" 2> /dev/null
fi
if python3 "$ROOT/tools/cpu_report.py" "$RUN" > "$RUN/cpu.md"; then
    echo ">> cpu: $RUN/cpu.md"
else
    echo ">> cpu report failed (samples in $RUN)"
fi
exit "$CODE"
