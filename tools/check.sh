#!/usr/bin/env bash
# Correctness gate for a change: one build, then every check, no GPU cooldown (temperature only matters
# for perf numbers, use tools/bench_ab.sh for those).
#
#   tools/check.sh [--no-build] [--spike] [--discovery] [--keep-going]
#
# Steps: build, direct load run with validation + page audit + discovery audit, origin parity (4 runs),
# integer-frame test, the QuadPagePool spike when --spike is given or src/quad_page_pool* differs from
# dev, and the discovery miss audit over 9 poses (tools/discovery_audit.sh) when --discovery is given
# or the discoverer / SDF sources differ from dev.
# Stops at the first failure unless --keep-going. Exit code: 0 all passed, 1 something failed.
# Needs a GPU and an unlocked session.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GODOT="${GODOT:-$HOME/.local/bin/godot}"
BUILD=1
SPIKE=0
DISCOVERY=0
KEEP_GOING=0
for a in "$@"; do
    case "$a" in
        --no-build) BUILD=0 ;;
        --spike) SPIKE=1 ;;
        --discovery) DISCOVERY=1 ;;
        --keep-going) KEEP_GOING=1 ;;
        *) echo "usage: $0 [--no-build] [--spike] [--discovery] [--keep-going]"; exit 1 ;;
    esac
done
if [[ $SPIKE == 0 ]] && git -C "$ROOT" diff --name-only dev -- 'src/quad_page_pool*' 2>/dev/null | grep -q .; then
    SPIKE=1
fi
if [[ $DISCOVERY == 0 ]] && git -C "$ROOT" diff --name-only dev -- 'src/chunk_discoverer*' 'src/sdf*' 'src/chunk_math*' 2>/dev/null | grep -q .; then
    DISCOVERY=1
fi

START=$(date +%s)
RESULTS=()
FAILED=0
step() {
    local name="$1"; shift
    local t0=$(date +%s)
    echo ">> $name"
    "$@"
    local code=$?
    local status="PASS"
    [[ $code -ne 0 ]] && { status="FAIL (exit $code)"; FAILED=1; }
    RESULTS+=("$(printf '%-10s %-16s %4ss' "$name" "$status" $(( $(date +%s) - t0 )))")
    if [[ $code -ne 0 && $KEEP_GOING == 0 ]]; then summary; exit 1; fi
}
summary() {
    echo
    printf '%s\n' "${RESULTS[@]}"
    echo ">> total $(( $(date +%s) - START ))s"
}

build() {
    local log="$ROOT/bench/results/check-build.log"
    mkdir -p "$ROOT/bench/results"
    (cd "$ROOT" && scons platform=linux target=template_debug -j"$(nproc)" > "$log" 2>&1) \
        || { grep -E "error|Error" "$log" | head -30; echo ">> full log: $log"; return 1; }
}
spike() {
    local log="$ROOT/bench/results/check-spike.log"
    timeout 300 "$GODOT" --path "$ROOT/project" --resolution 640x360 res://debug/spike.tscn > "$log" 2>&1
    local code=$?
    # Same known thread_model=2 exit abort as tools/bench.sh, accepted once the run finished.
    if [[ $code -eq 134 ]] && grep -q "BENCH: done" "$log"; then code=0; fi
    grep "SPIKE:" "$log" | tail -1
    [[ $code -eq 0 ]] && grep -q '"ok":true' "$log" || { echo ">> log: $log"; return 1; }
}

[[ $BUILD == 1 ]] && step build build
step direct "$ROOT/tools/bench.sh" --no-build --tag=check --mode=direct --scenario=load --steady=60 \
    --validate=64 --audit=1 --discovery_audit=1
step parity "$ROOT/tools/origin_parity.sh" -- --mode=direct --validate=64 --audit=1
step frame "$ROOT/tools/frame_test.sh" --no-build
[[ $SPIKE == 1 ]] && step spike spike
[[ $DISCOVERY == 1 ]] && step discovery "$ROOT/tools/discovery_audit.sh"
summary
exit $FAILED
