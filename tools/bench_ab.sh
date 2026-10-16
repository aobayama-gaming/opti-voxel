#!/usr/bin/env bash
# Before/after comparison of two checkouts (e.g. dev vs a feature worktree), same bench args for both.
#
#   tools/bench_ab.sh [--runs=3] [--scenarios="load fly"] BEFORE_ROOT AFTER_ROOT [-- bench-args...]
#   e.g. tools/bench_ab.sh ../opti-voxel . -- --mode=direct
# Each tree is built once (one after the other, never during a bench), then runs are interleaved
# before/after so slow drift (thermals, background load) hits both. Each tree benches with its own
# build and keeps its run dirs; the table lands in <AFTER_ROOT>/bench/results/<date>-ab.md.
set -uo pipefail

RUNS=3
SCENARIOS="load fly"
ROOTS=()
BARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs=*) RUNS="${1#--runs=}" ;;
        --scenarios=*) SCENARIOS="${1#--scenarios=}" ;;
        --) shift; BARGS=("$@"); break ;;
        *) ROOTS+=("$(cd "$1" && pwd)") ;;
    esac
    shift
done
[[ ${#ROOTS[@]} -eq 2 ]] || { echo "usage: $0 [--runs=N] [--scenarios=\"load fly\"] BEFORE_ROOT AFTER_ROOT [-- bench-args]"; exit 1; }
BEFORE="${ROOTS[0]}"
AFTER="${ROOTS[1]}"

for root in "$BEFORE" "$AFTER"; do
    echo ">> building $root ($(git -C "$root" rev-parse --abbrev-ref HEAD) $(git -C "$root" rev-parse --short HEAD))"
    (cd "$root" && scons platform=linux target=template_debug -j"$(nproc)" > /dev/null 2>&1) || { echo "build failed in $root"; exit 1; }
done

OUT="$AFTER/bench/results/$(date +%Y%m%d-%H%M%S)-ab.md"
mkdir -p "$AFTER/bench/results"
{
    echo "before: \`$(git -C "$BEFORE" rev-parse --abbrev-ref HEAD)\` $(git -C "$BEFORE" rev-parse --short HEAD)$(git -C "$BEFORE" diff --quiet HEAD || echo ' (dirty)')"
    echo "after: \`$(git -C "$AFTER" rev-parse --abbrev-ref HEAD)\` $(git -C "$AFTER" rev-parse --short HEAD)$(git -C "$AFTER" diff --quiet HEAD || echo ' (dirty)')"
    echo "bench args: ${BARGS[*]:-(none)}, $RUNS runs per side"
    echo
} > "$OUT.tmp"
for scenario in $SCENARIOS; do
    DIRS_BEFORE=""
    DIRS_AFTER=""
    for i in $(seq 1 "$RUNS"); do
        for side in before after; do
            root="$BEFORE"; [[ $side == after ]] && root="$AFTER"
            echo ">> [$scenario] $side run $i/$RUNS"
            if "$root/tools/bench.sh" --no-build --tag="$side-$scenario-$i" "${BARGS[@]}" --scenario="$scenario" > /dev/null 2>&1; then
                dir="$(cat "$root/bench/results/LAST")"
                if [[ $side == before ]]; then DIRS_BEFORE="${DIRS_BEFORE:+$DIRS_BEFORE,}$dir"; else DIRS_AFTER="${DIRS_AFTER:+$DIRS_AFTER,}$dir"; fi
            else
                echo "   run failed: $(cat "$root/bench/results/LAST")"
            fi
        done
    done
    echo "## Scenario: $scenario" >> "$OUT.tmp"
    python3 "$AFTER/tools/bench_compare.py" "$OUT.part" "before=$DIRS_BEFORE" "after=$DIRS_AFTER" --baseline=before > /dev/null
    cat "$OUT.part" >> "$OUT.tmp"
    echo >> "$OUT.tmp"
    rm -f "$OUT.part"
done
mv "$OUT.tmp" "$OUT"
cat "$OUT"
echo ">> written to $OUT"
