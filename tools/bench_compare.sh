#!/usr/bin/env bash
# Before/after comparison: runs each config N times per scenario and writes a markdown table.
#
#   tools/bench_compare.sh [--runs=3] [--scenarios="load fly"] CONFIG...
#   CONFIG = label:bench-args (bench-args comma-separated), e.g.
#     tools/bench_compare.sh "legacy:--mode=legacy" "legacy-trim:--mode=legacy,--trim=1" "direct:--mode=direct"
# The first config is the baseline. Output: bench/results/<date>-compare.md
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNS=3
SCENARIOS="load fly"
CONFIGS=()
for a in "$@"; do
    case "$a" in
        --runs=*) RUNS="${a#--runs=}" ;;
        --scenarios=*) SCENARIOS="${a#--scenarios=}" ;;
        *) CONFIGS+=("$a") ;;
    esac
done

(cd "$ROOT" && scons platform=linux target=template_debug -j"$(nproc)" > /dev/null 2>&1) || { echo "build failed"; exit 1; }

OUT="$ROOT/bench/results/$(date +%Y%m%d-%H%M%S)-compare.md"
: > "$OUT.tmp"
for scenario in $SCENARIOS; do
    # Interleave configs (A B A B ...) so slow drift (thermals, background load) hits all of them.
    declare -A DIRS=()
    for i in $(seq 1 "$RUNS"); do
        for cfg in "${CONFIGS[@]}"; do
            label="${cfg%%:*}"
            IFS=',' read -r -a bargs <<< "${cfg#*:}"
            echo ">> [$scenario] $label run $i/$RUNS"
            if "$ROOT/tools/bench.sh" --no-build --tag="$label-$scenario-$i" "${bargs[@]}" --scenario="$scenario" > /dev/null 2>&1; then
                DIRS[$label]="${DIRS[$label]:+${DIRS[$label]},}$(cat "$ROOT/bench/results/LAST")"
            else
                echo "   run failed: $(cat "$ROOT/bench/results/LAST")"
            fi
        done
    done
    AGG_ARGS=()
    for cfg in "${CONFIGS[@]}"; do
        label="${cfg%%:*}"
        AGG_ARGS+=("$label=${DIRS[$label]:-}")
    done
    unset DIRS
    echo "## Scenario: $scenario" >> "$OUT.tmp"
    python3 "$ROOT/tools/bench_compare.py" "$OUT.part" "${AGG_ARGS[@]}" > /dev/null
    cat "$OUT.part" >> "$OUT.tmp"
    echo >> "$OUT.tmp"
    rm -f "$OUT.part"
done
mv "$OUT.tmp" "$OUT"
cat "$OUT"
echo ">> written to $OUT"
