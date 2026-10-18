#!/usr/bin/env bash
# Discovery miss audit over several camera poses and settings: every run rebuilds its scan without
# culling and counts the quads of each culled chunk on the GPU (bench --discovery_audit=1). A culled
# chunk with quads is a miss, i.e. a hole. Also checks that each run settles (loaded == scanned).
#
#   tools/discovery_audit.sh [--only=a,b] [-- extra bench args, e.g. --discovery_cull=2]
#
# Cases: default pose, the main scene's lod_error 4, underground, another area, high above, deep
# below, a far world origin, 5 cm voxels, a fly run (audited once it settles back at the start), and
# 25 LODs (--lods=25) from 10 km up looking down and from 100 km up looking at the horizon. Heights are
# above the ground under the camera (--above_ground): the cascade's ground height varies by kilometres.
# Report: bench/results/<timestamp>-discovery.md. Exit code: 0 no miss anywhere, 1 a run failed or missed.
# No build (run scons first), no cooldown: this is a correctness check.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ONLY=""
EXTRA=()
SEEN_DASH=0
for a in "$@"; do
    if [[ $SEEN_DASH == 1 ]]; then
        EXTRA+=("$a")
    else
        case "$a" in
            --only=*) ONLY="${a#--only=}" ;;
            --) SEEN_DASH=1 ;;
            *) echo "usage: $0 [--only=a,b] [-- bench args]"; exit 1 ;;
        esac
    fi
done

# Far origin: ~1e15 m on x and z, a multiple of the root chunk span (2^15 ids of 14 voxels) and of
# 2^32 voxels. y stays 0: the terrain's datum is at absolute y ~ 300 m.
read -r FAR <<< "$(python3 -c "
w = 7 * 2**32
print(round(1e15 / 0.5 / w) * w)")"

CASES=(
    "default|"
    "lod4|--lod_error=4"
    "underground|--above_ground=-60"
    "east|--cam_pos=3000,340,2000 --above_ground=50"
    "high|--above_ground=1200"
    "deep|--above_ground=-1200"
    "far_origin|--origin=$FAR,0,-$FAR"
    "cm|--voxel=0.05 --lod_error=4"
    "fly|--scenario=fly"
    "lods25_10km|--lods=25 --above_ground=10000 --pitch=-45"
    "lods25_100km|--lods=25 --above_ground=100000 --pitch=-5"
)

OUT="$ROOT/bench/results/$(date +%Y%m%d-%H%M%S)-discovery.md"
mkdir -p "$(dirname "$OUT")"
FAILED=0
ROWS=()
for c in "${CASES[@]}"; do
    label="${c%%|*}"
    args="${c#*|}"
    if [[ -n "$ONLY" && ",$ONLY," != *",$label,"* ]]; then
        continue
    fi
    echo ">> $label: $args ${EXTRA[*]:-}"
    # shellcheck disable=SC2086
    "$ROOT/tools/bench.sh" --no-build --tag="disc-$label" --mode=direct --scenario=load --steady=10 \
        --discovery_audit=1 $args "${EXTRA[@]}" > /dev/null 2>&1
    code=$?
    dir=$(cat "$ROOT/bench/results/LAST")
    row=$(python3 - "$dir/stats.json" "$label" "$code" <<'PY'
import json, sys
p, label, code = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    d = json.load(open(p))
except Exception:
    print(f"| {label} | FAIL (exit {code}, no stats) |" + " |" * 11); sys.exit()
da = d.get("discovery_audit") or {}
ft = d.get("final_terrain", {})
ch, dsc = ft.get("chunks", {}), ft.get("discoverer", {})
sc = dsc.get("scan", {})
ok = code == 0 and bool(da.get("ok"))
status = "ok" if ok else f"FAIL (exit {code})"
print(f"| {label} | {status} | {da.get('misses')} | {da.get('scanned')} | {ch.get('loaded_non_empty')} | {da.get('culled')} | {da.get('reference')} "
      f"| {da.get('max_lod_jump')} | {ch.get('loaded_not_scanned')}/{ch.get('scanned_not_loaded')} | {sc.get('count')} | {sc.get('avg_ms', 0):.1f} "
      f"| {dsc.get('last_sdf_calls')} | {ft.get('frames_until_loaded')} |")
PY
)
    echo "   $row"
    ROWS+=("$row")
    [[ "$row" == *"FAIL"* ]] && { FAILED=1; echo "   run dir: $dir"; }
done

{
    echo "# Discovery audit $(date '+%Y-%m-%d %H:%M')"
    echo
    echo "Branch \`$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)\` @ $(git -C "$ROOT" rev-parse --short HEAD), extra args: \`${EXTRA[*]:-none}\`"
    echo
    echo "| case | result | misses | scanned | non-empty | culled | reference | max LOD jump | settle (loaded∖scanned / scanned∖loaded) | scans | scan avg ms | SDF calls (last scan) | frames until loaded |"
    echo "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    printf '%s\n' "${ROWS[@]}"
} > "$OUT"
echo ">> report: $OUT"
exit $FAILED
