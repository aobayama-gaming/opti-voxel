#!/usr/bin/env bash
# Runs main_scene (or the scene given as argument, e.g. res://main_scene_5cm.tscn) with a 1 Hz hardware log fsync'ed to disk, so a hard freeze leaves evidence.
cd "$(dirname "$0")/.." || exit 1
mkdir -p bench/results/play; LOG=bench/results/play/hw_$(date +%Y%m%d-%H%M%S).log
(
    while true; do
        GPU=$(nvidia-smi --query-gpu=temperature.gpu,clocks.gr,power.draw,memory.used,clocks_throttle_reasons.active --format=csv,noheader 2>&1)
        CPU=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)
        echo "$(date +%T) gpu[temp,clock,power,vram,throttle]=$GPU cpu_max_mC=$CPU" >> "$LOG"
        sync "$LOG"
        sleep 1
    done
) &
MON=$!
"$HOME/.local/bin/godot" --path adapter/godot/project "$@" > "${LOG%.log}_godot.log" 2>&1
kill $MON
echo "hardware log: $LOG"
