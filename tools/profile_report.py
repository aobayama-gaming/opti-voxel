#!/usr/bin/env python3
"""Markdown report of a bench run's profile.json (bench.sh --profile=1): where each phase's frame time goes.

    tools/profile_report.py <run dir> [budget ms, default 16.7]

Per phase: the frame (wall clock), the main thread (VoxelTerrain::_process by section),
the GPU per renderer pass and mesher step, the render thread's CPU, then the frames over budget against the
others. GPU frames are matched to main-thread frames by frame number, with the offset that best correlates
the GPU total with the frame time (printed).
"""
import json
import os
import statistics
import sys

MESHER = {
    "vx_c0": "mesher: cascade levels >= 3",
    "vx_c1": "mesher: cascade level 2",
    "vx_c2": "mesher: cascade level 1",
    "vx_p0": "mesher: sdf (level 0 / table fold)",
    "vx_p1": "mesher: edges",
    "vx_p2": "mesher: vertices",
    "vx_p3": "mesher: reserve",
    "vx_p4": "mesher: emit",
    "vx_copy": "mesher: page copy",
}
GEOMETRY = ("Render Depth Pre-Pass", "Render Opaque Pass", "Render Shadows")


def group_of(section):
    if section in MESHER:
        return "mesher"
    if section in GEOMETRY:
        return "geometry"
    return "other"


def pct(values, q):
    if not values:
        return 0.0
    v = sorted(values)
    return v[int(q * (len(v) - 1))]


def mean(values):
    return statistics.fmean(values) if values else 0.0


def row(name, values, n_frames):
    # Mean over all frames of the phase (a section absent from a frame counts 0), then percentiles.
    padded = values + [0.0] * (n_frames - len(values))
    return f"| {name} | {mean(padded):.2f} | {pct(padded, 0.5):.2f} | {pct(padded, 0.95):.2f} | {pct(padded, 0.99):.2f} | {max(padded, default=0):.2f} |"


def best_offset(main, gpu_by_frame):
    best = (0, -2.0)
    for off in range(-4, 5):
        pairs = [(m["wall_ms"], gpu_by_frame[m["engine_frame"] + off]["gpu_total_ms"])
                 for m in main if m["engine_frame"] + off in gpu_by_frame]
        if len(pairs) < 10:
            continue
        try:
            c = statistics.correlation([p[0] for p in pairs], [p[1] for p in pairs])
        except statistics.StatisticsError:
            continue
        if c > best[1]:
            best = (off, c)
    return best


def main():
    run = sys.argv[1]
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 16.7
    d = json.load(open(os.path.join(run, "profile.json")))
    stats = json.load(open(os.path.join(run, "stats.json")))
    main_frames = d["main"]
    gpu_by_frame = {g["frame"]: g for g in d["gpu"]}
    offset, corr = best_offset(main_frames, gpu_by_frame)
    tel = stats.get("telemetry", {})

    out = [f"# Profile: {os.path.basename(run)}", ""]
    out.append(f"Args: `{' '.join('--%s=%s' % kv for kv in stats['args'].items() if kv[0] != 'out')}`  ")
    out.append(f"Window {d.get('window')}, GPU clock median {tel.get('gpu_clock_mhz_median')} MHz, "
               f"max {tel.get('gpu_temp_c_max')} C, throttled {tel.get('throttled_share', 0):.0%}. "
               f"GPU frames matched at offset {offset:+d} (correlation {corr:.2f}). Times in ms per frame.")
    out.append("")

    phases = [p for p in d.get("phases", []) if any(m["phase"] == p for m in main_frames)]
    summary = ["| phase | frames | wall p50 | p95 | p99 | over budget | GPU p50 | GPU p99 | terrain p50 | render CPU p50 |",
               "|---|---|---|---|---|---|---|---|---|---|"]
    sections_out = []
    for phase in phases:
        ms = [m for m in main_frames if m["phase"] == phase][1:]  # the first frame spans the phase change
        gs = [g for g in d["gpu"] if g["phase"] == phase]
        if not ms:
            continue
        n = len(ms)
        wall = [m["wall_ms"] for m in ms]
        terr = [m["terrain"].get("total", 0.0) for m in ms]
        gtot = [g["gpu_total_ms"] for g in gs]
        ctot = [g["cpu_total_ms"] for g in gs]
        over = sum(w > budget for w in wall)
        summary.append(f"| {phase} | {n} | {pct(wall, .5):.2f} | {pct(wall, .95):.2f} | {pct(wall, .99):.2f} | "
                       f"{over} ({over / n:.0%}) | {pct(gtot, .5):.2f} | {pct(gtot, .99):.2f} | {pct(terr, .5):.2f} | {pct(ctot, .5):.2f} |")

        s = [f"## {phase}", "", "| | mean | p50 | p95 | p99 | max |", "|---|---|---|---|---|---|"]
        s.append(row("**frame (wall)**", wall, n))
        s.append(row("Godot delta", [m["frame_ms"] for m in ms], n))
        for k in sorted(ms[0]["terrain"], key=lambda k: -mean([m["terrain"].get(k, 0) for m in ms])):
            vals = [m["terrain"].get(k, 0.0) for m in ms]
            if k == "total" or max(vals) >= 0.05:
                s.append(row(f"main: terrain {k}", vals, n))
        ng = len(gs)
        s.append(row("**GPU total**", gtot, ng))
        by_section = {}
        for g in gs:
            for k, v in g["gpu"].items():
                by_section.setdefault(k, []).append(v)
        groups = {}
        for k, vals in by_section.items():
            groups.setdefault(group_of(k), []).append(k)
        for grp in ("geometry", "mesher", "other"):
            keys = groups.get(grp, [])
            if not keys:
                continue
            totals = [sum(g["gpu"].get(k, 0.0) for k in keys) for g in gs]
            s.append(row(f"GPU {grp}", totals, ng))
            for k in sorted(keys, key=lambda k: -sum(by_section[k])):
                if sum(by_section[k]) / max(ng, 1) >= 0.02:
                    s.append(row(f"&nbsp;&nbsp;{MESHER.get(k, k)}", by_section[k], ng))
        s.append(row("**render thread CPU total**", ctot, ng))
        cpu_sections = {}
        for g in gs:
            for k, v in g["cpu"].items():
                cpu_sections.setdefault(k, []).append(v)
        for k in sorted(cpu_sections, key=lambda k: -sum(cpu_sections[k]))[:6]:
            s.append(row(f"&nbsp;&nbsp;CPU {MESHER.get(k, k)}", cpu_sections[k], ng))
        s.append(row("draw calls (view)", [m["draw_calls_view"] for m in ms], n))
        s.append(row("draw calls (shadow)", [m["draw_calls_shadow"] for m in ms], n))
        s.append(row("pending chunks", [m["pending"] for m in ms], n))

        # Frames over budget against the others, GPU matched by frame number.
        slow = [m for m in ms if m["wall_ms"] > budget]
        fast = [m for m in ms if m["wall_ms"] <= budget]
        if slow and fast:
            def avg_gpu(frames, key):
                vals = []
                for m in frames:
                    g = gpu_by_frame.get(m["engine_frame"] + offset)
                    if g:
                        vals.append(key(g))
                return mean(vals)
            s += ["", f"Frames over {budget} ms ({len(slow)}) against the others ({len(fast)}), means:", "",
                  "| | over budget | within | difference |", "|---|---|---|---|"]
            lines = [("frame (wall)", lambda m: m["wall_ms"], None),
                     ("main: terrain total", lambda m: m["terrain"].get("total", 0), None),
                     ("main: terrain mesher_dispatch", lambda m: m["terrain"].get("mesher_dispatch", 0), None),
                     ("main: terrain scan_apply", lambda m: m["terrain"].get("scan_apply", 0), None),
                     ("main: terrain compact", lambda m: m["terrain"].get("compact", 0), None),
                     ("GPU total", None, lambda g: g["gpu_total_ms"]),
                     ("GPU geometry", None, lambda g: sum(v for k, v in g["gpu"].items() if group_of(k) == "geometry")),
                     ("GPU mesher", None, lambda g: sum(v for k, v in g["gpu"].items() if group_of(k) == "mesher")),
                     ("GPU other", None, lambda g: sum(v for k, v in g["gpu"].items() if group_of(k) == "other")),
                     ("render thread CPU", None, lambda g: g["cpu_total_ms"])]
            for name, fm, fg in lines:
                a = mean([fm(m) for m in slow]) if fm else avg_gpu(slow, fg)
                b = mean([fm(m) for m in fast]) if fm else avg_gpu(fast, fg)
                s.append(f"| {name} | {a:.2f} | {b:.2f} | {a - b:+.2f} |")
        sections_out += s + [""]

    out += summary + [""] + sections_out
    print("\n".join(out))


if __name__ == "__main__":
    main()
