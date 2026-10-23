#!/usr/bin/env python3
"""Markdown report of a bench run's profile.json (bench.sh --profile=1): where each phase's frame time goes.

    tools/profile_report.py <run dir> [budget ms, default 16.7]

Per phase: the frame (wall clock), the main thread (VoxelTerrain::_process by section),
the GPU per renderer pass and mesher step, the render thread's CPU, then the frames over budget against the
others. GPU frames are matched to main-thread frames by frame number, with the offset that best correlates
the GPU total with the frame time (printed). Then, from stats.json: the streamed instances per phase (keys computed /
evicted per kind and LOD, worker busy, main-thread stream time, park and hand-off latencies) and the mesher's GPU
time per batch against its instance list words. Stats missing from older runs read n/a.
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


def stream_of(terrain):
    return (terrain or {}).get("table", {}).get("stream")


def num(d, key):
    v = (d or {}).get(key)
    return v if isinstance(v, (int, float)) else None


def delta(cur, prev, key):
    a, b = num(cur, key), num(prev, key) if prev is not None else 0
    return None if a is None or b is None else a - b


def fmt(v, digits=1):
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def latency(d):
    """{count, p50, p95, max} as 'p50 / p95 / max (count)'."""
    if not isinstance(d, dict) or "count" not in d:
        return "n/a"
    return f"{d['p50']} / {d['p95']} / {d['max']} ({d['count']})"


def stream_phase_row(name, ph, prev_terrain):
    t = ph.get("terrain", {})
    st, prev = stream_of(t), stream_of(prev_terrain) if prev_terrain else None
    wall = ph.get("wall_ms", 0.0)
    compute = delta(st, prev, "compute_ms_total")
    workers = num(st, "workers")
    busy = f"{compute / wall / workers:.0%}" if compute is not None and workers and wall else "n/a"
    mesher, prev_mesher = t.get("mesher", {}), (prev_terrain or {}).get("mesher") if prev_terrain else None
    cells = [fmt(wall, 0), fmt(delta(st, prev, "computed")), fmt(delta(st, prev, "evicted")), busy,
             fmt(delta(st, prev, "admit_ms_total")), fmt(delta(st, prev, "poll_ms_total")), fmt(delta(st, prev, "land_ms_total")),
             fmt(delta(st, prev, "evict_ms_total")), fmt(delta(st, prev, "admit_keys_created")), fmt(num(st, "admit_keys_created_max")),
             fmt(num(st, "jobs_queued_max")), latency((st or {}).get("park_frames")), fmt(delta(mesher, prev_mesher, "stale_parked")),
             latency(t.get("destroy", {}).get("handoff_frames"))]
    return f"| {name} | " + " | ".join(cells) + " |"


def stream_keys_table(name, st, prev):
    by, prev_by = (st or {}).get("computed_by_key", {}), (prev or {}).get("computed_by_key", {})
    out = [f"#### {name}: keys per kind and LOD", "",
           "| key | computed | evicted | computed (cumul.) | evicted (cumul.) | recompute ratio (cumul.) | ms avg |",
           "|---|---|---|---|---|---|---|"]
    # SourceKey::kind_name: the walks, the records grown from them, the blobs.
    kinds = {"sites": 0, "proxies": 1, "parts": 2, "blobs": 3}
    for key in sorted(by, key=lambda k: (kinds.get(k.split()[0], len(kinds)), int(k.split()[1]))):
        k, p = by[key], prev_by.get(key, {})
        c, e = k.get("count", 0), k.get("evicted")
        ratio = f"{c / (c - e):.2f}" if e is not None and c > e else ("n/a" if e is None else "inf")
        de = None if e is None else e - p.get("evicted", 0)
        out.append(f"| {key} | {c - p.get('count', 0)} | {fmt(de)} | {c} | {fmt(e)} | {ratio} | {k.get('ms_avg', 0):.3f} |")
    return out + [""]


def stream_report(stats, phases):
    """Per phase (deltas from the previous phase's end; percentiles and maxima cumulative): the stream's work."""
    out = ["## Streamed instances", ""]
    if not any(stream_of(stats["phases"].get(p, {}).get("terrain")) for p in phases):
        return out + ["n/a (no instance stream in this run)", ""]
    out += ["Worker busy = compute ms / phase wall ms / workers. Latencies p50 / p95 / max (count), cumulative "
            "since the start: park = stream frames from a chunk's park to its hand-back, hand-off = frames a stalled "
            "parent stays drawn over its children.", "",
            "| phase | wall ms | computed | evicted | worker busy | admit ms | poll ms | land ms | evict ms | keys created "
            "by admits | max per admit | FIFO max | park frames | stale parked | hand-off frames |",
            "|" + "---|" * 15]
    prev, tables = None, []
    for p in phases:
        ph = stats["phases"].get(p)
        if not ph:
            continue
        out.append(stream_phase_row(p, ph, prev))
        tables += stream_keys_table(p, stream_of(ph.get("terrain")), stream_of(prev))
        prev = ph.get("terrain")
    return out + [""] + tables


def list_words_report(stats):
    """The mesher's GPU time per batch by its instance list words (log2 bins), over the whole run."""
    bins = stats.get("final_terrain", {}).get("mesher", {}).get("gpu_by_list_words")
    out = ["## Mesher GPU per batch vs list words", ""]
    if not bins:
        return out + ["n/a (no gpu_by_list_words in this run)", ""]
    out += ["| list words | batches | words avg | sdf pass us avg | batch us avg | sdf us per 1k words |",
            "|---|---|---|---|---|---|"]
    for b in bins:
        per_k = 1000.0 * b["sdf_us_avg"] / b["words_avg"] if b["words_avg"] else 0.0
        out.append(f"| {b['words_lo']}-{b['words_hi']} | {b['batches']} | {b['words_avg']:.0f} | {b['sdf_us_avg']:.0f} | "
                   f"{b['total_us_avg']:.0f} | {per_k:.1f} |")
    return out + [""]


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
                     ("main: terrain stream_poll", lambda m: m["terrain"].get("stream_poll", 0), None),
                     ("main: terrain mesher_dispatch", lambda m: m["terrain"].get("mesher_dispatch", 0), None),
                     ("main: terrain gate_admit", lambda m: m["terrain"].get("gate_admit", 0), None),
                     ("main: terrain build_lists", lambda m: m["terrain"].get("build_lists", 0), None),
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
    out += stream_report(stats, phases)
    out += list_words_report(stats)
    print("\n".join(out))


if __name__ == "__main__":
    main()
