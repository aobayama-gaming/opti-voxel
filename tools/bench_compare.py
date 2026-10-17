#!/usr/bin/env python3
"""Aggregate bench runs into a before/after markdown table.

    bench_compare.py OUT.md LABEL=run_dir1,run_dir2,... LABEL2=... [--baseline=LABEL]

Each run dir holds a stats.json written by project/debug/bench.gd. For every metric the
median across runs is shown, with the min-max spread, and the delta vs the baseline config.
"""
import json
import os
import statistics
import sys


def get(d, path, default=None):
    for key in path.split("."):
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


# (section, label, extractor, unit, lower_is_better)
METRICS = [
    ("Load (scenario A)", "time to full load", lambda d: get(d, "final_terrain.ms_until_loaded"), "ms", True),
    ("Load (scenario A)", "frames to full load", lambda d: get(d, "final_terrain.frames_until_loaded"), "frames", True),
    ("Load (scenario A)", "chunks meshed / s",
     lambda d: get(d, "phases.load.terrain.chunks.loaded", 0) / max(get(d, "final_terrain.ms_until_loaded", 1), 1) * 1000, "chunks/s", False),
    ("Load (scenario A)", "frame time p50 while loading", lambda d: get(d, "phases.load.frame_ms.p50"), "ms", True),
    ("Load (scenario A)", "frame time p99 while loading", lambda d: get(d, "phases.load.frame_ms.p99"), "ms", True),
    ("Load (scenario A)", "frame time max while loading", lambda d: get(d, "phases.load.frame_ms.max"), "ms", True),

    ("Mesher CPU", "main thread: _process total, avg", lambda d: get(d, "final_terrain.process.total.avg_ms"), "ms", True),
    ("Mesher CPU", "main thread: _process total, max", lambda d: get(d, "final_terrain.process.total.max_ms"), "ms", True),
    ("Mesher CPU", "main thread: mesher dispatch, max", lambda d: get(d, "final_terrain.process.mesher_dispatch.max_ms"), "ms", True),
    ("Mesher CPU", "main thread: page compaction, avg", lambda d: get(d, "final_terrain.process.compact.avg_ms"), "ms", True),
    ("Mesher CPU", "main thread: page compaction, max", lambda d: get(d, "final_terrain.process.compact.max_ms"), "ms", True),
    ("Mesher CPU", "batch wall time (worker/render thread), avg", lambda d: get(d, "final_terrain.mesher.batch_total.avg_ms"), "ms", True),
    ("Mesher CPU", "  legacy: GPU submit+sync, avg", lambda d: get(d, "final_terrain.mesher.gpu_submit_sync.avg_ms"), "ms", True),
    ("Mesher CPU", "  legacy: readback, avg", lambda d: get(d, "final_terrain.mesher.points_fetch.avg_ms"), "ms", True),
    ("Mesher CPU", "  legacy: CPU unpack, avg", lambda d: get(d, "final_terrain.mesher.points_process.avg_ms"), "ms", True),
    ("Mesher CPU", "  legacy: CPU triangulate, avg", lambda d: get(d, "final_terrain.mesher.triangulate.avg_ms"), "ms", True),
    ("Mesher CPU", "  legacy: mesh upload, avg", lambda d: get(d, "final_terrain.mesher.render_upload.avg_ms"), "ms", True),
    ("Mesher GPU", "  direct: GPU compute per batch, avg", lambda d: get(d, "final_terrain.mesher.gpu_compute.avg_ms"), "ms", True),
    ("Mesher GPU", "    of which SDF pass (fixed samples per batch), avg", lambda d: get(d, "final_terrain.mesher.gpu_pass_sdf.avg_ms"), "ms", True),
    ("Mesher GPU", "  direct: GPU copy per batch, avg", lambda d: get(d, "final_terrain.mesher.gpu_copy.avg_ms"), "ms", True),

    ("Other CPU", "update_destroy_from_scan, avg", lambda d: get(d, "final_terrain.process.destroy_from_scan.avg_ms"), "ms", True),
    ("Other CPU", "update_destroy_from_scan, total", lambda d: get(d, "final_terrain.process.destroy_from_scan.total_ms"), "ms", True),
    ("Other CPU", "scan ingest (main thread), total", lambda d: get(d, "final_terrain.process.scan_ingest.total_ms"), "ms", True),

    ("Discoverer (worker thread)", "scans", lambda d: get(d, "final_terrain.discoverer.scan.count"), "", None),
    ("Discoverer (worker thread)", "scan, avg", lambda d: get(d, "final_terrain.discoverer.scan.avg_ms"), "ms", True),
    ("Discoverer (worker thread)", "scan, max", lambda d: get(d, "final_terrain.discoverer.scan.max_ms"), "ms", True),
    ("Discoverer (worker thread)", "  cull share of scan time",
     lambda d: get(d, "final_terrain.discoverer.cull.total_ms", 0) / max(get(d, "final_terrain.discoverer.scan.total_ms", 1), 1e-9) * 100
     if get(d, "final_terrain.discoverer.cull.total_ms") is not None else None, "%", True),
    ("Discoverer (worker thread)", "camera move to published scan, max", lambda d: get(d, "final_terrain.discoverer.latency.max_ms"), "ms", True),
    ("Discoverer (worker thread)", "scans skipped (same camera chunk)", lambda d: get(d, "final_terrain.discoverer.scans_skipped_same_chunk"), "", None),
    ("Discoverer (worker thread)", "nodes visited, total", lambda d: get(d, "final_terrain.discoverer.nodes_visited_total"), "", None),
    ("Discoverer (worker thread)", "SDF calls, total", lambda d: get(d, "final_terrain.discoverer.sdf_calls_total"), "", True),
    ("Discoverer (worker thread)", "cull cache hits, total", lambda d: get(d, "final_terrain.discoverer.cache_hits_total"), "", None),
    ("Discoverer (worker thread)", "scanned chunks (last scan)", lambda d: get(d, "final_terrain.chunks.scanned"), "", None),
    ("Discoverer (worker thread)", "  of which non-empty", lambda d: get(d, "final_terrain.chunks.loaded_non_empty"), "", None),

    ("Steady state (scenario B)", "frame time p50", lambda d: get(d, "phases.steady.frame_ms.p50"), "ms", True),
    ("Steady state (scenario B)", "frame time p99", lambda d: get(d, "phases.steady.frame_ms.p99"), "ms", True),
    ("Steady state (scenario B)", "render GPU time p50", lambda d: get(d, "phases.steady.render_gpu_ms.p50"), "ms", True),
    ("Steady state (scenario B)", "render CPU time p50", lambda d: get(d, "phases.steady.render_cpu_ms.p50"), "ms", True),
    ("Steady state (scenario B)", "draw calls", lambda d: get(d, "phases.steady.draw_calls.p50"), "", True),
    ("Steady state (scenario B)", "primitives drawn (direct: page capacity, not triangles)", lambda d: get(d, "phases.steady.primitives.p50"), "", True),
    ("Steady state (scenario B)", "  draw calls, camera pass", lambda d: get(d, "phases.steady.draw_calls_view.p50"), "", True),
    ("Steady state (scenario B)", "  draw calls, shadow pass", lambda d: get(d, "phases.steady.draw_calls_shadow.p50"), "", True),

    ("Flythrough (scenario C)", "frame time p50 (out)", lambda d: get(d, "phases.fly_out.frame_ms.p50"), "ms", True),
    ("Flythrough (scenario C)", "frame time p99 (out)", lambda d: get(d, "phases.fly_out.frame_ms.p99"), "ms", True),
    ("Flythrough (scenario C)", "frame time max (out)", lambda d: get(d, "phases.fly_out.frame_ms.max"), "ms", True),
    ("Flythrough (scenario C)", "pending chunks mean (out)", lambda d: get(d, "phases.fly_out.pending_chunks.mean"), "", True),
    ("Flythrough (scenario C)", "pending chunks mean (back)", lambda d: get(d, "phases.fly_back.pending_chunks.mean"), "", True),
    ("Flythrough (scenario C)", "frames to settle after return", lambda d: get(d, "phases.settle.frames"), "frames", True),

    ("Geometry", "loaded chunks", lambda d: get(d, "phases.steady.terrain.chunks.loaded"), "", None),
    ("Geometry", "live triangles (real)", lambda d: get(d, "phases.steady.terrain.chunks.live_triangles"), "", None),
    ("Geometry", "triangles in index buffers (incl. padding)", lambda d: get(d, "phases.steady.terrain.chunks.drawn_triangles"), "", True),
    ("Geometry", "  direct: pages with live chunks", lambda d: get(d, "phases.steady.terrain.mesher.pages_with_live_chunks"), "", None),
    ("Geometry", "  direct: drawn quad slots per live quad",
     lambda d: (get(d, "phases.steady.terrain.mesher.drawn_quad_slots") or 0) / max(get(d, "phases.steady.terrain.mesher.live_quads", 1), 1)
     if get(d, "phases.steady.terrain.mesher.drawn_quad_slots") is not None else None, "", True),
    ("Geometry", "  direct: pages re-packed by compaction", lambda d: get(d, "phases.steady.terrain.mesher.pages_compacted"), "", None),
    ("Geometry", "  direct: page AABB longest axis, mean", lambda d: get(d, "phases.steady.terrain.mesher.page_aabb_extent_chunks_mean"), "chunks", True),
    ("Geometry", "  direct: page AABB longest axis, max", lambda d: get(d, "phases.steady.terrain.mesher.page_aabb_extent_chunks_max"), "chunks", True),

    ("Test conditions", "GPU clock, median (MHz)", lambda d: get(d, "telemetry.gpu_clock_mhz_median"), "", None),
    ("Test conditions", "GPU clock, min (MHz)", lambda d: get(d, "telemetry.gpu_clock_mhz_min"), "", None),
    ("Test conditions", "GPU temperature at start (C)", lambda d: get(d, "telemetry.start_temp_c"), "", None),
    ("Test conditions", "GPU temperature, max (C)", lambda d: get(d, "telemetry.gpu_temp_c_max"), "", None),
    ("Test conditions", "share of samples throttled", lambda d: (get(d, "telemetry.throttled_share") or 0) * 100, "%", None),

    ("Memory", "RSS (end of steady)", lambda d: get(d, "phases.steady.memory.rss_mb"), "MB", True),
    ("Memory", "VRAM used (end of steady)", lambda d: get(d, "phases.steady.memory.video_mem_mb"), "MB", True),
    ("Memory", "RD buffer memory (end of steady)", lambda d: get(d, "phases.steady.memory.buffer_mem_mb"), "MB", True),
    ("Memory", "  direct: page pool VRAM (end of steady)", lambda d: get(d, "phases.steady.terrain.mesher.pool.vram_mb"), "MB", True),
    ("Memory", "  direct: page pool, pages allocated", lambda d: get(d, "phases.steady.terrain.mesher.pool.pages_total"), "", True),
]


def fmt(v):
    if v is None:
        return "–"
    if abs(v) >= 1e6:
        return f"{v / 1e6:.2f}M"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.2f}"


def main():
    out_path = sys.argv[1]
    configs = []
    baseline = None
    for a in sys.argv[2:]:
        if a.startswith("--baseline="):
            baseline = a.split("=", 1)[1]
            continue
        label, dirs = a.split("=", 1)
        runs = []
        for rd in dirs.split(","):
            p = os.path.join(rd, "stats.json")
            if os.path.exists(p):
                runs.append(json.load(open(p)))
        configs.append((label, runs))
    baseline = baseline or configs[0][0]
    base_runs = dict(configs)[baseline]

    lines = []
    n_runs = ", ".join(f"{label}: {len(runs)} runs" for label, runs in configs)
    lines.append(f"Median of runs ({n_runs}); `[min–max]` spread. Δ is vs **{baseline}**.\n")
    header = "| Metric | " + " | ".join(label for label, _ in configs) + " |"
    for label, _ in configs:
        if label != baseline:
            header += f" Δ {label} |"
    sep = "|" + "---|" * (header.count("|") - 1)

    section = None
    for sec, name, fn, unit, lower_better in METRICS:
        values = []
        for label, runs in configs:
            vals = [fn(r) for r in runs]
            vals = [v for v in vals if isinstance(v, (int, float)) and v is not None]
            values.append(vals)
        if all(not v for v in values):
            continue
        if sec != section:
            lines.append(f"\n**{sec}**\n")
            lines.append(header)
            lines.append(sep)
            section = sec
        cells = []
        medians = []
        for vals in values:
            if not vals:
                cells.append("–")
                medians.append(None)
                continue
            m = statistics.median(vals)
            medians.append(m)
            spread = f" [{fmt(min(vals))}–{fmt(max(vals))}]" if len(vals) > 1 and min(vals) != max(vals) else ""
            cells.append(f"{fmt(m)}{spread}")
        row = f"| {name}{' (' + unit + ')' if unit else ''} | " + " | ".join(cells) + " |"
        b = medians[[l for l, _ in configs].index(baseline)]
        for (label, _), m in zip(configs, medians):
            if label == baseline:
                continue
            if b in (None, 0) or m is None:
                row += " – |"
                continue
            delta = (m - b) / abs(b) * 100
            mark = ""
            if lower_better is not None and abs(delta) >= 5:
                better = delta < 0 if lower_better else delta > 0
                mark = " ✅" if better else " ⚠️"
            row += f" {delta:+.0f}%{mark} |"
        lines.append(row)

    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
