#!/usr/bin/env python3
"""The engine map of a traced bench run (bench.sh --trace=1): which thread computes what, how much of each frame it
costs, and how work hands off between threads.

    tools/engine_map.py <run dir> [--out=<dir>]

Reads <run>/trace.json (the engine trace, src/trace.hpp), <run>/trace_phases.json (the bench's phases) and
<run>/stats.json (the run's args), writes <out>/engine_map.html (one swimlane per thread: each system's stages with
their ms per frame, and the hand-offs between them as arrows) and <out>/engine_map.md (the same numbers as tables).
<out> defaults to the run dir.

What the numbers mean:
- frames: the main thread's VoxelTerrain::_process calls in the phase; ms/frame = a node's total time / frames (on the
  worker pool, the CPU time of all its threads: it can exceed the frame).
- GPU: the captured frames' intervals (mesher batches always, every renderer pass with --profile=1), per GPU frame.
  Their placement on the CPU clock is the earliest consistent with when they were recorded.
- A thread's busy share counts traced work only: the render thread's own (Godot's) work is not in the trace.
- A flow's latency is the time between consecutive points of the same piece of work (a scan from the discoverer to
  the main thread; a mesh batch main -> render thread -> GPU -> readback -> main; a stream job; a field request).
"""
import bisect
import html
import json
import math
import os
import statistics
import sys
from collections import defaultdict

# Swimlanes. Unnamed threads are the worker pool. The map's columns, left to right, put every hand-off between
# neighbours: the threads feeding the main thread (scans, stream jobs; field requests stay in the column) stacked
# first, then main -> render -> GPU.
LANES = ["discoverer", "worker", "field_gpu", "main", "render", "GPU"]
COLUMNS = [["discoverer", "worker", "field_gpu"], ["main"], ["render"], ["GPU"]]
LANE_TITLE = {
    "discoverer": "Discoverer thread",
    "main": "Main thread",
    "worker": "Worker pool",
    "field_gpu": "FieldGPU thread",
    "render": "Render thread",
    "GPU": "GPU",
}
FLOW_TITLE = {
    "scan": "scan",
    "batch": "mesh batch",
    "stream_job": "stream job",
    "field": "field request",
    "kill": "page kills",
    "compact": "page compaction",
    "hidden": "hidden sub-cubes",
}
# Renderer and mesher GPU section names, as tools/profile_report.py reads them.
GPU_LABEL = {
    "vx_c0": "cascade levels >= 3",
    "vx_c1": "cascade level 2",
    "vx_c2": "cascade level 1",
    "vx_p0": "sdf / table fold",
    "vx_p1": "edges",
    "vx_p2": "vertices",
    "vx_p3": "reserve",
    "vx_p4": "emit",
    "vx_copy": "page copy",
}
FRAME_BUDGET_MS = 16.7
MIN_ROW_MS = 0.005  # rows under this per frame fold into "+N small"
MAX_DEPTH = 3


def pct(values, q):
    if not values:
        return 0.0
    v = sorted(values)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]


# ---------------------------------------------------------------------------------------------------------- reading

def read_trace(path):
    """(thread names {tid: name}, spans {tid: [(ts, dur, cat, name, n, seq, parent)]},
    flow points [(ph, name, id, ts, tid, parent)]). seq / parent: the span's number on its thread and the one it runs in
    (0 at the top), recorded by the engine (trace.hpp): the call tree doesn't depend on the clock's resolution."""
    names, spans, flows = {}, defaultdict(list), []

    def take(e):
        ph = e.get("ph")
        args = e.get("args", {})
        if ph == "X":
            spans[e["tid"]].append((e["ts"], e["dur"], e["cat"], e["name"], args.get("n", 0), args["s"], args["p"]))
        elif ph in ("s", "t", "f"):
            flows.append((ph, e["name"], e["id"], e["ts"], e["tid"], args["p"]))
        elif ph == "M" and e.get("name") == "thread_name":
            names[e["tid"]] = e["args"]["name"]

    # trace.cpp writes one event per line: parse them one by one (a whole trace can be hundreds of MB).
    with open(path) as f:
        first = f.readline()
        if not first.startswith('{"traceEvents":['):
            f.seek(0)
            for e in json.load(f)["traceEvents"]:
                take(e)
        else:
            for line in f:
                line = line.strip().rstrip(",")
                if line.startswith("{\"ph\""):
                    take(json.loads(line))
    return names, spans, flows


def lane_of(name):
    return name if name in LANES and name != "worker" else "worker"


# --------------------------------------------------------------------------------------------------------- analysis

class Node:
    __slots__ = ("total", "self", "durs")

    def __init__(self):
        self.total = 0.0
        self.self = 0.0
        self.durs = []


class Phase:
    def __init__(self, name, start, end):
        self.name, self.start, self.end = name, start, end
        self.nodes = defaultdict(Node)  # (lane, path) -> Node; path = ((cat, name), ...)
        self.lane_busy = defaultdict(float)  # depth-0 time per lane, usec
        self.lane_threads = defaultdict(set)
        self.process_starts = []
        self.gpu_frames = 0
        self.edges = defaultdict(list)  # (flow, (lane, path), (lane, path)) -> [latency usec]
        self.flow_total = defaultdict(list)  # flow -> [first point to last point, usec] of finished chains
        self.flow_unfinished = defaultdict(int)
        # (lane, parent path) -> {(a, b): parent calls in which child a first started before child b}
        self.sibling_order = defaultdict(lambda: defaultdict(int))

    @property
    def frames(self):
        return max(len(self.process_starts), 1)

    def per_frame(self, lane, usec):
        frames = max(self.gpu_frames, 1) if lane == "GPU" else self.frames
        return usec / 1000.0 / frames


def analyse(names, spans, flows, phases):
    """Fills each Phase (plus "all") from the spans and flows."""
    starts = [p.start for p in phases]

    def phase_of(ts):
        i = bisect.bisect_right(starts, ts) - 1
        return phases[i] if i >= 0 else None

    whole = Phase("all", min(starts) if starts else 0, max(p.end for p in phases) if phases else 0)
    # Per thread: each span's path from its parent's (a parent starts first: a smaller seq), then each flow point's
    # enclosing span by its parent.
    path_of = {}  # (tid, seq) -> path = ((cat, name), ...)
    for tid, list_ in spans.items():
        lane = lane_of(names.get(tid, "worker"))
        list_.sort(key=lambda s: s[5])
        child_time = defaultdict(float)
        kids = defaultdict(dict)  # parent seq -> its children's (cat, name), in the order they first started
        for ts, dur, cat, name, n, seq, parent in list_:
            path_of[(tid, seq)] = path_of.get((tid, parent), ()) + ((cat, name),)
            child_time[parent] += dur
            if parent:
                kids[parent].setdefault((cat, name), None)
        for parent, order in kids.items():
            if len(order) > 1 and (tid, parent) in path_of:
                counts = whole.sibling_order[(lane, path_of[(tid, parent)])]
                order = list(order)
                for i, a in enumerate(order):
                    for b in order[i + 1:]:
                        counts[(a, b)] += 1
        for ts, dur, cat, name, n, seq, parent in list_:
            path = path_of[(tid, seq)]
            for p in (phase_of(ts), whole):
                if p is None:
                    continue
                node = p.nodes[(lane, path)]
                node.total += dur
                node.self += max(dur - child_time[seq], 0)
                node.durs.append(dur)
                p.lane_threads[lane].add(tid)
                if parent == 0 and path != (("gpu", "frame"),):
                    p.lane_busy[lane] += dur
                if path == (("terrain", "process"),):
                    p.process_starts.append(ts)
                if path == (("gpu", "frame"),):
                    p.gpu_frames += 1
                    p.lane_busy[lane] += dur
    # A span that closed after the trace was written isn't in it: a point inside it counts at its thread's top.
    enclosing = [(lane_of(names.get(tid, "worker")), path_of.get((tid, parent), ())) for ph, name, fid, ts, tid, parent in flows]

    chains = defaultdict(list)
    for i, (ph, name, fid, ts, tid, parent) in enumerate(flows):
        chains[(name, fid)].append((ts, "stf".index(ph), i))
    for (name, fid), pts in chains.items():
        # A chain: s, then t..., then f; an id can be reused once its chain ended (field requests on the stack).
        pts.sort()
        chain = []
        for ts, kind, i in pts:
            if kind == 0 and chain:
                _record_chain(name, chain, flows, enclosing, phase_of, whole, finished=False)
                chain = []
            chain.append(i)
            if kind == 2:
                _record_chain(name, chain, flows, enclosing, phase_of, whole, finished=True)
                chain = []
        if chain:
            _record_chain(name, chain, flows, enclosing, phase_of, whole, finished=False)
    return whole


def _record_chain(name, chain, flows, enclosing, phase_of, whole, finished):
    first_ts = flows[chain[0]][3]
    for p in (phase_of(first_ts), whole):
        if p is None:
            continue
        for a, b in zip(chain, chain[1:]):
            p.edges[(name, enclosing[a], enclosing[b])].append(flows[b][3] - flows[a][3])
        if finished and len(chain) > 1:
            p.flow_total[name].append(flows[chain[-1]][3] - first_ts)
        elif not finished:
            p.flow_unfinished[name] += 1


def read_phases(run, spans):
    path = os.path.join(run, "trace_phases.json")
    first = min((s[0] for l in spans.values() for s in l), default=0)
    last = max((s[0] + s[1] for l in spans.values() for s in l), default=0)
    if not os.path.exists(path):
        return [Phase("run", first, last)]
    data = json.load(open(path))
    ph = data.get("phases", [])
    end = data.get("end_usec", last)
    out = []
    for i, p in enumerate(ph):
        start = first if i == 0 else p["start_usec"]  # the trace starts before the load phase's mark
        stop = ph[i + 1]["start_usec"] if i + 1 < len(ph) else end
        out.append(Phase(p["phase"], start, stop))
    return out or [Phase("run", first, last)]


# ----------------------------------------------------------------------------------------------------------- report

def label(lane, key):
    cat, name = key
    if lane == "GPU":
        return "frame total" if key == ("gpu", "frame") else GPU_LABEL.get(name, name)
    return name


def frame_summary(p):
    intervals = [b - a for a, b in zip(p.process_starts, p.process_starts[1:])]
    frame_ms = statistics.median(intervals) / 1000.0 if intervals else 0.0
    main_ms = p.per_frame("main", p.nodes[("main", (("terrain", "process"),))].total)
    gpu_ms = p.per_frame("GPU", p.nodes[("GPU", (("gpu", "frame"),))].total) if p.gpu_frames else 0.0
    if frame_ms <= 0:
        bound = "n/a"
    elif gpu_ms >= 0.85 * frame_ms:
        bound = "GPU"
    elif main_ms >= 0.85 * frame_ms:
        bound = "main thread"
    else:
        bound = "neither (render thread, vsync or untraced)"
    return {"frame_ms": frame_ms, "main_ms": main_ms, "gpu_ms": gpu_ms, "bound": bound,
            "frames": len(p.process_starts), "gpu_frames": p.gpu_frames, "wall_s": (p.end - p.start) / 1e6}


def tree_rows(p, lane):
    """Rows of a lane, depth first, children by ms/frame descending: (depth, path, Node) or (depth, None, count)."""
    children = defaultdict(list)
    for (l, path), node in p.nodes.items():
        if l == lane:
            children[path[:-1]].append((path, node))
    rows = []

    def walk(prefix, depth):
        kids = sorted(children.get(prefix, []), key=lambda c: -c[1].total)
        small = 0
        for path, node in kids:
            if p.per_frame(lane, node.total) < MIN_ROW_MS:
                small += 1
                continue
            rows.append((depth, path, node))
            if depth + 1 < MAX_DEPTH:
                walk(path, depth + 1)
        if small:
            rows.append((depth, None, small))

    walk((), 0)
    return rows


def fmt_ms(v):
    return f"{v:.2f}" if v < 10 else f"{v:.1f}"


def edge_label(name, lat):
    return f"{FLOW_TITLE.get(name, name)}: p50 {fmt_ms(pct(lat, 0.5) / 1000)} ms"


def markdown(run, args, phases):
    out = [f"# Engine map: {os.path.basename(os.path.normpath(run))}", ""]
    out.append("Generated by `tools/engine_map.py` from `trace.json` (`bench.sh --trace=1`). ms/frame: the time a node "
               "takes per main-thread frame (GPU: per captured GPU frame). Self: without its children.")
    out.append("")
    if args:
        keep = {k: v for k, v in args.items() if k not in ("out", "shot")}
        out.append("Args: `" + " ".join(f"--{k}={v}" for k, v in keep.items()) + "`")
        out.append("")
    for p in phases:
        s = frame_summary(p)
        out.append(f"## {p.name}")
        out.append("")
        out.append(f"{s['frames']} frames in {s['wall_s']:.1f} s, frame p50 {fmt_ms(s['frame_ms'])} ms; main thread "
                   f"`_process` {fmt_ms(s['main_ms'])} ms/frame; GPU {fmt_ms(s['gpu_ms'])} ms per captured frame "
                   f"({s['gpu_frames']} captured). Bound by: **{s['bound']}**.")
        out.append("")
        out.append("| thread | threads | traced work ms/frame | busy |")
        out.append("|---|---:|---:|---:|")
        for lane in LANES:
            if lane not in p.lane_threads:
                continue
            n = len(p.lane_threads[lane])
            busy = p.lane_busy[lane]
            wall = max(p.end - p.start, 1) * n
            out.append(f"| {LANE_TITLE[lane]} | {n} | {fmt_ms(p.per_frame(lane, busy))} | {100 * busy / wall:.0f}% |")
        out.append("")
        out.append("Hot spots (self time):")
        out.append("")
        out.append("| thread | stage | self ms/frame | total ms/frame | calls/frame | mean ms | p95 ms |")
        out.append("|---|---|---:|---:|---:|---:|---:|")
        hot = sorted(((lane, path, n) for (lane, path), n in p.nodes.items() if path != (("gpu", "frame"),)),
                     key=lambda x: -p.per_frame(x[0], x[2].self))[:20]
        for lane, path, n in hot:
            stage = " / ".join(label(lane, k) for k in path)
            frames = max(p.gpu_frames, 1) if lane == "GPU" else p.frames
            out.append(f"| {LANE_TITLE[lane]} | {stage} | {fmt_ms(p.per_frame(lane, n.self))} | "
                       f"{fmt_ms(p.per_frame(lane, n.total))} | {len(n.durs) / frames:.2f} | "
                       f"{statistics.fmean(n.durs) / 1000:.3f} | {pct(n.durs, 0.95) / 1000:.3f} |")
        out.append("")
        if p.flow_total or p.flow_unfinished:
            out.append("Hand-offs (first point to last):")
            out.append("")
            out.append("| flow | finished | unfinished | p50 ms | p95 ms | max ms |")
            out.append("|---|---:|---:|---:|---:|---:|")
            for name in sorted(set(p.flow_total) | set(p.flow_unfinished)):
                lat = p.flow_total.get(name, [])
                out.append(f"| {FLOW_TITLE.get(name, name)} | {len(lat)} | {p.flow_unfinished.get(name, 0)} | "
                           f"{fmt_ms(pct(lat, 0.5) / 1000)} | {fmt_ms(pct(lat, 0.95) / 1000)} | "
                           f"{fmt_ms(max(lat, default=0) / 1000)} |")
            out.append("")
            out.append("| flow | from | to | count | p50 ms | p95 ms |")
            out.append("|---|---|---|---:|---:|---:|")
            for (name, a, b), lat in sorted(p.edges.items(), key=lambda kv: -len(kv[1]))[:15]:
                out.append(f"| {FLOW_TITLE.get(name, name)} | {_where(a)} | {_where(b)} | {len(lat)} | "
                           f"{fmt_ms(pct(lat, 0.5) / 1000)} | {fmt_ms(pct(lat, 0.95) / 1000)} |")
            out.append("")
    return "\n".join(out) + "\n"


def _where(node):
    lane, path = node
    return LANE_TITLE[lane] + (": " + " / ".join(label(lane, k) for k in path) if path else "")


# ------------------------------------------------------------------------------------------------------------- html

COL_W = 290
GAP = 130
LANE_GAP = 14
ROW_H = 22
HEAD_H = 46
PAD = 16
SYSTEM_TOKEN = {"terrain": "--c-terrain", "mesher": "--c-mesher", "discoverer": "--c-disc", "stream": "--c-stream",
                "table": "--c-table", "pool": "--c-pool", "field": "--c-field", "pages": "--c-pages",
                "render": "--c-render", "gpu": "--c-render"}
FLOW_TOKEN = {"scan": "--c-disc", "batch": "--c-mesher", "stream_job": "--c-stream", "field": "--c-field",
              "kill": "--c-pages", "compact": "--c-pages", "hidden": "--c-pages"}
# Fonts, colour tokens (light, dark) and the page frame, shared with engine_structure.py. Opens a <style> block.
THEME_HEAD = '''<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {
  --bg: #f4f6f8; --surface: #ffffff; --lane: #eaeef2; --fg: #17202a; --muted: #5b6876; --rule: #d3dae1;
  --c-terrain: #4f7cac; --c-mesher: #c0663a; --c-disc: #6c8f3d; --c-stream: #8a5aa8; --c-table: #b08a2e;
  --c-pool: #7d8a96; --c-field: #2f8f8a; --c-pages: #a0525e; --c-render: #5a6fc0;
  --ok: #2f7d4f; --warn: #a86a12; --bad: #b23b3b;
  --font-ui: "IBM Plex Sans", system-ui, sans-serif; --font-data: "IBM Plex Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #11161c; --surface: #18202a; --lane: #1d2631; --fg: #e3e9ef; --muted: #93a1b0; --rule: #2c3846;
  --c-terrain: #6f9bd0; --c-mesher: #e08a5c; --c-disc: #8fb85a; --c-stream: #b183cf; --c-table: #d4ad4c;
  --c-pool: #9aa7b3; --c-field: #4fb8b2; --c-pages: #cf7682; --c-render: #8597e6;
  --ok: #5cb57f; --warn: #d79a3c; --bad: #e06a6a; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #11161c; --surface: #18202a; --lane: #1d2631; --fg: #e3e9ef; --muted: #93a1b0; --rule: #2c3846;
  --c-terrain: #6f9bd0; --c-mesher: #e08a5c; --c-disc: #8fb85a; --c-stream: #b183cf; --c-table: #d4ad4c;
  --c-pool: #9aa7b3; --c-field: #4fb8b2; --c-pages: #cf7682; --c-render: #8597e6;
  --ok: #5cb57f; --warn: #d79a3c; --bad: #e06a6a; color-scheme: dark; }
body { background: var(--bg); color: var(--fg); font: 14px/1.45 var(--font-ui);
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  padding-block: 20px 40px; }
header { display: flex; flex-wrap: wrap; gap: 8px 24px; align-items: baseline; }
h1 { font-size: 1.35rem; font-weight: 600; margin: 0; text-wrap: balance; }
.meta { color: var(--muted); font-family: var(--font-data); font-size: 12px; }
.help { color: var(--muted); max-width: 72ch; margin: 6px 0 14px; }
'''


def esc(s):
    return html.escape(str(s), quote=True)


def _svg_lane(p, lane, x, top, pos, parts):
    """One thread's box at (x, top): header, then its stages' rows. Records each row's anchor in pos; returns the
    box's bottom."""
    n = len(p.lane_threads[lane])
    busy = p.lane_busy[lane]
    share = 100 * busy / (max(p.end - p.start, 1) * n)
    sub = f"{n} threads · " if n > 1 else ""
    unit = "per GPU frame" if lane == "GPU" else "per frame"
    rows = []
    y = top + HEAD_H + 6
    pos[(lane, ())] = (x, x + COL_W, top + 22, x)
    for depth, path, node in tree_rows(p, lane):
        ind = 12 + 14 * depth
        if path is None:
            rows.append(f'<text x="{x + ind}" y="{y + 15}" class="row-small">+{node} small</text>')
            y += ROW_H
            continue
        ms = p.per_frame(lane, node.total)
        w = (COL_W - ind - 10) * min(ms / FRAME_BUDGET_MS, 1.0)
        token = SYSTEM_TOKEN.get(path[-1][0], "--c-pool")
        frames = max(p.gpu_frames, 1) if lane == "GPU" else p.frames
        tip = (f"{' / '.join(label(lane, k) for k in path)}\n{fmt_ms(ms)} ms/frame "
               f"(self {fmt_ms(p.per_frame(lane, node.self))}), {len(node.durs) / frames:.2f} calls/frame, "
               f"mean {statistics.fmean(node.durs) / 1000:.3f} ms, p95 {pct(node.durs, 0.95) / 1000:.3f} ms")
        name = label(lane, path[-1])
        if depth == 0 and lane != "GPU":
            name = f"{path[-1][0]}.{name}"
        rows.append(f'<g class="row"><title>{esc(tip)}</title>'
                    f'<rect x="{x + ind}" y="{y + 3}" width="{max(w, 1.5):.1f}" height="{ROW_H - 6}" rx="2" '
                    f'style="fill: var({token})" class="bar"/>'
                    f'<text x="{x + ind + 4}" y="{y + 15}" class="row-name{" row-top" if depth == 0 else ""}">'
                    f'{esc(name)}</text>'
                    f'<text x="{x + COL_W - 10}" y="{y + 15}" class="row-ms">{fmt_ms(ms)}</text></g>')
        pos[(lane, path)] = (x + ind, x + COL_W, y + ROW_H / 2, x)
        y += ROW_H
    bottom = y + 6
    parts.append(f'<g class="lane"><rect x="{x}" y="{top}" width="{COL_W}" height="{bottom - top}" rx="6" '
                 f'class="lane-bg"/><text x="{x + 12}" y="{top + 20}" class="lane-title">{esc(LANE_TITLE[lane])}</text>'
                 f'<text x="{x + 12}" y="{top + 37}" class="lane-sub">{sub}{fmt_ms(p.per_frame(lane, busy))} ms '
                 f'{unit} traced · {share:.0f}% busy</text></g>')
    parts.extend(rows)
    return bottom


def svg_phase(p):
    columns = [[l for l in col if l in p.lane_threads] for col in COLUMNS]
    columns = [col for col in columns if col]
    pos = {}  # (lane, path) -> (row left, column right, y, column left)
    parts = []
    height = 0
    for c, col in enumerate(columns):
        x = PAD + c * (COL_W + GAP)
        top = PAD
        for lane in col:
            top = _svg_lane(p, lane, x, top, pos, parts) + LANE_GAP
        height = max(height, top - LANE_GAP + PAD)

    def anchor(node):
        lane, path = node
        while (lane, path) not in pos and path:
            path = path[:-1]
        return pos.get((lane, path))

    # Hand-offs: one arrow per (flow, from, to), width by count; the busiest arrow of a flow between two threads
    # carries the median latency.
    edges = sorted(p.edges.items(), key=lambda kv: -len(kv[1]))[:24]
    labelled = set()
    for (name, a, b), lat in edges:
        pa, pb = anchor(a), anchor(b)
        if pa is None or pb is None or pa == pb:
            continue
        token = FLOW_TOKEN.get(name, "--muted")
        width = 1.2 + 1.2 * math.log10(1 + len(lat) / max(p.frames, 1) * 10)
        if pa[3] == pb[3]:  # same column: a loop on the right
            x1, x2 = pa[1], pb[1]
            d = f"M{x1},{pa[2]} C{x1 + 70},{pa[2]} {x2 + 70},{pb[2]} {x2 + 4},{pb[2]}"
            lx, ly = x1 + 56, (pa[2] + pb[2]) / 2
        elif pa[3] < pb[3]:
            x1, x2 = pa[1], pb[0] - 6
            d = f"M{x1},{pa[2]} C{x1 + GAP / 2},{pa[2]} {x2 - GAP / 2},{pb[2]} {x2},{pb[2]}"
            lx, ly = (x1 + x2) / 2, (pa[2] + pb[2]) / 2
        else:
            x1, x2 = pa[0] - 6, pb[1] + 4
            d = f"M{x1},{pa[2]} C{x1 - GAP / 2},{pa[2]} {x2 + GAP / 2},{pb[2]} {x2},{pb[2]}"
            lx, ly = (x1 + x2) / 2, (pa[2] + pb[2]) / 2
        tip = f"{edge_label(name, lat)}, p95 {fmt_ms(pct(lat, 0.95) / 1000)} ms, {len(lat)} hand-offs\n{_where(a)} -> {_where(b)}"
        text = ""
        if (name, a[0], b[0]) not in labelled:
            labelled.add((name, a[0], b[0]))
            # Backward arrows (to the left) label below their line, so a pair of opposite arrows doesn't collide.
            dy = 14 if pa[3] > pb[3] else -5
            text = (f'<text x="{lx:.0f}" y="{ly + dy:.0f}" class="edge-label">'
                    f'{fmt_ms(pct(lat, 0.5) / 1000)} ms</text>')
        parts.append(f'<g class="edge"><title>{esc(tip)}</title><path d="{d}" style="stroke: var({token}); '
                     f'stroke-width: {width:.1f}" marker-end="url(#arrow)"/>{text}</g>')
    total_w = PAD * 2 + len(columns) * COL_W + (len(columns) - 1) * GAP
    return (f'<svg viewBox="0 0 {total_w} {height}" width="{total_w}" height="{height}" role="img" '
            f'aria-label="Engine map, phase {esc(p.name)}"><defs><marker id="arrow" viewBox="0 0 8 8" refX="7" '
            f'refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" class="arrow"/>'
            f'</marker></defs>' + "".join(parts) + "</svg>")


def html_page(run, args, phases):
    run_name = os.path.basename(os.path.normpath(run))
    preset = args.get("preset") or args.get("terrain") or "default"
    scenario = args.get("scenario", "load")
    tabs, panels = [], []
    for i, p in enumerate(phases):
        s = frame_summary(p)
        pid = f"phase-{i}"
        tabs.append(f'<label class="tab"><input type="radio" name="phase" value="{pid}"'
                    f'{" checked" if i == 0 else ""}><span>{esc(p.name)}</span></label>')
        hot = sorted(((lane, path, n) for (lane, path), n in p.nodes.items() if path != (("gpu", "frame"),)),
                     key=lambda x: -p.per_frame(x[0], x[2].self))[:10]
        hot_rows = "".join(
            f'<tr><td>{esc(LANE_TITLE[lane])}</td><td>{esc(" / ".join(label(lane, k) for k in path))}</td>'
            f'<td class="num">{fmt_ms(p.per_frame(lane, n.self))}</td><td class="num">{fmt_ms(p.per_frame(lane, n.total))}</td></tr>'
            for lane, path, n in hot)
        flow_rows = "".join(
            f'<tr><td>{esc(FLOW_TITLE.get(name, name))}</td><td class="num">{len(p.flow_total.get(name, []))}</td>'
            f'<td class="num">{p.flow_unfinished.get(name, 0)}</td>'
            f'<td class="num">{fmt_ms(pct(p.flow_total.get(name, []), 0.5) / 1000)}</td>'
            f'<td class="num">{fmt_ms(pct(p.flow_total.get(name, []), 0.95) / 1000)}</td></tr>'
            for name in sorted(set(p.flow_total) | set(p.flow_unfinished)))
        bound_class = "gpu" if s["bound"] == "GPU" else ("main" if s["bound"] == "main thread" else "other")
        panels.append(f'''<section class="panel" id="{pid}"{"" if i == 0 else " hidden"}>
  <div class="summary">
    <div class="stat"><span class="k">frame p50</span><span class="v">{fmt_ms(s["frame_ms"])} ms</span></div>
    <div class="stat"><span class="k">main <code>_process</code></span><span class="v">{fmt_ms(s["main_ms"])} ms</span></div>
    <div class="stat"><span class="k">GPU per frame</span><span class="v">{fmt_ms(s["gpu_ms"])} ms</span></div>
    <div class="stat"><span class="k">bound by</span><span class="v pill {bound_class}">{esc(s["bound"])}</span></div>
    <div class="stat"><span class="k">frames</span><span class="v">{s["frames"]} in {s["wall_s"]:.1f} s</span></div>
  </div>
  <div class="map">{svg_phase(p)}</div>
  <div class="tables">
    <div class="table-wrap"><h3>Hot spots (self time)</h3><table><thead><tr><th>thread</th><th>stage</th>
      <th class="num">self ms/fr</th><th class="num">total ms/fr</th></tr></thead><tbody>{hot_rows}</tbody></table></div>
    <div class="table-wrap"><h3>Hand-offs, first point to last</h3><table><thead><tr><th>flow</th><th class="num">done</th>
      <th class="num">open</th><th class="num">p50 ms</th><th class="num">p95 ms</th></tr></thead><tbody>{flow_rows}</tbody></table></div>
  </div>
</section>''')
    legend = "".join(f'<span class="key"><i style="background: var({FLOW_TOKEN[k]})"></i>{esc(v)}</span>'
                     for k, v in FLOW_TITLE.items()) + '<span class="key-note">arrow label: median hand-off latency</span>'
    return f'''<title>Engine Map · {esc(run_name)}</title>
<meta name="claude:viewport-fit-sides" content="cover">
{THEME_HEAD}
/* Layout: a header strip, one swimlane per thread drawn to the 16.7 ms frame budget, then the tables. */
.tabs {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }}
.tab input {{ position: absolute; opacity: 0; }}
.tab span {{ display: inline-block; padding: 4px 12px; border: 1px solid var(--rule); border-radius: 999px;
  font-family: var(--font-data); font-size: 12px; cursor: pointer; background: var(--surface); }}
.tab input:checked + span {{ background: var(--fg); color: var(--bg); border-color: var(--fg); }}
.tab input:focus-visible + span {{ outline: 2px solid var(--c-terrain); outline-offset: 2px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 18px; margin-bottom: 12px; font-size: 12px; color: var(--muted); }}
.key {{ display: inline-flex; align-items: center; gap: 6px; }}
.key i {{ display: inline-block; width: 18px; height: 3px; border-radius: 2px; }}
.summary {{ display: flex; flex-wrap: wrap; gap: 10px 28px; margin-bottom: 12px; }}
.stat {{ display: flex; flex-direction: column; min-width: 0; }}
.stat .k {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }}
.stat .v {{ font-family: var(--font-data); font-size: 15px; font-variant-numeric: tabular-nums; }}
.pill {{ padding: 0 8px; border-radius: 4px; font-size: 13px; }}
.pill.gpu {{ background: color-mix(in srgb, var(--c-render) 22%, transparent); }}
.pill.main {{ background: color-mix(in srgb, var(--c-terrain) 22%, transparent); }}
.pill.other {{ background: color-mix(in srgb, var(--warn) 22%, transparent); }}
.map {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; }}
.map svg {{ display: block; font-family: var(--font-data); }}
.lane-bg {{ fill: var(--lane); }}
.lane-title {{ fill: var(--fg); font: 600 13px var(--font-ui); }}
.lane-sub {{ fill: var(--muted); font-size: 11px; }}
.bar {{ opacity: .28; }}
.row:hover .bar {{ opacity: .5; }}
.row-name {{ fill: var(--fg); font-size: 11.5px; }}
.row-top {{ font-weight: 500; }}
.row-ms {{ fill: var(--fg); font-size: 11.5px; text-anchor: end; font-variant-numeric: tabular-nums; }}
.row-small {{ fill: var(--muted); font-size: 11px; font-style: italic; }}
.edge path {{ fill: none; opacity: .75; }}
.edge:hover path {{ opacity: 1; }}
.edge-label {{ fill: var(--muted); font-size: 10.5px; text-anchor: middle; paint-order: stroke; stroke: var(--surface);
  stroke-width: 3px; }}
.arrow {{ fill: var(--muted); }}
.tables {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap: 16px; margin-top: 16px; }}
.table-wrap {{ min-width: 0; overflow-x: auto; }}
h3 {{ font-size: 13px; font-weight: 600; margin: 0 0 6px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12.5px; }}
th, td {{ text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--rule); }}
th {{ color: var(--muted); font-weight: 500; }}
td.num, th.num {{ text-align: right; font-family: var(--font-data); font-variant-numeric: tabular-nums; }}
code {{ font-family: var(--font-data); font-size: .92em; }}
</style>
<header><h1>Engine map</h1><span class="meta">{esc(run_name)} · preset {esc(preset)} · scenario {esc(scenario)} · mode {esc(args.get("mode", "?"))}</span></header>
<p class="help">One lane per thread. Bars are drawn against the 16.7 ms frame budget; the number is ms per frame
(GPU: per captured GPU frame; worker pool: all its threads together). Arrows are hand-offs of the same piece of work
between threads, labelled with their median latency. Hover a row or an arrow for counts and percentiles. The
timeline behind this map is <code>trace.json</code> in the run directory: open it in ui.perfetto.dev.</p>
<div class="tabs" role="radiogroup" aria-label="Phase">{"".join(tabs)}</div>
<div class="legend">{legend}</div>
{"".join(panels)}
<script>
document.querySelectorAll('input[name="phase"]').forEach(function (input) {{
  input.addEventListener('change', function () {{
    document.querySelectorAll('.panel').forEach(function (p) {{ p.hidden = p.id !== input.value; }});
  }});
}});
</script>
'''


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run = sys.argv[1]
    out = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--out=")), run)
    trace = os.path.join(run, "trace.json")
    if not os.path.exists(trace):
        sys.exit(f"no trace.json in {run}: run tools/bench.sh --trace=1")
    names, spans, flows = read_trace(trace)
    phases = read_phases(run, spans)
    whole = analyse(names, spans, flows, phases)
    shown = [p for p in phases if p.nodes] + ([whole] if len(phases) > 1 else [])
    stats_path = os.path.join(run, "stats.json")
    args = json.load(open(stats_path)).get("args", {}) if os.path.exists(stats_path) else {}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "engine_map.md"), "w") as f:
        f.write(markdown(run, args, shown))
    with open(os.path.join(out, "engine_map.html"), "w") as f:
        f.write(html_page(run, args, shown))


if __name__ == "__main__":
    main()
