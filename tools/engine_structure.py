#!/usr/bin/env python3
"""How the engine works, drawn from what a traced run executed: per thread, which stage runs inside which and in what
order, how often, and how work hands off between threads; each stage linked to the code that traces it. No timings
(engine_map.py has them): this is the structure.

    tools/engine_structure.py <run dir> [--src=<dir>] [--out=<dir>]

Reads <run>/trace.json (bench.sh --trace=1; the call tree comes from the parent ids the engine records, src/trace.hpp)
and scans <src> (default: src/ next to this tool) for the trace points: OV_TRACE / Trace::Scope / Section spans,
parallel_for names, Trace::flow hand-offs and the mesher's GPU timestamps. Each stage gets its file:line, the function
that holds it and the comments there (above the trace point, above and at the top of the function, on its declaration
in the header). Writes <out>/engine_structure.html (a board of nested blocks per thread, hand-offs as arrows; click a
block for its code) and <out>/engine_structure.md (the same tree as text: diff it between runs). <out> defaults to the
run dir.

A stage only shows up when it is traced and ran in the run: what isn't instrumented, or didn't run, isn't drawn. The
.md lists the trace points of src/ that no run reached.
"""
import html
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_map as em  # noqa: E402

MAX_COMMENT_LINES = 6
MIN_EDGE_SHARE = 0.02  # hand-offs under this share of their flow's points are left out of the board
# Godot's GPU timestamps include region markers ("> Render Viewport 0", "< Render Canvas", "vp_begin_<id>", the
# "(untracked)" gap): the interval ending at one is no pass of its own.
GPU_MARKER = (">", "<", "vp_begin_", "vp_end_", "(untracked)")

# --------------------------------------------------------------------------------------------------------- the code

SPAN_RE = re.compile(r'(?:OV_TRACE\(|Trace::Scope\s+\w+\()\s*"(\w+)"\s*,\s*(.*?)\)\s*;')
SECTION_RE = re.compile(r'\bSection\s+\w+\([^,]+,\s*"(\w+)"\)')
FLOW_RE = re.compile(r'Trace::(?:gpu_)?flow\(\s*Trace::Kind::FLOW_(\w+)\s*,\s*"(\w+)"')
PFOR_RE = re.compile(r'\bparallel_for\([^,]+,\s*"([^"]+)"')
STAMP_RE = re.compile(r'capture_timestamp\(')
FUNC_RE = re.compile(r'([~\w:]+)\s*\(')
NOT_FUNC = ("static const", "static constexpr", "constexpr", "namespace", "using", "template", "struct", "class", "enum",
            "inline thread_local", "#", "//", "}", "{")


class Site:
    def __init__(self, file, line, function, comments, text=""):
        self.file, self.line, self.function, self.comments, self.text = file, line, function, comments, text

    def ref(self):
        return f"{self.file}:{self.line}"

    def as_json(self):
        return {"ref": self.ref(), "function": self.function, "comments": self.comments}


def _comment_above(lines, i):
    """The // block right above line i (0-based), top to bottom."""
    out = []
    k = i - 1
    while k >= 0 and lines[k].strip().startswith("//"):
        out.append(lines[k].strip()[2:].strip())
        k -= 1
    return out[::-1]


def _function_at(lines, i):
    """(line index, name) of the function holding line i: the nearest line above at column 0 that opens one."""
    for j in range(i, -1, -1):
        line = lines[j]
        if not line or line[0].isspace() or line.startswith(NOT_FUNC):
            continue
        m = FUNC_RE.search(line)
        if m and not line.rstrip().endswith(";"):
            return j, m.group(1)
    return -1, ""


def _leading_comment(lines, j):
    """The // block at the top of the function opened at line j."""
    k = j
    while k < len(lines) and not lines[k].rstrip().endswith("{"):
        k += 1
    out = []
    k += 1
    while k < len(lines) and lines[k].strip().startswith("//"):
        out.append(lines[k].strip()[2:].strip())
        k += 1
    return out


def _declaration_comment(header_lines, name):
    """The comment on the declaration of `name` (a method's last component) in its header: above it, or after it."""
    if not header_lines:
        return []
    short = name.split("::")[-1]
    pat = re.compile(r"[\s\*&]" + re.escape(short) + r"\(")
    for i, line in enumerate(header_lines):
        if pat.search(line) and not line.strip().startswith("//"):
            above = _comment_above(header_lines, i)
            trailing = line.split("//", 1)[1].strip() if "//" in line else ""
            return above + ([trailing] if trailing else [])
    return []


def _site(rel, lines, header_lines, i):
    j, function = _function_at(lines, i)
    comments = []
    for block in (_comment_above(lines, i), _declaration_comment(header_lines, function) if function else [],
                  _comment_above(lines, j) if j >= 0 else [], _leading_comment(lines, j) if j >= 0 else []):
        for c in block:
            if c and c not in comments:
                comments.append(c)
    return Site(rel, i + 1, function, comments[:MAX_COMMENT_LINES], lines[i])


def _names(expr, text):
    """The span names a name expression can take: its literals, or a static array's when it indexes one."""
    names = re.findall(r'"([^"]+)"', expr)
    for array in re.findall(r"(\w+)\[", expr):
        arr = re.search(re.escape(array) + r"\[\]\s*=\s*\{([^}]*)\}", text)
        if arr:
            names += re.findall(r'"([^"]+)"', arr.group(1))
    return names


def scan_sources(src):
    """({(system, name): [Site]}, {flow: {kind: [Site]}}, [Site] of the GPU timestamps)."""
    spans, flows, stamps = defaultdict(list), defaultdict(lambda: defaultdict(list)), []
    root = os.path.dirname(os.path.abspath(src))
    for dirpath, dirnames, files in os.walk(src):
        dirnames[:] = [d for d in dirnames if d != "tests"]
        for f in sorted(files):
            if not f.endswith((".cpp", ".hpp", ".h")) or f.startswith("trace."):
                continue
            path = os.path.join(dirpath, f)
            rel = os.path.relpath(path, root)
            text = open(path, encoding="utf-8", errors="replace").read()
            lines = text.split("\n")
            header = os.path.splitext(path)[0] + ".hpp"
            header_lines = open(header, encoding="utf-8").read().split("\n") if header != path and os.path.exists(header) else []
            for i, line in enumerate(lines):
                if line.strip().startswith("//"):
                    continue
                for m in SPAN_RE.finditer(line):
                    for name in _names(m.group(2), text):
                        spans[(m.group(1), name)].append(_site(rel, lines, header_lines, i))
                for m in SECTION_RE.finditer(line):
                    spans[("terrain", m.group(1))].append(_site(rel, lines, header_lines, i))
                for m in PFOR_RE.finditer(line):
                    spans[("pool", m.group(1))].append(_site(rel, lines, header_lines, i))
                for m in FLOW_RE.finditer(line):
                    flows[m.group(2)][m.group(1)].append(_site(rel, lines, header_lines, i))
                if STAMP_RE.search(line) and "vx" not in f:
                    stamps.append(_site(rel, lines, header_lines, i))
    return spans, flows, stamps


# ------------------------------------------------------------------------------------------------------ the run's tree

class Block:
    def __init__(self, lane, path, node, parent_calls, frames):
        self.lane, self.path, self.node = lane, path, node
        self.calls = len(node.durs)
        self.rate = self.calls / parent_calls if parent_calls else self.calls / max(frames, 1)
        self.children = []

    @property
    def key(self):
        return self.lane + "|" + "/".join(f"{c}:{n}" for c, n in self.path)

    def label(self):
        return em.label(self.lane, self.path[-1])


def build_trees(whole):
    """{lane: [root Block]}, children in the order they start within their parent: a child that started before another
    in more of their parent's calls than after it goes first (ranked by those wins and losses: a stage that only runs
    in some calls keeps its place)."""
    frames = {lane: (max(whole.gpu_frames, 1) if lane == "GPU" else whole.frames) for lane in em.LANES}
    blocks = {}
    for (lane, path), node in sorted(whole.nodes.items(), key=lambda kv: len(kv[0][1])):
        if lane == "GPU" and path[-1][1].startswith(GPU_MARKER):
            continue
        if len(path) > 1 and (lane, path[:-1]) not in blocks:
            continue  # under a marker
        parent = blocks.get((lane, path[:-1]))
        b = Block(lane, path, node, parent.calls if parent else 0, frames[lane])
        blocks[(lane, path)] = b
        if parent:
            parent.children.append(b)
    roots = defaultdict(list)
    for (lane, path), b in blocks.items():
        counts = whole.sibling_order.get((lane, path), {})
        keys = [c.path[-1] for c in b.children]
        score = {k: sum(counts.get((k, o), 0) > counts.get((o, k), 0) for o in keys) -
                 sum(counts.get((o, k), 0) > counts.get((k, o), 0) for o in keys) for k in keys}
        b.children.sort(key=lambda c: (-score[c.path[-1]], -c.calls))
        if len(path) == 1:
            roots[lane].append(b)
    for lane in roots:
        roots[lane].sort(key=lambda b: -b.calls)
    return roots, blocks


def rate_text(b):
    """How often a block runs: per frame at the top, per call of its parent below."""
    per = "frame" if len(b.path) == 1 else "call"
    if len(b.path) == 1 and b.lane == "GPU":
        per = "GPU frame"
    r = b.rate
    if r >= 1.5:
        return f"×{r:.0f} per {per}" if r >= 10 else f"×{r:.1f} per {per}"
    if r >= 0.95:
        return f"every {per}"
    if r >= 0.1:
        return f"{r * 100:.0f}% of {per}s"
    return f"1 in {1 / r:.0f} {per}s" if r > 0 else ""


def gpu_site(b, stamps):
    """A mesher GPU pass vx_<step> is the interval ending at the timestamp "vx<batch>_<step>": the line capturing it
    ("_<step>" spelled out, or its letter with a computed number: "_c" + group, "_p" + pass)."""
    name = b.path[-1][1]
    if not name.startswith("vx_"):
        return []
    step = name[3:]
    mesher = [s for s in stamps if "GPU_direct_mesher" in s.file]
    return ([s for s in mesher if f'"_{step}"' in s.text] or
            [s for s in mesher if f'"_{step.rstrip("0123456789")}")' in s.text.replace(" ", "")])


def edges_of(whole):
    """[(flow, src key, dst key, count)], the rare ones left out."""
    total = defaultdict(int)
    for (name, a, b), lats in whole.edges.items():
        total[name] += len(lats)
    out = []
    for (name, a, b), lats in sorted(whole.edges.items(), key=lambda kv: -len(kv[1])):
        if not a[1] or not b[1] or len(lats) < MIN_EDGE_SHARE * total[name]:
            continue
        ka = a[0] + "|" + "/".join(f"{c}:{n}" for c, n in a[1])
        kb = b[0] + "|" + "/".join(f"{c}:{n}" for c, n in b[1])
        out.append((name, ka, kb, len(lats)))
    return out


# ------------------------------------------------------------------------------------------------------------- report

def markdown(run, args, roots, edges, spans, flows, reached):
    out = [f"# Engine structure: {os.path.basename(os.path.normpath(run))}", "",
           "Generated by `tools/engine_structure.py` from `trace.json` (`bench.sh --trace=1`) and the trace points in "
           "`src/`. Per thread, each stage nested in the one it runs in, in execution order, with how often it runs.",
           "", "Args: `" + " ".join(f"--{k}={v}" for k, v in sorted(args.items()) if k != "out") + "`", ""]

    def walk(b, depth):
        sites = spans.get(b.path[-1], [])
        where = ", ".join(s.ref() for s in sites[:2])
        out.append(f"{'  ' * depth}- **{b.label()}** ({b.path[-1][0]}) {rate_text(b)}" + (f" · `{where}`" if where else ""))
        for c in b.children:
            walk(c, depth + 1)

    for lane in em.LANES:
        if not roots.get(lane):
            continue
        out += [f"## {em.LANE_TITLE[lane]}", ""]
        for b in roots[lane]:
            walk(b, 0)
        out.append("")
    out += ["## Hand-offs", "", "| flow | from | to | count |", "|---|---|---|---:|"]
    def where(key):
        lane, path = key.split("|", 1)
        return em.LANE_TITLE[lane] + ": " + " › ".join(p.split(":", 1)[1] for p in path.split("/"))

    for name, a, b, n in edges:
        out.append(f"| {em.FLOW_TITLE.get(name, name)} | {where(a)} | {where(b)} | {n} |")
    missing = sorted(k for k in spans if k not in reached)
    if missing:
        out += ["", "## Traced in src/, not reached by this run", ""]
        out += [f"- {c} / {n} · `{spans[(c, n)][0].ref()}`" for c, n in missing]
    unseen = sorted(f for f in flows if f not in {e[0] for e in edges})
    if unseen:
        out += ["", "Hand-offs traced in src/ and not drawn: " + ", ".join(unseen)]
    return "\n".join(out) + "\n"


def html_page(run, args, roots, blocks, edges, spans, flows, stamps):
    esc = em.esc
    run_name = os.path.basename(os.path.normpath(run))
    info = {}

    def block_html(b, depth):
        token = em.SYSTEM_TOKEN.get(b.path[-1][0], "--c-pool")
        sites = spans.get(b.path[-1], []) or gpu_site(b, stamps)
        info[b.key] = {"name": b.label(), "system": b.path[-1][0], "thread": em.LANE_TITLE[b.lane],
                       "rate": rate_text(b), "calls": b.calls, "path": " › ".join(em.label(b.lane, k) for k in b.path),
                       "sites": [s.as_json() for s in sites]}
        kids = "".join(block_html(c, depth + 1) for c in b.children)
        code = f'<span class="fn">{esc(sites[0].function)}</span>' if sites and sites[0].function else ""
        return (f'<div class="blk" data-k="{esc(b.key)}" style="--sys: var({token})" tabindex="0" role="button">'
                f'<div class="hd"><span class="nm">{esc(b.label())}</span><span class="rt">{esc(rate_text(b))}</span></div>'
                f'{code}{f"<div class=kids>{kids}</div>" if kids else ""}</div>')

    cols = []
    for column in em.COLUMNS:
        lanes = []
        for lane in column:
            if not roots.get(lane):
                continue
            body = "".join(block_html(b, 0) for b in roots[lane])
            lanes.append(f'<section class="lane"><h2>{esc(em.LANE_TITLE[lane])}</h2>{body}</section>')
        if lanes:
            cols.append(f'<div class="col">{"".join(lanes)}</div>')
    flow_info = {name: {"title": em.FLOW_TITLE.get(name, name),
                        "sites": {k: [s.as_json() for s in v] for k, v in kinds.items()}} for name, kinds in flows.items()}
    legend = "".join(f'<span class="key"><i style="background: var({em.FLOW_TOKEN.get(k, "--muted")})"></i>'
                     f'{esc(em.FLOW_TITLE.get(k, k))}</span>' for k in sorted({e[0] for e in edges}))
    data = json.dumps({"blocks": info, "edges": edges, "flows": flow_info,
                       "tokens": {k: em.FLOW_TOKEN.get(k, "--muted") for k in flow_info}}).replace("</", "<\\/")
    return f'''<title>Engine Structure · {esc(run_name)}</title>
<meta name="claude:viewport-fit-sides" content="cover">
{em.THEME_HEAD}
/* Layout: one column per thread group; each thread a stack of nested blocks (a stage inside the stage that runs it,
   in execution order); hand-offs drawn over the board; the clicked block's code below. */
.help {{ color: var(--muted); max-width: 78ch; margin: 6px 0 12px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 18px; margin-bottom: 12px; font-size: 12px; color: var(--muted); }}
.key i {{ display: inline-block; width: 14px; height: 3px; vertical-align: middle; margin-right: 6px; border-radius: 2px; }}
.scroll {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; }}
.board {{ position: relative; display: flex; gap: 120px; padding: 16px 16px 16px 90px; width: max-content;
  align-items: flex-start; }}
.col {{ display: flex; flex-direction: column; gap: 18px; width: 300px; }}
.lane {{ background: var(--lane); border-radius: 8px; padding: 10px; display: flex; flex-direction: column; gap: 6px; }}
.lane h2 {{ font-size: 13px; font-weight: 600; margin: 0 0 4px; }}
.blk {{ background: var(--surface); border: 1px solid var(--rule); border-left: 3px solid var(--sys); border-radius: 5px;
  padding: 4px 6px 5px; cursor: pointer; display: flex; flex-direction: column; gap: 2px; }}
.blk:focus-visible {{ outline: 2px solid var(--sys); outline-offset: 1px; }}
.blk.sel {{ box-shadow: 0 0 0 2px var(--sys); }}
.blk.dim {{ opacity: .45; }}
.hd {{ display: flex; justify-content: space-between; gap: 8px; align-items: baseline; }}
.nm {{ font-family: var(--font-data); font-size: 12px; font-weight: 500; overflow-wrap: anywhere; }}
.rt {{ color: var(--muted); font-size: 11px; white-space: nowrap; }}
.fn {{ color: var(--muted); font-family: var(--font-data); font-size: 10.5px; overflow-wrap: anywhere; }}
.kids {{ display: flex; flex-direction: column; gap: 4px; margin-top: 3px; padding-left: 8px;
  border-left: 1px dashed var(--rule); }}
svg.wires {{ position: absolute; inset: 0; pointer-events: none; overflow: visible; }}
svg.wires path {{ fill: none; stroke-width: 1.6; opacity: .75; }}
svg.wires text {{ font: 10.5px var(--font-data); fill: var(--muted); paint-order: stroke; stroke: var(--surface);
  stroke-width: 3px; }}
#detail {{ margin-top: 14px; background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 12px 14px;
  min-height: 60px; }}
#detail h3 {{ margin: 0 0 2px; font: 600 14px var(--font-data); }}
#detail .sub {{ color: var(--muted); font-size: 12px; margin-bottom: 8px; }}
#detail .site {{ border-top: 1px solid var(--rule); padding: 8px 0 4px; }}
#detail .ref {{ font-family: var(--font-data); font-size: 12px; }}
#detail .cm {{ color: var(--muted); font-size: 12.5px; margin: 4px 0 0; max-width: 90ch; }}
#detail .flows {{ font-size: 12.5px; margin-top: 6px; }}
</style>
<header><h1>Engine structure</h1><span class="meta">{esc(run_name)} · preset {esc(args.get("preset", "?"))} · scenario {esc(args.get("scenario", "?"))}</span></header>
<p class="help">What the engine ran, by thread: each block is a traced stage, nested in the stage that runs it, children
in the order they start. The note on a block is how often it runs (per frame at the top, per call of its parent below).
Arrows are hand-offs of one piece of work between threads. Click a block for the code that traces it: file:line, its
function and the comments there. Drawn by <code>tools/engine_structure.py</code> from <code>trace.json</code> and
<code>src/</code>; only traced stages that ran appear.</p>
<div class="legend">{legend}</div>
<div class="scroll"><div class="board" id="board">{"".join(cols)}<svg class="wires" id="wires"></svg></div></div>
<div id="detail" aria-live="polite"><span class="sub">Click a block to see its code.</span></div>
<script>
const DATA = {data};
const board = document.getElementById('board'), wires = document.getElementById('wires');
const el = k => board.querySelector('.blk[data-k="' + CSS.escape(k) + '"]');
function head(b) {{ return b.querySelector(':scope > .hd'); }}
function draw() {{
  const r0 = board.getBoundingClientRect();
  wires.setAttribute('width', board.scrollWidth); wires.setAttribute('height', board.scrollHeight);
  let svg = '';
  const seen = new Set();
  for (const [flow, a, b, n] of DATA.edges) {{
    const ea = el(a), eb = el(b);
    if (!ea || !eb || ea === eb) continue;
    const ra = head(ea).getBoundingClientRect(), rb = head(eb).getBoundingClientRect();
    const ya = ra.top + ra.height / 2 - r0.top, yb = rb.top + rb.height / 2 - r0.top;
    const right = rb.left >= ra.right - 1;
    const sameCol = Math.abs(ra.left - rb.left) < 200 && !right && rb.right > ra.left;
    let xa, xb, d;
    if (sameCol) {{  // stacked lanes: loop out on the left
      xa = ra.left - r0.left - 6; xb = rb.left - r0.left - 6;
      d = `M${{xa}},${{ya}} C${{xa - 60}},${{ya}} ${{xb - 60}},${{yb}} ${{xb}},${{yb}}`;
    }} else if (right) {{
      xa = ra.right - r0.left; xb = rb.left - r0.left;
      const m = (xb - xa) / 2; d = `M${{xa}},${{ya}} C${{xa + m}},${{ya}} ${{xb - m}},${{yb}} ${{xb}},${{yb}}`;
    }} else {{
      xa = ra.left - r0.left; xb = rb.right - r0.left;
      const m = (xa - xb) / 2; d = `M${{xa}},${{ya}} C${{xa - m}},${{ya}} ${{xb + m}},${{yb}} ${{xb}},${{yb}}`;
    }}
    const color = `var(${{DATA.tokens[flow] || '--muted'}})`;
    svg += `<path d="${{d}}" stroke="${{color}}" marker-end="url(#ah-${{flow}})" data-a="${{a}}" data-b="${{b}}"/>`;
    const lbl = flow + '|' + a.split('|')[0] + '|' + b.split('|')[0];
    if (!seen.has(lbl)) {{
      seen.add(lbl);
      const lx = sameCol ? Math.min(xa, xb) - 64 : (xa + xb) / 2, ly = (ya + yb) / 2 - 4;
      svg += `<text x="${{lx}}" y="${{ly}}" text-anchor="middle">${{(DATA.flows[flow] || {{}}).title || flow}}</text>`;
    }}
  }}
  let defs = '<defs>';
  for (const f in DATA.tokens) defs += `<marker id="ah-${{f}}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(${{DATA.tokens[f]}})" stroke="none"/></marker>`;
  wires.innerHTML = defs + '</defs>' + svg;
}}
function esc(s) {{ return String(s).replace(/[&<>"]/g, c => ({{'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}})[c]); }}
function show(k) {{
  const b = DATA.blocks[k];
  board.querySelectorAll('.blk.sel').forEach(x => x.classList.remove('sel'));
  el(k).classList.add('sel');
  let h = `<h3>${{esc(b.name)}}</h3><div class="sub">${{esc(b.thread)}} · ${{esc(b.path)}} · ${{esc(b.rate)}} · ${{b.calls}} calls in the run</div>`;
  const outs = DATA.edges.filter(e => e[1] === k), ins = DATA.edges.filter(e => e[2] === k);
  const name = key => DATA.blocks[key] ? DATA.blocks[key].name + ' (' + DATA.blocks[key].thread + ')' : key;
  if (outs.length || ins.length) {{
    h += '<div class="flows">';
    for (const e of ins) h += `<div>← ${{esc((DATA.flows[e[0]] || {{}}).title || e[0])}} from <b>${{esc(name(e[1]))}}</b> (${{e[3]}})</div>`;
    for (const e of outs) h += `<div>→ ${{esc((DATA.flows[e[0]] || {{}}).title || e[0])}} to <b>${{esc(name(e[2]))}}</b> (${{e[3]}})</div>`;
    h += '</div>';
  }}
  if (!b.sites.length) h += '<div class="site cm">No trace point found in src/ for this name (a renderer pass of Godot\\'s).</div>';
  for (const s of b.sites) {{
    h += `<div class="site"><div class="ref">${{esc(s.ref)}} · <b>${{esc(s.function)}}</b></div>`;
    if (s.comments.length) h += `<p class="cm">${{s.comments.map(esc).join('<br>')}}</p>`;
    h += '</div>';
  }}
  document.getElementById('detail').innerHTML = h;
}}
board.addEventListener('click', ev => {{ const b = ev.target.closest('.blk'); if (b) {{ ev.stopPropagation(); show(b.dataset.k); }} }});
board.addEventListener('keydown', ev => {{
  if ((ev.key === 'Enter' || ev.key === ' ') && ev.target.classList.contains('blk')) {{ ev.preventDefault(); show(ev.target.dataset.k); }}
}});
window.addEventListener('resize', draw);
if (document.fonts) document.fonts.ready.then(draw);
draw();
</script>
'''


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run = sys.argv[1]
    opt = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a)
    out = opt.get("out", run)
    src = opt.get("src", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    trace = os.path.join(run, "trace.json")
    if not os.path.exists(trace):
        sys.exit(f"no trace.json in {run}: run tools/bench.sh --trace=1")
    names, span_events, flow_events = em.read_trace(trace)
    phases = em.read_phases(run, span_events)
    whole = em.analyse(names, span_events, flow_events, phases)
    roots, blocks = build_trees(whole)
    edges = edges_of(whole)
    spans, flows, stamps = scan_sources(src)
    reached = {path[-1] for (lane, path) in whole.nodes}
    stats_path = os.path.join(run, "stats.json")
    args = json.load(open(stats_path)).get("args", {}) if os.path.exists(stats_path) else {}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "engine_structure.md"), "w") as f:
        f.write(markdown(run, args, roots, edges, spans, flows, reached))
    with open(os.path.join(out, "engine_structure.html"), "w") as f:
        f.write(html_page(run, args, roots, blocks, edges, spans, flows, stamps))


if __name__ == "__main__":
    main()
