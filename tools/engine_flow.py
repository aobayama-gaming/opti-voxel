#!/usr/bin/env python3
"""What each function of the engine does, read from the code by the compiler: its loops, its branches, the calls it
makes into src/ in the order it makes them, the work it hands to another thread or to the GPU, and who calls it.
Start at a thread's entry (VoxelTerrain::_process, the discoverer's loop, a render-thread callback...) and unfold the
calls to follow one frame, one scan or one mesh batch down to the GPU dispatches.

    tools/engine_flow.py [<run dir>] [--src=<dir>] [--out=<dir>] [--jobs=N]

Parses every src/ translation unit with libclang (the Python bindings, `pip install --user clang`, against the
system's libclang.so; flags from the SConstruct: C++17, godot-cpp's include paths) and keeps, per function defined in
src/ (tests excluded):
  - loops (`for` / `while` header as written) and branches (`if` / `switch` condition), when they hold anything below;
  - calls to functions defined in src/, in evaluation order (arguments first); a virtual call lists the overrides;
  - hand-offs: a function or lambda passed to a call (callable_mp(this, &X::f) into call_on_render_thread, a lambda
    into parallel_for or the stream executor, std::thread(&X::loop)), labelled with where it runs (HANDOFF);
  - RenderingDevice work: compute dispatches (with the local variables their sizes come from), pipeline binds,
    uploads, copies, clears, readbacks;
  - trace points (OV_TRACE, Section, parallel_for names, Trace::flow), so the outline lines up with the trace.
Entry points are what nothing in src/ calls: Godot's callbacks (_process, _ready...), the hand-off targets, the methods
bound to GDScript.

With a run dir holding trace.json (bench.sh --trace=1), each traced span gets how often it ran (per frame, or per call
of the traced span around it) and the size it recorded (chunks per batch, wave sizes...).

Writes <out>/engine_flow.html (pick an entry point, unfold calls, click a line for its code and its callers) and
<out>/engine_flow.md (every entry's outline, each function unfolded once: diff it between commits). <out> defaults to
the run dir, or bench/results/flow/ without one.
"""
import html
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_map as em  # noqa: E402
import engine_structure as es  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLANG_ARGS = ["-x", "c++", "-std=c++17", "-Isrc", "-Igodot-cpp/include", "-Igodot-cpp/gen/include",
              "-Igodot-cpp/gdextension", "-Ithirdparty/doctest", "-DOPTI_VOXEL_TESTS", "-DLINUX_ENABLED",
              "-DUNIX_ENABLED", "-Wno-everything"]
# Where a function or lambda handed to these calls runs. A call not listed runs it on the same thread (a callback).
HANDOFF = {
    "call_on_render_thread": "render thread",
    "buffer_get_data_async": "render thread, when the GPU data is back",
    "texture_get_data_async": "render thread, when the GPU data is back",
    "parallel_for": "worker pool, in parallel (the caller waits)",
    "add_native_group_task": "worker pool, in parallel",
    "add_group_task": "worker pool, in parallel",
    "add_native_task": "worker pool",
    "add_task": "worker pool",
    "StreamExecutor::run": "worker pool (stream executor)",
    "Thread::start": "its own thread",
    "thread": "its own thread",
    "call_deferred": "main thread, later",
}
# RenderingDevice calls kept in the outline, and how they read.
GPU_CALLS = {
    "compute_list_dispatch": "GPU dispatch", "compute_list_dispatch_indirect": "GPU dispatch (indirect)",
    "compute_list_bind_compute_pipeline": "bind pipeline", "buffer_update": "upload", "texture_update": "upload",
    "buffer_copy": "GPU copy", "texture_copy": "GPU copy", "buffer_clear": "GPU clear", "texture_clear": "GPU clear",
    "buffer_get_data_async": "readback", "texture_get_data_async": "readback", "buffer_get_data": "readback (blocking)",
    "texture_get_data": "readback (blocking)", "submit": "submit", "sync": "sync (waits for the GPU)",
    "draw_list_draw": "draw",
}
GODOT_ENTRY = {"_process", "_physics_process", "_ready", "_enter_tree", "_exit_tree", "_notification", "_input",
               "_unhandled_input", "_init", "_draw"}
LOOPS = ("FOR_STMT", "CXX_FOR_RANGE_STMT", "WHILE_STMT", "DO_STMT")
FUNCS = ("FUNCTION_DECL", "CXX_METHOD", "CONSTRUCTOR", "DESTRUCTOR", "FUNCTION_TEMPLATE", "CONVERSION_FUNCTION")
SCOPES = ("NAMESPACE", "CLASS_DECL", "STRUCT_DECL", "CLASS_TEMPLATE", "UNEXPOSED_DECL", "LINKAGE_SPEC")
MAX_HEAD = 150
MAX_NOTE = 160
SMALL_LINES = 4  # a function this short with nothing to show is left out of the outline (getters, math helpers)
# Plumbing left out of the outline: the trace (its points show as ⏱), the error sink and core containers, logging.
PLUMBING = ("src/trace.", "src/ov/", "src/debug_log.")


def load_clang():
    import clang.cindex as ci
    if not ci.Config.loaded:
        try:
            ci.conf.lib
        except Exception:
            import glob
            libs = sorted(glob.glob("/usr/lib*/libclang.so*") + glob.glob("/usr/lib*/llvm*/lib*/libclang.so*"))
            if not libs:
                sys.exit("engine_flow: no libclang.so found (install clang-libs / libclang)")
            ci.Config.loaded = False
            ci.Config.set_library_file(libs[-1])
    return ci


def resource_dir():
    import glob
    dirs = sorted(glob.glob("/usr/lib*/clang/*")) + sorted(glob.glob("/usr/lib*/llvm*/lib*/clang/*"))
    return ["-resource-dir", dirs[-1]] if dirs else []


# ------------------------------------------------------------------------------------------------------ one TU's code

class Extractor:
    """Walks one translation unit: {usr: function record}, {class usr: [base usrs]}."""

    def __init__(self, sites):
        self.sites = sites  # {(file, line): [(system, name)]}: the trace points
        self.funcs, self.bases, self.texts, self.lambdas = {}, {}, {}, {}

    def rel(self, f):
        return os.path.relpath(os.path.abspath(f.name), ROOT) if f else ""

    def in_src(self, c):
        r = self.rel(c.location.file)
        return r.startswith("src/") and not r.startswith("src/tests/")

    def text(self, path):
        if path not in self.texts:
            self.texts[path] = open(os.path.join(ROOT, path), "rb").read()
        return self.texts[path]

    def src_of(self, c, end=None):
        """The code of c (up to `end`'s start), whitespace collapsed."""
        s = c.extent.start
        e = end.extent.start if end is not None else c.extent.end
        if not s.file or not e.file or s.file.name != e.file.name:
            return ""
        raw = self.text(self.rel(s.file))[s.offset:e.offset].decode("utf-8", "replace")
        raw = re.sub(r"//[^\n]*", "", raw)
        return re.sub(r"\s+", " ", raw).strip()

    def visit_scope(self, c):
        for k in c.get_children():
            if not self.in_src(k):
                continue
            kind = k.kind.name
            if kind in SCOPES:
                if kind in ("CLASS_DECL", "STRUCT_DECL", "CLASS_TEMPLATE"):
                    bases = [b.type.get_declaration().get_usr() for b in k.get_children() if b.kind.name == "CXX_BASE_SPECIFIER"]
                    if bases:
                        self.bases[k.get_usr()] = bases
                self.visit_scope(k)
            elif kind in FUNCS and k.is_definition():
                self.function(k)

    def function(self, c):
        usr = c.get_usr()
        if usr in self.funcs:
            return
        body = next((k for k in c.get_children() if k.kind.name == "COMPOUND_STMT"), None)
        if body is None:
            return
        lines = self.text(self.rel(c.location.file)).split(b"\n")
        if c.spelling.encode() not in lines[c.location.line - 1] and c.kind.name != "DESTRUCTOR":
            return  # written by a macro
        parent = c.semantic_parent
        cls = parent.spelling if parent is not None and parent.kind.name in ("CLASS_DECL", "STRUCT_DECL", "CLASS_TEMPLATE") else ""
        self.lambdas = {}
        self.funcs[usr] = {
            "name": f"{cls}::{c.spelling}" if cls else c.spelling,
            "short": c.spelling,
            "cls": parent.get_usr() if cls else "",
            "file": self.rel(c.location.file),
            "line": c.extent.start.line,
            "end": c.extent.end.line,
            "body": self.block(body),
        }

    # Statements and expressions -> outline nodes.
    def block(self, c):
        """A { } block's outline. A trace span lives until the block's end: what follows it goes under it."""
        out = []
        into = out
        for k in c.get_children():
            for n in self.node(k):
                into.append(n)
                if n["t"] == "span" and not n["cat"].startswith("flow:") and k.kind.name == "DECL_STMT":
                    n["c"] = []
                    into = n["c"]
        return out

    def node(self, c):
        kind = c.kind.name
        if kind == "COMPOUND_STMT":
            return self.block(c)
        if kind in LOOPS:
            kids = list(c.get_children())
            if not kids:
                return []
            body = kids[0] if kind == "DO_STMT" else kids[-1]
            if self.from_macro(c, "do" if kind == "DO_STMT" else "while" if kind == "WHILE_STMT" else "for"):
                return [n for k in kids if k is not body for n in self.node(k)]
            head = "do … while (" + self.src_of(kids[-1]) + ")" if kind == "DO_STMT" else self.src_of(c, body)
            inner = []
            for k in kids:
                if k is not body:
                    inner += self.node(k)  # the range / condition's calls, each pass
            inner += self.node(body)
            return [{"t": "loop", "head": head[:MAX_HEAD], "line": c.location.line, "c": inner}] if inner else []
        if kind == "IF_STMT":
            kids = list(c.get_children())
            has_else = len(kids) >= 3 and "else" in self.between(kids[-2], kids[-1])
            then = kids[-2] if has_else else kids[-1]
            pre = []
            for k in kids[: kids.index(then)]:
                pre += self.node(k)  # the condition's calls run first
            if self.from_macro(c, "if"):
                return pre  # OV_ERR_FAIL_COND & co: the condition, not the error path
            yes = self.node(then)
            no = self.node(kids[-1]) if has_else else []
            head = self.src_of(c, then)
            if not yes and not no:
                return pre
            return pre + [{"t": "if", "head": head[:MAX_HEAD], "line": c.location.line, "c": yes, "e": no}]
        if kind == "SWITCH_STMT":
            kids = list(c.get_children())
            pre = []
            for k in kids[:-1]:
                pre += self.node(k)
            cases, cur = [], None
            for k in kids[-1].get_children() if kids else []:
                if k.kind.name in ("CASE_STMT", "DEFAULT_STMT"):
                    label = []
                    while k is not None and k.kind.name in ("CASE_STMT", "DEFAULT_STMT"):
                        ch = list(k.get_children())
                        label.append("default" if k.kind.name == "DEFAULT_STMT" else "case " + self.src_of(ch[0]))
                        k = ch[-1] if ch else None
                    cur = {"label": ", ".join(label), "c": []}
                    cases.append(cur)
                if k is not None and cur is not None:
                    cur["c"] += self.node(k)
            cases = [x for x in cases if x["c"]]
            head = self.src_of(c, kids[-1]) if kids else "switch"
            return pre + ([{"t": "switch", "head": head[:MAX_HEAD], "line": c.location.line, "cases": cases}] if cases else [])
        if kind == "VAR_DECL":
            out = self.trace_points(c)
            kids = list(c.get_children())
            lam = next((k for k in kids if k.kind.name == "LAMBDA_EXPR"), None) or self.find_lambda(kids)
            if lam is not None:
                self.lambdas[c.get_usr()] = self.lambda_body(lam)
                return out
            for k in kids:
                out += self.node(k)
            return out
        if kind == "CALL_EXPR":
            return self.call(c)
        if kind == "LAMBDA_EXPR":
            return [{"t": "hand", "via": "", "name": "lambda", "line": c.location.line, "c": self.lambda_body(c)}]
        if kind in ("DECL_REF_EXPR", "MEMBER_REF_EXPR"):
            ref = c.referenced
            if ref is not None and ref.kind.name in FUNCS and self.in_src(ref):
                return [{"t": "hand", "via": "", "usr": ref.get_usr(), "name": self.qual(ref), "line": c.location.line}]
            if ref is not None and ref.kind.name == "VAR_DECL":
                body = self.lambdas.get(ref.get_usr())
                return [{"t": "hand", "via": "", "name": ref.spelling + " (lambda)", "line": c.location.line, "c": body}] if body else []
        out = []
        for k in c.get_children():
            out += self.node(k)
        return out

    def from_macro(self, c, keyword):
        """A statement written by a macro: its code doesn't start with its keyword."""
        return not re.match(keyword + r"\b", self.src_of(c)[:len(keyword) + 1])

    def find_lambda(self, kids):
        """A lambda under implicit wrappers (`const auto f = [&] {...};`)."""
        for k in kids:
            if k.kind.name == "LAMBDA_EXPR":
                return k
            if k.kind.name in ("UNEXPOSED_EXPR", "CALL_EXPR") and k.spelling in ("", k.type.spelling):
                hit = self.find_lambda(list(k.get_children()))
                if hit is not None:
                    return hit
        return None

    def lambda_body(self, lam):
        body = next((k for k in lam.get_children() if k.kind.name == "COMPOUND_STMT"), None)
        return self.block(body) if body is not None else []

    def between(self, a, b):
        e, s = a.extent.end, b.extent.start
        if not e.file or not s.file or e.file.name != s.file.name:
            return ""
        return self.text(self.rel(e.file))[e.offset:s.offset].decode("utf-8", "replace")

    def qual(self, ref):
        p = ref.semantic_parent
        cls = p.spelling if p is not None and p.kind.name in ("CLASS_DECL", "STRUCT_DECL", "CLASS_TEMPLATE") else ""
        return f"{cls}::{ref.spelling}" if cls else ref.spelling

    def trace_points(self, c):
        key = (self.rel(c.location.file), c.location.line)
        out = []
        for cat, name in self.sites.get(key, []):
            if out and out[-1]["cat"] == cat:
                out[-1]["alts"] = out[-1].get("alts", [out[-1]["name"]]) + [name]
            else:
                out.append({"t": "span", "cat": cat, "name": name, "line": key[1]})
        return out

    def call(self, c):
        ref = c.referenced
        kids = list(c.get_children())
        name = ref.spelling if ref is not None else c.spelling
        qual = self.qual(ref) if ref is not None else name
        out = self.trace_points(c) if name == "parallel_for" else []
        # The callee expression (an object's member, a free function) and the arguments.
        args = []
        for i, k in enumerate(kids):
            if i == 0 and k.kind.name in ("MEMBER_REF_EXPR", "DECL_REF_EXPR", "UNEXPOSED_EXPR") and self.is_callee(k, ref):
                k = self.unwrap(k)
                for g in k.get_children():
                    out += self.node(g)  # the object the method is called on: a->b()->c()
                continue
            args += self.node(k)
        local = None
        if kids and ref is not None and ref.spelling == "operator()":
            target = self.decl_ref(kids[0])
            local = self.lambdas.get(target.get_usr()) if target is not None else None
        # Functions and lambdas in the arguments run elsewhere (or later): label them by this call.
        via = HANDOFF.get(qual) or HANDOFF.get(name)
        own = ref.get_usr() if ref is not None else None
        args = [a for a in args if not (a["t"] == "hand" and a.get("usr") == own)]
        hands = [a for a in args if a["t"] == "hand"]
        rest = [a for a in args if a["t"] != "hand"]
        for h in hands:
            if not h["via"] or (h["via"] not in HANDOFF.values() and via):
                h["via"] = via or (h["via"] or f"through {qual}()")
        out += rest
        line = c.location.line
        if local is not None:
            out.append({"t": "call", "name": self.src_of(kids[0]) + " (local lambda)", "line": line, "inline": local})
            return out
        if ref is not None and ref.kind.name in FUNCS and self.in_src(ref):
            if self.rel(ref.location.file).startswith(PLUMBING) or ref.spelling.startswith(("operator", "(lambda")):
                return out + hands
            node = {"t": "call", "usr": ref.get_usr(), "name": qual, "line": line}
            if ref.kind.name == "CXX_METHOD" and ref.is_virtual_method():
                node["virt"] = [ref.semantic_parent.get_usr(), ref.spelling]
            if hands:
                node["h"] = hands
            out.append(node)
            return out
        gpu = GPU_CALLS.get(name) if ref is not None and ref.semantic_parent is not None and \
            ref.semantic_parent.spelling == "RenderingDevice" else None
        if gpu:
            node = {"t": "gpu", "what": gpu, "args": self.args_text(c)[:MAX_HEAD], "line": line,
                    "notes": self.notes(kids[2:] if name.startswith("compute_list") else kids[1:])}
            if hands:
                node["h"] = hands
            out.append(node)
            return out
        if hands and via:
            out.append({"t": "ext", "name": qual, "line": line, "h": hands})
            return out
        return out + hands

    def is_callee(self, k, ref):
        r = self.decl_ref(k)
        return r is not None and ref is not None and (r == ref or r.spelling == ref.spelling)

    def unwrap(self, k):
        while k is not None and k.kind.name == "UNEXPOSED_EXPR":
            ch = list(k.get_children())
            k = ch[0] if ch else None
        return k

    def decl_ref(self, k):
        k = self.unwrap(k)
        return k.referenced if k is not None and k.kind.name in ("DECL_REF_EXPR", "MEMBER_REF_EXPR") else None

    def args_text(self, c):
        t = self.src_of(c)
        i = t.find("(")
        return t[i:] if i >= 0 else t

    def notes(self, args):
        """`x = <init>` for the local variables an argument reads (a dispatch's group counts, a pipeline's index)."""
        out, seen = [], set()
        for a in args:
            for k in a.walk_preorder():
                if k.kind.name != "DECL_REF_EXPR":
                    continue
                d = k.referenced
                if d is None or d.kind.name != "VAR_DECL" or d.get_usr() in seen or not self.in_src(d):
                    continue
                seen.add(d.get_usr())
                init = self.src_of(d)
                if ("=" in init or "{" in init) and not re.fullmatch(r"[\w:]+ \w+ = 0", init):
                    out.append(init[:MAX_NOTE])
        return out


def clang_view():
    """src/ as clang should read it. `using namespace godot` brings godot::Material next to our namespace Material: GCC
    takes `Material::X` as ours, clang calls it ambiguous and drops the declarations (and what derives from them). Our
    namespace gets a name of the same length, so every byte offset still points into the real file."""
    out = []
    for d, dirs, files in os.walk("src"):
        dirs[:] = [x for x in dirs if x != "tests"]
        for f in files:
            path = os.path.join(d, f)
            text = open(path, encoding="utf-8", errors="replace").read()
            if "Material" in text:
                text = re.sub(r"(?<![\w:])Material::", "Mater1al::", re.sub(r"\bnamespace Material\b", "namespace Mater1al", text))
                out.append((path, text))
    return out


def trace_sites():
    """{(file, line): [(system, name)]} of src/'s trace points (engine_structure's scan); hand-off points as
    ("flow:<start|step|end>", flow)."""
    spans, flows, _ = es.scan_sources(os.path.join(ROOT, "src"))
    sites = defaultdict(list)
    for (cat, name), ss in spans.items():
        for s in ss:
            sites[(s.file, s.line)].append((cat, name))
    for flow, kinds in flows.items():
        for kind, ss in kinds.items():
            for s in ss:
                sites[(s.file, s.line)].append(("flow:" + kind.lower(), flow))
    return sites


_worker = {}


def _init_worker():
    _worker.update(ci=load_clang(), sites=trace_sites(), view=clang_view(), args=CLANG_ARGS + resource_dir())


def parse_one(path):
    """One translation unit: (path, {usr: function}, {class usr: [base usrs]}, [clang errors])."""
    if not _worker:
        _init_worker()
    tu = _worker["ci"].Index.create().parse(path, args=_worker["args"], unsaved_files=_worker["view"])
    errors = [str(d) for d in tu.diagnostics if d.severity >= 3]
    x = Extractor(_worker["sites"])
    x.visit_scope(tu.cursor)
    return path, x.funcs, x.bases, errors


# ------------------------------------------------------------------------------------------------------- the program

def merge(results):
    funcs, bases, errors = {}, {}, []
    for path, f, b, err in results:
        for usr, rec in f.items():
            funcs.setdefault(usr, rec)
        bases.update(b)
        errors += [f"{path}: {e}" for e in err]
    return funcs, bases, errors


def derived_from(bases, cls, base, seen=None):
    if cls == base:
        return True
    seen = seen or set()
    for b in bases.get(cls, []):
        if b not in seen:
            seen.add(b)
            if derived_from(bases, b, base, seen):
                return True
    return False


def resolve(funcs, bases):
    """Virtual calls get their src/ overrides; calls to short functions with nothing to show are dropped, loops and
    branches left empty too. Returns {usr: [caller usrs]}."""
    by_name = defaultdict(list)
    for usr, f in funcs.items():
        by_name[f["short"]].append(usr)
    trivial = {usr for usr, f in funcs.items() if not f["body"] and f["end"] - f["line"] < SMALL_LINES}

    def fix(nodes, caller, callers):
        out = []
        for n in nodes:
            t = n["t"]
            for key in ("c", "e", "inline"):
                if key in n:
                    n[key] = fix(n[key], caller, callers)
            if "cases" in n:
                for case in n["cases"]:
                    case["c"] = fix(case["c"], caller, callers)
            if "h" in n:
                n["h"] = fix(n["h"], caller, callers)
            if t == "call" and "usr" in n:
                if "virt" in n:
                    cls, short = n.pop("virt")
                    over = [u for u in by_name[short] if u != n["usr"] and funcs[u]["cls"] and derived_from(bases, funcs[u]["cls"], cls)]
                    if over:
                        n["over"] = over
                if n["usr"] not in funcs and "over" not in n:
                    n.pop("usr")
                    if "h" not in n:
                        continue  # declared in src/, defined nowhere we parse
                if n.get("usr") in trivial and "h" not in n and "over" not in n:
                    continue
                for u in [n.get("usr")] + n.get("over", []):
                    if u:
                        callers[u].add(caller)
            if t == "hand" and n.get("usr"):
                if n["usr"] not in funcs:
                    n.pop("usr")
                else:
                    callers[n["usr"]].add(caller)
            if t in ("loop", "if") and not n["c"] and not n.get("e"):
                continue
            if t == "switch":
                n["cases"] = [case for case in n["cases"] if case["c"]]
                if not n["cases"]:
                    continue
            out.append(n)
        return out

    callers = defaultdict(set)
    for usr, f in funcs.items():
        f["body"] = fix(f["body"], usr, callers)
    return {u: sorted(c) for u, c in callers.items()}


def handoff_targets(funcs):
    """{usr: where it runs} for every function handed to another thread."""
    out = {}

    def walk(nodes):
        for n in nodes:
            if n["t"] == "hand" and n.get("usr"):
                out.setdefault(n["usr"], n["via"])
            for key in ("c", "e", "inline", "h"):
                walk(n.get(key, []))
            for case in n.get("cases", []):
                walk(case["c"])
    for f in funcs.values():
        walk(f["body"])
    return out


def reach_sizes(funcs):
    """{usr: how many functions it reaches}: the entries that run the most code come first."""
    out = {}

    def calls(nodes, acc):
        for n in nodes:
            for u in [n.get("usr")] + n.get("over", []):
                if u in funcs and u not in acc:
                    acc.add(u)
                    calls(funcs[u]["body"], acc)
            for key in ("c", "e", "inline", "h"):
                calls(n.get(key, []), acc)
            for case in n.get("cases", []):
                calls(case["c"], acc)
    for usr, f in funcs.items():
        acc = set()
        calls(f["body"], acc)
        out[usr] = len(acc)
    return out


def entries(funcs, callers, hands):
    """[(group, [usr])]: Godot's callbacks, the hand-off targets by thread, the rest nothing in src/ calls."""
    godot, bound, by_via = [], [], defaultdict(list)
    for usr, f in funcs.items():
        if hands.get(usr) in HANDOFF.values():
            by_via[hands[usr]].append(usr)
        elif f["short"] in GODOT_ENTRY and f["body"]:
            godot.append(usr)
        elif usr not in callers and f["body"] and not f["short"].startswith(("~", "operator", "_bind_methods")) and \
                f["short"] != f["name"].split("::")[0]:
            bound.append(usr)
    size = reach_sizes(funcs)
    key = lambda u: (-size[u], funcs[u]["name"])
    rank = {name: i for i, name in enumerate(("_process", "_physics_process", "_ready", "_enter_tree", "_exit_tree"))}
    cls_size = defaultdict(int)
    for u in godot:
        cls_size[funcs[u]["cls"]] = max(cls_size[funcs[u]["cls"]], size[u])
    godot.sort(key=lambda u: (-cls_size[funcs[u]["cls"]], rank.get(funcs[u]["short"], 9), funcs[u]["name"]))
    groups = [("Main thread: Godot calls these", godot)]
    for via in sorted(by_via, key=lambda v: (not v.startswith("render"), v)):
        groups.append((f"Handed off: {via}", sorted(by_via[via], key=key)))
    groups.append(("Not called from src/ (GDScript, tools, tests)", sorted(bound, key=key)))
    return [g for g in groups if g[1]]


# -------------------------------------------------------------------------------------------------------- the trace

def trace_stats(run):
    """{(cat, name): {"per_frame", "parent": (cat, name), "per_parent", "n": median arg}} from <run>/trace.json."""
    path = os.path.join(run, "trace.json")
    if not run or not os.path.exists(path):
        return {}
    names, spans, flows = em.read_trace(path)
    frames = 0
    calls, args, under = defaultdict(int), defaultdict(list), defaultdict(lambda: defaultdict(int))
    for tid, events in spans.items():
        by_seq = {e[5]: e for e in events if e[5]}
        for ts, dur, cat, name, n, seq, parent in events:
            key = (cat, name)
            if key == ("terrain", "process"):
                frames += 1
            calls[key] += 1
            if n:
                args[key].append(n)
            p = by_seq.get(parent)
            if p is not None:
                under[key][(p[2], p[3])] += 1
    out = {}
    for key, c in calls.items():
        rec = {"calls": c, "per_frame": c / max(frames, 1)}
        if under[key]:
            pkey, pc = max(under[key].items(), key=lambda kv: kv[1])
            if pc >= 0.8 * c and calls.get(pkey):
                rec["parent"], rec["per_parent"] = pkey, c / calls[pkey]
        if args[key]:
            rec["n"] = statistics.median(args[key])
            rec["n_max"] = max(args[key])
        out[f"{key[0]}/{key[1]}"] = rec
    return out


# -------------------------------------------------------------------------------------------------------- the pages

def rate(r):
    if r >= 10:
        return f"×{r:.0f}"
    if r >= 1.5:
        return f"×{r:.1f}"
    if r >= 0.95:
        return "×1"
    return f"{r * 100:.0f}%" if r >= 0.1 else f"1 in {1 / r:.0f}" if r > 0 else "never"


def span_text(n, stats, encl):
    """A trace point and how often it ran: per run of the traced span around it in the outline when the trace has it
    there, per frame otherwise; the size it records (span.set_arg)."""
    if n["cat"].startswith("flow:"):
        return f"{n['cat'][5:]} of hand-off “{n['name']}”"
    parts = []
    for name in n.get("alts", [n["name"]]):
        s = stats.get(f"{n['cat']}/{name}")
        text = name
        if s:
            if s.get("parent") and list(s["parent"]) == encl:
                text += f" {rate(s['per_parent'])} per {s['parent'][1]}"
            else:
                text += f" {rate(s['per_frame'])} per frame"
            if "n" in s:
                text += f", size {s['n']:.0f} (max {s['n_max']})"
        elif stats:
            text += " did not run"
        parts.append(text)
    return f"traced {n['cat']}/" + " · ".join(parts)


def md_outline(funcs, stats, usr, depth, seen, lines, indent=""):
    f = funcs[usr]

    def emit(nodes, ind, encl=None):
        for n in nodes:
            t = n["t"]
            if t == "loop":
                lines.append(f"{ind}- ↻ `{n['head']}`")
                emit(n["c"], ind + "  ", encl)
            elif t == "if":
                lines.append(f"{ind}- ◇ `{n['head']}`")
                emit(n["c"], ind + "  ", encl)
                if n["e"]:
                    lines.append(f"{ind}- ◇ else")
                    emit(n["e"], ind + "  ", encl)
            elif t == "switch":
                lines.append(f"{ind}- ◇ `{n['head']}`")
                for case in n["cases"]:
                    lines.append(f"{ind}  - {case['label']}")
                    emit(case["c"], ind + "    ", encl)
            elif t == "span":
                lines.append(f"{ind}- ⏱ {span_text(n, stats, encl)}")
                emit(n.get("c", []), ind + "  ", [n["cat"], n["name"]])
            elif t == "gpu":
                notes = f" — {'; '.join(n['notes'])}" if n["notes"] else ""
                lines.append(f"{ind}- ▣ {n['what']} `{n['args']}`{notes}")
                emit(n.get("h", []), ind + "  ", encl)
            elif t == "hand":
                target = n.get("name", "lambda")
                arrow = "⇢" if n["via"] in HANDOFF.values() else "↳"
                lines.append(f"{ind}- {arrow} {n['via'] or 'callback'}: {target}")
                if n.get("usr"):
                    call(n["usr"], ind + "  ", encl=encl)
                emit(n.get("c", []), ind + "  ", encl)
            elif t == "ext":
                lines.append(f"{ind}- {n['name']}()")
                emit(n.get("h", []), ind + "  ", encl)
            elif t == "call":
                over = n.get("over", [])
                lines.append(f"{ind}- **{n['name']}**" + (f" (virtual: {', '.join(funcs[o]['name'] for o in over)})" if over else ""))
                emit(n.get("h", []), ind + "  ", encl)
                if "inline" in n:
                    emit(n["inline"], ind + "  ", encl)
                for u in ([n["usr"]] if n.get("usr") in funcs and funcs[n["usr"]]["body"] else []) + over:
                    call(u, ind + "  ", label=bool(over), encl=encl)

    def call(u, ind, label=False, encl=None):
        if u in seen:
            if funcs[u]["body"]:
                lines.append(f"{ind}- *(unfolded above: {funcs[u]['name']})*")
            return
        if depth + len(ind) // 2 > 40:
            return
        seen.add(u)
        if label:
            lines.append(f"{ind}- {funcs[u]['name']}: `{funcs[u]['file']}:{funcs[u]['line']}`")
            ind += "  "
        emit(funcs[u]["body"], ind, encl)

    seen.add(usr)
    lines.append(f"{indent}`{f['file']}:{f['line']}`")
    lines.append("")
    emit(f["body"], indent)


def markdown(funcs, groups, stats, run):
    lines = ["# Engine flow", "",
             "Generated by `tools/engine_flow.py` from the code of `src/` (libclang)" +
             (f" and the trace of `{os.path.basename(os.path.normpath(run))}`" if stats else "") + ". "
             "Per entry point: loops ↻, branches ◇, calls into src/ (bold, unfolded in place the first time), "
             "hand-offs ⇢ to another thread, RenderingDevice work ▣, trace points ⏱.", ""]
    seen = set()
    for title, usrs in groups:
        lines += [f"## {title}", ""]
        done = [u for u in usrs if u in seen]
        if done:
            lines += ["Unfolded above: " + ", ".join(funcs[u]["name"] for u in done) + ".", ""]
        if title.startswith("Not called"):
            lines += [f"- {funcs[u]['name']} `{funcs[u]['file']}:{funcs[u]['line']}`" for u in usrs if u not in seen]
            continue
        for u in usrs:
            if u in seen:
                continue
            lines += [f"### {funcs[u]['name']}", ""]
            md_outline(funcs, stats, u, 0, seen, lines)
            lines.append("")
    return "\n".join(lines) + "\n"


def sources(funcs):
    """{file: text} of the files holding the functions (the page shows each function's code)."""
    out = {}
    for f in funcs.values():
        if f["file"] not in out:
            out[f["file"]] = open(os.path.join(ROOT, f["file"]), encoding="utf-8", errors="replace").read()
    return out


def html_page(funcs, callers, groups, stats, run):
    data = {
        "funcs": {u: {k: f[k] for k in ("name", "file", "line", "end", "body")} for u, f in funcs.items()},
        "callers": callers,
        "threads": sorted(set(HANDOFF.values())),
        "groups": groups,
        "stats": stats,
        "files": sources(funcs),
    }
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    title = "Engine flow" + (f": {os.path.basename(os.path.normpath(run))}" if stats else "")
    return PAGE.replace("@TITLE@", html.escape(title)).replace("@THEME@", em.THEME_HEAD).replace("@DATA@", blob)


PAGE = """<title>@TITLE@</title>
<meta name="claude:viewport-fit-sides" content="cover">
@THEME@
<style>
:root{--hl:#e9e4f7;--accent:#5b45b0;--k-loop:#a35f00;--k-if:#6f45ad;--k-gpu:#b8322a;--k-hand:#16707f}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--hl:#2d2a3d;--accent:#b0a2ee;--k-loop:#f0b35a;--k-if:#b99be6;--k-gpu:#ff8a7a;--k-hand:#6cc8d8}}
:root[data-theme=dark]{--hl:#2d2a3d;--accent:#b0a2ee;--k-loop:#f0b35a;--k-if:#b99be6;--k-gpu:#ff8a7a;--k-hand:#6cc8d8}
.wrap{display:grid;grid-template-columns:minmax(220px,300px) minmax(0,1fr) minmax(0,44%);gap:0;height:calc(100vh - 92px)}
@media (max-width:900px){.wrap{grid-template-columns:1fr;height:auto}}
.pane{overflow:auto;padding:10px 14px;min-width:0;border-right:1px solid var(--rule)}
#entries h3{font:600 11px/1.3 var(--font-ui);letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:14px 0 4px}
#entries a{display:block;padding:2px 6px;border-radius:4px;color:var(--fg);text-decoration:none;font:12px/1.5 var(--font-data);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#entries a:hover,#entries a.on{background:var(--hl)}
#find{width:100%;box-sizing:border-box;padding:5px 8px;border:1px solid var(--rule);border-radius:5px;background:var(--bg);color:var(--fg);font:12px var(--font-data)}
.tools{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px;align-items:center}
.tools button{font:12px var(--font-ui);padding:3px 9px;border:1px solid var(--rule);border-radius:5px;background:var(--surface);color:var(--fg);cursor:pointer}
#tree{font:13px/1.55 var(--font-data)}
#tree ul{list-style:none;margin:0;padding-left:18px;border-left:1px dotted var(--rule)}
#tree>ul{padding-left:0;border:0}
.r{cursor:pointer;padding:0 4px;border-radius:3px;white-space:pre-wrap;word-break:break-word}
.r:hover{background:var(--hl)}.r.sel{outline:1px solid var(--accent)}
.tw{display:inline-block;width:14px;color:var(--muted)}
.call{color:var(--fg);font-weight:600}.loop{color:var(--k-loop)}.if{color:var(--k-if)}.gpu{color:var(--k-gpu)}.hand{color:var(--k-hand);font-weight:600}
.span{color:var(--muted);font-style:italic}.cb{color:var(--muted)}.ext{color:var(--muted)}
.loc{color:var(--muted);font-weight:400;font-size:11px}
.note{color:var(--muted);font-size:11px;display:block;padding-left:18px}
#code h2{font:600 15px var(--font-data);margin:4px 0}
#code pre{font:12px/1.45 var(--font-data);margin:0;overflow-x:auto}
#code .ln{display:inline-block;width:40px;color:var(--muted);text-align:right;padding-right:8px;user-select:none}
#code .hit{background:var(--hl)}
#code .callers a{font:12px var(--font-data);margin-right:10px;color:var(--accent);cursor:pointer}
.legend{font:12px var(--font-ui);color:var(--muted)}
</style>
<header><h1>@TITLE@</h1><div class="meta">Read from <code>src/</code> by the compiler (libclang): pick an entry point, unfold the calls. <span class="legend"><b class="loop">↻ loop</b> · <b class="if">◇ branch</b> · <b class="call">call</b> · <b class="hand">⇢ hand-off to another thread</b> · <span class="cb">↳ callback</span> · <b class="gpu">▣ GPU work</b> · <span class="span">⏱ trace point</span></span></div></header>
<div class="wrap">
<div class="pane" id="entries"><input id="find" placeholder="find a function…"><div id="list"></div></div>
<div class="pane"><div class="tools"><span class="legend">unfold calls:</span><button data-d="1">1 deep</button><button data-d="2">2</button><button data-d="4">4</button><button data-d="8">8</button><button data-d="0">fold all</button></div><div id="tree"></div></div>
<div class="pane" id="code"><p class="legend">Click a line to see its code and who calls the function.</p></div>
</div>
<script>
const D = @DATA@;
const F = D.funcs, S = D.stats, THREADS = new Set(D.threads);
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function rate(r){ return r>=10?'×'+r.toFixed(0):r>=1.5?'×'+r.toFixed(1):r>=0.95?'×1':r>=0.1?Math.round(r*100)+'%':r>0?'1 in '+Math.round(1/r):'never'; }
function spanText(n, en){
  if (n.cat.startsWith('flow:')) return n.cat.slice(5)+' of hand-off “'+n.name+'”';
  const parts = (n.alts || [n.name]).map(name => {
    const s = S[n.cat+'/'+name]; let t = name;
    if (s){ t += s.parent && en && s.parent[0] === en[0] && s.parent[1] === en[1] ? ' '+rate(s.per_parent)+' per '+s.parent[1] : ' '+rate(s.per_frame)+' per frame';
            if (s.n !== undefined) t += ', size '+Math.round(s.n)+' (max '+s.n_max+')'; }
    else if (Object.keys(S).length) t += ' did not run';
    return t; });
  return 'traced '+n.cat+'/'+parts.join(' · ');
}
function loc(file, line){ return '<span class="loc"> '+esc(file.replace(/^src\\//,''))+':'+line+'</span>'; }
// One outline line: its label and what unfolds under it.
function item(n, file, stack, en){
  const li = document.createElement('li'); let html = '', kids = null, sel = {file, line:n.line};
  switch (n.t){
    case 'loop': html = '<span class="loop">↻ '+esc(n.head)+'</span>'; kids = () => list(n.c, file, stack, en); break;
    case 'if': html = '<span class="if">◇ '+esc(n.head)+'</span>'; kids = () => { const u = list(n.c, file, stack, en); if (n.e.length){ const e = document.createElement('li'); e.innerHTML = '<div class="r"><span class="tw"></span><span class="if">◇ else</span></div>'; e.appendChild(list(n.e, file, stack, en)); u.appendChild(e);} return u; }; break;
    case 'switch': html = '<span class="if">◇ '+esc(n.head)+'</span>'; kids = () => { const u = document.createElement('ul'); for (const c of n.cases){ const e = document.createElement('li'); e.innerHTML = '<div class="r"><span class="tw"></span><span class="if">'+esc(c.label)+'</span></div>'; e.appendChild(list(c.c, file, stack, en)); u.appendChild(e);} return u; }; break;
    case 'span': html = '<span class="span">⏱ '+esc(spanText(n, en))+'</span>'; if (n.c && n.c.length) kids = () => list(n.c, file, stack, [n.cat, n.name]); break;
    case 'gpu': html = '<span class="gpu">▣ '+esc(n.what)+'</span> '+esc(n.args)+(n.notes.length?'<span class="note">'+n.notes.map(esc).join('<br>')+'</span>':''); if (n.h) kids = () => list(n.h, file, stack, en); break;
    case 'ext': html = '<span class="ext">'+esc(n.name)+'()</span>'; if (n.h) kids = () => list(n.h, file, stack, en); break;
    case 'hand': {
      html = '<span class="'+(THREADS.has(n.via) ? 'hand' : 'cb')+'">'+(THREADS.has(n.via) ? '⇢ ' : '↳ ')+esc(n.via || 'callback')+':</span> <span class="call">'+esc(n.name || 'lambda')+'</span>';
      if (n.usr && F[n.usr]) { html += loc(F[n.usr].file, F[n.usr].line); sel = {usr:n.usr}; kids = () => body(n.usr, stack, en); }
      else if (n.c) kids = () => list(n.c, file, stack, en);
      break; }
    case 'call': {
      const f = n.usr && F[n.usr];
      html = '<span class="call">'+esc(n.name)+'</span>' + (f ? loc(f.file, f.line) : '');
      if (n.over) html += ' <span class="loc">virtual → '+n.over.map(o => esc(F[o].name)).join(', ')+'</span>';
      if (f) sel = {usr:n.usr, file, line:n.line};
      const parts = [];
      if (n.h) parts.push(() => list(n.h, file, stack, en));
      if (n.inline) parts.push(() => list(n.inline, file, stack, en));
      if (f && f.body.length && !stack.includes(n.usr)) parts.push(() => body(n.usr, stack, en));
      for (const o of n.over || []) parts.push(() => { const u = document.createElement('ul'); u.appendChild(item({t:'call', usr:o, name:F[o].name, line:F[o].line}, F[o].file, stack, en)); return u; });
      if (f && stack.includes(n.usr)) html += ' <span class="loc">(recursion)</span>';
      if (parts.length) kids = () => { const u = document.createElement('ul'); for (const p of parts) for (const c of [...p().children]) u.appendChild(c); return u; };
      break; }
  }
  const row = document.createElement('div'); row.className = 'r';
  row.innerHTML = '<span class="tw">'+(kids ? '▸' : '')+'</span>'+html;
  li.appendChild(row);
  if (kids){ li._kids = kids; li._fn = n.t === 'call' || (n.t === 'hand' && !!n.usr); }
  row.onclick = e => { if (kids && (e.target.classList.contains('tw') || li._open === undefined || !e.altKey)) toggle(li); show(sel, n); document.querySelectorAll('.r.sel').forEach(x => x.classList.remove('sel')); row.classList.add('sel'); };
  return li;
}
function toggle(li, force){
  const open = force !== undefined ? force : !li._open;
  if (open === !!li._open) return;
  li._open = open;
  li.firstChild.firstChild.textContent = open ? '▾' : '▸';
  if (open){ li._ul = li._kids(); li.appendChild(li._ul); } else if (li._ul){ li._ul.remove(); li._ul = null; }
}
function list(nodes, file, stack, en){ const u = document.createElement('ul'); for (const n of nodes) u.appendChild(item(n, file, stack, en)); return u; }
function body(usr, stack, en){ return list(F[usr].body, F[usr].file, stack.concat([usr]), en); }
function unfold(depth){
  const walk = (ul, d) => { for (const li of ul.children){ if (!li._kids) continue; toggle(li, !li._fn || d < depth); if (li._ul) walk(li._ul, d + (li._fn ? 1 : 0)); } };
  walk(document.querySelector('#tree>ul'), 0);
}
let current = null;
function open(usr){
  current = usr;
  const t = document.getElementById('tree'); t.innerHTML = '';
  const u = document.createElement('ul'); u.appendChild(item({t:'call', usr, name:F[usr].name, line:F[usr].line}, F[usr].file, [], null));
  t.appendChild(u); unfold(2);
  document.querySelectorAll('#list a').forEach(a => a.classList.toggle('on', a.dataset.u === usr));
  show({usr}, null);
  try { location.hash = ''; } catch(e) {}
}
function show(sel, n){
  const pane = document.getElementById('code');
  const usr = sel.usr, file = usr ? F[usr].file : sel.file;
  const text = D.files[file]; if (!text) return;
  const lines = text.split('\\n');
  const from = usr ? F[usr].line : Math.max(1, sel.line - 12), to = usr ? F[usr].end : Math.min(lines.length, sel.line + 12);
  let top = Math.max(1, from - 6);
  while (top < from && !/^\\s*\\/\\//.test(lines[top - 1]) && top < from) top++;
  for (let k = from - 1; k >= 1 && /^\\s*\\/\\//.test(lines[k - 1]); k--) top = k;
  const hit = sel.line && (!usr || sel.file === file) ? sel.line : -1;
  let h = '<h2>'+esc(usr ? F[usr].name : file)+'</h2><div class="loc">'+esc(file)+':'+from+'</div>';
  if (usr){
    const cs = D.callers[usr] || [];
    h += '<p class="callers"><b class="legend">called by:</b> '+(cs.length ? cs.map(c => '<a data-u="'+c+'">'+esc(F[c].name)+'</a>').join('') : '<span class="legend">nothing in src/ (an entry point)</span>')+'</p>';
  }
  h += '<pre>'; for (let i = top; i <= to; i++) h += '<span class="'+(i === hit ? 'hit' : '')+'"><span class="ln">'+i+'</span>'+esc(lines[i - 1])+'</span>\\n';
  pane.innerHTML = h + '</pre>';
  pane.querySelectorAll('.callers a').forEach(a => a.onclick = () => open(a.dataset.u));
  const el = pane.querySelector('.hit'); if (el) el.scrollIntoView({block:'center'});
}
function entries(filter){
  const l = document.getElementById('list'); let h = '';
  if (filter){
    const q = filter.toLowerCase();
    const hits = Object.keys(F).filter(u => F[u].name.toLowerCase().includes(q)).sort((a, b) => F[a].name.localeCompare(F[b].name)).slice(0, 200);
    h += '<h3>matches</h3>' + hits.map(u => '<a data-u="'+u+'" title="'+esc(F[u].file)+'">'+esc(F[u].name)+'</a>').join('');
  } else for (const [title, us] of D.groups) h += '<h3>'+esc(title)+'</h3>' + us.map(u => '<a data-u="'+u+'" title="'+esc(F[u].file)+'">'+esc(F[u].name)+'</a>').join('');
  l.innerHTML = h;
  l.querySelectorAll('a').forEach(a => { a.onclick = () => open(a.dataset.u); a.classList.toggle('on', a.dataset.u === current); });
}
document.getElementById('find').oninput = e => entries(e.target.value.trim());
document.querySelectorAll('.tools button').forEach(b => b.onclick = () => unfold(+b.dataset.d));
entries('');
const first = Object.keys(F).find(u => F[u].name === 'VoxelTerrain::_process') || D.groups[0][1][0];
open(first);
</script>
"""


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    run = pos[0] if pos else ""
    out = opt.get("out", run or os.path.join(ROOT, "bench", "results", "flow"))
    src = os.path.abspath(opt.get("src", os.path.join(ROOT, "src")))
    os.chdir(ROOT)
    units = sorted(os.path.relpath(os.path.join(d, f), ROOT) for d, dirs, files in os.walk(src)
                   for f in files if f.endswith(".cpp") and "/tests" not in os.path.relpath(d, ROOT) + "/")
    try:
        load_clang()
    except ImportError:
        sys.exit("engine_flow: needs libclang's Python bindings: pip install --user clang")
    with Pool(int(opt.get("jobs", os.cpu_count() or 4)), initializer=_init_worker) as pool:
        results = pool.map(parse_one, units)
    funcs, bases, errors = merge(results)
    for e in errors[:10]:
        print("engine_flow: clang:", e, file=sys.stderr)
    callers = resolve(funcs, bases)
    groups = entries(funcs, callers, handoff_targets(funcs))
    stats = trace_stats(run)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "engine_flow.md"), "w") as f:
        f.write(markdown(funcs, groups, stats, run))
    with open(os.path.join(out, "engine_flow.html"), "w") as f:
        f.write(html_page(funcs, callers, groups, stats, run))
    print(f"engine_flow: {len(units)} files, {len(funcs)} functions, {len(errors)} clang errors -> {out}/engine_flow.html")


if __name__ == "__main__":
    main()
