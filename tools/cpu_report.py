#!/usr/bin/env python3
"""Markdown report of a CPU-sampled bench run (tools/cpu_profile.sh): where each thread's time goes.

    tools/cpu_report.py <run dir> [top, default 40]

Reads <run>/perf.script.txt (perf script -F comm,pid,tid,ip,sym,dso: on-CPU samples) or <run>/stacks.txt
(cpu_profile.sh's eu-stack loop: every thread every 0.5 s, wall clock; samples whose top frames wait in the kernel are
counted as waiting and left out of the tables; perf's are all kept). Tables: samples per thread, the top self symbols (leaf frames) of the
main thread (tid == pid) and of the other threads together, the top inclusive symbols (a symbol counted once per
sample it appears in) of every thread. Unnamed frames read "[unknown] <module>".
"""
import collections
import os
import re
import sys

# Leaf frames of a thread blocked in the kernel (eu-stack samples every thread, running or not).
WAITING = re.compile(r"futex|__internal_syscall_cancel|epoll_wait|\bpoll\b|__poll|nanosleep|cond_wait|cond_timedwait|"
                     r"sem_wait|sem_timedwait|\bselect\b|pselect|ppoll|syscall_cancel|__GI___wait|sched_yield")

Sample = collections.namedtuple("Sample", "pid tid comm frames")  # frames: leaf first


def frame_name(sym, module):
    sym = sym.strip()
    if sym and sym not in ("[unknown]", "??"):
        return sym
    return f"[unknown] {os.path.basename(module.strip()) or '?'}"


def parse_perf_script(lines):
    """perf script -F comm,pid,tid,ip,sym,dso: a header 'comm pid/tid', then one indented line per frame."""
    header = re.compile(r"^(\S.*?)\s+(\d+)/(\d+)\s*$")
    frame = re.compile(r"^\s+[0-9a-f]+\s+(.*?)\s*\(([^()]*)\)\s*$")
    current = None
    for line in lines:
        h = header.match(line)
        if h:
            if current and current.frames:
                yield current
            current = Sample(int(h.group(2)), int(h.group(3)), h.group(1).strip(), [])
            continue
        f = frame.match(line)
        if f and current is not None:
            current.frames.append(frame_name(f.group(1), f.group(2)))
    if current and current.frames:
        yield current


def parse_eu_stack(lines):
    """cpu_profile.sh's stacks.txt: '=== SAMPLE', 'COMM tid name' lines, then eu-stack -m output."""
    frame = re.compile(r"^#\d+\s+0x[0-9a-f]+\s*(.*)$")
    comms, pid, current = {}, 0, None
    for line in lines:
        line = line.rstrip("\n")
        if line.startswith("=== SAMPLE"):
            comms = {}
        elif line.startswith("COMM "):
            parts = line.split(" ", 2)
            if parts[1].isdigit():  # not the glob of a process that exited meanwhile
                comms[int(parts[1])] = parts[2] if len(parts) > 2 else "?"
        elif line.startswith("PID "):
            pid = int(line.split()[1])
        elif line.startswith("TID "):
            if current and current.frames:
                yield current
            tid = int(line[4:].rstrip(":"))
            current = Sample(pid, tid, comms.get(tid, "?"), [])
        else:
            f = frame.match(line)
            if f and current is not None:
                sym, _, module = f.group(1).rpartition(" - ") if " - " in f.group(1) else (f.group(1), "", "")
                current.frames.append(frame_name(sym, module))
    if current and current.frames:
        yield current


def is_waiting(sample):
    # libc's and the kernel's wrappers only (a C++ name like InstanceStream::poll() is work).
    return any(WAITING.search(f) and "::" not in f for f in sample.frames[:3])


def table(title, counter, total, inclusive=None, top=40, extra=None, label="self %"):
    out = [f"### {title}", ""]
    if not total:
        return out + ["n/a (no samples)", ""]
    cols = f"| symbol | samples | {label} |" + (" inclusive % |" if inclusive is not None else "") + (" threads |" if extra else "")
    out += [cols, "|" + "---|" * (cols.count("|") - 1)]
    for sym, n in counter.most_common(top):
        row = f"| `{sym[:110]}` | {n} | {100.0 * n / total:.1f} |"
        if inclusive is not None:
            row += f" {100.0 * inclusive.get(sym, n) / total:.1f} |"
        if extra:
            row += f" {extra(sym)} |"
        out.append(row)
    return out + [""]


def aggregate(samples):
    self_c, incl_c = collections.Counter(), collections.Counter()
    for s in samples:
        self_c[s.frames[0]] += 1
        for f in set(s.frames):
            incl_c[f] += 1
    return self_c, incl_c


def thread_summary(samples, waiting):
    by = collections.defaultdict(lambda: [set(), 0, 0])
    for s in samples:
        by[s.comm][0].add(s.tid)
        by[s.comm][1] += 1
    for s in waiting:
        by[s.comm][0].add(s.tid)
        by[s.comm][2] += 1
    total = len(samples)
    out = ["### Samples per thread name", "", "| thread (comm) | threads | busy samples | % of busy | waiting samples |",
           "|---|---|---|---|---|"]
    for comm, (tids, busy, wait) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        out.append(f"| {comm} | {len(tids)} | {busy} | {100.0 * busy / max(total, 1):.1f} | {wait} |")
    return out + [""]


def report(run, top=40):
    script, stacks = os.path.join(run, "perf.script.txt"), os.path.join(run, "stacks.txt")
    if os.path.exists(script):
        source = "perf (on-CPU samples)"
        samples = list(parse_perf_script(open(script, errors="replace")))
    elif os.path.exists(stacks):
        source = "eu-stack (wall-clock samples of every thread)"
        samples = list(parse_eu_stack(open(stacks, errors="replace")))
    else:
        return f"# CPU: {os.path.basename(run)}\n\nn/a: no perf.script.txt or stacks.txt\n"
    on_cpu = source.startswith("perf")  # perf samples running threads only
    waiting = [] if on_cpu else [s for s in samples if is_waiting(s)]
    busy = samples if on_cpu else [s for s in samples if not is_waiting(s)]
    main = [s for s in busy if s.tid == s.pid]
    others = [s for s in busy if s.tid != s.pid]
    out = [f"# CPU: {os.path.basename(run)}", "",
           f"Sampler: {source}. {len(samples)} samples, {len(busy)} busy, {len(waiting)} waiting (left out). "
           "Self = the leaf frame; inclusive = the symbol anywhere in the stack (once per sample).", ""]
    out += thread_summary(busy, waiting)
    m_self, m_incl = aggregate(main)
    out += table("Main thread: top self symbols", m_self, len(main), m_incl, top)
    o_self, _ = aggregate(others)
    comms = collections.defaultdict(collections.Counter)
    for s in others:
        comms[s.frames[0]][s.comm] += 1
    out += table("Other threads (render, worker pool, ...): top self symbols", o_self, len(others), None, top,
                 extra=lambda sym: ", ".join(f"{c} {n}" for c, n in comms[sym].most_common(3)))
    _, a_incl = aggregate(busy)
    out += table("All threads: top inclusive symbols", a_incl, len(busy), None, top, label="inclusive %")
    return "\n".join(out)


def main():
    run = sys.argv[1]
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    print(report(run, top))


if __name__ == "__main__":
    main()
