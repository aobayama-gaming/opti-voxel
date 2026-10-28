#!/usr/bin/env python3
"""Biome lab: a local web page to edit climate tables, preview them and fly them (tools/biome_lab.sh starts it).

    python3 tools/biome_lab/server.py [--port=8765] [--no-open]

Python's standard library only, bound to 127.0.0.1. The page (web/) edits the table in memory; this server:
- lists, reads and saves the tables under adapter/godot/project/climate/ (saved like match.py / world.py write them,
  tools/landforms/table_json.py);
- previews a table through tools/climate_report.sh (headless, CPU; the engine's own parse and validate, so a refused
  table shows the engine's message), one job at a time, a check replaced by a newer one;
- runs the landform tools (world.py, match.py) as jobs;
- launches the player on a copy of the table (a windowed Godot: only when the page's button is pressed).
Its files (drafts, runs, launch copies and logs) go to bench/results/biome-lab/.
"""
import itertools
import json
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PROJECT = ROOT / "adapter/godot/project"
CLIMATE = PROJECT / "climate"
LAB = ROOT / "bench/results/biome-lab"
WEB = HERE / "web"
GODOT = os.environ.get("GODOT", str(Path.home() / ".local/bin/godot"))  # as the other tools (climate_report.sh)
sys.path.insert(0, str(ROOT / "tools/landforms"))
from table_json import WORLD_TABLE, dump_table  # noqa: E402

KEEP_RUNS = 30
KEEP_JOBS = 200
JOB_IDS = itertools.count(1)  # jobs, runs and drafts: unique names within a second


def table_path(p_rel: str) -> Path:
    """A table under CLIMATE from its path relative to it; ValueError for anything else."""
    path = (CLIMATE / p_rel).resolve()
    if not path.is_relative_to(CLIMATE) or path.suffix != ".json":
        raise ValueError(f"not a table under {CLIMATE.relative_to(ROOT)}: {p_rel}")
    return path


def stem_of(p_rel: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", Path(p_rel or "untitled").with_suffix("").as_posix())


def same_data(p_a, p_b) -> bool:
    """JSON equality as the engine reads it (Python's == has True == 1 == 1.0)."""
    return json.dumps(p_a, sort_keys=True) == json.dumps(p_b, sort_keys=True)


def write_draft(p_dir: Path, p_rel: str, p_data) -> Path:
    path = p_dir / f"{stem_of(p_rel)}-{time.strftime('%Y%m%d-%H%M%S')}-{next(JOB_IDS)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_table(p_data) + "\n")
    return path


def sites() -> list:
    """The real heightmap sites of tools/landforms/sites.json (match.py's first argument)."""
    return list(json.loads((ROOT / "tools/landforms/sites.json").read_text())["sites"])


def presets() -> list:
    """The preset names of debug/presets.gd (shared by the player and the bench)."""
    names = re.findall(r'^\s+"([A-Za-z0-9_]+)": \[', (PROJECT / "debug/presets.gd").read_text(), re.M)
    if not names:
        raise ValueError("no preset found in debug/presets.gd (its layout changed: update presets() in server.py)")
    return names


def format_drift() -> list:
    """The keys the engine's parser accepts (its keys() lists, src/climate_parse.cpp) that web/schema.js has no form
    for, and the reverse: the page lays its forms out from schema.js, so an engine key it misses is only reachable
    through the JSON views. The page shows the list."""
    engine = set()
    for call in re.findall(r"keys\([^{;]*\{([^}]*)\}", (ROOT / "src/climate_parse.cpp").read_text()):
        engine.update(re.findall(r'"([a-z_0-9]+)"', call))
    schema = set(re.findall(r"^\s+([a-z_0-9]+): \{ kind:", (WEB / "schema.js").read_text(), re.M)) - {"comment"}
    engine.discard("comment")  # every object takes one (ignored)
    return [f"the engine accepts '{k}', schema.js has no form for it" for k in sorted(engine - schema)] + \
           [f"schema.js offers '{k}', the engine's parser doesn't list it" for k in sorted(schema - engine)]


def flag_args(p_values: dict, p_pattern: str) -> list:
    """--key=value arguments of a tool (the tool refuses a key it doesn't know), each value matching p_pattern."""
    out = []
    for k, v in p_values.items():
        v = str(v).strip()
        if not v:
            continue
        if not re.fullmatch(r"[a-z_]+", k) or not re.fullmatch(p_pattern, v):
            raise ValueError(f"--{k}={v}")
        out.append(f"--{k}={v}")
    return out


# --- jobs: one runs at a time (climate reports load the CPU; the landform tools rewrite tables) ---


class Job:
    def __init__(self, p_kind: str, p_cmd: list, p_out: Path | None = None):
        self.id = str(next(JOB_IDS))
        self.kind, self.cmd, self.out = p_kind, p_cmd, p_out
        self.state, self.code, self.lines, self.result = "queued", None, [], None
        self.proc = None

    @property
    def output(self) -> str:
        return "".join(self.lines)

    def stop(self):
        """The job's whole process group: climate_report.sh runs Godot under timeout, a child of its own."""
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)

    def view(self) -> dict:
        return {"id": self.id, "kind": self.kind, "state": self.state, "code": self.code, "output": self.output[-20000:],
                "result": self.result, "cmd": " ".join(str(c) for c in self.cmd)}


JOBS: dict[str, Job] = {}
QUEUE: list[Job] = []
LOCK = threading.Condition()


def submit(p_job: Job) -> Job:
    with LOCK:
        if p_job.kind == "check":  # only the latest check matters: drop the queued ones, stop the running one
            for old in [j for j in QUEUE if j.kind == "check"]:
                QUEUE.remove(old)
                old.state = "dropped"
            for old in JOBS.values():
                if old.kind == "check" and old.state == "running":
                    old.stop()
        JOBS[p_job.id] = p_job
        for old in list(JOBS)[:-KEEP_JOBS]:
            del JOBS[old]
        QUEUE.append(p_job)
        LOCK.notify()
    return p_job


def report_result(p_job: Job) -> dict:
    """What the page shows of a climate report run: climate_report.sh's verdict (exit 3: refused, the CLIMATE line it
    prints carries the engine's message), the report, the images."""
    global LAST_OK_RUN
    out = p_job.out
    error = None
    if p_job.code == 0:
        LAST_OK_RUN = out
    elif p_job.code == 3:
        line = next((x for x in p_job.lines if x.startswith("CLIMATE: ")), "")
        try:
            error = json.loads(line[len("CLIMATE: "):]).get("error")
        except json.JSONDecodeError:
            error = p_job.output
    elif p_job.code != 0:
        error = f"climate_report.sh exit {p_job.code}: " + "".join(p_job.lines[-15:])
    report = (out / "report.md").read_text() if (out / "report.md").exists() else ""
    heights = json.loads((out / "heights.json").read_text()) if (out / "heights.json").exists() else None
    files = sorted(f.name for f in out.glob("*.png"))
    return {"run": out.name, "ok": p_job.code == 0, "refused": p_job.code == 3, "error": error, "report": report,
            "heights": heights, "images": files}


LAST_OK_RUN = None  # the page keeps showing the last accepted run's maps while the table is refused


def prune_runs():
    runs = sorted((LAB / "runs").glob("*"), key=lambda p: p.stat().st_mtime)
    for old in runs[:-KEEP_RUNS]:
        if old != LAST_OK_RUN:
            shutil.rmtree(old, ignore_errors=True)


def worker():
    while True:
        with LOCK:
            while not QUEUE:
                LOCK.wait()
            job = QUEUE.pop(0)
            try:  # started under the lock: a running job always has its process, for submit() to stop
                job.proc = subprocess.Popen([str(c) for c in job.cmd], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, errors="replace", start_new_session=True)
            except OSError as e:  # a missing tool (uv, godot): the job fails, the server goes on
                job.lines.append(str(e))
                job.code, job.state = -1, "done"
                continue
            job.state = "running"
        try:
            for line in job.proc.stdout:
                job.lines.append(line)
            job.code = job.proc.wait()
            if job.out:
                job.result = report_result(job)
                prune_runs()
        except Exception as e:  # a report that can't be read: the job fails, the server goes on
            job.lines.append(f"\n{e}")
            job.code = -1
        job.state = "done"


def report_job(p_kind: str, p_rel: str, p_data, p_params: dict) -> Job:
    out = LAB / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{next(JOB_IDS)}-{p_kind}"
    draft = write_draft(out, p_rel, p_data)  # in the run: pruned with it
    cmd = [ROOT / "tools/climate_report.sh", "--no-build", f"--out={out}", f"--climate={draft}"]
    return submit(Job(p_kind, cmd + flag_args(p_params, r"[-0-9.,]+"), out))


def tool_job(p_body: dict) -> Job:
    """The landform tools (tools/landforms/, uv for numpy): world.py recomposes world.json, match.py fits a table."""
    tool = p_body.get("tool")
    if tool == "world":
        return submit(Job("world", ["uv", "run", ROOT / "tools/landforms/world.py"]))
    if tool == "match":
        table = table_path(p_body["path"])
        site = p_body["site"]
        if site not in sites():
            raise ValueError(f"unknown site {site}")
        cmd = ["uv", "run", ROOT / "tools/landforms/match.py", site, table, "--no-build"]
        cmd += flag_args(p_body.get("flags", {}), r"[-A-Za-z0-9_.,:]+") + (["--hypso"] if p_body.get("hypso") else [])
        return submit(Job("match", cmd))
    raise ValueError(f"unknown tool {tool}")


def launch(p_body: dict) -> dict:
    """The player on a copy of the table (windowed: the user's own run)."""
    table = write_draft(LAB / "launch", p_body.get("path", ""), p_body["data"])
    engine = str(p_body.get("engine_args", "")).split()
    game = str(p_body.get("game_args", "")).split()
    scene = str(p_body.get("scene", "")).strip()
    cmd = [GODOT, "--path", str(PROJECT)] + engine + ([scene] if scene else []) + ["--"] + game + [f"--climate={table}"]
    log = table.with_suffix(".log")
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    return {"pid": proc.pid, "log": str(log.relative_to(ROOT)), "cmd": " ".join(cmd)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, p_format, *p_args):
        pass

    def send(self, p_code: int, p_body: bytes, p_type: str = "application/json"):
        self.send_response(p_code)
        self.send_header("Content-Type", p_type)
        self.send_header("Content-Length", str(len(p_body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(p_body)

    def json(self, p_value, p_code: int = 200):
        self.send(p_code, json.dumps(p_value).encode())

    def file(self, p_path: Path):
        if not p_path.is_file():
            return self.json({"error": "not found"}, 404)
        self.send(200, p_path.read_bytes(), mimetypes.guess_type(p_path.name)[0] or "application/octet-stream")

    def local(self) -> bool:
        """A request from the lab's own page: other sites the browser visits can't reach it (DNS rebinding: Host;
        cross-site posts: Origin; simple text/plain posts that skip the preflight: the JSON content type)."""
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origin = self.headers.get("Origin")
        if self.headers.get("Host") not in hosts or origin is not None and origin.split("//", 1)[-1] not in hosts:
            return False
        return self.command == "GET" or self.headers.get("Content-Type", "").startswith("application/json")

    def do_GET(self):
        if not self.local():
            return self.json({"error": "not from the lab's page"}, 403)
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/tables":
                tables = [{"path": p.relative_to(CLIMATE).as_posix(), "generated": p == ROOT / WORLD_TABLE}
                          for p in sorted(CLIMATE.rglob("*.json"))]
                return self.json({"tables": tables, "presets": presets(), "sites": sites(), "drift": format_drift()})
            if url.path == "/api/table":
                path = table_path(q["path"])
                return self.json({"path": path.relative_to(CLIMATE).as_posix(), "data": json.loads(path.read_text())})
            if url.path == "/api/job":
                job = JOBS.get(q.get("id", ""))
                return self.json(job.view() if job else {"error": "no such job"}, 200 if job else 404)
            if url.path == "/api/run":  # a file of a report run
                run = (LAB / "runs" / q["run"] / Path(q["name"]).name).resolve()
                return self.file(run) if run.is_relative_to(LAB / "runs") else self.json({"error": "bad path"}, 400)
            name = url.path.lstrip("/") or "index.html"
            path = (WEB / name).resolve()
            return self.file(path) if path.is_relative_to(WEB) else self.json({"error": "bad path"}, 400)
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as e:
            return self.json({"error": str(e)}, 400)

    def do_POST(self):
        if not self.local():
            return self.json({"error": "not from the lab's page"}, 403)
        url = urlparse(self.path)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            if url.path == "/api/save":
                path = table_path(body["path"])
                rel = path.relative_to(ROOT).as_posix()
                # A table whose data didn't change keeps its text (the hand-written tables have their own layout).
                if path.exists() and same_data(json.loads(path.read_text()), body["data"]):
                    return self.json({"saved": rel, "path": path.relative_to(CLIMATE).as_posix(), "unchanged": True})
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(dump_table(body["data"]) + "\n")
                return self.json({"saved": rel, "path": path.relative_to(CLIMATE).as_posix()})
            if url.path in ("/api/check", "/api/preview"):
                kind = url.path.rsplit("/", 1)[1]
                return self.json(report_job(kind, body.get("path", ""), body["data"], body.get("params", {})).view())
            if url.path == "/api/tool":
                return self.json(tool_job(body).view())
            if url.path == "/api/cancel":
                job = JOBS.get(body.get("id", ""))
                if job:
                    job.stop()
                return self.json({"ok": bool(job)})
            if url.path == "/api/launch":
                return self.json(launch(body))
            return self.json({"error": "unknown endpoint"}, 404)
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as e:
            return self.json({"error": str(e)}, 400)


def main():
    port, open_browser = 8765, True
    for a in sys.argv[1:]:
        if a.startswith("--port="):
            port = int(a.split("=", 1)[1])
        elif a == "--no-open":
            open_browser = False
        else:
            sys.exit(__doc__)
    threading.Thread(target=worker, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"biome lab: {url} (files in {LAB.relative_to(ROOT)})", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
