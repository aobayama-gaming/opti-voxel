#!/usr/bin/env python3
"""Compares the runs of tools/origin_parity.sh: OUT label=run_dir ... (zero first, zero2 = noise floor)."""
import json
import sys

import numpy as np
from PIL import Image

EXACT = [
    ("chunks.scanned", lambda t: t["chunks"]["scanned"]),
    ("chunks.loaded", lambda t: t["chunks"]["loaded"]),
    ("chunks.loaded_non_empty", lambda t: t["chunks"]["loaded_non_empty"]),
    ("chunks.live_triangles", lambda t: t["chunks"]["live_triangles"]),
    ("chunks.drawn_triangles", lambda t: t["chunks"]["drawn_triangles"]),
    ("discoverer.last_scan_size", lambda t: t["discoverer"]["last_scan_size"]),
    ("mesher.live_quads", lambda t: t["mesher"].get("live_quads")),
    ("mesher.missing_neighbor", lambda t: t["mesher"].get("missing_neighbor")),
]
SHOTS = ["loaded.png", "steady.png"]


def image_diff(a, b):
    ia = np.asarray(Image.open(a).convert("RGB")).astype(int)
    ib = np.asarray(Image.open(b).convert("RGB")).astype(int)
    d = np.abs(ia - ib).max(axis=2)
    return float((d > 3).mean()), int(d.max())


def main():
    out = sys.argv[1]
    runs = dict(arg.split("=", 1) for arg in sys.argv[2:])
    stats = {k: json.load(open(f"{d}/stats.json")) for k, d in runs.items()}
    lines = ["# Origin parity", "", "| run | origin (voxels) | " + " | ".join(n for n, _ in EXACT) + " | nan | validation | audit |",
             "|" + "---|" * (len(EXACT) + 5)]
    ok = True
    base = {n: f(stats["zero"]["final_terrain"]) for n, f in EXACT}
    for k, s in stats.items():
        t = s["final_terrain"]
        vals = [f(t) for _, f in EXACT]
        nan = t["mesher"].get("nan_vertices", 0)
        val = (s.get("validation") or {}).get("ok", "-")
        aud = (s.get("audit") or {}).get("ok", "-")
        same = all(v == base[n] for (n, _), v in zip(EXACT, vals))
        ok &= same and not nan and val in (True, "-") and aud in (True, "-")
        lines.append(f"| {k} | {t['world_origin']['voxel']} | " + " | ".join(str(v) for v in vals) +
                     f" | {nan} | {val} | {aud} |" + ("" if same else " **differs**"))
    lines += ["", "Screenshots vs `zero`: share of pixels with a channel difference > 3 (max difference).", "",
              "| run | " + " | ".join(SHOTS) + " |", "|" + "---|" * (len(SHOTS) + 1)]
    floor = {shot: image_diff(f"{runs['zero']}/{shot}", f"{runs['zero2']}/{shot}")[0] for shot in SHOTS}
    for k in runs:
        if k == "zero":
            continue
        cells = []
        for shot in SHOTS:
            share, mx = image_diff(f"{runs['zero']}/{shot}", f"{runs[k]}/{shot}")
            limit = max(2 * floor[shot], 1e-4)
            cells.append(f"{share:.6f} ({mx})" + ("" if share <= limit else " **over**"))
            ok &= share <= limit
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines += ["", f"**{'PARITY OK' if ok else 'PARITY BROKEN'}**", ""]
    lines += [f"- {k}: `{d}`" for k, d in runs.items()]
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f">> report: {out}")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
