#!/usr/bin/env python3
"""Compares the runs of tools/origin_parity.sh: OUT label=run_dir ...

Pairs (same world point, different origins) must load the same non-empty chunks with identical
per-chunk mesh digests (chunks.json of the page audit). Discovery only depends on the camera's chunk
id, so the sets match unless a camera sits within float error of a chunk boundary (the parity cameras
don't). zero / zero2 (the same run twice) must also match on the exact counts."""
import json
import sys

import numpy as np
from PIL import Image

PAIRS = [("zero", "zero2"), ("zero", "near"), ("far_a", "far_b"), ("wrap_a", "wrap_b")]
COUNTS = [
    ("chunks.loaded", lambda t: t["chunks"]["loaded"]),
    ("chunks.loaded_non_empty", lambda t: t["chunks"]["loaded_non_empty"]),
    ("chunks.live_triangles", lambda t: t["chunks"]["live_triangles"]),
    ("mesher.live_quads", lambda t: t["mesher"].get("live_quads")),
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
    digests = {k: json.load(open(f"{d}/chunks.json")) for k, d in runs.items()}
    ok = True

    lines = ["# Origin parity", "", "| run | origin (voxels) | " + " | ".join(n for n, _ in COUNTS) + " | nan | validation | audit |",
             "|" + "---|" * (len(COUNTS) + 5)]
    for k, s in stats.items():
        t = s["final_terrain"]
        nan = t["mesher"].get("nan_vertices", 0)
        val = (s.get("validation") or {}).get("ok", "-")
        aud = (s.get("audit") or {}).get("ok", "-")
        run_ok = not nan and val in (True, "-") and aud is True
        ok &= run_ok
        lines.append(f"| {k} | {t['world_origin']['voxel']} | " + " | ".join(str(f(t)) for _, f in COUNTS) +
                     f" | {nan} | {val} | {aud} |" + ("" if run_ok else " **failed**"))

    lines += ["", "Same world point, different origins: per-chunk mesh digests.", "",
              "| pair | chunks a | chunks b | shared | identical | differing | " + " | ".join(SHOTS) + " |",
              "|" + "---|" * (6 + len(SHOTS))]
    floor = {shot: image_diff(f"{runs['zero']}/{shot}", f"{runs['zero2']}/{shot}")[0] for shot in SHOTS}
    examples = []
    for a, b in PAIRS:
        da, db = digests[a], digests[b]
        shared = set(da) & set(db)
        differing = sorted(c for c in shared if da[c] != db[c])
        share = len(shared) / max(len(da), len(db), 1)
        pair_ok = not differing and set(da) == set(db) and len(shared) > 0
        if (a, b) == ("zero", "zero2"):
            pair_ok &= all(f(stats[a]["final_terrain"]) == f(stats[b]["final_terrain"]) for _, f in COUNTS)
        ok &= pair_ok
        examples += [f"{a}/{b} {c}: {da[c]} vs {db[c]}" for c in differing[:5]]
        cells = []
        for shot in SHOTS:
            s, mx = image_diff(f"{runs[a]}/{shot}", f"{runs[b]}/{shot}")
            cells.append(f"{s:.6f} ({mx})")
        lines.append(f"| {a} / {b} | {len(da)} | {len(db)} | {share:.4f} | {len(shared) - len(differing)} | {len(differing)} | " +
                     " | ".join(cells) + " |" + ("" if pair_ok else " **broken**"))
    lines += ["", "Screenshots: share of pixels with a channel difference > 3 (max difference); noise floor (zero / zero2): " +
              ", ".join(f"{s} {floor[s]:.6f}" for s in SHOTS) + ". Informational for the pairs.", ""]
    if examples:
        lines += ["Differing chunks (quads:digest):", ""] + [f"- {e}" for e in examples] + [""]
    lines += [f"**{'PARITY OK' if ok else 'PARITY BROKEN'}**", ""]
    lines += [f"- {k}: `{d}`" for k, d in runs.items()]
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f">> report: {out}")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
