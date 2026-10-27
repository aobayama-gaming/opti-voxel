#!/usr/bin/env python3
"""Which loaded chunks one frame never shows: tools/occlusion_census.py <run> [<run> ...]

Needs a bench run with --depth_grab=1 --audit=1 (depth.f32 + depth.json from DepthGrab, chunks.json from the
page audit: every non-empty loaded chunk and its quad count). Each chunk's box (its id's span grown by
--grow cells of its LOD, for skirts and vertex placement) is tested against the camera that drew the depth:
- beyond_far: the box's nearest point is farther from the camera than the far plane's corners (its depth over
  the cosine of the diagonal half-FOV), in any direction: never drawn whichever way the camera turns (the far
  plane follows the altitude with --lods above 15, docs/integer-frame.md);
- offscreen: outside the view frustum's side planes or behind the camera;
- near: crosses the camera plane (counted as seen);
- hidden: every depth pixel of its screen rect (grown by a pixel) is nearer than the box's nearest corner,
  the conservative test a Hi-Z occlusion cull of that frame would make;
- visible: the rest.
The first line checks that depth and boxes share one frame: the share of the depth's points inside a non-empty
chunk box of LOD <= 8 (coarser boxes hold everything). Expect ~95-100%.

Two rules, cumulative, estimated on that frame:
1. far: a node entirely beyond the far distance is not loaded;
2. hidden: a node whose box grown by its own size on every side (its 3x3x3 neighbourhood, --cut_grow cells) is
   hidden stops splitting. That keeps face neighbours within one LOD (a finer neighbour lies inside the hidden
   neighbourhood) and a coarse mesh inside a hidden box (no flicker). Each such chunk is replaced by its
   coarsest such ancestor; its quads at an LOD are that LOD's mean.
Mesher GPU ms come from the run's cost model (us per chunk per LOD): compare them, the clock they were measured
at may not be full speed. Empty chunks (meshed, never drawn) are not in chunks.json and not counted.
Writes <run>/occlusion.md and occlusion.json.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

SANITY_MAX_LOD = 8  # the frame check uses the chunks up to this LOD
CLASSES = ["near", "offscreen", "beyond_far", "hidden", "visible"]
NEAR, OFFSCREEN, BEYOND_FAR, HIDDEN, VISIBLE = range(len(CLASSES))
REPORT_ORDER = ["beyond_far", "offscreen", "hidden", "near", "visible"]  # the report's columns


def lod_of(cid):
    v = cid[0]
    return (v & -v).bit_length() - 1


def chunk_at(voxel, lod, half):
    """ChunkMath::chunk_at_voxel on one axis: the LOD-lod chunk id holding an LOD0 voxel (ints or int arrays)."""
    unit = 1 << lod
    return ((voxel // (2 * half * unit)) * 2 + 1) * unit


def parent_of(cid, lod, half):
    return tuple(chunk_at(c * half, lod + 1, half) for c in cid)


class Frame:
    """The grabbed depth and camera. The projection is the one the renderer used: Vulkan clip space with
    reverse Z in [0, 1] (1 at the near plane, 0 for the sky) and y down the image."""

    def __init__(self, run):
        meta = json.load(open(os.path.join(run, "depth.json")))
        self.w, self.h = int(meta["width"]), int(meta["height"])
        self.depth = np.fromfile(os.path.join(run, "depth.f32"), dtype="<f4").reshape(self.h, self.w)
        self.proj = np.array(meta["projection_columns"], dtype=np.float64).T
        self.cam_to_world = np.eye(4)
        self.cam_to_world[:3, :3] = np.array(meta["basis_columns"], dtype=np.float64).T
        self.cam_to_world[:3, 3] = meta["origin"]
        self.cam = self.cam_to_world[:3, 3]
        self.view = np.linalg.inv(self.cam_to_world)
        self.vs = float(meta["voxel_size"])
        self.half = int(meta["half_chunk_voxels"])  # ChunkParam::half_chunk_voxels(): one chunk-id unit
        self.far = float(meta["far"])
        # The far plane is a depth: its corners are at far * sqrt(1 + tan^2 of each half-FOV) from the camera.
        self.far_reach = self.far * float(np.sqrt(1.0 + 1.0 / self.proj[0, 0] ** 2 + 1.0 / self.proj[1, 1] ** 2))
        if any(int(v) != 0 for v in meta["world_origin_voxel"]):
            sys.exit("occlusion_census: only a world origin of 0 is supported")

    def clip(self, pts):  # pts (n, 3) render metres -> clip (n, 4)
        h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
        return (self.proj @ (self.view @ h.T)).T

    def to_pixels(self, clip):
        x = (clip[:, 0] / clip[:, 3] + 1.0) * 0.5 * self.w
        y = (clip[:, 1] / clip[:, 3] + 1.0) * 0.5 * self.h
        return x, y

    def unproject(self, step):  # world points of every step-th pixel that isn't sky
        ys, xs = np.mgrid[0:self.h:step, 0:self.w:step]
        d = self.depth[ys, xs]
        keep = d > 0.0
        xs, ys, d = xs[keep] + 0.5, ys[keep] + 0.5, d[keep]
        ndc = np.stack([xs / self.w * 2.0 - 1.0, ys / self.h * 2.0 - 1.0, d, np.ones_like(d)], axis=1)
        p = (self.cam_to_world @ np.linalg.inv(self.proj) @ ndc.T).T
        return p[:, :3] / p[:, 3:4]


def chunk_boxes(ids, fr, grow):
    ids = np.array(ids, dtype=np.float64).reshape(-1, 3)
    lods = np.array([lod_of(tuple(int(c) for c in i)) for i in ids], dtype=np.int64)
    half = (2.0 ** lods)[:, None] * fr.half
    pad = (2.0 ** lods)[:, None] * grow
    lo = (ids * fr.half - half - pad) * fr.vs
    hi = (ids * fr.half + half + pad) * fr.vs
    return lo, hi, lods


def inside_share(fr, ids):
    """The share of the depth's points inside a non-empty chunk up to SANITY_MAX_LOD (coarser chunks hold
    everything): the check that depth and chunks are in one frame."""
    pts = fr.unproject(8)
    if not len(pts):
        return 0.0
    voxels = np.floor(pts / fr.vs).astype(np.int64)
    known = set(ids)
    inside = np.zeros(len(pts), dtype=bool)
    for L in range(SANITY_MAX_LOD + 1):
        held = chunk_at(voxels, L, fr.half)
        inside |= np.array([tuple(int(c) for c in v) in known for v in held])
    return float(inside.mean())


def classify(fr, lo, hi):
    """One of CLASSES per box."""
    n = len(lo)
    out = np.full(n, VISIBLE, dtype=np.int8)
    gap = np.maximum(np.maximum(lo - fr.cam, fr.cam - hi), 0.0)  # the box's nearest point, per axis
    beyond = np.linalg.norm(gap, axis=1) > fr.far_reach
    corners = np.stack([np.where(np.array([(k >> a) & 1 for a in range(3)], dtype=bool), hi, lo) for k in range(8)],
                       axis=1)  # (n, 8, 3)
    clip = fr.clip(corners.reshape(-1, 3)).reshape(n, 8, 4)
    x, y, w = clip[:, :, 0], clip[:, :, 1], clip[:, :, 3]
    # Outside the frustum: all 8 corners beyond one plane (left, right, bottom, top, behind the camera).
    outside = ((x < -w).all(axis=1) | (x > w).all(axis=1) | (y < -w).all(axis=1) | (y > w).all(axis=1) |
               (w <= 1e-6).all(axis=1))
    near = (w <= 1e-6).any(axis=1) & ~outside
    out[near] = NEAR
    out[outside] = OFFSCREEN
    out[beyond] = BEYOND_FAR
    for i in np.nonzero(out == VISIBLE)[0]:
        c = clip[i]
        px, py = fr.to_pixels(c)
        x0, x1 = int(np.floor(px.min())) - 1, int(np.ceil(px.max())) + 1
        y0, y1 = int(np.floor(py.min())) - 1, int(np.ceil(py.max())) + 1
        if x1 < 0 or y1 < 0 or x0 >= fr.w or y0 >= fr.h:
            out[i] = OFFSCREEN
            continue
        rect = fr.depth[max(y0, 0):min(y1, fr.h - 1) + 1, max(x0, 0):min(x1, fr.w - 1) + 1]
        if rect.min() > (c[:, 2] / c[:, 3]).max():  # reverse Z: larger is nearer; every pixel nearer than the box
            out[i] = HIDDEN
    return out


def coarsest_hidden(fr, cid, lod, max_lod, cut_grow, cache):
    best = (cid, lod)
    while best[1] + 1 < max_lod:
        p = parent_of(*best, fr.half)
        if p not in cache:
            lo, hi, _ = chunk_boxes([p], fr, cut_grow)
            cache[p] = classify(fr, lo, hi)[0] == HIDDEN
        if not cache[p]:
            break
        best = (p, best[1] + 1)
    return best


def cost_of(cost, per_lod):
    return sum(n * (cost[L] if L < len(cost) else 0.0) for L, n in enumerate(per_lod)) / 1000.0


def census(run, grow, cut_grow):
    fr = Frame(run)
    cut_grow = 2.0 * fr.half if cut_grow is None else cut_grow  # default: the node's own size
    digests = json.load(open(os.path.join(run, "chunks.json")))
    stats = json.load(open(os.path.join(run, "stats.json")))
    max_lod = int(stats["args"].get("lods", 15))
    cost = stats["final_terrain"]["mesher"].get("cost_us_per_chunk", [])
    ids, quads = [], []
    for k, v in digests.items():
        q = int(v.split(":")[0])
        if q > 0:
            ids.append(tuple(int(c) for c in re.findall(r"-?\d+", k)))
            quads.append(q)
    quads = np.array(quads)
    lo, hi, lods = chunk_boxes(ids, fr, grow)
    sanity = inside_share(fr, [i for i, L in zip(ids, lods) if L <= SANITY_MAX_LOD])
    cls = classify(fr, lo, hi)

    dist = np.linalg.norm((lo + hi) * 0.5 - fr.cam, axis=1)
    rows, mean_quads = [], {}
    for L in range(max_lod):
        m = lods == L
        if not m.any():
            continue
        mean_quads[L] = float(quads[m].mean())
        rows.append({"lod": L, "chunks": int(m.sum()), "quads": int(quads[m].sum()), "dist_m": float(np.median(dist[m])),
                     **{name: [int((m & (cls == c)).sum()), int(quads[m & (cls == c)].sum())]
                        for c, name in enumerate(CLASSES)}})

    def tally(keep, extra_lods=None):
        per_lod = np.bincount(lods[keep], minlength=max_lod).astype(float)
        q = float(quads[keep].sum())
        if extra_lods is not None:
            per_lod += extra_lods
            q += sum(extra_lods[L] * mean_quads.get(L, 0.0) for L in range(max_lod))
        return {"chunks": int(per_lod.sum()), "quads": q, "mesh_ms": cost_of(cost, per_lod)}

    before = tally(np.ones(len(ids), dtype=bool))
    far_kept = cls != BEYOND_FAR
    after_far = tally(far_kept)
    # Rule 2 on what rule 1 keeps: a node whose grown box is hidden stops splitting, at its coarsest such ancestor.
    glo, ghi, _ = chunk_boxes(ids, fr, cut_grow)
    stop = (classify(fr, glo, ghi) == HIDDEN) & far_kept
    cache, cut = {}, {}
    for i in np.nonzero(stop)[0]:
        a = coarsest_hidden(fr, ids[i], int(lods[i]), max_lod, cut_grow, cache)
        cut[a[0]] = a[1]
    cut_lods = np.bincount(np.array(list(cut.values()), dtype=np.int64), minlength=max_lod).astype(float) \
        if cut else np.zeros(max_lod)
    after_hidden = tally(far_kept & ~stop, cut_lods)
    return {"run": run, "sanity_inside": sanity, "size": [fr.w, fr.h], "far_m": fr.far, "far_reach_m": fr.far_reach, "grow_cells": grow,
            "cut_grow_cells": cut_grow, "rows": rows,
            "rules": {"before": before, "far": after_far, "far+hidden": after_hidden}}


def pct(a, b):
    return f"{100 * (a / b - 1):+.0f}%" if b else "-"


def report(r):
    out = [f"## {os.path.basename(r['run'].rstrip('/'))}", "",
           f"{r['size'][0]}x{r['size'][1]}, far plane {r['far_m'] / 1000:.0f} km (corners {r['far_reach_m'] / 1000:.0f} km), boxes grown by {r['grow_cells']:g} cells; "
           f"{100 * r['sanity_inside']:.1f}% of the depth's points inside a non-empty chunk box of LOD <= {SANITY_MAX_LOD}.", "",
           "Share of each LOD's quads per class:", "",
           "| LOD | median dist km | chunks | quads | beyond far | offscreen | hidden | near | visible |",
           "|---|---|---|---|---|---|---|---|---|"]
    tot = {k: [0, 0] for k in CLASSES}
    tq = 0
    for row in r["rows"]:
        q = max(row["quads"], 1)
        out.append(f"| {row['lod']} | {row['dist_m'] / 1000:.2f} | {row['chunks']} | {row['quads']} | " +
                   " | ".join(f"{100 * row[k][1] / q:.0f}%" for k in REPORT_ORDER) + " |")
        tq += row["quads"]
        for k in CLASSES:
            tot[k] = [tot[k][0] + row[k][0], tot[k][1] + row[k][1]]
    out.append(f"| all | | {sum(x['chunks'] for x in r['rows'])} | {tq} | " +
               " | ".join(f"{100 * tot[k][1] / max(tq, 1):.0f}%" for k in REPORT_ORDER) + " |")
    b, f, h = r["rules"]["before"], r["rules"]["far"], r["rules"]["far+hidden"]
    out += ["", f"Rules (cumulative; hidden = grown by {r['cut_grow_cells']:g} cells, non-empty chunks only):", "",
            "| | loaded now | 1. far: not beyond the far plane | 2. + hidden: no split |", "|---|---|---|---|"]
    for key, name, fmt in (("chunks", "chunks", "{:.0f}"), ("quads", "quads", "{:.0f}"),
                           ("mesh_ms", "mesher GPU ms (cost model)", "{:.0f}")):
        out.append(f"| {name} | {fmt.format(b[key])} | {fmt.format(f[key])} ({pct(f[key], b[key])}) | "
                   f"{fmt.format(h[key])} ({pct(h[key], b[key])}) |")
    return "\n".join(out + [""])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--grow", type=float, default=4.0, help="cells of its LOD each box grows by (default 4)")
    ap.add_argument("--cut_grow", type=float, default=None,
                    help="cells the no-split test grows a box by (default: its own size, 2 * half_chunk_voxels)")
    a = ap.parse_args()
    for run in a.runs:
        r = census(run, a.grow, a.cut_grow)
        md = report(r)
        open(os.path.join(run, "occlusion.md"), "w").write(md)
        json.dump(r, open(os.path.join(run, "occlusion.json"), "w"), indent=1)
        print(md)


if __name__ == "__main__":
    main()
