#!/usr/bin/env python3
"""Finds the lines along which the bark pattern jumps, from the bench's probe views (bench.gd --probe):
PREFIX_10.png holds per pixel the bark's grain (R: the smooth noise its plates and grooves are made of, so that a
jump in it is a seam, not a groove), a hash of the chunk (G) and 8 * LOD + 4 (B, 0 = no bark at full detail);
PREFIX_11.png log2 of the depth (R, G: 6 bits each, times 4) and the normal's azimuth (B).

Pairs of neighbouring bark pixels (right and down) on one surface (depths within 8%), where the bark is resolved (the
normal turns less than GRAZING_DEG per pixel next to the pair: not at a trunk's silhouette), are classed as inside a chunk,
across a border between two chunks of one LOD, or across a LOD change. A pair jumps when its grain step is visible
(> VISIBLE: the plates are |2 grain - 1| * 1.8, so 0.033 is a plate step of 0.12) and SHARP times the steps just
before and after it: a continuous grain steps about as much as its neighbours. A seam is a line of jumps: a jump with
at least RUN of the RUN_WINDOW pairs along the line through it (across the pair) jumping too; lone jumps are noise. Reported per pose and class: pairs, the share that jump, the pairs on
lines, and the normal's mean azimuth step in degrees (where the bark's coordinate comes from).

Usage: bark_seams.py PREFIX [PREFIX ...] [--md out.md] [--overlay] [--max_line N] [--max_border F]
  --overlay: PREFIX_seams.png, the grain in grey, chunk borders blue, jumps dark red, lines bright red.
Exit code 1 when the poses hold more than --max_line (default 100) pairs on lines, in any class, or more than
--max_border (default 5%) of the pairs across same-LOD chunk borders jump (a border's slips show only where a groove
crosses it, so they don't make lines; inside a chunk ~0.02% jump). LOD changes are reported, not gated: the close-up
poses see few of them at full detail.
"""
import argparse
import sys

import numpy as np
from PIL import Image

VISIBLE = 0.033
SHARP = 4.0
DEPTH_TOL = np.log2(1.08)
RUN_WINDOW = 12
RUN = 9
GRAZING_DEG = 2.0
CLASSES = ("inside", "same_lod", "lod_change")


def load(prefix):
    a = np.asarray(Image.open(prefix + "_10.png").convert("RGB")).astype(np.int32)
    b = np.asarray(Image.open(prefix + "_11.png").convert("RGB")).astype(np.int32)
    bark = a[..., 2] > 0
    grain = a[..., 0] / 255.0
    chunk = a[..., 1] * 256 + a[..., 2]
    lod = (a[..., 2] - 4) // 8
    depth = ((b[..., 0] // 4) * 64 + b[..., 1] // 4) / 4095.0 * 16.0 - 4.0
    azimuth = b[..., 2] / 255.0 * 360.0
    return bark, grain, chunk, lod, depth, azimuth


def shift(x, k, axis):
    """x moved so that the result at p is x[p + k] along axis (zero / False past the image's edge)."""
    out = np.zeros_like(x)
    n = x.shape[axis]
    src = [slice(None)] * 2
    dst = [slice(None)] * 2
    if k >= 0:
        src[axis], dst[axis] = slice(k, n), slice(0, n - k)
    else:
        src[axis], dst[axis] = slice(0, n + k), slice(-k, n)
    out[tuple(dst)] = x[tuple(src)]
    return out


def on_lines(jump, axis):
    """The jumps with at least RUN jumps among the RUN_WINDOW pairs centred on them across axis."""
    along = 1 - axis
    count = np.zeros(jump.shape, dtype=np.int32)
    for k in range(-(RUN_WINDOW // 2), RUN_WINDOW - RUN_WINDOW // 2):
        count += shift(jump, k, along)
    return jump & (count >= RUN)


def measure(prefix, overlay=False):
    bark, grain, chunk, lod, depth, azimuth = load(prefix)
    stats = {name: [0, 0, 0, 0.0] for name in CLASSES}  # pairs, jumps, on lines, azimuth step sum
    jumps = np.zeros(grain.shape, dtype=bool)
    lines = np.zeros(grain.shape, dtype=bool)
    border = np.zeros(grain.shape, dtype=bool)
    for axis in (0, 1):
        sh = lambda x, k: shift(x, k, axis)
        ok = bark & sh(bark, 1) & sh(bark, -1) & sh(bark, 2)
        for k in (-1, 0, 1):
            ok &= np.abs(sh(depth, k + 1) - sh(depth, k)) < DEPTH_TOL
        # The pairs before and after lie inside one chunk each: they measure the texture, not a border.
        ok &= (sh(chunk, -1) == chunk) & (sh(chunk, 2) == sh(chunk, 1))
        daz = np.abs(sh(azimuth, 1) - azimuth)
        daz = np.minimum(daz, 360.0 - daz)
        # The bark is resolved there: towards a trunk's silhouette its normal (the bark's coordinate) sweeps several
        # degrees per pixel and the grain aliases. Judged on the pairs before and after: a seam's own step may be large.
        ok &= (sh(daz, -1) < GRAZING_DEG) & (sh(daz, 1) < GRAZING_DEG)
        step = np.abs(sh(grain, 1) - grain)
        around = 0.5 * (np.abs(grain - sh(grain, -1)) + np.abs(sh(grain, 2) - sh(grain, 1)))
        jump = ok & (step > VISIBLE) & (step > SHARP * around)
        line = on_lines(jump, axis)
        seam = chunk != sh(chunk, 1)
        lod_change = lod != sh(lod, 1)
        masks = {"inside": ok & ~seam, "same_lod": ok & seam & ~lod_change, "lod_change": ok & seam & lod_change}
        for name, m in masks.items():
            s = stats[name]
            s[0] += int(m.sum())
            s[1] += int((m & jump).sum())
            s[2] += int((m & line).sum())
            s[3] += float(daz[m].sum())
        jumps |= jump
        lines |= line
        border |= ok & seam
    if overlay:
        g = (grain * 0.6 * bark)[..., None].repeat(3, axis=2)
        g[border] = (0.0, 0.5, 1.0)
        g[jumps] = (0.5, 0.0, 0.0)
        g[lines] = (1.0, 0.1, 0.1)
        Image.fromarray((g * 255.0).astype(np.uint8)).save(prefix + "_seams.png")
    return stats


def row(pose, name, s):
    n = max(s[0], 1)
    return "| %s | %s | %d | %.2f%% | %d | %.1f |" % (pose, name, s[0], 100.0 * s[1] / n, s[2], s[3] / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--md")
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--max_line", type=int, default=100)
    ap.add_argument("--max_border", type=float, default=0.05)
    a = ap.parse_args()
    out = ["| pose | class | pairs | jumps | on lines | normal step (deg) |", "|---|---|---:|---:|---:|---:|"]
    total = {name: [0, 0, 0, 0.0] for name in CLASSES}
    for prefix in a.prefixes:
        stats = measure(prefix, a.overlay)
        for name in CLASSES:
            out.append(row(prefix.split("/")[-1], name, stats[name]))
            total[name] = [t + s for t, s in zip(total[name], stats[name])]
    for name in CLASSES:
        out.append(row("**all**", name, total[name]))
    on_line = sum(t[2] for t in total.values())
    border = total["same_lod"][1] / max(total["same_lod"][0], 1)
    failed = on_line > a.max_line or border > a.max_border
    report = "\n".join(out)
    report += "\n\n%s: %d pairs on lines (at most %d), %.2f%% of same-LOD border pairs jump (at most %.0f%%)\n" % (
        "FAIL" if failed else "PASS", on_line, a.max_line, 100.0 * border, 100.0 * a.max_border)
    print(report)
    if a.md:
        with open(a.md, "w") as f:
            f.write(report)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
