#!/usr/bin/env python3
"""Compare two screenshots of one pose: the share of pixels that differ and the mean difference.

    tools/shot_diff.py a.png b.png [--max_share=0.01] [--tol=8] [--diff=out.png]

A pixel differs when a channel moves by more than --tol (0-255). Exit 1 when more than --max_share of the pixels
differ. Used for the render-origin check: a fly far out and back (bench.sh --scenario=fly --fly_dist=20000 --still=1)
rebases many times, and its steady_end shot must match its loaded shot.
"""

import argparse
import json
import sys

import numpy as np
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--max_share", type=float, default=0.01)
    parser.add_argument("--tol", type=int, default=8)
    parser.add_argument("--diff", help="write the differing pixels (white) here")
    args = parser.parse_args()

    a = np.asarray(Image.open(args.a).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(args.b).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        print(json.dumps({"ok": False, "error": f"sizes differ: {a.shape} vs {b.shape}"}))
        return 1
    delta = np.abs(a - b)
    differs = delta.max(axis=2) > args.tol
    share = float(differs.mean())
    report = {
        "ok": share <= args.max_share,
        "differing_share": round(share, 6),
        "mean_abs": round(float(delta.mean()), 4),
        "max_abs": int(delta.max()),
        "tol": args.tol,
        "max_share": args.max_share,
    }
    if args.diff:
        Image.fromarray((differs * 255).astype(np.uint8)).save(args.diff)
    print(json.dumps(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
