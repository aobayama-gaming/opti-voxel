#!/usr/bin/env python3
"""Compare two screenshots of one pose: the share of pixels that differ (origin_parity.image_diff).

    tools/shot_diff.py a.png b.png [--max_share=0.01] [--tol=8]

A pixel differs when a channel moves by more than --tol (0-255). Exit 1 when more than --max_share of the pixels
differ. Used for the render-origin check: a fly far out and back (bench.sh --scenario=fly --fly_dist=20000 --still=1)
rebases many times, and its steady_end shot must match its loaded shot.
"""

import argparse
import json
import sys

from origin_parity import image_diff


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--max_share", type=float, default=0.01)
    parser.add_argument("--tol", type=int, default=8)
    args = parser.parse_args()
    share, max_abs = image_diff(args.a, args.b, args.tol)
    ok = share <= args.max_share
    print(json.dumps({"ok": ok, "differing_share": round(share, 6), "max_abs": max_abs, "tol": args.tol, "max_share": args.max_share}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
