#!/usr/bin/env python3
"""Compares Luau port output against the C++ oracle (tools/cpp_oracle).

  compare.py sdf  <luau_points> <oracle_values>
  compare.py mesh <luau_mesh>   <oracle_mesh>

Mesh comparison: every Luau vertex is matched to the nearest oracle vertex (one-to-one expected),
then triangles are compared through that mapping, winding included. Prints one JSON line.
"""
import json
import math
import sys


def load_mesh(path):
    verts, tris = [], []
    with open(path) as f:
        n = int(f.readline().split()[1])
        for _ in range(n):
            verts.append([float(v) for v in f.readline().split()])
        m = int(f.readline().split()[1])
        for _ in range(m):
            tris.append(tuple(int(v) for v in f.readline().split()))
    return verts, tris


def canon(t):
    # Rotation that starts at the smallest index keeps the winding.
    i = t.index(min(t))
    return t[i:] + t[:i]


def compare_mesh(luau_path, oracle_path, tol):
    lv, lt = load_mesh(luau_path)
    ov, ot = load_mesh(oracle_path)

    grid = {}
    for i, v in enumerate(ov):
        grid.setdefault(tuple(math.floor(c) for c in v[:3]), []).append(i)

    mapping, dists, normal_dots = [], [], []
    for v in lv:
        cx, cy, cz = (math.floor(c) for c in v[:3])
        best, best_d = -1, float("inf")
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in grid.get((cx + dx, cy + dy, cz + dz), ()):
                        d = math.dist(v[:3], ov[j][:3])
                        if d < best_d:
                            best, best_d = j, d
        mapping.append(best)
        dists.append(best_d)
        if best >= 0:
            normal_dots.append(sum(a * b for a, b in zip(v[3:], ov[best][3:])))

    unique = len(set(m for m in mapping if m >= 0))
    oracle_tris = {canon(t) for t in ot}
    flipped_oracle = {canon((t[0], t[2], t[1])) for t in ot}
    matched = flipped = missing = 0
    for t in lt:
        mt = canon(tuple(mapping[i] for i in t))
        if mt in oracle_tris:
            matched += 1
        elif mt in flipped_oracle:
            flipped += 1
        else:
            missing += 1

    dists.sort()
    far = sum(1 for d in dists if d > tol)
    return {
        "luau_vertices": len(lv),
        "oracle_vertices": len(ov),
        "one_to_one": unique == len(lv) == len(ov),
        "vertex_dist_max": round(dists[-1], 6) if dists else 0,
        "vertex_dist_p99": round(dists[int(len(dists) * 0.99)], 6) if dists else 0,
        "vertices_over_tol": far,
        "normal_dot_min": round(min(normal_dots), 6) if normal_dots else 1,
        "luau_triangles": len(lt),
        "oracle_triangles": len(ot),
        "triangles_matched": matched,
        "triangles_flipped": flipped,
        "triangles_missing": missing,
    }


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "mesh":
        tol = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
        print(json.dumps(compare_mesh(sys.argv[2], sys.argv[3], tol)))
    elif len(sys.argv) >= 4 and sys.argv[1] == "sdf":
        lu = [float(l.split()[3]) for l in open(sys.argv[2])]
        orc = [float(l) for l in open(sys.argv[3])]
        d = sorted(abs(a - b) for a, b in zip(lu, orc))
        flips = sum(1 for a, b in zip(lu, orc) if (a < 0) != (b < 0))
        print(json.dumps({"points": len(d), "max": d[-1], "p99": d[int(len(d) * 0.99)], "sign_flips": flips}))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
