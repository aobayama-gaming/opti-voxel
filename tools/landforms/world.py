# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Compose the landform tables into one world table: each landform behind a region of macro climate axes.

    uv run tools/landforms/world.py [--out=project/climate/world.json]
    uv run tools/landforms/world.py --axes=RUN     # the axes' quantiles in a climate_report.sh --lattice run

The landform tables (project/climate/landforms/) stay the one source: refit one, then run this again. The world table
takes every landform's fields, height terms and biomes:

- a field defined the same way in several tables is shared (the warp triple, elev); a field defined differently is
  renamed <landform>_<name>, with every reference to it in that landform;
- each landform's height terms are gated by its region ("region" on the term), and its biomes take the region's
  edges before their own; mountain's biome comes last and has no edges: the default, which takes what the priority
  layering leaves in the transitions;
- the tables must agree on "ground_stretch" (one stretch for the world), and turn off the ground octaves the axes can't
  gate (COARSE_M and up).

The regions are boxes of temperature x humidity x wind (LAYOUT). The thresholds are the axes' quantiles over the ground
(--axes on a report of a world written with any thresholds), so each landform gets the area LAYOUT gives it. Two
neighbours share a boundary t: the lower one fades out over [t - FADE / 2, t + FADE / 2] and the upper one fades in
over the same range, so their edges add up to about 1.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import match  # noqa: E402
import terrain_stats as ts  # noqa: E402

LANDFORMS = ROOT / "project/climate/landforms"

# The macro climate axes: two octaves each, at 25 LODs and 5 cm (wavelengths are 2^k voxels).
AXES = [
    {"name": "temperature", "seed": 2001, "bands": [{"wavelength_m": 419430.4, "amp": 1.0}, {"wavelength_m": 209715.2, "amp": 0.5}]},
    {"name": "humidity", "seed": 2002, "bands": [{"wavelength_m": 419430.4, "amp": 1.0}, {"wavelength_m": 209715.2, "amp": 0.5}]},
    {"name": "wind", "seed": 2003, "bands": [{"wavelength_m": 209715.2, "amp": 1.0}, {"wavelength_m": 104857.6, "amp": 0.5}]},
]
# Boundaries on each axis (quantiles of the axis over the ground, from --axes on a 512 x 512 report of lattice 17, 3,355
# km across): temperature in quartiles (cold, cool, warm, hot), humidity and wind at their medians.
T = [-0.22, 0.04, 0.27]
H = -0.02
W = -0.02
FADE = 0.04
# The "to the top" bound of the tables' octave ranges (the ground's coarsest octave at any LOD count is under it).
TOP_M = 3e7
# The ground octaves at or above the finest axis' wavelength are inserted before the axes exist (an octave reads the
# climate of the octaves coarser than itself, Cascade::Modulation): no region can gate them, so every landform turns them
# off (multiplier 0) and the world and each table agree there.
COARSE_M = min(b["wavelength_m"] for a in AXES for b in a["bands"])
# Each landform's box: (temperature lo, hi), (humidity lo, hi), (wind lo, hi); None is open. Mountain is the default
# biome: it comes last.
LAYOUT = {
    "valley": ((T[0], T[1]), (H, None), (None, None)),
    "caldera": ((T[0], T[1]), (None, H), (None, None)),
    "island": ((T[1], T[2]), (H, None), (None, None)),
    "cliff": ((T[1], T[2]), (None, H), (None, None)),
    "mangrove": ((T[2], None), (H, None), (None, None)),
    "dunes": ((T[2], None), (None, H), (W, None)),
    "desert": ((T[2], None), (None, H), (None, W)),
    "mountain": ((None, T[0]), (None, None), (None, None)),
}


def region_edges(box) -> list:
    edges = []
    for axis, (lo, hi) in zip(("temperature", "humidity", "wind"), box):
        if lo is not None:
            edges.append({"field": axis, "min": round(lo + FADE / 2, 6), "fade": FADE})
        if hi is not None:
            edges.append({"field": axis, "max": round(hi - FADE / 2, 6), "fade": FADE})
    return edges


def check_coarse(name: str, t: dict) -> None:
    """Exit unless every biome of table t has ground multiplier 0 from COARSE_M up."""
    def lo(a):
        return a.get("from_m", a.get("wavelength_m", 0.0))

    def hi(a):
        return a.get("to_m", a.get("wavelength_m", TOP_M))

    for b in t["biomes"]:
        ground = [a for a in b.get("amp", []) if a.get("field") == "ground"]
        off = any(a.get("x", 1) == 0 and lo(a) <= COARSE_M and hi(a) >= TOP_M for a in ground)
        on = any(a.get("x", 1) != 0 and hi(a) >= COARSE_M for a in ground)
        if on or not off:
            sys.exit(f"world: {name}.json biome '{b['name']}' needs ground multiplier 0 from {COARSE_M} m up (no region can gate it)")


def strip(d: dict, *keys: str) -> dict:
    return {k: v for k, v in d.items() if k not in keys}


def definition(f: dict) -> str:
    return json.dumps(strip(f, "name", "comment"), sort_keys=True)


def rename_refs(t: dict, names: dict) -> None:
    """Every field reference in table t through names (old -> new)."""
    def ref(n):
        return names.get(n, n)
    for f in t.get("fields", []):
        for term in f.get("terms", []):
            term["source"] = ref(term["source"])
    for h in t.get("heights", []):
        h["field"] = ref(h["field"])
    for b in t["biomes"]:
        for e in b.get("where", []):
            e["field"] = ref(e["field"])
        for a in b.get("amp", []):
            a["field"] = ref(a["field"])
            if "warp_fields" in a:
                a["warp_fields"] = [ref(w) for w in a["warp_fields"]]


def compose() -> dict:
    fields = [dict(a) for a in AXES]
    by_name = {f["name"]: definition(f) for f in fields}
    heights, biomes, regions = [], [], []
    stretch = unset = object()
    for name, box in LAYOUT.items():
        t = json.loads((LANDFORMS / f"{name}.json").read_text())
        check_coarse(name, t)
        if stretch is unset:
            stretch = t.get("ground_stretch")
        elif t.get("ground_stretch") != stretch:
            sys.exit(f"world: {name}.json's ground_stretch differs from the others' (the world has one)")
        if "regions" in t or any("region" in x for x in t.get("heights", []) + t["biomes"]):
            sys.exit(f"world: {name}.json has regions of its own (a world's landform is a plain table)")
        defaults = [i for i, b in enumerate(t["biomes"]) if "where" not in b]
        if name == list(LAYOUT)[-1] and defaults != [len(t["biomes"]) - 1]:
            sys.exit(f"world: {name}.json is the world's default: one biome without edges, its last")
        # A field is shared when its definition, with its references renamed, is the world's: renaming one can change
        # another's (a term reading it), so until nothing changes.
        renames = {}
        while True:
            u = json.loads(json.dumps(t))
            rename_refs(u, renames)
            more = {f["name"]: f"{name}_{f['name']}" for f in u.get("fields", [])
                    if f["name"] not in renames and by_name.get(f["name"], definition(f)) != definition(f)}
            if not more:
                break
            renames.update(more)
        rename_refs(t, renames)
        for f in t.get("fields", []):
            f = strip(f, "comment")
            f["name"] = renames.get(f["name"], f["name"])
            if f["name"] not in by_name:
                by_name[f["name"]] = definition(f)
                fields.append(f)
        regions.append({"name": name, "where": region_edges(box)})
        for h in t.get("heights", []):
            heights.append({**h, "region": name})
        for b in t["biomes"]:
            b = strip(b, "comment")
            default = name == list(LAYOUT)[-1] and "where" not in b
            if not default:
                b = {"name": b["name"], "region": name, **strip(b, "name")}
            biomes.append(b)
    world = {
        "comment": "The eight landforms in one world, generated by tools/landforms/world.py from project/climate/landforms/ "
        "(edit those and run it again). Each landform sits behind a region of temperature x humidity x wind (the LAYOUT "
        "there); mountain's biome is the default. 25 LODs at 5 cm: the axes' octaves are 105-419 km.",
        "fields": fields,
        "regions": regions,
        "heights": heights,
    }
    if stretch:
        world["ground_stretch"] = stretch
    world["biomes"] = biomes
    return world


def axes(run: Path) -> None:
    import numpy as np
    import compare
    meta, _, f, _ = compare.load_run(run)
    for a in AXES:
        v = f[:, meta["fields"].index(a["name"])]
        q = np.quantile(v, [0.25, 0.5, 0.75])
        print(f"{a['name']}: quartiles {q[0]:.3f} {q[1]:.3f} {q[2]:.3f}, range {v.min():.3f} .. {v.max():.3f}")


def main() -> None:
    _, opts = ts.parse_args(sys.argv[1:])
    if "axes" in opts:
        axes(Path(opts["axes"]))
        return
    out = ROOT / opts.get("out", "project/climate/world.json")
    out.write_text(match.dump_table(compose()) + "\n")
    print(f">> {out}")


if __name__ == "__main__":
    main()
