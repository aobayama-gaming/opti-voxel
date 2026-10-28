# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Generate a climate table's heightmap on a reference site's grid, compare it (compare.py), optionally fit it.

    uv run tools/landforms/match.py <site> <table.json> [--fit=N] [--hypso] [--gain=0.8] [--biome=NAME] [--center=U,V]
        [--texture=BIOME [--copy=OTHER:F,...]] [--octaves=0]
        [--lods=25] [--voxel=0.05] [--size=N] [--no-build] [--out=DIR]

The heightmap comes from tools/climate_report.sh on the lattice whose spacing is the site's (sites.json; at 5 cm
voxels, 2^m voxels), centred on --center (metres). --fit=N runs N rounds of: generate, compare, then multiply the
"x" of each ground multiplier with a single "wavelength_m" in the biome (default: the last one, the table's default
biome) by (real / generated amplitude at that octave)^gain, for the octaves the map resolves (4 cells to a quarter
of the map), capped where the engine would refuse it (heights.json's max_x). The noise's own band shape and the
shapes' gains make the octaves leak into their neighbours, hence the rounds and the gain below 1. --hypso also
refits the table's height remap (a height term on a band-less field reading only the ground, fit_hypsometry) so the
heights take the real map's distribution; --octaves=0 leaves the multipliers alone (the two fits fight when run
together: fit the octaves, then the remap, then the octaves gently). --texture=BIOME fits that biome's octaves on a
copy of the table without height terms and with that biome alone (a sharp feature puts energy into every octave, and
a whole-map fit would starve the texture), then copies them back, and --copy=OTHER:0.6 into other biomes times a
factor; the final comparison runs on the whole table. The table is rewritten in place; the last sheet and stats land
in --out (default bench/results/<site>-match).
"""
import json
import math
import subprocess
import sys

import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import compare  # noqa: E402
import terrain_stats as ts  # noqa: E402
from table_json import dump_table  # noqa: E402


def generate(table: Path, site: dict, opts: dict, build: bool) -> Path:
    voxel = float(opts.get("voxel", "0.05"))
    lattice = round(math.log2(site["spacing_m"] / voxel))
    path = table.resolve()
    res = f"res://{path.relative_to(ROOT / 'adapter/godot/project')}" if path.is_relative_to(ROOT / "adapter/godot/project") else str(path)
    args = [str(ROOT / "tools/climate_report.sh"), f"--climate={res}",
            f"--voxel={voxel}", f"--lattice={lattice}", f"--size={opts.get('size', site['n'])}", f"--lods={opts.get('lods', '25')}",
            f"--center={opts.get('center', '0,0')}"]
    if not build:
        args.insert(1, "--no-build")
    out = subprocess.run(args, capture_output=True, text=True)
    lines = out.stdout.strip().splitlines()
    if out.returncode != 0 or not lines or not lines[-1].startswith(">> "):
        sys.exit(f"climate_report.sh failed ({out.returncode}):\n" + "\n".join(lines[-15:]))
    return Path(lines[-1][3:])


def octave(octs: list, w: float) -> float | None:
    """The amplitude of octs' octave at wavelength w (within 1%: the table's metres vs the grid's spacing * 2^j)."""
    return next((a for x, a in octs if abs(math.log2(x / w)) < 0.015), None)


def fit_round(data: dict, real: dict, gen: dict, biome: str | None, gain: float, n_m: float, spacing: float, meta: dict) -> int:
    b = next((b for b in data["biomes"] if b["name"] == biome), None) if biome else data["biomes"][-1]
    if b is None:
        sys.exit(f"no biome '{biome}' in the table")
    caps = meta["max_x"][b["name"]]
    changed = 0
    for a in b.get("amp", []):
        w = a.get("wavelength_m")
        if a.get("field", "ground") != "ground" or w is None or not ts.resolved(w, spacing, n_m):
            continue
        r, g, cap = octave(real["octaves"], w), octave(gen["octaves"], w), octave(caps, w)
        if r is None or not g or cap is None:
            print(f"  {w} m: not an octave of the grid or the table, left alone")
            continue
        x = float(a.get("x", 1.0)) * (r / g) ** gain
        if x > 0.98 * cap:
            print(f"  {w} m: x {x:.3f} capped at {0.98 * cap:.3f} (the ground's amplitude cap)")
            x = 0.98 * cap
        a["x"] = round(x, 4)
        changed += 1
    return changed


QUANTILES = [0.0, 0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.94, 0.97, 0.99, 1.0]
CURVE_POINTS = 10
REMAP_SLOPE_MIN, REMAP_SLOPE_MAX = 0.0, 1.25


def remap_term(data: dict) -> tuple[dict, int, float] | None:
    """The table's height remap: a height term on a band-less field whose only term reads the ground (per_m):
    (the height term, the field's index, per_m)."""
    for t in data.get("heights", []):
        for i, f in enumerate(data["fields"]):
            terms = f.get("terms", [])
            if f["name"] == t["field"] and not f.get("bands") and len(terms) == 1 and terms[0]["source"] == "ground":
                return t, i, terms[0]["per_m"]
    return None


def fit_hypsometry(data: dict, fields: np.ndarray, real_h: np.ndarray) -> bool:
    """Refit the remap's curve so the generated heights take the real map's quantiles: the ground h becomes
    h + H c(e), e = per_m h the field, so the pairs (e_q, (real_q - h_q) / H) at the same quantiles q (h and e rank
    alike) make the output's distribution the real one's. A one-step match, up to the 3D noise (the surface moves with
    the remap, so e at the new surface differs): hence the rounds."""
    found = remap_term(data)
    if not found:
        return False
    term, idx, per_m = found
    e = fields[:, idx].astype(float)
    target = real_h.ravel().astype(float)
    eq = np.quantile(e, QUANTILES)
    tq = np.quantile(target, QUANTILES)
    # libnoise's cubic interpolation takes its points as evenly spaced: on uneven ones it kinks into terraces, so the
    # rank mapping is resampled on an even grid of e.
    xs = np.linspace(eq[0], eq[-1], CURVE_POINTS)
    t = np.interp(xs, eq, tq)
    # The ground is 3D: the remap multiplies dh/dy by its slope T', and past dh/dy = 1 the surface folds into needles
    # and overhangs: T' <= REMAP_SLOPE_MAX. A slope near 0 is safe (flat floors: dune corridors, valley bottoms).
    mid = CURVE_POINTS // 2
    dx = (xs[1] - xs[0]) / per_m
    for i in range(mid + 1, CURVE_POINTS):
        t[i] = t[i - 1] + np.clip(t[i] - t[i - 1], REMAP_SLOPE_MIN * dx, REMAP_SLOPE_MAX * dx)
    for i in range(mid - 1, -1, -1):
        t[i] = t[i + 1] - np.clip(t[i + 1] - t[i], REMAP_SLOPE_MIN * dx, REMAP_SLOPE_MAX * dx)
    ys = (t - xs / per_m) / term["height_m"]
    # Centred: the real map's mean altitude is a constant lift (every statistic here is relative to the mean), and
    # carried by the curve it would push the term past the ground's amplitude cap, which the engine refuses.
    ys -= (ys.max() + ys.min()) / 2
    step = xs[1] - xs[0]
    pts = [[round(float(xs[0] - step), 5), round(float(ys[0]), 5)]]
    pts += [[round(float(x), 5), round(float(y), 5)] for x, y in zip(xs, ys)]
    pts.append([round(float(xs[-1] + step), 5), round(float(ys[-1]), 5)])
    term["curve"] = pts
    return True


def texture_table(data: dict, biome: str) -> dict:
    """The table without its features: no height terms, the one biome everywhere (its own octaves, the texture)."""
    t = json.loads(json.dumps(data))
    t.pop("heights", None)
    t["biomes"] = [b for b in t["biomes"] if b["name"] == biome]
    if not t["biomes"]:
        sys.exit(f"no biome '{biome}'")
    t["biomes"][0].pop("where", None)
    return t


def copy_multipliers(src: dict, dst: dict, biome: str, copies: dict[str, float]) -> None:
    """The fitted single-wavelength multipliers of src's biome into dst's biome, and times a factor into others."""
    fit = {a["wavelength_m"]: a["x"] for a in src["biomes"][0]["amp"] if "wavelength_m" in a}
    unknown = set(copies) - {b["name"] for b in dst["biomes"]}
    if unknown:
        sys.exit(f"--copy names no biome of the table: {', '.join(sorted(unknown))}")
    for b in dst["biomes"]:
        f = 1.0 if b["name"] == biome else copies.get(b["name"])
        for a in b.get("amp", []) if f is not None else []:
            if a.get("wavelength_m") in fit:
                a["x"] = round(f * fit[a["wavelength_m"]], 4)


def main() -> None:
    pos, opts = ts.parse_args(sys.argv[1:])
    if len(pos) != 2:
        sys.exit(__doc__)
    site_key, table = pos[0], Path(pos[1])
    site = json.loads((HERE / "sites.json").read_text())["sites"][site_key]
    dem = Path(opts.get("dem", ts.DEM_DIR)).expanduser()
    h_real, spacing, real_name = compare.load_real(site_key, dem)
    site["n"] = h_real.shape[0]  # the generated map covers the real one
    real = ts.summary(h_real.astype(float), spacing)
    out = Path(opts.get("out", ROOT / f"bench/results/{site_key}-match"))
    out.mkdir(parents=True, exist_ok=True)
    build = "no-build" not in opts
    target = table
    if "texture" in opts:
        # A sharp feature (a cliff, a trough, a coast) puts energy into every octave and the whole-map fit starves the
        # texture: fit the biome's octaves on the table without its features, then copy them back.
        target = out / "texture.json"
        target.write_text(dump_table(texture_table(json.loads(table.read_text()), opts["texture"])) + "\n")
        opts.pop("biome", None)
    rounds = int(opts.get("fit", "0"))
    for r in range(rounds):
        meta, h_gen, fields, _ = compare.load_run(generate(target, site, opts, build and r == 0))
        gen = ts.summary(h_gen.astype(float), spacing)
        data = json.loads(target.read_text())
        hyp = "hypso" in opts and fit_hypsometry(data, fields, h_real)
        n = 0
        if opts.get("octaves", "1") != "0":
            n = fit_round(data, real, gen, opts.get("biome"), float(opts.get("gain", "0.8")), h_real.shape[0] * spacing, spacing, meta)
        target.write_text(dump_table(data) + "\n")
        print(f"round {r + 1}: {n} multipliers refit" + (", remap curve refit" if hyp else ""))
    if target != table:
        data = json.loads(table.read_text())
        copies = {k: float(v) for k, v in (c.split(":") for c in opts.get("copy", "").split(",") if c)}
        copy_multipliers(json.loads(target.read_text()), data, opts["texture"], copies)
        table.write_text(dump_table(data) + "\n")
    run = generate(table, site, opts, build and rounds == 0)
    meta, h_gen, _, dhdy = compare.load_run(run)
    text = compare.compare([("real", h_real, real_name), ("gen", h_gen, Path(meta["climate"]).name)], spacing, out / "compare.png", dhdy,
                           out / "compare.json", [real, ts.summary(h_gen.astype(float), spacing)])
    print(text)
    (out / "compare.md").write_text(text + "\n")
    print(f">> {run}")


if __name__ == "__main__":
    main()
