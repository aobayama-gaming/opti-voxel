# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Generate a climate table's heightmap on a reference site's grid, compare it (compare.py), optionally fit it.

    uv run tools/landforms/match.py <site> <table.json> [--fit=N] [--hypso] [--gain=0.8] [--biome=NAME] [--center=X,Z]
        [--texture=BIOME [--copy=OTHER:F,...]] [--octaves=0]
        [--lods=25] [--voxel=0.05] [--size=N] [--no-build] [--out=DIR]

The heightmap comes from tools/climate_report.sh on the lattice whose spacing is the site's (sites.json; at 5 cm
voxels, 2^m voxels), centred on --center (metres). --fit=N runs N rounds of: generate, compare, then multiply the
"x" of each ground multiplier with a single "wavelength_m" in the biome (default: the last one, the table's
default biome) by (real / generated amplitude at that octave)^gain, for the octaves the map resolves (4 cells to
half the map). The noise's own band shape and the shapes' gains make the octaves leak into their neighbours, hence
the rounds and the gain below 1. --hypso also refits the table's height remap (a height term on a band-less field
reading only the ground, fit_hypsometry) so the heights take the real map's distribution; --octaves=0 leaves the
multipliers alone (the two fits fight when run together: fit the octaves, then the remap, then the octaves gently).
--texture=BIOME fits that biome's octaves on a copy of the table without height terms and with that biome alone (a
sharp feature puts energy into every octave, and a whole-map fit would starve the texture), then copies them back,
and --copy=OTHER:0.6 into other biomes times a factor; the final comparison runs on the whole table. The table is rewritten in place; the last sheet and stats land in --out
(default bench/results/<site>-match).
"""
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import compare  # noqa: E402
import terrain_stats as ts  # noqa: E402


def generate(table: Path, site: dict, opts: dict, build: bool) -> Path:
    voxel = float(opts.get("voxel", "0.05"))
    lattice = round(math.log2(site["spacing_m"] / voxel))
    path = table.resolve()
    res = f"res://{path.relative_to(ROOT / 'project')}" if path.is_relative_to(ROOT / "project") else str(path)
    args = [str(ROOT / "tools/climate_report.sh"), f"--climate={res}",
            f"--voxel={voxel}", f"--lattice={lattice}", f"--size={opts.get('size', site.get('n', 512))}", f"--lods={opts.get('lods', '25')}",
            f"--center={opts.get('center', '0,0')}"]
    if not build:
        args.insert(1, "--no-build")
    out = subprocess.run(args, capture_output=True, text=True)
    lines = out.stdout.strip().splitlines()
    if out.returncode != 0 or not lines or not lines[-1].startswith(">> "):
        sys.exit(f"climate_report.sh failed ({out.returncode}):\n" + "\n".join(lines[-15:]))
    return Path(lines[-1][3:])


def dump_table(v, indent: int = 0) -> str:
    """JSON with one line per object or list that holds no object (a band, a multiplier, a curve point)."""
    flat = json.dumps(v)
    if not isinstance(v, (dict, list)) or len(flat) <= 110 and "{" not in flat[1:]:
        return flat
    pad = " " * (indent + 1)
    if isinstance(v, dict):
        items = [f"{pad}{json.dumps(k)}: {dump_table(x, indent + 1)}" for k, x in v.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    if all(not isinstance(x, (dict, list)) for x in v) or len(flat) <= 110 and all(isinstance(x, list) for x in v):
        return flat
    return "[\n" + ",\n".join(pad + dump_table(x, indent + 1) for x in v) + "\n" + " " * indent + "]"




def shape_coeffs(a: dict) -> list[float]:
    """Climate::Shape of a multiplier entry (climate_parse.cpp), plain without a shape."""
    if "coeffs" in a:
        return a["coeffs"]
    s, sc, o = a.get("shape", "plain"), a.get("scale", 1.0), a.get("offset")
    if s == "billow":
        return [(o or 0.0) - 1.0, 0.0, 2.0 * sc, 0.0]
    if s == "ridged":
        o = 1.0 if o is None else o
        return [o * o, 0.0, -2.0 * o * sc, sc * sc]
    if s == "square":
        return [o or 0.0, 0.0, 0.0, sc * sc]
    return [o or 0.0, sc, 0.0, 0.0]


def max_x(a: dict, k: dict) -> float:
    """The largest multiplier the ground's cap allows: x * the octave's amplitude * shape_bound <= the cap (Climate's
    validate_amp). k: the cascade constants of the run (heights.json)."""
    c = shape_coeffs(a)
    e = k["smooth_abs_e"]
    n = [-1.0 + i / 1000.0 for i in range(2001)]
    bound = max(abs(c[1] * v + c[2] * (math.sqrt(v * v + e * e) - e - k["noise_abs_mean"]) + c[3] * (v * v - k["noise_sq_mean"])) for v in n)
    return 0.98 * k["amp_cap_m"] / (min(a["wavelength_m"] / 8.0, k["amp_cap_m"]) * max(bound, 1e-9))


def fit_round(data: dict, real: dict, gen: dict, biome: str | None, gain: float, n_m: float, spacing: float, consts: dict) -> int:
    real_amp = {round(w): a for w, a in real["octaves"]}
    gen_amp = {round(w): a for w, a in gen["octaves"]}
    b = next(b for b in data["biomes"] if b["name"] == biome) if biome else data["biomes"][-1]
    changed = 0
    for a in b.get("amp", []):
        w = a.get("wavelength_m")
        if a.get("field", "ground") != "ground" or w is None or not (4 * spacing <= w <= n_m / 2):
            continue
        key = round(w)
        if key not in real_amp or gen_amp.get(key, 0) <= 0:
            continue
        x = float(a.get("x", 1.0)) * (real_amp[key] / gen_amp[key]) ** gain
        cap = max_x(a, consts)
        if x > cap:
            print(f"  {w} m: x {x:.3f} capped at {cap:.3f} (the ground's amplitude cap)")
            x = cap
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


def fit_hypsometry(data: dict, run: Path, real_h) -> bool:
    """Refit the remap's curve so the generated heights take the real map's quantiles: the ground h becomes
    h + H c(e), e = per_m h the field, so the pairs (e_q, (real_q - h_q) / H) at the same quantiles q (h and e rank
    alike) make the output's distribution the real one's. A one-step match, up to the 3D noise (the surface moves with
    the remap, so e at the new surface differs): hence the rounds."""
    import numpy as np

    found = remap_term(data)
    if not found:
        return False
    term, idx, per_m = found
    meta = json.loads((run / "heights.json").read_text())
    nf = len(meta["fields"])
    e = np.fromfile(run / "fields.f32", dtype="<f4").reshape(-1, nf)[:, idx].astype(float)
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
    for b in dst["biomes"]:
        f = 1.0 if b["name"] == biome else copies.get(b["name"])
        for a in b.get("amp", []) if f is not None else []:
            if a.get("wavelength_m") in fit:
                a["x"] = round(f * fit[a["wavelength_m"]], 4)


def main() -> None:
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict((a[2:].split("=", 1) + ["1"])[:2] for a in sys.argv[1:] if a.startswith("--"))
    if len(pos) != 2:
        sys.exit(__doc__)
    site_key, table = pos[0], Path(pos[1])
    site = json.loads((HERE / "sites.json").read_text())["sites"][site_key]
    dem = Path(opts.get("dem", compare.DEM_DIR)).expanduser()
    h_real, spacing, _ = compare.load_real(site_key, dem)
    real = ts.summary(h_real.astype(float), spacing)
    out = Path(opts.get("out", ROOT / f"bench/results/{site_key}-match"))
    out.mkdir(parents=True, exist_ok=True)
    rounds = int(opts.get("fit", "0"))
    build = "no-build" not in opts
    target = table
    if "texture" in opts:
        # A sharp feature (a cliff, a trough, a coast) puts energy into every octave and the whole-map fit starves the
        # texture: fit the biome's octaves on the table without its features, then copy them back.
        target = out / "texture.json"
        target.write_text(dump_table(texture_table(json.loads(table.read_text()), opts["texture"])) + "\n")
        opts.pop("biome", None)
    for r in range(rounds + 1):
        if r == rounds and target != table:
            data = json.loads(table.read_text())
            copies = {k: float(v) for k, v in (c.split(":") for c in opts.get("copy", "").split(",") if c)}
            copy_multipliers(json.loads(target.read_text()), data, opts["texture"], copies)
            table.write_text(dump_table(data) + "\n")
            target = table
        run = generate(target, site, opts, build and r == 0)
        if r == rounds:
            break
        h_gen, _, _ = compare.load_gen(run)
        gen = ts.summary(h_gen.astype(float), spacing)
        data = json.loads(target.read_text())
        hyp = "hypso" in opts and fit_hypsometry(data, run, h_real)
        n = 0
        if opts.get("octaves", "1") != "0":
            consts = json.loads((run / "heights.json").read_text())
            n = fit_round(data, real, gen, opts.get("biome"), float(opts.get("gain", "0.8")), h_real.shape[0] * spacing, spacing, consts)
        target.write_text(dump_table(data) + "\n")
        print(f"round {r + 1}: {n} multipliers refit" + (", remap curve refit" if hyp else ""))
    cmp = subprocess.run([sys.executable, str(HERE / "compare.py"), f"--real={site_key}", f"--gen={run}", f"--dem={dem}",
                          f"--out={out / 'compare.png'}", f"--json={out / 'compare.json'}"], capture_output=True, text=True)
    print(cmp.stdout)
    if cmp.returncode != 0:
        sys.exit(cmp.stderr)
    (out / "compare.md").write_text(cmp.stdout)
    print(f">> {run}")


if __name__ == "__main__":
    main()
