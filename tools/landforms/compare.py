# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Compare a generated heightmap (climate_report.sh's heights.f32) with a real one (fetch_dem.py), or show one alone.

    uv run tools/landforms/compare.py --real=<site> [--gen=<climate report run dir>] [--dem=DIR] [--out=PNG]

Prints the statistics side by side (terrain_stats.py) with the per-octave amplitude ratio gen / real (the factor a
band's multiplier is off by) and the scores, and writes a sheet: hillshades on the same height colours (same metres per
colour from each map's lowest point), oblique views, a cross-section through the middle row, and the octave spectrum,
slope, hypsometry and curvature plots. Scores (0 = the same):
- spectrum: RMS of log2(gen / real) of the octave amplitudes from 4 cells to a quarter of the map;
- slopes: the mean |difference| of the two slope distributions' percentiles (degrees, a Wasserstein-1 distance);
- hypsometry: the mean |difference| of the hypsometric curves;
- curvature: |difference| of the curvature skewness.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import terrain_stats as ts  # noqa: E402

def load_real(site: str, dem: Path) -> tuple[np.ndarray, float, str]:
    meta = json.loads((dem / f"{site}.json").read_text())
    return np.load(dem / f"{site}.npy"), meta["spacing_m"], meta["name"]


def load_run(run: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """A climate_report.sh --lattice run: (heights.json, heights n x n, fields n^2 x fields, dh/dy n^2)."""
    meta = json.loads((run / "heights.json").read_text())
    n = meta["n"]
    h = np.fromfile(run / "heights.f32", dtype="<f4").reshape(n, n)
    fields = np.fromfile(run / "fields.f32", dtype="<f4").reshape(n * n, len(meta["fields"]))
    return meta, h, fields, np.fromfile(run / "dhdy.f32", dtype="<f4")


def hillshade(h: np.ndarray, spacing: float, lo: float, span: float) -> np.ndarray:
    gz, gx = np.gradient(h, spacing)
    nx, ny, nz = -gx, np.ones_like(h), -gz
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    light = np.array([-0.577, 0.577, -0.577])
    shade = np.clip((nx * light[0] + ny * light[1] + nz * light[2]) / ln, 0, 1)
    t = np.clip((h - lo) / span, 0, 1)
    cmap = matplotlib.colormaps["gist_earth"]
    rgb = cmap(0.25 + 0.75 * t)[..., :3]
    return np.clip(rgb * (0.35 + 0.85 * shade[..., None]), 0, 1)


def details(h: np.ndarray, spacing: float) -> dict:
    """The distributions the scores and the sheet read, once per map."""
    return {"slopes": ts.slopes_deg(h, spacing), "hypsometry": ts.hypsometry(h), "curvature": ts.curvature(h, spacing)}


def scores(a: dict, b: dict, da: dict, db: dict, spacing: float, n_m: float) -> dict:
    ratios = [np.log2(max(x[1], 1e-9) / max(y[1], 1e-9)) for x, y in zip(a["octaves"], b["octaves"]) if ts.resolved(x[0], spacing, n_m)]
    q = np.linspace(0.5, 99.5, 100)
    return {
        "spectrum": float(np.sqrt(np.mean(np.square(ratios)))),
        "slopes": float(np.mean(np.abs(np.percentile(da["slopes"], q) - np.percentile(db["slopes"], q)))),
        "hypsometry": float(np.mean(np.abs(da["hypsometry"][0] - db["hypsometry"][0]))),
        "curvature": abs(a["curvature_skew"] - b["curvature_skew"]),
    }


def oblique(ax, h: np.ndarray, spacing: float, lo: float, span: float, title: str) -> None:
    step = max(1, h.shape[0] // 160)
    hs = h[::step, ::step]
    n = hs.shape[0]
    x, z = np.meshgrid(np.arange(n) * spacing * step, np.arange(n) * spacing * step)
    colours = hillshade(hs, spacing * step, lo, span)
    ax.plot_surface(x, -z, hs, facecolors=colours, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_box_aspect((1, 1, max(span / (n * spacing * step), 0.02)))
    ax.view_init(elev=28, azim=-60)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def stats_table(maps: list, stats: list[dict], dhdy, sc: dict | None) -> list[str]:
    """The statistics side by side, the gen map's vertical slope, the octave amplitudes and the scores (markdown)."""
    head = "| statistic | " + " | ".join(f"{k} ({name})" for k, _, name in maps) + " |"
    lines = [head, "|---" * (len(maps) + 1) + "|"]
    for k in (k for k in stats[0] if k != "octaves"):
        lines.append(f"| {k} | " + " | ".join(f"{s[k]:.3f}" for s in stats) + " |")
    if dhdy is not None:
        # The 3D ground's vertical slope (a real heightmap's is 0): the distance falls as 1 - dh/dy along the up, so past
        # dh/dy = 1 the surface folds over (needles, overhangs); a negative dh/dy only steepens it.
        cells = ["-"] * (len(maps) - 1)
        lines.append("| dhdy_p90 (gen) | " + " | ".join(cells + [f"{np.percentile(dhdy, 90):.3f}"]) + " |")
        lines.append("| fold_share (gen, dh/dy > 1) | " + " | ".join(cells + [f"{(dhdy > 1).mean():.4f}"]) + " |")
    lines += ["", "| wavelength m | " + " | ".join(f"{k} amp m" for k, *_ in maps) + (" | gen / real |" if len(maps) == 2 else " |"),
              "|---" * (len(maps) + 1 + (len(maps) == 2)) + "|"]
    for i, (w, _) in enumerate(stats[0]["octaves"]):
        amps = [s["octaves"][i][1] for s in stats]
        ratio = f" | {amps[1] / max(amps[0], 1e-9):.2f}" if len(maps) == 2 else ""
        lines.append(f"| {w:.0f} | " + " | ".join(f"{a:.2f}" for a in amps) + ratio + " |")
    if sc:
        lines += ["", "scores: " + ", ".join(f"{k} {v:.3f}" for k, v in sc.items())]
    return lines


def sheet(maps: list, stats: list[dict], dets: list[dict], spacing: float, sc: dict | None, out: Path) -> None:
    n = maps[0][1].shape[0]
    span = max(np.ptp(h) for _, h, _ in maps)
    cols = max(len(maps), 2)
    fig = plt.figure(figsize=(6.2 * cols, 22))
    for i, (k, h, name) in enumerate(maps):
        ax = fig.add_subplot(5, cols, 1 + i)
        ax.imshow(hillshade(h, spacing, h.min(), span), extent=(0, n * spacing / 1000, n * spacing / 1000, 0))
        ax.set_title(f"{k}: {name}  ({np.ptp(h):.0f} m relief)", fontsize=9)
        ax.set_xlabel("km")
        oblique(fig.add_subplot(5, cols, 1 + cols + i, projection="3d"), h, spacing, h.min(), span, f"{k}: oblique")
    ax = fig.add_subplot(5, 1, 3)
    for k, h, name in maps:
        ax.plot(np.arange(n) * spacing / 1000, h[n // 2] - h.min(), label=f"{k}: {name}")
    ax.set_xlabel("km (middle row)")
    ax.set_ylabel("m above the map's lowest point")
    ax.set_aspect("auto")
    ax.legend(fontsize=8)
    ax = fig.add_subplot(5, 2, 7)
    for (k, *_), s in zip(maps, stats):
        w, a = np.array(s["octaves"]).T
        ax.loglog(w, a, "o-", label=k)
    ax.set_xlabel("octave wavelength m")
    ax.set_ylabel("amplitude m")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax = fig.add_subplot(5, 2, 8)
    for (k, *_), d in zip(maps, dets):
        ax.hist(d["slopes"].ravel(), bins=90, range=(0, 90), histtype="step", density=True, label=k)
    ax.set_xlabel("slope degrees")
    ax.legend(fontsize=8)
    ax = fig.add_subplot(5, 2, 9)
    for (k, *_), d in zip(maps, dets):
        frac, hi = d["hypsometry"]
        ax.plot(frac, np.linspace(0, 1, len(frac)), label=f"{k} (integral {hi:.2f})")
    ax.set_xlabel("area fraction above")
    ax.set_ylabel("normalised height")
    ax.legend(fontsize=8)
    ax = fig.add_subplot(5, 2, 10)
    for (k, *_), s, d in zip(maps, stats, dets):
        c = d["curvature"] / max(d["curvature"].std(), 1e-12)
        ax.hist(c.ravel(), bins=120, range=(-5, 5), histtype="step", density=True, log=True, label=f"{k} (skew {s['curvature_skew']:.2f})")
    ax.set_xlabel("curvature / std (negative: crests)")
    ax.legend(fontsize=8)
    if sc:
        fig.suptitle("scores: " + ", ".join(f"{k} {v:.3f}" for k, v in sc.items()), fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=72)
    plt.close(fig)


def compare(maps: list, spacing: float, out: Path, dhdy=None, json_path: Path | None = None, stats: list[dict] | None = None) -> str:
    """maps: [(label, heights, name)] ("real" first, then "gen") on one spacing; their middle squares are compared.
    dhdy: the gen map's dh/dy (load_run), cropped like its heights.
    Writes the sheet to out (and the stats and scores to json_path), returns the markdown table. stats: each map's
    terrain_stats.summary if already computed (used when no map is cropped)."""
    n = min(h.shape[0] for _, h, _ in maps)
    if any(h.shape[0] != n for _, h, _ in maps):
        stats = None
    if dhdy is not None:
        g = maps[-1][1].shape[0]
        dhdy = dhdy.reshape(g, g)[(g - n) // 2:(g - n) // 2 + n, (g - n) // 2:(g - n) // 2 + n]
    # Both maps are centred on their site: crop the middle n x n of each.
    maps = [(k, h[(h.shape[0] - n) // 2:(h.shape[0] - n) // 2 + n, (h.shape[1] - n) // 2:(h.shape[1] - n) // 2 + n].astype(np.float64), name)
            for k, h, name in maps]
    stats = stats or [ts.summary(h, spacing) for _, h, _ in maps]
    dets = [details(h, spacing) for _, h, _ in maps]
    sc = scores(stats[0], stats[1], dets[0], dets[1], spacing, n * spacing) if len(maps) == 2 else None
    lines = stats_table(maps, stats, dhdy, sc)
    sheet(maps, stats, dets, spacing, sc, out)
    lines.append(f">> {out}")
    if json_path:
        json_path.write_text(json.dumps({"stats": dict(zip([m[0] for m in maps], stats)), "scores": sc}, indent=1))
    return "\n".join(lines)


def main() -> None:
    _, opts = ts.parse_args(sys.argv[1:])
    dem = Path(opts.get("dem", ts.DEM_DIR)).expanduser()
    maps, spacings, dhdy = [], [], None
    if "real" in opts:
        h, spacing, name = load_real(opts["real"], dem)
        maps.append(("real", h, name))
        spacings.append(spacing)
    if "gen" in opts:
        meta, h, _, dhdy = load_run(Path(opts["gen"]))
        maps.append(("gen", h, Path(meta["climate"]).name))
        spacings.append(meta["spacing_m"])
    if not maps:
        sys.exit(__doc__)
    if len(maps) == 2 and abs(spacings[0] - spacings[1]) > 1e-6:
        sys.exit(f"spacing differs: real {spacings[0]} m, gen {spacings[1]} m (climate_report.sh --lattice=...)")
    out = Path(opts.get("out", (Path(opts["gen"]) if "gen" in opts else dem) / f"compare_{opts.get('real', 'gen')}.png"))
    print(compare(maps, spacings[0], out, dhdy, Path(opts["json"]) if "json" in opts else None))


if __name__ == "__main__":
    main()
