# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""The statistics of every fetched reference site in one table (terrain_stats.py), plus each site's sheet.

    uv run tools/landforms/survey.py [--dem=DIR] [site ...]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare  # noqa: E402
import terrain_stats as ts  # noqa: E402


def main() -> None:
    pos, opts = ts.parse_args(sys.argv[1:])
    dem = Path(opts.get("dem", ts.DEM_DIR)).expanduser()
    sites = pos or json.loads((HERE / "sites.json").read_text())["sites"]
    lines, spectra = [], ["", "Octave amplitudes (m), wavelength m: amplitude:", ""]
    for site in sites:
        if not (dem / f"{site}.npy").exists():
            print(f"{site}: not fetched (fetch_dem.py {site}), skipped")
            continue
        h, spacing, name = compare.load_real(site, dem)
        s = ts.summary(h.astype(float), spacing)
        keys = [k for k in s if k != "octaves"]
        if not lines:
            lines = ["| site | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
        lines.append(f"| {site} | " + " | ".join(f"{s[k]:.2f}" for k in keys) + " |")
        spectra.append(f"- {site}: " + ", ".join(f"{w:.0f}: {a:.2f}" for w, a in s["octaves"]))
        compare.compare([("real", h, name)], spacing, dem / f"compare_{site}.png", stats=[s])
    lines += spectra
    print("\n".join(lines))
    (dem / "survey.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
