# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""The statistics of every fetched reference site in one table (terrain_stats.py), plus each site's sheet.

    uv run tools/landforms/survey.py [--dem=DIR] [site ...]
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare  # noqa: E402
import terrain_stats as ts  # noqa: E402

KEYS = ["relief_m", "std_m", "hypsometric_integral", "slope_p50", "slope_p90", "slope_p99", "flat_share", "cliff_share",
        "curvature_skew", "spectral_slope_fine", "spectral_slope_coarse", "anisotropy"]


def main() -> None:
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    dem = Path(opts.get("dem", compare.DEM_DIR)).expanduser()
    sites = [a for a in sys.argv[1:] if not a.startswith("--")] or json.loads((HERE / "sites.json").read_text())["sites"]
    lines = ["| site | " + " | ".join(KEYS) + " |", "|---" * (len(KEYS) + 1) + "|"]
    spectra = ["", "Octave amplitudes (m), wavelength m: amplitude:", ""]
    for site in sites:
        h, spacing, _ = compare.load_real(site, dem)
        s = ts.summary(h.astype(float), spacing)
        lines.append(f"| {site} | " + " | ".join(f"{s[k]:.2f}" for k in KEYS) + " |")
        spectra.append(f"- {site}: " + ", ".join(f"{w:.0f}: {a:.2f}" for w, a in s["octaves"]))
        subprocess.run([sys.executable, str(HERE / "compare.py"), f"--real={site}", f"--dem={dem}"], check=True, capture_output=True)
    lines += spectra
    print("\n".join(lines))
    (dem / "survey.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
