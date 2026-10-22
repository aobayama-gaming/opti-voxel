# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "tifffile", "imagecodecs"]
# ///
"""Reference heightmaps of real landforms (sites.json) from the Copernicus GLO-30 DSM (public, no account: the AWS open
data bucket copernicus-dem-30m), cropped to each site's square and resampled bilinearly to a local grid of metres
(x east, z south: the climate report's image axes) at the site's spacing.

    uv run tools/landforms/fetch_dem.py [--out=DIR] [--cache=DIR] [site ...]

Writes <out>/<site>.npy (float32 metres above sea level, row 0 north) and <out>/<site>.json (name, centre, spacing).
Tiles are cached in --cache (default ~/.cache/opti-voxel/dem): one 1x1 degree GeoTIFF of ~30 m each.
"""
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import tifffile

BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
EARTH_M_PER_DEG = 111_320.0


def tile_name(lat: int, lon: int) -> str:
    ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


def load_tile(lat: int, lon: int, cache: Path) -> np.ndarray | None:
    """The tile whose south-west corner is (lat, lon), row 0 north; None over open sea (no tile)."""
    name = tile_name(lat, lon)
    path = cache / f"{name}.tif"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        url = f"{BUCKET}/{name}/{name}.tif"
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            raise
        path.write_bytes(data)
    return tifffile.imread(path).astype(np.float32)


def sample(lat: np.ndarray, lon: np.ndarray, cache: Path) -> np.ndarray:
    """Bilinear DSM heights at the points (pixel-is-point grid: the tile's row 0 is its north edge)."""
    out = np.zeros(lat.shape, np.float32)
    for tlat in range(math.floor(lat.min()), math.floor(lat.max()) + 1):
        for tlon in range(math.floor(lon.min()), math.floor(lon.max()) + 1):
            mask = (np.floor(lat) == tlat) & (np.floor(lon) == tlon)
            if not mask.any():
                continue
            tile = load_tile(tlat, tlon, cache)
            if tile is None:
                continue  # sea: 0 m
            rows, cols = tile.shape
            fy = (tlat + 1 - lat[mask]) * rows
            fx = (lon[mask] - tlon) * cols
            y0 = np.clip(np.floor(fy).astype(int), 0, rows - 2)
            x0 = np.clip(np.floor(fx).astype(int), 0, cols - 2)
            ty = np.clip(fy - y0, 0, 1)
            tx = np.clip(fx - x0, 0, 1)
            h = (tile[y0, x0] * (1 - tx) * (1 - ty) + tile[y0, x0 + 1] * tx * (1 - ty) + tile[y0 + 1, x0] * (1 - tx) * ty
                 + tile[y0 + 1, x0 + 1] * tx * ty)
            out[mask] = h
    return out


def fetch(key: str, site: dict, out: Path, cache: Path) -> None:
    spacing = site["spacing_m"]
    n = int(round(site["size_km"] * 1000.0 / spacing))
    offs = (np.arange(n) - n / 2 + 0.5) * spacing
    x, z = np.meshgrid(offs, offs)  # x east, z south
    lat = site["lat"] - z / EARTH_M_PER_DEG
    lon = site["lon"] + x / (EARTH_M_PER_DEG * math.cos(math.radians(site["lat"])))
    h = sample(lat, lon, cache)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{key}.npy", h)
    meta = {"site": key, "name": site["name"], "lat": site["lat"], "lon": site["lon"], "spacing_m": spacing, "n": n,
            "source": "Copernicus GLO-30 DSM"}
    (out / f"{key}.json").write_text(json.dumps(meta, indent=1))
    print(f"{key}: {n}x{n} at {spacing} m, {h.min():.0f} .. {h.max():.0f} m")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    here = Path(__file__).resolve().parent
    sites = json.loads((here / "sites.json").read_text())["sites"]
    out = Path(opts.get("out", Path.home() / "Drive/perso/opti-voxel/landforms-2026-10-22/dem")).expanduser()
    cache = Path(opts.get("cache", Path.home() / ".cache/opti-voxel/dem")).expanduser()
    for key in args or sites:
        fetch(key, sites[key], out, cache)


if __name__ == "__main__":
    main()
