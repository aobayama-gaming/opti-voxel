"""Statistics of a heightmap that tell landforms apart (compare.py): what a climate table has to match for the generated
ground to look and walk like the real one. Heights in metres on a square grid of `spacing` metres (row = z, col = x).

- octave_amplitudes: the std of the height band-passed to each octave (wavelength spacing * 2^j, a one-octave
  annulus of the windowed spectrum). The cascade's ground is one Perlin octave per wavelength, so this is the per-octave
  multiplier a biome needs, up to the noise's own band shape (fit by iteration).
- slopes: the slope angle distribution (degrees, central differences): flat, walkable, steep, cliff.
- hypsometry: the area fraction above each normalised height (Strahler's curve; its integral is low for incised
  plateaux and valleys, high for convex domes).
- curvature: the Laplacian at 2 cells, in units of its own std; its skewness tells sharp ridges over round valleys
  (negative: ridged mountains, crests) from sharp valleys over round tops (positive: billows, incised channels).
- anisotropy: the ratio of the spectral energy's principal axes (1 isotropic, large for linear dunes, ridges).
"""
from pathlib import Path

import numpy as np

# fetch_dem.py's default --out, where compare.py, match.py and survey.py read: under the repo's (gitignored) bench results.
DEM_DIR = Path(__file__).resolve().parents[2] / "bench/results/landforms/dem"


def parse_args(argv: list[str]) -> tuple[list[str], dict[str, str]]:
    """(positional arguments, --key=value options; a bare --flag is "1")."""
    pos = [a for a in argv if not a.startswith("--")]
    opts = dict((a[2:].split("=", 1) + ["1"])[:2] for a in argv if a.startswith("--"))
    return pos, opts


def resolved(wavelength_m: float, spacing: float, n_m: float) -> bool:
    """An octave the map measures well: 4 cells to a quarter of the map (the spectrum score and match.py's fit)."""
    return 4 * spacing <= wavelength_m <= n_m / 4


def detrend(h: np.ndarray) -> np.ndarray:
    n = h.shape[0]
    z, x = np.mgrid[0:n, 0:n].astype(np.float64)
    a = np.stack([np.ones(h.size), x.ravel(), z.ravel()], 1)
    coef, *_ = np.linalg.lstsq(a, h.ravel().astype(np.float64), rcond=None)
    return h - (a @ coef).reshape(h.shape)


def _spectrum(h: np.ndarray):
    n = h.shape[0]
    w = np.hanning(n)
    win = np.outer(w, w)
    f = np.fft.fft2(detrend(h) * win) / np.sqrt((win**2).mean())
    k = np.fft.fftfreq(n)
    kx, kz = np.meshgrid(k, k)
    return f, kx, kz


def octave_amplitudes(h: np.ndarray, spacing: float, spectrum=None) -> list[tuple[float, float]]:
    """[(wavelength m, amplitude m)] from 2 cells to half the map (spectrum: _spectrum(h), if already computed)."""
    n = h.shape[0]
    f, kx, kz = spectrum or _spectrum(h)
    kr = np.hypot(kx, kz)
    p = np.abs(f) ** 2 / (n * n) ** 2
    out = []
    j = 1
    while 2**j <= n // 2:
        fc = 1.0 / 2**j
        band = (kr >= fc / np.sqrt(2)) & (kr < fc * np.sqrt(2))
        out.append((spacing * 2**j, float(np.sqrt(p[band].sum()))))
        j += 1
    return out


def spectral_slope(oct_amps, lo_m: float, hi_m: float) -> float:
    """d log(amplitude) / d log(wavelength) between lo_m and hi_m (1: self-similar, the ground's default)."""
    pts = [(np.log2(w), np.log2(max(a, 1e-9))) for w, a in oct_amps if lo_m <= w <= hi_m]
    if len(pts) < 2:
        return float("nan")
    x, y = np.array(pts).T
    return float(np.polyfit(x, y, 1)[0])


def slopes_deg(h: np.ndarray, spacing: float) -> np.ndarray:
    gz, gx = np.gradient(h.astype(np.float64), spacing)
    return np.degrees(np.arctan(np.hypot(gx, gz)))[1:-1, 1:-1]


def hypsometry(h: np.ndarray, bins: int = 50) -> tuple[np.ndarray, float]:
    """(area fraction above each of `bins` normalised heights, hypsometric integral)."""
    t = (h - h.min()) / max(np.ptp(h), 1e-9)
    below = np.searchsorted(np.sort(t.ravel()), np.linspace(0, 1, bins), side="left")
    return 1.0 - below / t.size, float(t.mean())


def curvature(h: np.ndarray, spacing: float, step: int = 2) -> np.ndarray:
    s = step
    c = h[s:-s, s:-s]
    lap = (h[2 * s:, s:-s] + h[:-2 * s, s:-s] + h[s:-s, 2 * s:] + h[s:-s, :-2 * s] - 4 * c) / (s * spacing) ** 2
    return lap


def skew(v: np.ndarray) -> float:
    v = v.ravel().astype(np.float64)
    sd = v.std()
    return float(((v - v.mean()) ** 3).mean() / sd**3) if sd > 0 else 0.0


def anisotropy(h: np.ndarray, spectrum=None) -> tuple[float, float]:
    """(ratio of the spectral energy's principal axes, direction of the long features in degrees from x)."""
    f, kx, kz = spectrum or _spectrum(h)
    p = np.abs(f) ** 2
    kr = np.hypot(kx, kz)
    m = (kr > 2.0 / h.shape[0]) & (kr < 0.25)
    w = p[m] / kr[m] ** 2  # direction only: unit wavevectors
    ux, uz = kx[m], kz[m]
    cov = np.array([[np.sum(w * ux * ux), np.sum(w * ux * uz)], [np.sum(w * ux * uz), np.sum(w * uz * uz)]])
    ev, evec = np.linalg.eigh(cov)
    ratio = float(np.sqrt(ev[1] / max(ev[0], 1e-30)))
    kdir = evec[:, 1]  # dominant wavevector: features run across it
    return ratio, float((np.degrees(np.arctan2(kdir[1], kdir[0])) + 90.0) % 180.0)


def summary(h: np.ndarray, spacing: float) -> dict:
    s = slopes_deg(h, spacing)
    spectrum = _spectrum(h)
    oct_amps = octave_amplitudes(h, spacing, spectrum)
    _, hi = hypsometry(h)
    curv = curvature(h, spacing)
    ratio, direction = anisotropy(h, spectrum)
    n_m = h.shape[0] * spacing
    return {
        "relief_m": float(np.ptp(h)),
        "std_m": float(h.std()),
        "height_skew": skew(h),
        "hypsometric_integral": hi,
        "slope_p10": float(np.percentile(s, 10)),
        "slope_p50": float(np.percentile(s, 50)),
        "slope_p90": float(np.percentile(s, 90)),
        "slope_p99": float(np.percentile(s, 99)),
        "flat_share": float((s < 2.0).mean()),
        "steep_share": float((s > 35.0).mean()),
        "cliff_share": float((s > 55.0).mean()),
        "needle_share": float((s > 75.0).mean()),
        "curvature_skew": skew(curv),
        "spectral_slope_fine": spectral_slope(oct_amps, 4 * spacing, n_m / 16),
        "spectral_slope_coarse": spectral_slope(oct_amps, n_m / 16, n_m / 2),
        "anisotropy": ratio,
        "anisotropy_dir_deg": direction,
        "octaves": oct_amps,
    }
