"""Height-term curves for climate tables, on evenly spaced points: libnoise's Curve (Climate::make_curve) interpolates
cubically as if its points were evenly spaced, so uneven points overshoot between them (terraces, bumps on flat land)."""


def smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def step(x0: float, x1: float, y0: float, y1: float, lo: float, hi: float, n: int = 16) -> list[list[float]]:
    """y0 below x0, y1 above x1, a smoothstep between; n + 1 points evenly spaced over [lo, hi] (the curve's ends hold
    their value beyond: keep [lo, hi] around the transition)."""
    pts = []
    for i in range(n + 1):
        x = lo + (hi - lo) * i / n
        pts.append([round(x, 5), round(y0 + (y1 - y0) * smoothstep((x - x0) / (x1 - x0)), 5)])
    return pts


def sample(f, lo: float, hi: float, n: int = 16) -> list[list[float]]:
    """Any function of the field, sampled on n + 1 evenly spaced points."""
    return [[round(lo + (hi - lo) * i / n, 5), round(f(lo + (hi - lo) * i / n), 5)] for i in range(n + 1)]
