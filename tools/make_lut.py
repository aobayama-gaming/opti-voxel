#!/usr/bin/env python3
"""Writes the scenes' colour-grading LUT: adapter/godot/project/grade_lut.png, a 32^3 cube as 32 slices side by side
(1024 x 32, blue = slice, red = x, green = y), imported as a Texture3D (grade_lut.png.import: 32 horizontal
slices) and set as Environment.adjustment_color_correction. Godot applies it after the tonemapper, sampling the
filtered texture at uvw = colour with no half-texel remap: texel i is read at colour (i + 0.5) / N, so that is the
colour it holds the grade of.

The grade (warm, leaning red, a filmic look on the tonemapper's flat output):
- on luma only (hues keep their place): a gamma of 1.2 against the milky mids, an S-curve, lifted blacks;
- split toning: shadows towards a plum red, mids towards red, highlights towards amber;
- greens turned a little towards olive, so the reds and the bark read against them;
- vibrance: pale colours gain the most saturation, saturated ones (lava) little.

Run it again after changing the grade: python3 tools/make_lut.py
Preview a grade on a screenshot without Godot: python3 tools/make_lut.py --preview in.png out.png
"""
import colorsys
import os
import sys

from PIL import Image

N = 32


def smoothstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def s_curve(x):
    # A gamma of 1.2 (the tonemapper's output sits high: milky mids), contrast through the mids, then blacks lifted
    # a little so the canopy's shadows keep their colour.
    x = min(max(x, 0.0), 1.0) ** 1.2
    return 0.035 + 0.965 * (x + 0.25 * (x * x * (3.0 - 2.0 * x) - x))


def grade(r, g, b):
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    graded = s_curve(luma)
    k = graded / max(luma, 1e-4)
    r, g, b = r * k, g * k, b * k
    luma = graded
    shadow = 1.0 - smoothstep(0.0, 0.4, luma)
    high = smoothstep(0.55, 1.0, luma)
    mid = 1.0 - shadow - high
    # Split toning: plum-red shadows, red mids, amber highlights.
    r += 0.035 * shadow + 0.055 * mid + 0.030 * high
    g += -0.005 * shadow - 0.015 * mid + 0.005 * high
    b += 0.020 * shadow - 0.035 * mid - 0.040 * high
    h, s, v = colorsys.rgb_to_hsv(min(max(r, 0.0), 1.0), min(max(g, 0.0), 1.0), min(max(b, 0.0), 1.0))
    hue = h * 360.0
    # Greens (hue ~70-170 degrees) a touch towards olive.
    green = smoothstep(60.0, 90.0, hue) * (1.0 - smoothstep(160.0, 190.0, hue))
    hue -= 7.0 * green
    # Vibrance (pale colours gain the most), a little more on reds and oranges, less in the highlights.
    warm = max(1.0 - smoothstep(35.0, 55.0, hue), smoothstep(320.0, 345.0, hue))
    s = min(s * (1.08 + (0.45 + 0.15 * warm) * (1.0 - s) * (1.0 - 0.3 * high)), 1.0)
    return colorsys.hsv_to_rgb((hue % 360.0) / 360.0, s, v)


def make_lut():
    return [[[grade((rx + 0.5) / N, (gy + 0.5) / N, (bz + 0.5) / N) for rx in range(N)] for gy in range(N)]
            for bz in range(N)]


def to_byte(c):
    return tuple(int(round(min(max(v, 0.0), 1.0) * 255.0)) for v in c)


def preview(lut, src, dst):
    # Trilinear, as Godot samples the Texture3D (needs numpy).
    import numpy as np

    cube = np.array(lut, dtype=np.float32)  # [b][g][r][rgb]
    c = np.asarray(Image.open(src).convert("RGB"), dtype=np.float32) / 255.0
    t = np.clip(c * N - 0.5, 0.0, N - 1.0)
    i0 = np.minimum(t.astype(np.int32), N - 2)
    f = t - i0
    out = np.zeros_like(c)
    for db in (0, 1):
        for dg in (0, 1):
            for dr in (0, 1):
                w = ((f[..., 2] if db else 1 - f[..., 2]) * (f[..., 1] if dg else 1 - f[..., 1]) *
                     (f[..., 0] if dr else 1 - f[..., 0]))
                out += w[..., None] * cube[i0[..., 2] + db, i0[..., 1] + dg, i0[..., 0] + dr]
    Image.fromarray((np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(dst)


def main():
    lut = make_lut()
    if len(sys.argv) == 4 and sys.argv[1] == "--preview":
        preview(lut, sys.argv[2], sys.argv[3])
        print(sys.argv[3])
        return
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "adapter", "godot", "project", "grade_lut.png")
    img = Image.new("RGB", (N * N, N))
    px = img.load()
    for bz in range(N):
        for gy in range(N):
            for rx in range(N):
                px[bz * N + rx, gy] = to_byte(lut[bz][gy][rx])
    img.save(out)
    print(out)


if __name__ == "__main__":
    main()
