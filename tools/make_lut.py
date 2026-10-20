#!/usr/bin/env python3
"""Writes the scenes' colour-grading LUT: project/grade_lut.png, a 32^3 cube as 32 slices side by side
(1024 x 32, blue = slice, red = x, green = y), imported as a Texture3D (grade_lut.png.import: 32 horizontal
slices) and set as Environment.adjustment_color_correction. Godot applies it after the tonemapper and the
brightness / contrast / saturation adjustments, on [0, 1] colours.

The grade (the cute look, kept gentle: the scenes already push saturation):
- shadows lifted a little and tinted cool, so dark crowns and bark keep their colour instead of going black;
- mids warmed (bark, ground, sun-lit leaves read friendlier);
- greens and browns a bit more saturated than the rest, highlights rolled off softly.

Run it again after changing the grade: python3 tools/make_lut.py
"""
import colorsys
import os
import sys

from PIL import Image

N = 32


def smoothstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def grade(r, g, b):
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    shadow = 1.0 - smoothstep(0.0, 0.35, luma)
    mid = smoothstep(0.1, 0.45, luma) * (1.0 - smoothstep(0.55, 0.95, luma))
    # Lifted, cool shadows.
    r += shadow * 0.025
    g += shadow * 0.030
    b += shadow * 0.045
    # Warm mids.
    r *= 1.0 + 0.06 * mid
    g *= 1.0 + 0.02 * mid
    b *= 1.0 - 0.05 * mid
    # Saturation: +15% on greens and browns (hue 20-160 degrees), +5% elsewhere.
    h, s, v = colorsys.rgb_to_hsv(min(max(r, 0.0), 1.0), min(max(g, 0.0), 1.0), min(max(b, 0.0), 1.0))
    hue = h * 360.0
    natural = smoothstep(10.0, 30.0, hue) * (1.0 - smoothstep(150.0, 175.0, hue))
    s = min(s * (1.05 + 0.10 * natural), 1.0)
    # Soft highlight roll-off.
    v = v - 0.08 * smoothstep(0.8, 1.0, v) * (v - 0.8)
    return colorsys.hsv_to_rgb(h, s, v)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "project", "grade_lut.png")
    img = Image.new("RGB", (N * N, N))
    px = img.load()
    for bz in range(N):
        for gy in range(N):
            for rx in range(N):
                c = grade(rx / (N - 1), gy / (N - 1), bz / (N - 1))
                px[bz * N + rx, gy] = tuple(int(round(min(max(v, 0.0), 1.0) * 255.0)) for v in c)
    img.save(out)
    print(out)


if __name__ == "__main__":
    main()
