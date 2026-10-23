#!/usr/bin/env python3
"""Writes the cloud layer's texture: project/clouds.png, a tileable 512 x 512 grey noise read by the sky shader (the
clouds) and by the page shader (their shadows on the ground), both through VoxelSky's GLSL (src/voxel_sky.cpp):
the same texel at the same world point, so a shadow lies under its cloud.

The noise is a spectral synthesis (random phases, power ~ 1 / f^2.6, inverse FFT), tileable by construction, then
rank-normalised: the values are uniform on [0, 1], so a cover of c in cloud_density covers about c of the sky.

Run it again after changing the spectrum: python3 tools/make_clouds.py
"""
import os

import numpy as np
from PIL import Image

N = 512
BETA = 2.6  # spectral slope: higher = smoother, larger cumulus; 2 is plain fBm
SEED = 7


def main() -> None:
    rng = np.random.default_rng(SEED)
    f = np.fft.fftfreq(N)[:, None] ** 2 + np.fft.fftfreq(N)[None, :] ** 2
    f[0, 0] = 1.0  # no DC term
    amplitude = f ** (-BETA / 4.0)  # power ~ f^-beta, so amplitude ~ |f|^(-beta / 2) with |f|^2 = f
    amplitude[0, 0] = 0.0
    phase = np.exp(2j * np.pi * rng.random((N, N)))
    noise = np.real(np.fft.ifft2(amplitude * phase))
    ranks = np.argsort(np.argsort(noise.ravel())).reshape(N, N)
    uniform = (ranks + 0.5) / (N * N)
    out = os.path.join(os.path.dirname(__file__), "..", "project", "clouds.png")
    Image.fromarray(np.round(uniform * 255.0).astype(np.uint8), mode="L").save(out)
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
