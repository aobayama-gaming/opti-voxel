# opti-voxel on Dawn (raw C++/WebGPU)

The second consumer of the de-Godot core: a plain C++ application, no engine, that links the same
`src/` modules as the Godot GDExtension and draws the voxel terrain with [Dawn](https://dawn.googlesource.com/dawn)
(Google's WebGPU implementation). The Godot adapter keeps the GPU-direct mesher; this host meshes on the
CPU with the core's surface-nets code and renders with Dawn, which makes the two integrations
independent: a change to the core (cascade, chunk math, page format, shader text) has to keep both
working.

## Build

Dawn is not vendored (it is large); build it once and point CMake at the install prefix:

```sh
git clone --depth 1 --branch v20261023.194543 https://github.com/google/dawn.git
cmake -S dawn -B dawn/out -G Ninja \
    -DDAWN_FETCH_DEPENDENCIES=ON -DDAWN_ENABLE_VULKAN=ON \
    -DDAWN_ENABLE_D3D11=OFF -DDAWN_ENABLE_D3D12=OFF -DDAWN_ENABLE_METAL=OFF \
    -DDAWN_ENABLE_OPENGLES=OFF -DDAWN_BUILD_SAMPLES=OFF -DDAWN_BUILD_TESTS=OFF \
    -DDAWN_USE_GLFW=OFF -DDAWN_USE_WAYLAND=OFF -DDAWN_USE_X11=ON \
    -DDAWN_ENABLE_INSTALL=ON -DCMAKE_BUILD_TYPE=Release
ninja -C dawn/out webgpu_dawn
cmake --install dawn/out --prefix "$HOME/dawn-install"

# glslang compiles the core's generated GLSL to SPIR-V at runtime (Dawn's SPIR-V input).
git clone --depth 1 --branch 15.3.0 https://github.com/KhronosGroup/glslang.git
cmake -S glslang -B glslang/out -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_OPT=OFF -DBUILD_TESTING=OFF
ninja -C glslang/out
cmake --install glslang/out --prefix "$HOME/glslang-install"

cmake -S dawn -B dawn/build -G Ninja \
    -DOV_DAWN_DIR="$HOME/dawn-install" -DOV_GLSLANG_DIR="$HOME/glslang-install"
ninja -C dawn/build
./dawn/build/optivoxel_dawn
```

`DAWN_USE_X11=ON` needs the X11 headers; `DAWN_USE_WAYLAND=OFF` avoids the wayland-scanner
dependency. On a machine without `libvulkan.so`'s development symlink, CMake links
`-l:libvulkan.so.1` (the runtime loader).

## Controls

| Input | Action |
|---|---|
| WASD | move |
| Space / Ctrl | up / down |
| Shift | sprint |
| mouse | look (the pointer is captured; Escape releases it) |
| left click | **big capsule**: a 2.5 m subtract capsule from the surface along the view |
| right click | **dig**: a 0.8 m subtract capsule from the surface 6 m along the view |
| C | clear every carve edit |
| F | toggle the upscaler (FSR3 / native) |
| P | pixelate 1x / 2x / 4x |
| Escape | capture / release the pointer |

`--no-fsr` starts at full resolution; `--pixelate=N` starts with N×N blocks; `--carve` applies one
sphere and one capsule at startup (a headless smoke test of the edit path).

## Pipeline

1. **Chunk selection.** A quadtree in x/z around the camera; a leaf picks the chunks its surface
   range touches (a height field can leave one chunk's y band on a steep slope). The quadtree
   partitions the plane, so a coarse chunk never overlaps a fine one.
2. **Meshing.** `src/cpu_mesher`'s surface nets: one vertex per sign-changing cell, one quad per
   sign-changing edge, with a one-cell border so adjacent chunks share the same crossing vertices
   and the seam closes. Normals are central differences of the SDF.
3. **Field.** `DawnSDF` is the core's Hermite cascade (`CascadeTerrain::height_sdf`) on a flat base
   plus a small list of sphere/capsule edits. The same edit list is uploaded to the shader and used
   by the CPU ray march, so the tools and the mesh see one field.
4. **Render.** The terrain is drawn into an `Rgba16Float` target with depth and motion vectors at
   half resolution, then upscaled to the swapchain. The final pass applies the pixelate blocks.
5. **Upscaling.** `dawn/src/fsr3.*` is a port of AMD's FSR 3.1 upscaler (the `wgpu-ffx` pass order,
   resources and constants) with the reference SPIR-V shaders vendored under `thirdparty/fsr3`. The
   shaders are rewritten from SPIR-V 1.5 to 1.3 and stripped of constructs Tint cannot translate
   (the generated header documents each rewrite); all eight passes run: prepare inputs, luma
   pyramid, shading change pyramid/change, prepare reactivity, luma instability, accumulate and
   RCAS. The device requests `TextureFormatsTier2` and `maxStorageTexturesPerShaderStage = 8`.
   `--no-fsr` (or `F`) falls back to the bilinear present; the FSR3 output is on by default.

## Layout

- `src/gpu.*`: the Dawn instance/device/surface and the buffer/texture/readback helpers.
- `src/glsl.*`: glslang's C++ API, GLSL → SPIR-V.
- `src/terrain_sdf.*`: the cascade + edits field (CPU and GLSL).
- `src/cpu_mesher.*`: the surface-nets mesher.
- `src/terrain.*`: chunk quadtree, meshing, ray march, carve tools.
- `src/renderer.*`: pipelines, offscreen targets, the present/pixelate pass.
- `src/fsr3.*`: the FSR3 upscaler; `thirdparty/fsr3/` is its MIT-licensed shader set.
