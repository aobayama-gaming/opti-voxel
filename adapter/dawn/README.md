# Dawn adapter

The second adapter of the same engine: the core in `src/` (every file of `src/core_files.txt`, the meshers and
page pool included) driven through Dawn (native WebGPU) instead of Godot. `DawnDevice` is the engine's
`ov::gpu::Device` (glslang compiles the engine's GLSL to SPIR-V, Tint takes it from there); `ov_dawn --check`
runs `GPUDeviceCheck` on it (buffers, textures, readbacks, then the exact noise bit for bit), `--frame-check[=VOXEL_M]`
the integer frame and every noise probe (`FrameCheck`), `--cascade-check[=JSON options]` the GPU cascade against its CPU
oracle (`CascadeCheck`); each prints its report as JSON.

The viewer drives the core's world loop (`ov::World`, GPU Direct) with a fly camera and draws its pages with
`DawnPageHost` (the corners decoded by the same `PageFormat` GLSL as Godot's page shader; slope colours, a sun, fog, no
materials yet):

```
adapter/dawn/out/ov_dawn --viewer                          # an X11 window (XWayland), the Godot player's controls: mouse look,
                                                           # WASD / arrows, space up, shift down, wheel speed, Escape frees the mouse
                                                           # (a click takes it back), V cycles the views (1-3), Ctrl+Q quits
adapter/dawn/out/ov_dawn --shot=out.png --view=2 --pitch=-35   # offscreen: a PNG once the load is complete, JSON report
```

Options: `--size=WxH --voxel=M --lod_error=F --lods=N --terrain=cascade|continents --above_ground=M --pitch=DEG --yaw=DEG
--view=0|1|2 --max_frames=N`, and upscaling after Godot's `--upscale=fsr2 --render_scale=F --pixelate=N`
(`DawnUpscaler`): `--upscale=temporal` renders the 3D at `--render_scale` (default 0.5 per axis) and brings it to the
output with a compact temporal upscaler in FSR 2 / 3's form (jittered projection, depth reprojection, clamped
Catmull-Rom history, Lanczos-2, RCAS at `--sharpness=STOPS`, default 0.2), not AMD's shaders; `--pixelate=N` makes the
output the window over N and copies each pixel over an N x N block. The LOD reads the 3D render's pixels. The look of
the Godot video preset: `--upscale=temporal --pixelate=2`. Not there yet: instances and trees (the fold variant's barriers fail WGSL's uniformity
analysis: Tint only lifts it for WGSL source, `chromium_disable_uniformity_analysis`), materials, frustum culling, GPU
timestamps (no budget).

```
cmake -S adapter/dawn -B adapter/dawn/out -G Ninja -DOV_DAWN_DIR=<dawn install prefix> -DOV_GLSLANG_DIR=<glslang install prefix>
cmake --build adapter/dawn/out
adapter/dawn/out/ov_dawn --check
```

Dawn is built from github.com/google/dawn and installed with `cmake --install <build> --prefix <prefix>`, glslang from
github.com/KhronosGroup/glslang the same way. Without `OV_DAWN_DIR` only the core library (`ov_core`) builds.

Stock Tint drops SPIR-V's NoContraction (GLSL `precise`), so the driver fuses multiply-adds and the exact noise
differs from the CPU's by a few ulps: the check's noise part fails until that is fixed. `--frame-check` also fails on
the `base_at` probes' duplicate definition, as `tools/frame_test.sh` does on dev.

Keep this adapter thin: engine logic belongs in the core, where both adapters get it.
