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
(`DawnUpscaler`): `--upscale=fsr3` renders the 3D at `--render_scale` (default 0.5 per axis) and brings it to the output
with AMD's FSR 3.1 upscaler (`fsr3.*`: the reference passes, their SPIR-V vendored in `thirdparty/fsr3`, MIT; needs
the device's `TextureFormatsTier2`), RCAS at `--sharpness=F` (FSR's 0 .. 1, default 0.2); `--pixelate=N` makes the
output the window over N and copies each pixel over an N x N block. FSR's depth and motion vectors come from the 3D's
depth (the terrain is static: the camera's reprojection is the motion). The LOD reads the 3D render's pixels. The look
of the Godot video preset: `--upscale=fsr3 --pixelate=2`. The forest: `--trees=N --tree_area=M` (with `--climate=PATH`
for a climate world), placed like Godot's `forest5cm` preset by `--trees=4000 --tree_area=316 --voxel=0.05
--cam_pos=-52,320,-95 --above_ground=5 --pitch=-5 --yaw=-90`. Not there yet: materials (the trees are slope-shaded),
frustum culling, GPU timestamps (no budget).

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
