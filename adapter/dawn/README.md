# Dawn adapter

The second adapter of the same engine: the core in `src/` (every file of `src/core_files.txt`, the meshers and
page pool included) driven through Dawn (native WebGPU) instead of Godot. `DawnDevice` is the engine's
`ov::gpu::Device` (glslang compiles the engine's GLSL to SPIR-V, Tint takes it from there); `ov_dawn --check`
runs `GPUDeviceCheck` on it (buffers, textures, readbacks, then the exact noise bit for bit), `--frame-check[=VOXEL_M]`
the integer frame and every noise probe (`FrameCheck`), `--cascade-check[=JSON options]` the GPU cascade against its CPU
oracle (`CascadeCheck`); each prints its report as JSON.

The viewer drives the core's world loop (`ov::World`, GPU Direct) with a fly camera and draws its pages with
`DawnPageHost` (the corners decoded by the same `PageFormat` GLSL as Godot's page shader, and the same materials:
`page_shading_glsl.hpp`, bark, leaves, a biome's ground, the ground's detail; no contour lines, no wind), lit like Godot's
main scenes (`dawn_look_glsl.hpp`): their sun with a shadow map (4096 texels over `--shadow_m=M`, default 250 m around the
camera, 0 off), the sky of Godot's `VoxelSky` (`src/sky_glsl.hpp`: its PhysicalSkyMaterial, without the clouds, at 3x its
exposure) where no page is, the fog towards it, Godot's ACES tonemapper and the grading LUT (`--grade=PATH|off`, default
`adapter/godot/project/grade_lut.rgb`, which `tools/make_lut.py` writes next to the PNG; preloaded on the web):

```
adapter/dawn/out/ov_dawn --viewer                          # an X11 window (XWayland), the Godot player's controls: mouse look,
                                                           # WASD / arrows, space up, shift down, wheel speed, Escape frees the mouse
                                                           # (a click takes it back), V cycles the views (1-3), Ctrl+Q quits;
                                                           # right click digs a hole, the left button held sprays (DigTool)
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
for a climate world; the web build preloads the Godot project's climate folder at `/climate`, so the Godot video preset's world is `?lods=25&climate=/climate/world_video.json`), placed like Godot's `forest5cm` preset by `--trees=4000 --tree_area=316 --voxel=0.05
--cam_pos=-52,320,-95 --above_ground=5 --pitch=-5 --yaw=-90`. Not there yet: frustum culling, GPU timestamps (no budget),
SSAO, the clouds, the sky's light from the sky itself (a constant haze colour stands in).

```
cmake -S adapter/dawn -B adapter/dawn/out -G Ninja -DOV_DAWN_DIR=<dawn install prefix> -DOV_GLSLANG_DIR=<glslang install prefix>
cmake --build adapter/dawn/out
adapter/dawn/out/ov_dawn --check
```

Dawn is built from github.com/google/dawn and installed with `cmake --install <build> --prefix <prefix>`, glslang from
github.com/KhronosGroup/glslang the same way. Without `OV_DAWN_DIR` only the core library (`ov_core`) builds.

## In a browser

The same viewer, compiled to WebAssembly against the browser's WebGPU (emdawnwebgpu), with glslang and Tint linked into
the wasm: the engine's GLSL goes to SPIR-V and then WGSL at run time, FSR's vendored SPIR-V too.

```
adapter/dawn/web/build.sh                 # once: Dawn, glslang, emdawnwebgpu, Tint, SPIR-V Tools for wasm (build_deps.sh)
python3 adapter/dawn/web/serve.py         # http://127.0.0.1:8095/: the scenes (web/index.html), or /ov_dawn.html
```

The page uses threads, which a browser only allows on a cross-origin isolated page: any other server must send
`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` as `serve.py` does. Where the
server can't (GitHub Pages), `web/coi-serviceworker.js` (vendored, MIT), which the page loads first, adds them from a
service worker: the first visit reloads once, isolated (not in a private window, which has no service workers).

GitHub Pages: `.github/workflows/pages.yml` builds the viewer on each pull request that touches it and publishes it on
each push to dev, at https://aobayama-gaming.github.io/opti-voxel/ (the landing page, `web/index.html`: a few scenes as
query strings). The wasm dependencies are cached per pin and `web/build_deps.sh` (the first build ~30 min).

Chrome
on Linux needs WebGPU switched on, and Vulkan, which its Wayland backend lacks, so it runs under X11 (XWayland):

```
google-chrome --ozone-platform=x11 --enable-unsafe-webgpu \
    --enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan --use-angle=vulkan http://127.0.0.1:8095/ov_dawn.html
```

Under Wayland its WebGPU canvas loses the device on the first frame (`vkAllocateMemory failed with
VK_ERROR_OUT_OF_DEVICE_MEMORY`); so does headless Chrome's, whose offscreen `?shot=` still works.

The URL query gives the flags: `ov_dawn.html?upscale=fsr3&pixelate=2&trees=1000000&tree_area=5000&voxel=0.05` is
`--viewer --upscale=fsr3 --pixelate=2 --trees=1000000 --tree_area=5000 --voxel=0.05` (the 1M forest at 5 cm; the
default voxel is 0.5 m); `?check` (or `frame-check`, `cascade-check`) runs that
check and `?shot=/x.png` renders offscreen into the page's file system (`Module.FS.readFile('/x.png')`). A click locks
the mouse, Escape frees it. `build.sh` needs emsdk (`EMSDK`, at the version of `web/emsdk_version.txt`) and fetches the
Dawn of `web/dawn_revision.txt` with its dependencies into `OV_DAWN_SRC` when there is no checkout there (it warns on
another revision); `web/deps` is the CMake project that adds the SPIRV-Tools Dawn's CMake leaves out under Emscripten.

What differs from native:
- **The blocking waits** (device creation, error scopes, readbacks, `sync`) suspend the wasm (JSPI: Firefox 139+,
  Chrome 137+); each frame awaits the browser's animation frame (`WebCanvas`).
- **Threads are Web Workers** (pthreads): the core's `ThreadPoolExecutor` as natively, one worker per core started with
  the page, and the discoverer's and the field's local devices each in their worker (a worker has its own WebGPU
  device; its waits suspend it like the main thread's). The 1M forest (FSR 3 x2, pixelate 2, lod error 1 | 3)
  loads in 10 s in Chrome (43 s without threads), 31 s in Firefox (89 s; no FSR there, so at the output size), 3 s
  natively, and the frames no longer stall while it does.
- **Slow readbacks**: Firefox 156 delivers a map ~100 ms after the GPU is done (Chrome ~9 ms, native well under 1). The
  site walk's GPU field waits for one per submission, so `FieldGPU` times one on its device when it starts and leaves
  the walks on the CPU past `SLOW_READBACK_MS` (30): the forest took 150 s in Firefox on the GPU field.
  `--walk_gpu=0` keeps the field on the CPU everywhere.
- **Push constants**: browsers have no immediates (Firefox) or not in the WGSL Tint writes: `compile_glsl` turns the
  push block into a uniform at set 1 and `DawnDevice` feeds it from a ring buffer, one 256-byte slot per dispatch.
- **FSR 3** needs `texture-formats-tier2`: Chrome has it, Firefox not yet (pixelate only, the 3D at the output size).

Tested here: Firefox 156 (headless, Playwright) draws the viewer at 60 fps; Chrome 154 under X11 too, FSR 3 included;
Chrome's offscreen `?shot=` matches Firefox and native Dawn pixel for pixel on the default scene.

Stock Tint drops SPIR-V's NoContraction (GLSL `precise`), so the driver fuses multiply-adds and the exact noise
differs from the CPU's by a few ulps: the check's noise part fails until that is fixed. `--frame-check` also fails on
the `base_at` probes' duplicate definition, as `tools/frame_test.sh` does on dev.

Keep this adapter thin: engine logic belongs in the core, where both adapters get it.
