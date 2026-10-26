# Dawn adapter

The second adapter of the same engine: the core in `src/` (every file of `src/core_files.txt`, the meshers and
page pool included) driven through Dawn (native WebGPU) instead of Godot. `DawnDevice` is the engine's
`ov::gpu::Device` (glslang compiles the engine's GLSL to SPIR-V, Tint takes it from there); `ov_dawn --check`
runs `GPUDeviceCheck` on it (buffers, textures, readbacks, then the exact noise bit for bit). Drawing (a
`PageHost`) and the world loop come next.

```
cmake -S adapter/dawn -B adapter/dawn/out -G Ninja -DOV_DAWN_DIR=<dawn install prefix> -DOV_GLSLANG_DIR=<glslang install prefix>
cmake --build adapter/dawn/out
adapter/dawn/out/ov_dawn --check
```

Dawn is built from github.com/google/dawn and installed with `cmake --install <build> --prefix <prefix>`, glslang from
github.com/KhronosGroup/glslang the same way. Without `OV_DAWN_DIR` only the core library (`ov_core`) builds.

Stock Tint drops SPIR-V's NoContraction (GLSL `precise`), so the driver fuses multiply-adds and the exact noise
differs from the CPU's by a few ulps: the check's noise part fails until that is fixed.

Keep this adapter thin: engine logic belongs in the core, where both adapters get it.
