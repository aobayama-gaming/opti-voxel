# Dawn adapter

The second adapter of the same engine: the core in `src/` (the files of `src/core_files.txt`) driven
through Dawn (native WebGPU) instead of Godot. Today it brings the core up on a headless Dawn device; the
GPU interface both adapters implement comes next, so the meshers run here unchanged.

```
cmake -S adapter/dawn -B adapter/dawn/out -G Ninja -DOV_DAWN_DIR=<dawn install prefix>
cmake --build adapter/dawn/out
adapter/dawn/out/ov_dawn
```

Dawn is built from github.com/google/dawn and installed with `cmake --install <build> --prefix <prefix>`.
Without `OV_DAWN_DIR` only the core library (`ov_core`) builds.

Keep this adapter thin: engine logic belongs in the core, where both adapters get it.
