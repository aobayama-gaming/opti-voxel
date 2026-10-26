# FSR 3.1 upscaler shaders

The nine SPIR-V modules here are the reference FidelityFX Super Resolution 3.1 upscaler passes,
compiled by the [`wgpu-ffx`](https://github.com/cwfitzgerald/wgpu-ffx) port from the
[AMD FidelityFX SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK) v1.1.4
(`ffx_fsr3upscaler_*_pass_*.spv`). The non-HDR, non-inverted-depth, non-jittered-motion-vector,
low-resolution-motion-vector, reference-Lanczos, FP32, non-wave64 variant is vendored.

License: MIT, see `LICENSE.MIT` (the wgpu-ffx port's license; the FidelityFX SDK is MIT too).
`dawn/src/fsr3_shaders.hpp` embeds these files as byte arrays so the executable needs no asset
directory at runtime.
