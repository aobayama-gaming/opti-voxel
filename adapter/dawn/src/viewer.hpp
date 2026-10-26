#pragma once

// The Dawn viewer: ov::World driven by a fly camera, its pages drawn by DawnPageHost, in an X11 window or offscreen into
// a PNG (--shot: once the load is complete, then exit).

#include "dawn_context.hpp"
#include "dawn_upscaler.hpp"

#include <string>

namespace ov::dawn {

struct ViewerOptions {
    int width = 1280;
    int height = 720;
    std::string shot; // a PNG path: offscreen, no window
    int max_frames = 6000; // a shot is taken by then even if the load isn't complete
    float voxel_size = 0.5f;
    float lod_error = 2.0f;
    int lods = 15;
    int terrain_mode = 1; // World::Settings::terrain_mode
    float above_ground = 50.0f; // the camera starts this many metres above the terrain under it
    float pitch_deg = -15.0f;
    float yaw_deg = 0.0f;
    int view = 0; // 0 shaded, 1 normals, 2 LOD
    DawnUpscaler::Settings upscale; // FSR or pixelate > 1: through DawnUpscaler
};

// 0 on success, 1 on a failure (no window, no device, the shot not written).
int run_viewer(DawnContext &p_context, const ViewerOptions &p_options);

} // namespace ov::dawn
