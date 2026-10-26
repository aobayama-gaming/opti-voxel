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
    float lod_error_back = 0.0f; // 0: lod_error everywhere
    int lods = 15;
    int terrain_mode = 1; // World::Settings::terrain_mode
    float cam_pos[3] = {}; // render metres; y is then moved to above_ground over the terrain
    float above_ground = 50.0f; // the camera starts this many metres above the terrain under it
    int trees = 0; // World::Settings::archetype_trees, over a tree_area metres square
    float tree_area = 60.0f;
    std::string climate; // World::Settings::climate_path
    float pitch_deg = -15.0f;
    float yaw_deg = 0.0f;
    int view = 0; // 0 shaded, 1 normals, 2 LOD
    float fly_speed = 0.0f; // metres per second straight ahead, a scripted fly on top of the controls
    int discoverer_mode = -1; // World::Settings::discoverer_mode
    bool walk_gpu = true; // World::Settings::walk_gpu
    DawnUpscaler::Settings upscale; // FSR or pixelate > 1: through DawnUpscaler
};

// 0 on success, 1 on a failure (no window, no device, the shot not written).
int run_viewer(DawnContext &p_context, const ViewerOptions &p_options);

} // namespace ov::dawn
