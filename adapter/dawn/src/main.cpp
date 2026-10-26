// The Dawn adapter's host: the engine on a headless Dawn device.
//
//   ov_dawn            brings the device up
//   ov_dawn --check                   runs the engine's GPU device check (GPUDeviceCheck)
//   ov_dawn --frame-check[=VOXEL_M]   the integer frame and exact noise, GPU vs CPU (FrameCheck)
//   ov_dawn --cascade-check[=JSON]    the GPU cascade vs its CPU oracle (CascadeCheck; JSON: its options, e.g.
//                                     '{"lods":15,"timing":false}')
// Each prints its report and exits 1 on a failure.
//   ov_dawn --viewer [options]        the terrain in a window, the Godot player's controls (README.md)
//   ov_dawn --shot=PATH [options]     the same offscreen: a PNG once the load is complete, its report as JSON
// Viewer options: --size=WxH --voxel=M --lod_error=F --lods=N --terrain=cascade|continents --above_ground=M
// --pitch=DEG --yaw=DEG --view=0|1|2 --max_frames=N --upscale=temporal --render_scale=F --pixelate=N --sharpness=STOPS.

#include "dawn_context.hpp"
#include "dawn_device.hpp"

#include "cascade_check.hpp"
#include "frame_check.hpp"
#include "gpu_device_check.hpp"
#include "ov/json.hpp"
#include "ov/log.hpp"
#include "viewer.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

// argv[i] as "--name" or "--name=value": true and r_value set (empty without '=').
bool flag(const char *p_arg, const char *p_name, std::string &r_value) {
    const size_t n = std::strlen(p_name);
    if (std::strncmp(p_arg, p_name, n) != 0 || (p_arg[n] != '\0' && p_arg[n] != '=')) {
        return false;
    }
    r_value = p_arg[n] == '=' ? p_arg + n + 1 : "";
    return true;
}

int print_report(const ov::Value &p_report) {
    std::printf("%s\n", ov::to_json(p_report).c_str());
    return p_report["ok"].as_bool() ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
    ov::log_level() = ov::LOG_INFO;
    ov::dawn::DawnContext context;
    if (!context.initialize()) {
        return 1;
    }
    ov::dawn::ViewerOptions viewer;
    bool run_viewer = false;
    for (int i = 1; i < argc; i++) {
        std::string value;
        if (flag(argv[i], "--viewer", value)) {
            run_viewer = true;
        } else if (flag(argv[i], "--shot", value)) {
            viewer.shot = value;
            run_viewer = true;
        } else if (flag(argv[i], "--size", value)) {
            std::sscanf(value.c_str(), "%dx%d", &viewer.width, &viewer.height);
        } else if (flag(argv[i], "--voxel", value)) {
            viewer.voxel_size = std::strtof(value.c_str(), nullptr);
        } else if (flag(argv[i], "--lod_error", value)) {
            viewer.lod_error = std::strtof(value.c_str(), nullptr);
        } else if (flag(argv[i], "--lods", value)) {
            viewer.lods = std::atoi(value.c_str());
        } else if (flag(argv[i], "--terrain", value)) {
            viewer.terrain_mode = value == "continents" ? 2 : 1;
        } else if (flag(argv[i], "--above_ground", value)) {
            viewer.above_ground = std::strtof(value.c_str(), nullptr);
        } else if (flag(argv[i], "--pitch", value)) {
            viewer.pitch_deg = std::strtof(value.c_str(), nullptr);
        } else if (flag(argv[i], "--yaw", value)) {
            viewer.yaw_deg = std::strtof(value.c_str(), nullptr);
        } else if (flag(argv[i], "--view", value)) {
            viewer.view = std::atoi(value.c_str());
        } else if (flag(argv[i], "--max_frames", value)) {
            viewer.max_frames = std::atoi(value.c_str());
        } else if (flag(argv[i], "--upscale", value)) {
            viewer.upscale.temporal = value == "temporal";
            if (!viewer.upscale.temporal && value != "off") {
                std::fprintf(stderr, "--upscale=%s: expected temporal or off\n", value.c_str());
                return 1;
            }
        } else if (flag(argv[i], "--render_scale", value)) {
            viewer.upscale.render_scale = std::strtof(value.c_str(), nullptr);
        } else if (flag(argv[i], "--pixelate", value)) {
            viewer.upscale.pixelate = std::atoi(value.c_str());
        } else if (flag(argv[i], "--sharpness", value)) {
            viewer.upscale.sharpness = std::strtof(value.c_str(), nullptr);
        }
    }
    if (run_viewer) {
        return ov::dawn::run_viewer(context, viewer);
    }
    for (int i = 1; i < argc; i++) {
        std::string value;
        if (flag(argv[i], "--check", value)) {
            ov::dawn::DawnDevice device(context);
            const GPUDeviceCheck::Report report = GPUDeviceCheck::run(device);
            std::printf("%s\n", report.summary().c_str());
            return report.ok() ? 0 : 1;
        }
        if (flag(argv[i], "--frame-check", value)) {
            ov::dawn::DawnDevice device(context);
            return print_report(FrameCheck::run(device, value.empty() ? 0.5f : std::strtof(value.c_str(), nullptr)));
        }
        if (flag(argv[i], "--cascade-check", value)) {
            ov::str error;
            const ov::Value options = value.empty() ? ov::Value::object() : ov::parse_json(value, error);
            if (!error.empty()) {
                std::fprintf(stderr, "--cascade-check: %s\n", error.c_str());
                return 1;
            }
            ov::dawn::DawnDevice device(context);
            return print_report(CascadeCheck::run(device, options));
        }
    }
    return 0;
}
