// The Dawn adapter's host: the engine on a headless Dawn device.
//
//   ov_dawn            brings the device up
//   ov_dawn --check                   runs the engine's GPU device check (GPUDeviceCheck)
//   ov_dawn --frame-check[=VOXEL_M]   the integer frame and exact noise, GPU vs CPU (FrameCheck)
//   ov_dawn --cascade-check[=JSON]    the GPU cascade vs its CPU oracle (CascadeCheck; JSON: its options, e.g.
//                                     '{"lods":15,"timing":false}')
// Each prints its report and exits 1 on a failure.

#include "dawn_context.hpp"
#include "dawn_device.hpp"

#include "cascade_check.hpp"
#include "frame_check.hpp"
#include "gpu_device_check.hpp"
#include "ov/json.hpp"
#include "ov/log.hpp"

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
