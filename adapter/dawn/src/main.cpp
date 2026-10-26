// The Dawn adapter's host: the engine on a headless Dawn device.
//
//   ov_dawn            brings the device up
//   ov_dawn --check    runs the engine's GPU device check (GPUDeviceCheck), exit 1 on a failure

#include "dawn_context.hpp"
#include "dawn_device.hpp"

#include "gpu_device_check.hpp"
#include "ov/log.hpp"

#include <cstdio>
#include <cstring>

int main(int argc, char **argv) {
    ov::log_level() = ov::LOG_INFO;
    ov::dawn::DawnContext context;
    if (!context.initialize()) {
        return 1;
    }
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--check") == 0) {
            ov::dawn::DawnDevice device(context);
            const GPUDeviceCheck::Report report = GPUDeviceCheck::run(device);
            std::printf("%s\n", report.summary().c_str());
            return report.ok() ? 0 : 1;
        }
    }
    return 0;
}
