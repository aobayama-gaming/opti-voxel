#include "local_points_gpu.hpp"

#include "debug_log.hpp"
#include "ov/math.hpp"

bool LocalPointsGPU::create(const SDFBase *p_sdf, const ChunkParam &p_param, uint32_t p_max_points) {
    const uint64_t t0 = now_usec();
    device = ov::gpu::create_local_device();
    if (device == nullptr) {
        return false;
    }
    const uint64_t t_device = now_usec();
    if (!passes.create(device.get(), p_sdf, p_param, p_max_points)) {
        free();
        return false;
    }
    VLOG(1, "LocalPointsGPU: local device in ", double(t_device - t0) / 1000.0, " ms, point passes (shader compile) in ", double(now_usec() - t_device) / 1000.0,
            " ms, scratch ", double(passes.scratch_bytes()) / 1048576.0, " MB for ", int64_t(p_max_points), " points per batch");
    return true;
}

double LocalPointsGPU::readback_ms() {
    const uint32_t word = 0;
    const ov::gpu::Buffer buffer = device->create_buffer(ov::gpu::BufferKind::STORAGE, sizeof(word), &word);
    device->read_buffer(buffer, 0, sizeof(word)); // the first pays the device's warm-up (Chrome: ~50 ms, then ~9)
    double ms = 0.0;
    for (int i = 0; i < 3; i++) { // the fastest of three: a stall during the start-up (shader compiles) doesn't count
        const uint64_t t0 = now_usec();
        device->read_buffer(buffer, 0, sizeof(word));
        const double t = double(now_usec() - t0) / 1000.0;
        ms = i == 0 ? t : ov::min(ms, t);
    }
    device->free(buffer);
    return ms;
}

void LocalPointsGPU::free() {
    if (device == nullptr) {
        return;
    }
    passes.free();
    device.reset();
}
