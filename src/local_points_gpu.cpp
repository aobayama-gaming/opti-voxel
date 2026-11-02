#include "local_points_gpu.hpp"

#include "debug_log.hpp"
#include "ov/math.hpp"

bool LocalPointsGPU::create(const SDFBase *p_sdf, const ChunkParam &p_param, uint32_t p_max_points, LocalGpuThread *p_thread) {
    const uint64_t t0 = now_usec();
    if (p_thread != nullptr) {
        bool ok = false;
        p_thread->run([&](ov::gpu::Device *p_device) {
            // The passes a world stopped before freed (parking) are taken by these if built alike; the rest go.
            ok = p_device != nullptr && passes.create(p_device, p_sdf, p_param, p_max_points);
            if (p_device != nullptr) {
                p_device->drop_parked();
            }
        });
        if (!ok) {
            return false;
        }
        thread = p_thread;
        VLOG(1, "LocalPointsGPU: point passes on the kept device in ", double(now_usec() - t0) / 1000.0, " ms");
        return true;
    }
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

bool LocalPointsGPU::cull_values(const VoxelPos *p_points, int p_count, int p_lod, float *r_out, CascadePointsGPU::BatchStats &r_stats) {
    if (thread == nullptr) {
        return passes.cull_values(p_points, p_count, p_lod, r_out, r_stats);
    }
    bool ok = false;
    thread->run([&](ov::gpu::Device *) { ok = passes.cull_values(p_points, p_count, p_lod, r_out, r_stats); });
    return ok;
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
    if (thread != nullptr) {
        thread->run([this](ov::gpu::Device *p_device) {
            p_device->park(); // for the next world's create (drop_parked there, or with the device)
            passes.free();
        });
        thread = nullptr;
        return;
    }
    if (device == nullptr) {
        return;
    }
    passes.free();
    device.reset();
}
