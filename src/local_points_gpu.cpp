#include "local_points_gpu.hpp"

#include "debug_log.hpp"

#include <godot_cpp/classes/rendering_server.hpp>

bool LocalPointsGPU::create(const SDFBase *p_sdf, const ChunkParam &p_param, uint32_t p_max_points) {
    const uint64_t t0 = now_usec();
    rd = RenderingServer::get_singleton()->create_local_rendering_device();
    if (rd == nullptr) {
        return false;
    }
    const uint64_t t_device = now_usec();
    if (!passes.create(rd, p_sdf, p_param, p_max_points)) {
        free();
        return false;
    }
    VLOG(1, "LocalPointsGPU: local device in ", double(t_device - t0) / 1000.0, " ms, point passes (shader compile) in ", double(now_usec() - t_device) / 1000.0,
            " ms, scratch ", double(passes.scratch_bytes()) / 1048576.0, " MB for ", int64_t(p_max_points), " points per batch");
    return true;
}

void LocalPointsGPU::free() {
    if (rd == nullptr) {
        return;
    }
    passes.free();
    memdelete(rd);
    rd = nullptr;
}
