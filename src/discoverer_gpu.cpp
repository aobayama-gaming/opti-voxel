#include "discoverer_gpu.hpp"

#include "debug_log.hpp"

#include <godot_cpp/classes/rendering_server.hpp>

bool DiscovererGPU::create(const SDFBase *p_sdf, const ChunkParam &p_param) {
    const uint64_t t0 = now_usec();
    rd = RenderingServer::get_singleton()->create_local_rendering_device();
    if (rd == nullptr) {
        return false;
    }
    const uint64_t t_device = now_usec();
    if (!points.create(rd, p_sdf, p_param, MAX_POINTS)) {
        free();
        return false;
    }
    VLOG(1, "DiscovererGPU: local device in ", double(t_device - t0) / 1000.0, " ms, point passes (shader compile) in ", double(now_usec() - t_device) / 1000.0,
            " ms, scratch ", double(points.scratch_bytes()) / 1048576.0, " MB for ", int64_t(MAX_POINTS), " points per batch");
    return true;
}

void DiscovererGPU::free() {
    if (rd == nullptr) {
        return;
    }
    points.free();
    memdelete(rd);
    rd = nullptr;
}
