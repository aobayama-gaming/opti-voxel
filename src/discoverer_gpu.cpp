#include "discoverer_gpu.hpp"

#include "debug_log.hpp"
#include "godot_conv.hpp"
#include "gpu_frame.hpp"

#include <godot_cpp/classes/rendering_server.hpp>

#include <cstring>

bool DiscovererGPU::create(const SDFBase *p_sdf, const ChunkParam &p_param) {
    const uint64_t t0 = now_usec();
    rd = RenderingServer::get_singleton()->create_local_rendering_device();
    if (rd == nullptr) {
        return false;
    }
    const uint64_t t_device = now_usec();
    point_param = CascadeGLSL::point_param(p_param);
    input = rd->storage_buffer_create(MAX_POINTS * GPUFrame::CHUNK_INPUT_BYTES);
    const PackedByteArray param_bytes = godot_bytes(GPUFrame::chunk_param_bytes(point_param));
    param_ubo = rd->uniform_buffer_create(param_bytes.size(), param_bytes);
    values = rd->storage_buffer_create(MAX_POINTS * sizeof(float));
    if (!passes.create_points(rd, p_sdf, point_param, MAX_POINTS, input, param_ubo, values)) {
        free();
        return false;
    }
    input_bytes.resize(MAX_POINTS * GPUFrame::CHUNK_INPUT_BYTES);
    VLOG(1, "DiscovererGPU: local device in ", double(t_device - t0) / 1000.0, " ms, point passes (shader compile) in ", double(now_usec() - t_device) / 1000.0,
            " ms, scratch ", double(passes.scratch_bytes()) / 1048576.0, " MB for ", int64_t(MAX_POINTS), " points per batch");
    return true;
}

void DiscovererGPU::free() {
    if (rd == nullptr) {
        return;
    }
    // The passes' uniform set uses the buffers freed below.
    passes.free();
    for (const RID &rid : { input, param_ubo, values }) {
        if (rid.is_valid()) {
            rd->free_rid(rid);
        }
    }
    input = RID();
    param_ubo = RID();
    values = RID();
    memdelete(rd);
    rd = nullptr;
}

bool DiscovererGPU::evaluate(const VoxelPos *p_points, int p_count, int p_lod, float *r_out, BatchStats &r_stats) {
    ERR_FAIL_COND_V(!is_valid(), false);
    for (int base = 0; base < p_count; base += int(MAX_POINTS)) {
        if (!_run_batch(p_points + base, uint32_t(MIN(int(MAX_POINTS), p_count - base)), p_lod, r_out + base, r_stats)) {
            return false;
        }
    }
    return true;
}

bool DiscovererGPU::_run_batch(const VoxelPos *p_points, uint32_t p_count, int p_lod, float *r_out, BatchStats &r_stats) {
    for (uint32_t i = 0; i < p_count; i++) {
        if (!CascadeGLSL::encode_point_input(input_bytes, i, p_points[i], p_lod, point_param)) {
            return false; // not a node corner: no GPU value for it
        }
    }
    rd->buffer_update(input, 0, int64_t(p_count) * GPUFrame::CHUNK_INPUT_BYTES, input_bytes);
    const String serial = itos(int64_t(batch_serial++));
    const String name_begin = String("disc_b") + serial;
    const String name_end = String("disc_e") + serial;
    rd->capture_timestamp(name_begin);
    passes.record_all(Vector2i(p_lod, p_lod), p_count);
    rd->capture_timestamp(name_end);
    rd->submit();
    rd->sync();

    // A local device's timestamps are readable once its frame is synced, and only until the next frame
    // begins: buffer_get_data flushes one, so read them first.
    int64_t tb = -1;
    int64_t te = -1;
    for (uint32_t i = 0; i < rd->get_captured_timestamps_count(); i++) {
        const String name = rd->get_captured_timestamp_name(i);
        if (name == name_begin) {
            tb = int64_t(rd->get_captured_timestamp_gpu_time(i));
        } else if (name == name_end) {
            te = int64_t(rd->get_captured_timestamp_gpu_time(i));
        }
    }
    if (tb >= 0 && te >= tb && r_stats.gpu_ms >= 0.0) {
        r_stats.gpu_ms += double(te - tb) / 1e6;
    } else {
        r_stats.gpu_ms = -1.0;
    }

    const int64_t bytes = int64_t(p_count) * int64_t(sizeof(float));
    const PackedByteArray out = rd->buffer_get_data(values, 0, bytes);
    ERR_FAIL_COND_V(out.size() != bytes, false);
    std::memcpy(r_out, out.ptr(), size_t(bytes));
    r_stats.batches++;
    r_stats.readback_bytes += bytes;
    return true;
}
