#include "field_points_gpu.hpp"

#include "debug_log.hpp"
#include "godot_conv.hpp"
#include "gpu_frame.hpp"
#include "sdf_cascade.hpp"

bool FieldPointsGPU::create(RenderingDevice *p_rd, const SDFBase *p_sdf, const ChunkParam &p_param, uint32_t p_max_points) {
    ERR_FAIL_COND_V_MSG(p_rd == nullptr || p_sdf->get_cascade() == nullptr || p_max_points == 0, false,
            "FieldPointsGPU: needs a device, a cascade terrain and a batch size.");
    rd = p_rd;
    max_batch = p_max_points;
    const ChunkParam point_param = CascadeGLSL::point_param(p_param);
    input = rd->storage_buffer_create(int64_t(max_batch) * GPUFrame::CHUNK_INPUT_BYTES);
    const PackedByteArray param_bytes = godot_bytes(GPUFrame::chunk_param_bytes(point_param));
    param_ubo = rd->uniform_buffer_create(param_bytes.size(), param_bytes);
    values = rd->storage_buffer_create(int64_t(max_batch) * int64_t(sizeof(ExactNoise::Float4)));
    if (!passes.create_fields(rd, p_sdf, point_param, max_batch, input, param_ubo, values)) {
        free();
        return false;
    }
    input_bytes.resize(int64_t(max_batch) * GPUFrame::CHUNK_INPUT_BYTES);
    VLOG(1, "FieldPointsGPU: scratch ", double(passes.scratch_bytes()) / 1048576.0, " MB for ", int64_t(max_batch), " points per batch");
    return true;
}

void FieldPointsGPU::free() {
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
    rd = nullptr;
}

bool FieldPointsGPU::units(const VoxelPos *p_points, const int *p_lods, int p_count, ExactNoise::Float4 *r_out, BatchStats &r_stats) {
    ERR_FAIL_COND_V(!is_valid(), false);
    for (int base = 0; base < p_count; base += int(max_batch)) {
        if (!_run_batch(p_points + base, p_lods + base, uint32_t(MIN(int(max_batch), p_count - base)), r_out + base, r_stats)) {
            return false;
        }
    }
    return true;
}

bool FieldPointsGPU::ground_field(const VoxelPos *p_voxels, const int *p_lods, int p_count, ExactNoise::Float4 *r_out, BatchStats &r_stats) {
    lattice.resize(size_t(p_count));
    for (int i = 0; i < p_count; i++) {
        lattice[i] = SDFCascade::ground_lattice(p_voxels[i], p_lods[i]);
    }
    if (!units(lattice.data(), p_lods, p_count, r_out, r_stats)) {
        return false;
    }
    for (int i = 0; i < p_count; i++) {
        r_out[i] = SDFCascade::extend_field(r_out[i], lattice[i], p_voxels[i], p_lods[i]);
    }
    return true;
}

bool FieldPointsGPU::_run_batch(const VoxelPos *p_points, const int *p_lods, uint32_t p_count, ExactNoise::Float4 *r_out, BatchStats &r_stats) {
    // The batch's LOD range: every item runs its own levels (one above another's is zero until its own top).
    Vector2i lods(ChunkMath::MAX_LOD_NUMBERS, 0);
    for (uint32_t i = 0; i < p_count; i++) {
        const int lod = p_lods[i];
        const int64_t mask = (int64_t(1) << MAX(lod, 0)) - 1;
        ERR_FAIL_COND_V_MSG(lod < 0 || lod >= ChunkMath::MAX_LOD_NUMBERS || (p_points[i].x & mask) || (p_points[i].y & mask) || (p_points[i].z & mask), false,
                "FieldPointsGPU: a field point must be on its LOD's lattice.");
        GPUFrame::encode_field_item(input_bytes.ptrw() + int64_t(i) * GPUFrame::CHUNK_INPUT_BYTES, p_points[i], lod);
        lods.x = MIN(lods.x, lod);
        lods.y = MAX(lods.y, lod);
    }
    rd->buffer_update(input, 0, int64_t(p_count) * GPUFrame::CHUNK_INPUT_BYTES, input_bytes);
    const String serial = itos(int64_t(batch_serial++));
    const String name_begin = String("field_b") + serial;
    const String name_end = String("field_e") + serial;
    rd->capture_timestamp(name_begin);
    passes.record_all(lods, p_count);
    rd->capture_timestamp(name_end);
    rd->submit();
    rd->sync();

    // Timestamps are readable once the frame is synced, and only until the next frame begins: buffer_get_data flushes
    // one, so read them first (DiscovererGPU).
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

    const int64_t bytes = int64_t(p_count) * 16;
    const PackedByteArray out = rd->buffer_get_data(values, 0, bytes);
    ERR_FAIL_COND_V(out.size() != bytes, false);
    const float *f = reinterpret_cast<const float *>(out.ptr());
    for (uint32_t i = 0; i < p_count; i++) {
        r_out[i] = ExactNoise::Float4{ f[4 * i], f[4 * i + 1], f[4 * i + 2], f[4 * i + 3] };
    }
    r_stats.batches++;
    r_stats.points += p_count;
    r_stats.readback_bytes += bytes;
    return true;
}
