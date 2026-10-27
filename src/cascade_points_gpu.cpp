#include "cascade_points_gpu.hpp"

#include "debug_log.hpp"
#include "gpu_frame.hpp"
#include "sdf_cascade.hpp"

#include "ov/math.hpp"
#include "ov/on_error.hpp"

#include <cstring>

static_assert(sizeof(ExactNoise::Float4) == 16, "the point passes' register is a vec4");

using ov::gpu::BufferKind;

bool CascadePointsGPU::create(ov::gpu::Device *p_device, const SDFBase *p_sdf, const ChunkParam &p_param, uint32_t p_max_points) {
    OV_ERR_FAIL_COND_V_MSG(p_device == nullptr || p_sdf->get_cascade() == nullptr || p_max_points == 0, false,
            "CascadePointsGPU: needs a device, a cascade terrain and a batch size.");
    device = p_device;
    voxel_size = p_param.voxel_size;
    max_batch = p_max_points;
    const ChunkParam point_param = CascadeGLSL::point_param(p_param);
    input = device->create_buffer(BufferKind::STORAGE, uint64_t(max_batch) * GPUFrame::CHUNK_INPUT_BYTES);
    const ov::bytes param_bytes = GPUFrame::chunk_param_bytes(point_param);
    param_ubo = device->create_buffer(BufferKind::UNIFORM, param_bytes.size(), param_bytes.data());
    values = device->create_buffer(BufferKind::STORAGE, uint64_t(max_batch) * sizeof(ExactNoise::Float4));
    if (!passes.create_points(device, p_sdf, point_param, max_batch, input, param_ubo, values)) {
        free();
        return false;
    }
    input_bytes.resize(size_t(max_batch) * GPUFrame::CHUNK_INPUT_BYTES);
    VLOG(1, "CascadePointsGPU: scratch ", double(passes.scratch_bytes()) / 1048576.0, " MB for ", int64_t(max_batch), " points per batch");
    return true;
}

void CascadePointsGPU::free() {
    if (device == nullptr) {
        return;
    }
    // The passes' bind sets use the buffers freed below.
    passes.free();
    for (const ov::gpu::Buffer b : { input, param_ubo, values }) {
        if (b.is_valid()) {
            device->free(b);
        }
    }
    input = ov::gpu::Buffer();
    param_ubo = ov::gpu::Buffer();
    values = ov::gpu::Buffer();
    device = nullptr;
}

bool CascadePointsGPU::units(const VoxelPos *p_points, const int *p_lods, int p_count, ExactNoise::Float4 *r_out, BatchStats &r_stats) {
    OV_ERR_FAIL_COND_V(!is_valid(), false);
    for (int base = 0; base < p_count; base += int(max_batch)) {
        if (!_run_batch(p_points + base, p_lods + base, uint32_t(ov::min(int(max_batch), p_count - base)), r_out + base, r_stats)) {
            return false;
        }
    }
    return true;
}

bool CascadePointsGPU::ground_field(const VoxelPos *p_voxels, const int *p_lods, int p_count, ExactNoise::Float4 *r_out, BatchStats &r_stats) {
    std::vector<VoxelPos> lattice(static_cast<size_t>(p_count));
    for (int i = 0; i < p_count; i++) {
        lattice[i] = SDFBase::ground_lattice(p_voxels[i], p_lods[i]);
    }
    if (!units(lattice.data(), p_lods, p_count, r_out, r_stats)) {
        return false;
    }
    for (int i = 0; i < p_count; i++) {
        r_out[i] = SDFBase::extend_field(r_out[i], lattice[i], p_voxels[i], p_lods[i]);
    }
    return true;
}

bool CascadePointsGPU::cull_values(const VoxelPos *p_points, int p_count, int p_lod, float *r_out, BatchStats &r_stats) {
    const std::vector<int> lods(static_cast<size_t>(p_count), p_lod);
    std::vector<ExactNoise::Float4> u(static_cast<size_t>(p_count));
    if (!units(p_points, lods.data(), p_count, u.data(), r_stats)) {
        return false;
    }
    for (int i = 0; i < p_count; i++) {
        r_out[i] = SDFCascade::to_metres(u[i], p_lod, voxel_size).w;
    }
    return true;
}

bool CascadePointsGPU::_run_batch(const VoxelPos *p_points, const int *p_lods, uint32_t p_count, ExactNoise::Float4 *r_out, BatchStats &r_stats) {
    const uint64_t t0 = now_usec();
    // The batch's LOD range: every item runs its own levels (one above another's is zero until its own top).
    ov::ivec2 lods(ChunkMath::MAX_LOD_NUMBERS, 0);
    uint8_t *items = input_bytes.data();
    for (uint32_t i = 0; i < p_count; i++) {
        const int lod = p_lods[i];
        OV_ERR_FAIL_COND_V_MSG(!GPUFrame::encode_point_item(items + int64_t(i) * GPUFrame::CHUNK_INPUT_BYTES, p_points[i], lod), false,
                "CascadePointsGPU: a point must be on its LOD's lattice.");
        lods.x = ov::min(lods.x, lod);
        lods.y = ov::max(lods.y, lod);
    }
    passes.upload_items(items, p_count);
    const uint64_t batch = ++batch_serial;
    const ov::str name_begin = "points_b" + ov::itos(int64_t(batch));
    const ov::str name_end = "points_e" + ov::itos(int64_t(batch));
    device->timestamp(name_begin);
    passes.record_all(lods, p_count);
    device->timestamp(name_end);
    // The registers' copy rides in the same submission (its callback runs in sync()); a blocking buffer_get_data after
    // the sync costs a second round trip on a local device (~4 ms). A callback that comes late, after the fallback took
    // over, is told apart by the batch's serial.
    const int64_t bytes = int64_t(p_count) * int64_t(sizeof(ExactNoise::Float4));
    const bool queued = async_readback;
    if (queued) {
        Delivery *to = &delivery;
        device->read_buffer_async(values, 0, uint64_t(bytes), [to, batch](const ov::bytes &p_data) {
            to->bytes = p_data;
            to->serial = batch;
        });
    }
    const uint64_t t_recorded = now_usec();
    device->sync();
    const uint64_t t_synced = now_usec();

    // A local device's timestamps are readable once its frame is synced, and only until the next frame begins:
    // a blocking read_buffer flushes one, so read them first.
    int64_t tb = -1;
    int64_t te = -1;
    for (const ov::gpu::Timestamp &t : device->timestamps()) {
        if (t.name == name_begin) {
            tb = int64_t(t.gpu_ns);
        } else if (t.name == name_end) {
            te = int64_t(t.gpu_ns);
        }
    }
    if (tb >= 0 && te >= tb && r_stats.gpu_ms >= 0.0) {
        r_stats.gpu_ms += double(te - tb) / 1e6;
    } else {
        r_stats.gpu_ms = -1.0;
    }

    ov::bytes out;
    if (queued && delivery.serial == batch) {
        out = std::move(delivery.bytes);
    } else {
        if (queued) {
            OV_WARN_PRINT("CascadePointsGPU: the readback didn't come with the sync: read_buffer from now on.");
            async_readback = false;
        }
        out = device->read_buffer(values, 0, uint64_t(bytes));
    }
    OV_ERR_FAIL_COND_V(int64_t(out.size()) != bytes, false);
    std::memcpy(r_out, out.data(), size_t(bytes));
    r_stats.batches++;
    r_stats.readback_bytes += bytes;
    r_stats.record_ms += double(t_recorded - t0) / 1000.0;
    r_stats.sync_ms += double(t_synced - t_recorded) / 1000.0;
    r_stats.readback_ms += double(now_usec() - t_synced) / 1000.0;
    return true;
}
