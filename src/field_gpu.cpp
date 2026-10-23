#include "field_gpu.hpp"

#include "cascade_points_gpu.hpp"
#include "debug_log.hpp"

#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rendering_server.hpp>

#include <vector>

using namespace godot;

void FieldGPU::start(const SDFBase *p_sdf, const ChunkParam &p_param, uint32_t p_scratch_mb) {
    stop();
    {
        const std::lock_guard<std::mutex> lock(mutex);
        stopping = false;
    }
    state_now = STARTING;
    thread = std::thread(&FieldGPU::loop, this, p_sdf, p_param, uint64_t(p_scratch_mb) << 20);
}

void FieldGPU::stop() {
    {
        const std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
        state_now = FAILED; // no request queued from now on
    }
    wake.notify_all();
    if (thread.joinable()) {
        thread.join();
    }
}

bool FieldGPU::ground_field(const VoxelPos *p_voxels, const int *p_lods, int p_count, ExactNoise::Float4 *r_out) {
    if (p_count <= 0) {
        return true;
    }
    Request r;
    r.voxels = p_voxels;
    r.lods = p_lods;
    r.count = p_count;
    r.out = r_out;
    const uint64_t t0 = now_usec();
    std::unique_lock<std::mutex> lock(mutex);
    // Checked under the lock: a failing thread sets FAILED under it, then answers what it finds queued.
    if (stopping || state_now.load() != READY) {
        return false;
    }
    queue.push_back(&r);
    requests++;
    wake.notify_one();
    answered.wait(lock, [&r] { return r.done; });
    wait_ms += double(now_usec() - t0) / 1000.0;
    return r.ok;
}

void FieldGPU::loop(const SDFBase *p_sdf, ChunkParam p_param, uint64_t p_scratch_bytes) {
    const uint64_t t0 = now_usec();
    RenderingDevice *rd = RenderingServer::get_singleton()->create_local_rendering_device();
    CascadePointsGPU passes;
    const uint64_t per_point = CascadeGLSL::point_scratch_bytes(p_sdf);
    const uint32_t batch = uint32_t(CLAMP(p_scratch_bytes / MAX(per_point, uint64_t(1)), uint64_t(MIN_POINTS), uint64_t(MAX_POINTS)));
    const bool ok = rd != nullptr && passes.create(rd, p_sdf, p_param, batch);
    {
        const std::lock_guard<std::mutex> lock(mutex);
        init_ms = double(now_usec() - t0) / 1000.0;
        max_points = batch;
        scratch_mb = double(passes.scratch_bytes()) / 1048576.0;
        if (ok && !stopping) {
            state_now = READY;
        }
    }
    if (!ok) {
        WARN_PRINT(rd == nullptr ? "FieldGPU: no local RenderingDevice (headless or the Compatibility renderer): the site walk reads the field on the CPU."
                                 : "FieldGPU: the point passes failed to build: the site walk reads the field on the CPU.");
    } else {
        VLOG(1, "FieldGPU: ready in ", init_ms, " ms, ", int64_t(batch), " points per batch, scratch ", scratch_mb, " MB");
    }
    std::vector<VoxelPos> voxels;
    std::vector<int> lods;
    std::vector<ExactNoise::Float4> out;
    std::vector<Request *> taken;
    uint32_t cap = batch; // a submission's points (pacing)
    bool good = ok;
    while (good) {
        {
            std::unique_lock<std::mutex> lock(mutex);
            wake.wait(lock, [this] { return stopping || !queue.empty(); });
            if (stopping) {
                break;
            }
            // Every request waiting, up to a submission's points (a larger one alone: split below).
            taken.clear();
            int64_t n = 0;
            while (!queue.empty() && (taken.empty() || n + queue.front()->count <= int64_t(cap))) {
                n += queue.front()->count;
                taken.push_back(queue.front());
                queue.pop_front();
            }
        }
        voxels.clear();
        lods.clear();
        for (const Request *r : taken) {
            voxels.insert(voxels.end(), r->voxels, r->voxels + r->count);
            lods.insert(lods.end(), r->lods, r->lods + r->count);
        }
        out.resize(voxels.size());
        CascadePointsGPU::BatchStats stats;
        const uint64_t t_run = now_usec();
        for (size_t first = 0; good && first < voxels.size(); first += cap) {
            const int count = int(MIN(voxels.size() - first, size_t(cap)));
            good = passes.ground_field(voxels.data() + first, lods.data() + first, count, out.data() + first, stats);
        }
        size_t at = 0;
        for (Request *r : taken) {
            if (good) {
                std::copy(out.begin() + int64_t(at), out.begin() + int64_t(at + size_t(r->count)), r->out);
            }
            at += size_t(r->count);
        }
        {
            const std::lock_guard<std::mutex> lock(mutex);
            for (Request *r : taken) {
                r->done = true;
                r->ok = good;
            }
            submissions++;
            points += int64_t(voxels.size());
            gpu_ms += MAX(stats.gpu_ms, 0.0);
            if (stats.gpu_ms > 0.0) {
                // The points' cost, smoothed; the cap follows it.
                const double measured = stats.gpu_ms * 1000.0 / double(voxels.size());
                us_per_point = us_per_point > 0.0 ? 0.8 * us_per_point + 0.2 * measured : measured;
                cap = uint32_t(CLAMP(SUBMIT_GPU_MS * 1000.0 / us_per_point, double(MIN_POINTS), double(batch)));
            }
            busy_ms += double(now_usec() - t_run) / 1000.0;
            record_ms += stats.record_ms;
            sync_ms += stats.sync_ms;
            readback_ms += stats.readback_ms;
            if (!good) {
                state_now = FAILED; // under the lock: no caller queues after it
                ERR_PRINT("FieldGPU: a batch failed: the site walk reads the field on the CPU from now on.");
            }
        }
        answered.notify_all();
    }
    // Stopped or failed: answer what is still queued (the callers read the CPU's field), then free the device here.
    {
        const std::lock_guard<std::mutex> lock(mutex);
        state_now = FAILED;
        for (Request *r : queue) {
            r->done = true;
            r->ok = false;
        }
        queue.clear();
    }
    answered.notify_all();
    passes.free();
    if (rd != nullptr) {
        memdelete(rd);
    }
}

Dictionary FieldGPU::get_stats() const {
    const std::lock_guard<std::mutex> lock(mutex);
    Dictionary d;
    const State s = state_now.load();
    d["state"] = s == READY ? "ready" : (s == STARTING ? "starting" : "off");
    d["init_ms"] = init_ms;
    d["points_per_batch"] = int64_t(max_points);
    d["us_per_point"] = us_per_point;
    d["scratch_mb"] = scratch_mb;
    d["requests"] = requests;
    d["submissions"] = submissions;
    d["points"] = points;
    d["points_per_submission"] = submissions > 0 ? double(points) / double(submissions) : 0.0;
    d["gpu_ms"] = gpu_ms;
    d["busy_ms"] = busy_ms;
    d["record_ms"] = record_ms;
    d["sync_ms"] = sync_ms;
    d["readback_ms"] = readback_ms;
    d["wait_ms"] = wait_ms;
    return d;
}
