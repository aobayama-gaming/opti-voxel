#include "field_gpu.hpp"

#include "ov/log.hpp"
#include "ov/math.hpp"
#include "ov/on_error.hpp"
#include "ov/value.hpp"
#include "trace.hpp"

#include <vector>

void FieldGPU::start(const SDFBase *p_sdf, const ChunkParam &p_param, uint32_t p_scratch_mb) {
    stop();
    if (!ov::executor().threaded()) {
        return; // FAILED: the callers read the CPU's field
    }
    sdf = p_sdf;
    param = p_param;
    scratch_bytes = uint64_t(p_scratch_mb) << 20;
    state_now = STARTING;
    worker = ov::executor().start_worker("field_gpu", *this);
}

void FieldGPU::stop() {
    fail_all();
    if (worker) {
        worker->stop();
        worker.reset();
    }
}

void FieldGPU::fail_all() {
    {
        const std::lock_guard<std::mutex> lock(mutex);
        state_now = FAILED; // under the lock: no caller queues after it
        for (Request *r : queue) {
            r->done = true;
            r->ok = false;
        }
        queue.clear();
    }
    answered.notify_all();
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
    Trace::Scope span("field", "wait");
    span.set_arg(uint64_t(p_count));
    Trace::flow(Trace::Kind::FLOW_START, "field", uint64_t(reinterpret_cast<uintptr_t>(&r)));
    const uint64_t t0 = now_usec();
    std::unique_lock<std::mutex> lock(mutex);
    if (state_now.load() != READY) {
        return false;
    }
    queue.push_back(&r);
    worker->wake();
    answered.wait(lock, [&r] { return r.done; });
    waits.add(now_usec() - t0);
    return r.ok;
}

void FieldGPU::enter() {
    const uint64_t t0 = now_usec();
    uint64_t scratch_a = 0, scratch_b = 0;
    CascadeGLSL::scratch_bytes(sdf, 1, 1, scratch_a, scratch_b);
    batch = uint32_t(ov::clamp(scratch_bytes / ov::max(scratch_a + scratch_b, uint64_t(1)), uint64_t(MIN_POINTS), uint64_t(MAX_POINTS)));
    cap = batch;
    const bool ok = gpu.create(sdf, param, batch);
    {
        const std::lock_guard<std::mutex> lock(mutex);
        init_ms = double(now_usec() - t0) / 1000.0;
        max_points = batch;
        scratch_mb = double(gpu.scratch_bytes()) / 1048576.0;
        if (ok && state_now.load() == STARTING) { // not stopped meanwhile
            state_now = READY;
        }
    }
    if (!ok) {
        OV_WARN_PRINT("FieldGPU: no local RenderingDevice (headless, the Compatibility renderer) or the point passes failed: the site walks read the field on the CPU.");
        fail_all();
    }
}

bool FieldGPU::step() {
    {
        const std::lock_guard<std::mutex> lock(mutex);
        if (state_now.load() != READY || queue.empty()) {
            return false;
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
    Trace::Scope span("field", "submit");
    span.set_arg(voxels.size());
    for (const Request *r : taken) {
        Trace::flow(Trace::Kind::FLOW_END, "field", uint64_t(reinterpret_cast<uintptr_t>(r)));
    }
    CascadePointsGPU::BatchStats stats;
    const uint64_t t_run = now_usec();
    bool good = true;
    for (size_t first = 0; good && first < voxels.size(); first += cap) {
        const int count = int(ov::min(voxels.size() - first, size_t(cap)));
        good = gpu.points().ground_field(voxels.data() + first, lods.data() + first, count, out.data() + first, stats);
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
        submissions.add(now_usec() - t_run);
        points += int64_t(voxels.size());
        batches.batches += stats.batches;
        batches.gpu_ms += ov::max(stats.gpu_ms, 0.0);
        batches.record_ms += stats.record_ms;
        batches.sync_ms += stats.sync_ms;
        batches.readback_ms += stats.readback_ms;
        if (stats.gpu_ms > 0.0) {
            // The points' cost: the smoothed GPU time over the smoothed points (weighted by points: a small submission
            // timed behind the renderer's work weighs what its points do); the cap follows it.
            recent_gpu_ms = recent_points > 0.0 ? 0.8 * recent_gpu_ms + 0.2 * stats.gpu_ms : stats.gpu_ms;
            recent_points = recent_points > 0.0 ? 0.8 * recent_points + 0.2 * double(voxels.size()) : double(voxels.size());
            us_per_point = recent_gpu_ms * 1000.0 / recent_points;
            cap = uint32_t(ov::clamp(SUBMIT_GPU_MS * 1000.0 / us_per_point, double(MIN_POINTS), double(batch)));
        }
    }
    answered.notify_all();
    if (!good) {
        OV_ERR_PRINT("FieldGPU: a batch failed: the site walks read the field on the CPU from now on.");
        fail_all();
        return false;
    }
    return true;
}

void FieldGPU::exit() {
    fail_all(); // stopped or failed: what is still queued reads the CPU's field
    gpu.free(); // on this thread, which made it
}

ov::Value FieldGPU::get_stats() const {
    const std::lock_guard<std::mutex> lock(mutex);
    ov::Value d = ov::Value::object();
    const State s = state_now.load();
    d["state"] = s == READY ? "ready" : (s == STARTING ? "starting" : "off");
    d["init_ms"] = init_ms;
    d["points_per_batch"] = int64_t(max_points);
    d["scratch_mb"] = scratch_mb;
    d["us_per_point"] = us_per_point;
    d["points"] = points;
    d["points_per_submission"] = submissions.count > 0 ? double(points) / double(submissions.count) : 0.0;
    d["submissions"] = submissions.to_value();
    d["waits"] = waits.to_value();
    d["gpu_ms"] = batches.gpu_ms;
    d["record_ms"] = batches.record_ms;
    d["sync_ms"] = batches.sync_ms;
    d["readback_ms"] = batches.readback_ms;
    return d;
}
