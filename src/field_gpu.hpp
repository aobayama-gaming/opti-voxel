#pragma once

#include "exact_noise.hpp"
#include "sdf_base.hpp"

#include <godot_cpp/variant/dictionary.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>

// The site walk's field on the GPU (VoxelTerrain.walk_gpu): SDFBase::ground_field at many points, bit for bit, for the
// stream's drain tasks (their lockstep walks hand over a round of readings at a time, ScatterSource::walk_tiles).
//
// One thread owns a local RenderingDevice and the cascade's point passes (CascadePointsGPU), created when it starts and
// freed before it ends: no other thread touches them. Callers queue a request and wait; each submission takes every
// request waiting (up to the batch's points, the rest next), so the rounds of all the walks running share the GPU's
// latency. No device (headless, the Compatibility renderer) or a device that fails: the service ends and every request
// returns false, the caller then reads the field on the CPU (the same bits).
class FieldGPU {
public:
    // A batch's points: sized from the scratch asked for, within these.
    static constexpr uint32_t MIN_POINTS = 1024;
    static constexpr uint32_t MAX_POINTS = 65536;
    // A submission's GPU time, estimated from the µs per point measured so far: the renderer's frames wait behind at
    // most about this much (the device is shared). The estimate starts at the batch's points.
    static constexpr double SUBMIT_GPU_MS = 4.0;

    enum State {
        STARTING,
        READY,
        FAILED, // no device, or the passes didn't build, or stopped
    };

    // The thread starts creating its device and passes (shader compile: ~0.5 s); state() says when it is ready.
    // p_sdf: a cascade terrain, configured, alive until stop(). p_scratch_mb: the passes' scratch (sizes a batch).
    void start(const SDFBase *p_sdf, const ChunkParam &p_param, uint32_t p_scratch_mb);
    // Fails the requests waiting and every later one, then joins the thread.
    void stop();
    ~FieldGPU() { stop(); }
    State state() const { return state_now.load(); }

    // SDFBase::ground_field at p_count LOD0 voxels, voxel i as LOD p_lods[i] sees it. Thread-safe, blocking. False when
    // the service is not ready (or failed meanwhile): r_out is then incomplete and the caller reads the CPU's field.
    bool ground_field(const VoxelPos *p_voxels, const int *p_lods, int p_count, ExactNoise::Float4 *r_out);

    godot::Dictionary get_stats() const;

private:
    struct Request {
        const VoxelPos *voxels = nullptr;
        const int *lods = nullptr;
        int count = 0;
        ExactNoise::Float4 *out = nullptr;
        bool done = false;
        bool ok = false;
    };

    std::thread thread;
    std::atomic<State> state_now{ FAILED };
    mutable std::mutex mutex;
    std::condition_variable wake; // the thread: a request or stop
    std::condition_variable answered; // the callers: their request done
    std::deque<Request *> queue;
    bool stopping = false;
    uint32_t max_points = 0;

    // Under mutex.
    int64_t requests = 0;
    int64_t submissions = 0;
    int64_t points = 0;
    double gpu_ms = 0.0;
    double busy_ms = 0.0; // the thread's submissions, wall clock
    double record_ms = 0.0; // of which CascadePointsGPU's steps (BatchStats)
    double sync_ms = 0.0;
    double readback_ms = 0.0;
    double wait_ms = 0.0; // the callers', queued to answered
    double init_ms = 0.0;
    double us_per_point = 0.0; // measured (GPU timestamps / points), 0 until the first submission
    double scratch_mb = 0.0;

    void loop(const SDFBase *p_sdf, ChunkParam p_param, uint64_t p_scratch_bytes);
};
