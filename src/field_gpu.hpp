#pragma once

#include "cascade_points_gpu.hpp"
#include "debug_log.hpp"
#include "exact_noise.hpp"
#include "sdf_base.hpp"

#include <godot_cpp/variant/dictionary.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>

// The site walks' field on the GPU (VoxelTerrain.walk_gpu): SDFBase::ground_field at many points, bit for bit, for the
// stream's drain tasks (their lockstep walks hand over a round of readings at a time, ScatterSource::walk_tiles).
//
// One thread owns a LocalPointsGPU, created when it starts and freed before it ends. Callers queue a request and wait;
// each submission takes every request waiting, so the rounds of all the walks running share the device's round trip.
// Pacing: a submission's GPU time, estimated from the µs per point its timestamps measure, stays under SUBMIT_GPU_MS
// (the renderer shares the GPU). No device (headless, the Compatibility renderer), a failure or stop(): the service
// ends FAILED and every request returns false, the caller then reads the CPU's field (the same bits).
class FieldGPU {
public:
    // A batch's points: sized from the scratch asked for, within these.
    static constexpr uint32_t MIN_POINTS = 1024;
    static constexpr uint32_t MAX_POINTS = 65536;
    static constexpr double SUBMIT_GPU_MS = 4.0;

    enum State {
        STARTING,
        READY,
        FAILED, // no device, the passes didn't build, a batch failed, or stopped
    };

    // The thread starts creating its device and passes (shader compile: ~0.5 s); state() says when it is ready.
    // p_sdf: a cascade terrain, configured, alive until stop(). p_scratch_mb: the passes' scratch (sizes a batch).
    void start(const SDFBase *p_sdf, const ChunkParam &p_param, uint32_t p_scratch_mb);
    // Fails the requests waiting and every later one, then joins the thread (which may still be compiling).
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
    std::atomic<State> state_now{ FAILED }; // set under mutex (a caller reads it there before queueing)
    mutable std::mutex mutex;
    std::condition_variable wake; // the thread: a request, or FAILED
    std::condition_variable answered; // the callers: their request done
    std::deque<Request *> queue;

    // Under mutex.
    uint32_t max_points = 0;
    double init_ms = 0.0;
    double scratch_mb = 0.0;
    double us_per_point = 0.0; // measured (GPU timestamps / points), 0 until the first submission
    int64_t points = 0;
    TimingStat submissions; // the thread's, wall clock
    TimingStat waits; // the callers', queued to answered
    CascadePointsGPU::BatchStats batches; // the passes' steps, summed

    void loop(const SDFBase *p_sdf, ChunkParam p_param, uint64_t p_scratch_bytes);
    // FAILED under the lock, then every request still queued answered false.
    void fail_all();
};
