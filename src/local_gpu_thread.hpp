#pragma once

#include "gpu_device.hpp"
#include "ov/executor.hpp"

#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>

// A local device (ov::gpu::create_local_device) on a worker of its own, outliving the objects that use it: they run their
// GPU work on its thread (run), which the device needs, the caller waiting. A world hands it to the next (World::Kept):
// a restart then makes no device, and the pipelines and buffers its users free while the device parks (Device::park)
// are taken by those of the next with the same text or shape (the discoverer's point passes: ~2 s on a new device).
class LocalGpuThread final : public ov::WorkerLoop {
public:
    // Started: the device is made on the worker's thread (null in run's jobs without one: headless, no local device).
    static std::unique_ptr<LocalGpuThread> create();
    // Stops the worker: the device freed on its thread. No run may be waiting.
    ~LocalGpuThread() override;

    // p_job(device) on the device's thread, after the jobs run before it; returns once it has run.
    void run(const std::function<void(ov::gpu::Device *)> &p_job);

private:
    struct Job {
        const std::function<void(ov::gpu::Device *)> *fn = nullptr;
        bool done = false;
    };

    LocalGpuThread() = default;
    void enter() override;
    bool step() override;
    void exit() override;

    std::unique_ptr<ov::Worker> worker;
    std::unique_ptr<ov::gpu::Device> device; // the worker's thread only
    std::mutex mutex;
    std::condition_variable job_done;
    std::deque<Job *> jobs;
};
