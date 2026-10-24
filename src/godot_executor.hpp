#pragma once

#include "ov/executor.hpp"

// The core's executor on Godot (register_types installs it): tasks and data-parallel loops on Godot's WorkerThreadPool,
// workers on threads of their own (ov::ThreadWorker).
class GodotExecutor : public ov::Executor {
public:
    bool threaded() const override { return true; }
    // HIGH: the whole pool. LOW: the pool's low-priority share (threading/worker_pool/low_priority_thread_ratio, 0.5).
    int concurrency(ov::Priority p_priority) const override;
    ov::TaskId submit(std::function<void()> p_task, ov::Priority p_priority, const char *p_name) override;
    bool done(ov::TaskId p_task) override;
    void wait(ov::TaskId p_task) override;
    void parallel_for(uint32_t p_count, const char *p_name, void (*p_fn)(void *, uint32_t), void *p_ctx) override;
    std::unique_ptr<ov::Worker> start_worker(const char *p_name, ov::WorkerLoop &p_loop) override;
};
