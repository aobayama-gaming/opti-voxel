#pragma once

// The core's executor on plain std::threads, for the hosts without a worker pool of their own (the Dawn adapter; Godot
// uses its WorkerThreadPool, godot_executor.hpp). HIGH tasks go first; LOW ones run on at most half the threads, so a
// background drain never takes the threads a frame waits on. A wait on a task still queued runs it on the waiting
// thread, and a parallel_for's caller works through the indices with the pool: neither can deadlock on a busy pool.

#include "executor.hpp"

#include <map>

namespace ov {

class ThreadPoolExecutor final : public Executor {
public:
    // p_threads: 0 = one per core but the caller's (at least 1).
    explicit ThreadPoolExecutor(int p_threads = 0);
    // Joins the threads; the tasks still queued are dropped (every task is waited before, by the contract).
    ~ThreadPoolExecutor() override;

    bool threaded() const override { return true; }
    int concurrency(Priority p_priority) const override;
    TaskId submit(std::function<void()> p_task, Priority p_priority, const char *p_name) override;
    bool done(TaskId p_task) override;
    void wait(TaskId p_task) override;
    void parallel_for(uint32_t p_count, const char *p_name, void (*p_fn)(void *, uint32_t), void *p_ctx) override;
    std::unique_ptr<Worker> start_worker(const char *p_name, WorkerLoop &p_loop) override;

private:
    enum class State {
        QUEUED,
        RUNNING,
        DONE,
    };
    struct Task {
        std::function<void()> fn;
        Priority priority = Priority::HIGH;
        State state = State::QUEUED;
    };

    void run();
    // Under the lock: the next task a pool thread may take (HIGH first, LOW under its cap), or 0.
    TaskId _next();
    void _finish(TaskId p_task, Priority p_priority);

    std::mutex mutex;
    std::condition_variable work; // a task queued, or stopping
    std::condition_variable finished; // a task done
    std::map<TaskId, Task> tasks; // until waited
    std::deque<TaskId> high;
    std::deque<TaskId> low;
    int low_running = 0;
    int low_cap = 1;
    TaskId next_id = 1;
    bool stopping = false;
    std::vector<std::thread> threads; // last: they start once the members above exist
};

} // namespace ov
