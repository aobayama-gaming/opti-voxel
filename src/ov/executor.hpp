#pragma once

// The core's one threading interface: one-shot tasks, data-parallel loops and long-lived workers. The engine installs
// its backend (register_types: Godot's worker pool, godot_executor.hpp); the default runs everything on the calling
// thread (InlineExecutor: tests, a threads=no web build). The core never starts a thread itself.

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace ov {

enum class Priority {
    HIGH, // the frame waits on it (a group the caller helps with)
    LOW, // background (the instance stream's drain tasks)
};

using TaskId = int64_t;

// A long-lived worker's body, pumped: enter() once, then step() after each wake() until it returns false (idle until
// the next wake), then exit() once, enter and exit on the same thread (a local RenderingDevice lives between them).
struct WorkerLoop {
    virtual ~WorkerLoop() = default;
    virtual void enter() {}
    virtual bool step() = 0;
    virtual void exit() {}
};

class Worker {
public:
    virtual ~Worker() = default;
    // Asks for steps (a wake while it steps still counts: it steps again).
    virtual void wake() = 0;
    // Ends it: the step running finishes, no other starts, exit() runs. Idempotent.
    virtual void stop() = 0;
};

class Executor {
public:
    virtual ~Executor() = default;
    // Whether tasks and workers run on threads of their own (false: on the thread that pumps or waits).
    virtual bool threaded() const = 0;
    // How many tasks of p_priority may run side by side.
    virtual int concurrency(Priority p_priority) const = 0;
    // Runs p_task later, never on the calling thread before submit returns (it may hold a lock the task takes). Every
    // task is waited exactly once (wait), done() or not.
    virtual TaskId submit(std::function<void()> p_task, Priority p_priority, const char *p_name) = 0;
    // Whether the task has run (non-blocking).
    virtual bool done(TaskId p_task) = 0;
    // Returns once the task has run, and releases it.
    virtual void wait(TaskId p_task) = 0;
    // p_fn(p_ctx, i) for every i in [0, p_count), the caller waiting (and helping). Each i writes only what is its own.
    virtual void parallel_for(uint32_t p_count, const char *p_name, void (*p_fn)(void *, uint32_t), void *p_ctx) = 0;
    // Starts p_loop (it must outlive the worker): enter() now or on the worker's thread.
    virtual std::unique_ptr<Worker> start_worker(const char *p_name, WorkerLoop &p_loop) = 0;
    // Main thread, once per frame: an inline backend runs its queued tasks and steps its woken workers.
    virtual void pump() {}
};

// A worker on a std::thread of its own: the threaded backends' (wake / step until idle / wait). Out of line
// (executor.cpp).
class ThreadWorker final : public Worker {
public:
    explicit ThreadWorker(WorkerLoop &p_loop);
    ~ThreadWorker() override;
    void wake() override;
    void stop() override;

private:
    void run();

    WorkerLoop &loop;
    std::mutex mutex;
    std::condition_variable signal;
    bool woken = false;
    bool stopping = false;
    std::thread thread; // last: it starts running once the members above exist
};

// Everything on the calling thread: a task runs at the next pump() or at its wait(), a parallel_for is a loop, a worker
// enters at once and steps in pump() after a wake.
class InlineExecutor : public Executor {
public:
    bool threaded() const override { return false; }
    int concurrency(Priority) const override { return 1; }

    TaskId submit(std::function<void()> p_task, Priority, const char *) override {
        const std::lock_guard<std::mutex> lock(mutex);
        queue.push_back(Task{ next_id, std::move(p_task) });
        return next_id++;
    }
    bool done(TaskId p_task) override {
        const std::lock_guard<std::mutex> lock(mutex);
        for (const Task &t : queue) {
            if (t.id == p_task) {
                return false;
            }
        }
        return true;
    }
    void wait(TaskId p_task) override {
        std::function<void()> fn;
        {
            const std::lock_guard<std::mutex> lock(mutex);
            for (auto it = queue.begin(); it != queue.end(); ++it) {
                if (it->id == p_task) {
                    fn = std::move(it->fn);
                    queue.erase(it);
                    break;
                }
            }
        }
        if (fn) {
            fn();
        }
    }
    void parallel_for(uint32_t p_count, const char *, void (*p_fn)(void *, uint32_t), void *p_ctx) override {
        for (uint32_t i = 0; i < p_count; i++) {
            p_fn(p_ctx, i);
        }
    }
    std::unique_ptr<Worker> start_worker(const char *, WorkerLoop &p_loop) override {
        auto worker = std::make_unique<InlineWorker>(*this, p_loop);
        const std::lock_guard<std::mutex> lock(mutex);
        workers.push_back(worker.get());
        return worker;
    }
    // The queued tasks (those they submit too), then each woken worker's steps until idle.
    void pump() override {
        for (;;) {
            Task task;
            {
                const std::lock_guard<std::mutex> lock(mutex);
                if (queue.empty()) {
                    break;
                }
                task = std::move(queue.front());
                queue.pop_front();
            }
            task.fn();
        }
        std::vector<InlineWorker *> now;
        {
            const std::lock_guard<std::mutex> lock(mutex);
            now = workers;
        }
        for (InlineWorker *w : now) {
            w->pump();
        }
    }

private:
    struct Task {
        TaskId id = 0;
        std::function<void()> fn;
    };
    class InlineWorker final : public Worker {
    public:
        InlineWorker(InlineExecutor &p_owner, WorkerLoop &p_loop) :
                owner(p_owner), loop(p_loop) { loop.enter(); }
        ~InlineWorker() override { stop(); }
        void wake() override { woken = true; }
        void stop() override {
            if (stopped) {
                return;
            }
            stopped = true;
            loop.exit();
            const std::lock_guard<std::mutex> lock(owner.mutex);
            for (size_t i = 0; i < owner.workers.size(); i++) {
                if (owner.workers[i] == this) {
                    owner.workers.erase(owner.workers.begin() + int64_t(i));
                    break;
                }
            }
        }
        void pump() {
            while (woken && !stopped) {
                woken = false;
                while (!stopped && loop.step()) {
                }
            }
        }

    private:
        InlineExecutor &owner;
        WorkerLoop &loop;
        bool woken = false;
        bool stopped = false;
    };

    std::mutex mutex;
    std::deque<Task> queue;
    TaskId next_id = 0;
    std::vector<InlineWorker *> workers;
};

inline InlineExecutor &inline_executor() {
    static InlineExecutor s_inline;
    return s_inline;
}

// The installed backend (default: the inline one). Installed once, before any thread uses it (register_types).
inline Executor *&executor_slot() {
    static Executor *s_executor = &inline_executor();
    return s_executor;
}

inline Executor &executor() {
    return *executor_slot();
}

// p_executor must outlive every task and worker it runs; null restores the inline default.
inline void set_executor(Executor *p_executor) {
    executor_slot() = p_executor ? p_executor : &inline_executor();
}

} // namespace ov
