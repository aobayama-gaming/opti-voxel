#pragma once

#include <godot_cpp/classes/worker_thread_pool.hpp>

#include <cstdint>

// Set on a thread running a pool task (InlineParallelFor): parallel_for runs its loop there, in order.
inline thread_local bool parallel_for_inline = false;

// p_fn(i) for every i in [0, p_count) on Godot's worker pool, the calling thread waiting (and helping) until all
// are done. Each i must only write what is its own, so the loop inline gives the same results. Not from inside a pool
// task (the waits nest), unless the task holds an InlineParallelFor.
template <typename F>
void parallel_for(uint32_t p_count, const char *p_name, F &&p_fn) {
    if (p_count == 0) {
        return;
    }
    if (parallel_for_inline) {
        for (uint32_t i = 0; i < p_count; i++) {
            p_fn(i);
        }
        return;
    }
    struct Job {
        F *fn;
        static void run(void *p_job, uint32_t p_i) { (*static_cast<Job *>(p_job)->fn)(p_i); }
    };
    Job job{ &p_fn };
    godot::WorkerThreadPool *pool = godot::WorkerThreadPool::get_singleton();
    pool->wait_for_group_task_completion(pool->add_native_group_task(&Job::run, &job, int(p_count), -1, true, p_name));
}

// Held by a pool task that calls code using parallel_for (a streamed forest key): its loops run inline.
struct InlineParallelFor {
    bool was = parallel_for_inline;
    InlineParallelFor() { parallel_for_inline = true; }
    ~InlineParallelFor() { parallel_for_inline = was; }
    InlineParallelFor(const InlineParallelFor &) = delete;
    InlineParallelFor &operator=(const InlineParallelFor &) = delete;
};
