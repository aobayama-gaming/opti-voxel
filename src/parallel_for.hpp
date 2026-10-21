#pragma once

#include <godot_cpp/classes/worker_thread_pool.hpp>

#include <cstdint>

// p_fn(i) for every i in [0, p_count) on Godot's worker pool, the calling thread waiting (and helping) until all
// are done. Each i must only write what is its own. Not from inside a pool task (the waits nest).
template <typename F>
void parallel_for(uint32_t p_count, const char *p_name, F &&p_fn) {
    if (p_count == 0) {
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
