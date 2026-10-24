#include "godot_executor.hpp"

#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/worker_thread_pool.hpp>

using namespace godot;

namespace {

// A submitted task's function, freed once it has run (on the pool's thread).
struct PoolTask {
    std::function<void()> fn;
    static void call(void *p_task) {
        PoolTask *task = static_cast<PoolTask *>(p_task);
        task->fn();
        delete task;
    }
};

} // namespace

int GodotExecutor::concurrency(ov::Priority p_priority) const {
    const int cores = OS::get_singleton()->get_processor_count();
    return p_priority == ov::Priority::LOW ? MAX(1, cores / 2) : MAX(1, cores);
}

ov::TaskId GodotExecutor::submit(std::function<void()> p_task, ov::Priority p_priority, const char *p_name) {
    PoolTask *task = new PoolTask{ std::move(p_task) };
    return WorkerThreadPool::get_singleton()->add_native_task(&PoolTask::call, task, p_priority == ov::Priority::HIGH, p_name);
}

bool GodotExecutor::done(ov::TaskId p_task) {
    return WorkerThreadPool::get_singleton()->is_task_completed(p_task);
}

void GodotExecutor::wait(ov::TaskId p_task) {
    WorkerThreadPool::get_singleton()->wait_for_task_completion(p_task); // releases its id
}

void GodotExecutor::parallel_for(uint32_t p_count, const char *p_name, void (*p_fn)(void *, uint32_t), void *p_ctx) {
    WorkerThreadPool *pool = WorkerThreadPool::get_singleton();
    pool->wait_for_group_task_completion(pool->add_native_group_task(p_fn, p_ctx, int(p_count), -1, true, p_name));
}

std::unique_ptr<ov::Worker> GodotExecutor::start_worker(const char *, ov::WorkerLoop &p_loop) {
    return std::make_unique<ov::ThreadWorker>(p_loop);
}
