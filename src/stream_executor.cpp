// The stream's drain tasks' executors (instance_stream.hpp).
#include "instance_stream.hpp"

#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/worker_thread_pool.hpp>

namespace Archetype {

namespace {

class PoolExecutor : public StreamExecutor {
public:
    ~PoolExecutor() override { reap(true); }
    // The pool gives low-priority tasks threading/worker_pool/low_priority_thread_ratio (0.5) of its threads.
    int workers() const override { return MAX(1, OS::get_singleton()->get_processor_count() / 2); }
    void run(std::function<void()> p_task) override {
        auto task = std::make_unique<Task>(Task{ std::move(p_task) });
        const int64_t id = WorkerThreadPool::get_singleton()->add_native_task(&Task::call, task.get(), false, "InstanceStream");
        const std::lock_guard<std::mutex> lock(task_mutex);
        tasks.emplace_back(id, std::move(task));
    }
    void reap(bool p_wait) override {
        WorkerThreadPool *pool = WorkerThreadPool::get_singleton();
        for (;;) {
            std::vector<std::pair<int64_t, std::unique_ptr<Task>>> done;
            {
                const std::lock_guard<std::mutex> lock(task_mutex);
                for (size_t i = 0; i < tasks.size();) {
                    if (p_wait || pool->is_task_completed(tasks[i].first)) {
                        done.push_back(std::move(tasks[i]));
                        tasks[i] = std::move(tasks.back());
                        tasks.pop_back();
                    } else {
                        i++;
                    }
                }
            }
            // Released without the lock: a running task (p_wait) may start others meanwhile, taken by the next round.
            for (const std::pair<int64_t, std::unique_ptr<Task>> &t : done) {
                pool->wait_for_task_completion(t.first); // releases its id
            }
            if (!p_wait || done.empty()) {
                return;
            }
        }
    }

private:
    struct Task {
        std::function<void()> fn;
        static void call(void *p_task) { static_cast<Task *>(p_task)->fn(); }
    };
    std::mutex task_mutex; // run() is called from the drain tasks too
    std::vector<std::pair<int64_t, std::unique_ptr<Task>>> tasks;
};

} // namespace

std::unique_ptr<StreamExecutor> StreamExecutor::pool() {
    return std::make_unique<PoolExecutor>();
}

void StreamExecutor::run(std::function<void()> p_task) {
    const std::lock_guard<std::mutex> lock(queue_mutex);
    queued.push_back(std::move(p_task));
}

void StreamExecutor::reap(bool /* p_wait */) {
    for (;;) {
        std::vector<std::function<void()>> now;
        {
            const std::lock_guard<std::mutex> lock(queue_mutex);
            now.swap(queued);
        }
        if (now.empty()) {
            return;
        }
        for (const std::function<void()> &task : now) {
            task();
        }
    }
}

} // namespace Archetype
