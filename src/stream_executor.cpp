// The stream's drain tasks' executors (instance_stream.hpp).
#include "instance_stream.hpp"

#include "ov/executor.hpp"

namespace Archetype {

namespace {

// The drain tasks on the core's executor, low priority.
class PoolExecutor : public StreamExecutor {
public:
    ~PoolExecutor() override { reap(true); }
    int workers() const override { return ov::executor().concurrency(ov::Priority::LOW); }
    void run(std::function<void()> p_task) override {
        const ov::TaskId id = ov::executor().submit(std::move(p_task), ov::Priority::LOW, "InstanceStream");
        const std::lock_guard<std::mutex> lock(task_mutex);
        tasks.push_back(id);
    }
    void reap(bool p_wait) override {
        ov::Executor &executor = ov::executor();
        for (;;) {
            std::vector<ov::TaskId> done;
            {
                const std::lock_guard<std::mutex> lock(task_mutex);
                for (size_t i = 0; i < tasks.size();) {
                    if (p_wait || executor.done(tasks[i])) {
                        done.push_back(tasks[i]);
                        tasks[i] = tasks.back();
                        tasks.pop_back();
                    } else {
                        i++;
                    }
                }
            }
            // Released without the lock: a running task (p_wait) may start others meanwhile, taken by the next round.
            for (const ov::TaskId id : done) {
                executor.wait(id);
            }
            if (!p_wait || done.empty()) {
                return;
            }
        }
    }

private:
    std::mutex task_mutex; // run() is called from the drain tasks too
    std::vector<ov::TaskId> tasks;
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
