#include "thread_pool_executor.hpp"

#include "../trace.hpp"

#include <atomic>

namespace ov {

ThreadPoolExecutor::ThreadPoolExecutor(int p_threads) {
    const int cores = int(std::thread::hardware_concurrency());
    const int count = p_threads > 0 ? p_threads : std::max(1, cores - 1);
    low_cap = std::max(1, count / 2);
    threads.reserve(size_t(count));
    for (int i = 0; i < count; i++) {
        threads.emplace_back(&ThreadPoolExecutor::run, this);
    }
}

ThreadPoolExecutor::~ThreadPoolExecutor() {
    {
        const std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
    }
    work.notify_all();
    for (std::thread &t : threads) {
        t.join();
    }
}

int ThreadPoolExecutor::concurrency(Priority p_priority) const {
    return p_priority == Priority::LOW ? low_cap : int(threads.size());
}

TaskId ThreadPoolExecutor::submit(std::function<void()> p_task, Priority p_priority, const char *) {
    TaskId id = 0;
    {
        const std::lock_guard<std::mutex> lock(mutex);
        id = next_id++;
        tasks[id] = Task{ std::move(p_task), p_priority, State::QUEUED };
        (p_priority == Priority::HIGH ? high : low).push_back(id);
    }
    work.notify_one();
    return id;
}

bool ThreadPoolExecutor::done(TaskId p_task) {
    const std::lock_guard<std::mutex> lock(mutex);
    const auto it = tasks.find(p_task);
    return it == tasks.end() || it->second.state == State::DONE;
}

void ThreadPoolExecutor::wait(TaskId p_task) {
    std::function<void()> fn;
    {
        std::unique_lock<std::mutex> lock(mutex);
        const auto it = tasks.find(p_task);
        if (it == tasks.end()) {
            return;
        }
        if (it->second.state == State::QUEUED) {
            // Not started: run it here rather than wait for a thread (the pool may be busy with the caller's peers).
            std::deque<TaskId> &queue = it->second.priority == Priority::HIGH ? high : low;
            queue.erase(std::find(queue.begin(), queue.end(), p_task));
            fn = std::move(it->second.fn);
            tasks.erase(it);
        } else {
            finished.wait(lock, [&] { return tasks.at(p_task).state == State::DONE; });
            tasks.erase(p_task);
            return;
        }
    }
    fn();
}

void ThreadPoolExecutor::parallel_for(uint32_t p_count, const char *p_name, void (*p_fn)(void *, uint32_t), void *p_ctx) {
    if (p_count == 0) {
        return;
    }
    std::atomic<uint32_t> next{ 0 };
    const auto drain = [&] {
        for (uint32_t i = next++; i < p_count; i = next++) {
            p_fn(p_ctx, i);
        }
    };
    // The caller drains too: the helpers it waits on and finds still queued run (and find nothing left) here.
    const uint32_t helpers = std::min(p_count - 1, uint32_t(threads.size()));
    std::vector<TaskId> ids;
    ids.reserve(helpers);
    for (uint32_t h = 0; h < helpers; h++) {
        ids.push_back(submit(drain, Priority::HIGH, p_name));
    }
    drain();
    for (TaskId id : ids) {
        wait(id);
    }
}

std::unique_ptr<Worker> ThreadPoolExecutor::start_worker(const char *p_name, WorkerLoop &p_loop) {
    return std::make_unique<ThreadWorker>(p_loop, p_name);
}

TaskId ThreadPoolExecutor::_next() {
    if (!high.empty()) {
        const TaskId id = high.front();
        high.pop_front();
        return id;
    }
    if (!low.empty() && low_running < low_cap) {
        const TaskId id = low.front();
        low.pop_front();
        low_running++;
        return id;
    }
    return 0;
}

void ThreadPoolExecutor::_finish(TaskId p_task, Priority p_priority) {
    {
        const std::lock_guard<std::mutex> lock(mutex);
        tasks.at(p_task).state = State::DONE;
        if (p_priority == Priority::LOW) {
            low_running--;
        }
    }
    finished.notify_all();
    work.notify_one(); // a LOW slot may have freed
}

void ThreadPoolExecutor::run() {
    Trace::name_thread("ov pool");
    for (;;) {
        TaskId id = 0;
        std::function<void()> fn;
        Priority priority = Priority::HIGH;
        {
            std::unique_lock<std::mutex> lock(mutex);
            work.wait(lock, [&] { return stopping || (id = _next()) != 0; });
            if (id == 0) {
                return; // stopping
            }
            Task &task = tasks.at(id);
            task.state = State::RUNNING;
            fn = std::move(task.fn);
            priority = task.priority;
        }
        fn();
        _finish(id, priority);
    }
}

} // namespace ov
