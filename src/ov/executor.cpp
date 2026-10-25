// ThreadWorker out of line: inlined into a caller holding an InlineWorker behind a Worker pointer, GCC 16's speculative
// devirtualization warned array-bounds on it (-Werror).
#include "executor.hpp"

#include "../trace.hpp"

namespace ov {

ThreadWorker::ThreadWorker(WorkerLoop &p_loop, const char *p_name) :
        loop(p_loop), name(p_name), thread(&ThreadWorker::run, this) {}

ThreadWorker::~ThreadWorker() {
    stop();
}

void ThreadWorker::wake() {
    {
        const std::lock_guard<std::mutex> lock(mutex);
        woken = true;
    }
    signal.notify_one();
}

void ThreadWorker::stop() {
    {
        const std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
    }
    signal.notify_one();
    if (thread.joinable()) {
        thread.join();
    }
}

void ThreadWorker::run() {
    Trace::name_thread(name.c_str());
    loop.enter();
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(mutex);
            signal.wait(lock, [this] { return woken || stopping; });
            if (stopping) {
                break;
            }
            woken = false;
        }
        while (loop.step()) {
            const std::lock_guard<std::mutex> lock(mutex);
            if (stopping) {
                break;
            }
        }
    }
    loop.exit();
}

} // namespace ov
