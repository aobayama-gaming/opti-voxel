#include "test_common.hpp"

#include "godot_executor.hpp"
#include "ov/executor.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

// The core's executor (ov/executor.hpp): the inline backend's contract (a task never runs on submit, a worker steps
// only after a wake), the threaded worker, and the Godot backend the engine installs.

namespace {

struct CountingLoop : ov::WorkerLoop {
    std::atomic<int> entered{ 0 };
    std::atomic<int> exited{ 0 };
    std::atomic<int> steps{ 0 };
    std::atomic<int> work{ 0 }; // steps returning true left
    std::thread::id enter_thread;
    std::thread::id exit_thread;
    void enter() override {
        entered++;
        enter_thread = std::this_thread::get_id();
    }
    bool step() override {
        steps++;
        if (work.load() > 0) {
            work--;
            return true;
        }
        return false;
    }
    void exit() override {
        exited++;
        exit_thread = std::this_thread::get_id();
    }
};

} // namespace

TEST_CASE("ov::InlineExecutor: tasks run at pump or wait, never on submit") {
    ov::InlineExecutor e;
    CHECK_FALSE(e.threaded());
    int ran = 0;
    const ov::TaskId a = e.submit([&ran] { ran += 1; }, ov::Priority::LOW, "a");
    const ov::TaskId b = e.submit([&ran] { ran += 10; }, ov::Priority::HIGH, "b");
    CHECK(ran == 0);
    CHECK_FALSE(e.done(a));
    e.wait(b); // runs b alone
    CHECK(ran == 10);
    CHECK(e.done(b));
    CHECK_FALSE(e.done(a));
    // A task submitting another: both run in one pump.
    e.submit([&] { e.submit([&ran] { ran += 100; }, ov::Priority::LOW, "c"); }, ov::Priority::LOW, "d");
    e.pump();
    CHECK(ran == 111);
    CHECK(e.done(a));
    e.wait(a); // already run: returns at once
    CHECK(ran == 111);

    std::vector<uint32_t> seen(37, 0);
    struct Ctx {
        std::vector<uint32_t> *seen;
        static void run(void *p_ctx, uint32_t p_i) { (*static_cast<Ctx *>(p_ctx)->seen)[p_i]++; }
    } ctx{ &seen };
    e.parallel_for(37, "loop", &Ctx::run, &ctx);
    CHECK(std::count(seen.begin(), seen.end(), 1u) == 37);
}

TEST_CASE("ov::InlineExecutor: a worker enters at once, steps after a wake until idle, exits at stop") {
    ov::InlineExecutor e;
    CountingLoop loop;
    std::unique_ptr<ov::Worker> w = e.start_worker("w", loop);
    CHECK(loop.entered == 1);
    e.pump();
    CHECK(loop.steps == 0); // no wake: no step
    loop.work = 2;
    w->wake();
    e.pump();
    CHECK(loop.steps == 3); // two with work, one idle
    e.pump();
    CHECK(loop.steps == 3);
    w->stop();
    w->stop(); // idempotent
    CHECK(loop.exited == 1);
    w->wake();
    e.pump(); // stopped: no step
    CHECK(loop.steps == 3);
}

TEST_CASE("ov::ThreadWorker: enter and exit on its thread, steps after each wake") {
    CountingLoop loop;
    {
        std::unique_ptr<ov::Worker> w = std::make_unique<ov::ThreadWorker>(loop);
        for (int i = 0; i < 200 && loop.entered.load() == 0; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        CHECK(loop.entered == 1);
        CHECK(loop.steps == 0); // waits for a wake first
        loop.work = 3;
        w->wake();
        for (int i = 0; i < 2000 && loop.steps.load() < 4; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        CHECK(loop.steps == 4);
        w->stop();
    }
    CHECK(loop.exited == 1);
    CHECK((loop.enter_thread == loop.exit_thread));
    CHECK((loop.enter_thread != std::this_thread::get_id()));
}

TEST_CASE("GodotExecutor: tasks and loops on the worker pool") {
    GodotExecutor e;
    CHECK(e.threaded());
    CHECK(e.concurrency(ov::Priority::LOW) >= 1);
    CHECK(e.concurrency(ov::Priority::HIGH) >= e.concurrency(ov::Priority::LOW));
    std::atomic<int> sum{ 0 };
    std::vector<ov::TaskId> ids;
    for (int i = 1; i <= 8; i++) {
        ids.push_back(e.submit([&sum, i] { sum += i; }, i % 2 ? ov::Priority::LOW : ov::Priority::HIGH, "test"));
    }
    for (const ov::TaskId id : ids) {
        e.wait(id);
    }
    CHECK(sum == 36);
    std::vector<std::atomic<int>> seen(1000);
    struct Ctx {
        std::vector<std::atomic<int>> *seen;
        static void run(void *p_ctx, uint32_t p_i) { (*static_cast<Ctx *>(p_ctx)->seen)[p_i]++; }
    } ctx{ &seen };
    e.parallel_for(1000, "test", &Ctx::run, &ctx);
    int once = 0;
    for (const std::atomic<int> &s : seen) {
        once += s.load() == 1;
    }
    CHECK(once == 1000);
}
