#include "test_common.hpp"

#include "trace.hpp"

#include <godot_cpp/classes/json.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>

#include <thread>

using namespace godot;

namespace {

Array trace_events() {
    const Dictionary root = JSON::parse_string(String::utf8(Trace::to_json().c_str()));
    return root["traceEvents"];
}

// The events with this phase ("X", "s", "f", "M"...) and name.
std::vector<Dictionary> events(const Array &p_all, const char *p_phase, const char *p_name) {
    std::vector<Dictionary> out;
    for (int i = 0; i < p_all.size(); i++) {
        const Dictionary e = p_all[i];
        if (String(e["ph"]) == p_phase && String(e["name"]) == p_name) {
            out.push_back(e);
        }
    }
    return out;
}

// The tid whose thread_name metadata is p_name, -1 when none.
int64_t tid_named(const Array &p_all, const char *p_name) {
    for (const Dictionary &m : events(p_all, "M", "thread_name")) {
        if (String(Dictionary(m["args"])["name"]) == p_name) {
            return int64_t(m["tid"]);
        }
    }
    return -1;
}

} // namespace

TEST_CASE("Trace: nothing is recorded while stopped") {
    Trace::start();
    Trace::stop();
    {
        OV_TRACE("test", "ignored");
        Trace::flow(Trace::Kind::FLOW_START, "f", 1);
        Trace::gpu_span("test", "ignored_gpu", 10, 5);
    }
    CHECK(Trace::event_count() == 0);
    CHECK(events(trace_events(), "X", "ignored").empty());
}

TEST_CASE("Trace: spans and a flow across two named threads") {
    Trace::start();
    Trace::name_thread("test_main");
    {
        Trace::Scope span("test", "producer");
        span.set_arg(42);
        Trace::flow(Trace::Kind::FLOW_START, "handoff", 7);
    }
    std::thread consumer([] {
        Trace::name_thread("test_consumer");
        OV_TRACE("test", "consumer");
        Trace::flow(Trace::Kind::FLOW_END, "handoff", 7);
    });
    consumer.join();
    Trace::stop();

    const Array all = trace_events();
    const int64_t main_tid = tid_named(all, "test_main");
    const int64_t consumer_tid = tid_named(all, "test_consumer");
    CHECK_OR_RETURN(main_tid >= 0);
    CHECK_OR_RETURN(consumer_tid >= 0);
    CHECK(main_tid != consumer_tid);

    const std::vector<Dictionary> producer = events(all, "X", "producer");
    CHECK_OR_RETURN(producer.size() == 1);
    CHECK(int64_t(producer[0]["tid"]) == main_tid);
    CHECK(String(producer[0]["cat"]) == "test");
    CHECK(int64_t(Dictionary(producer[0]["args"])["n"]) == 42);
    const std::vector<Dictionary> consumed = events(all, "X", "consumer");
    CHECK_OR_RETURN(consumed.size() == 1);
    CHECK(int64_t(consumed[0]["tid"]) == consumer_tid);
    CHECK(int64_t(consumed[0]["ts"]) >= int64_t(producer[0]["ts"]) + int64_t(producer[0]["dur"]));

    // The flow starts inside the producer's span and ends inside the consumer's (bound to it: bp "e").
    const std::vector<Dictionary> start = events(all, "s", "handoff");
    const std::vector<Dictionary> end = events(all, "f", "handoff");
    CHECK_OR_RETURN(start.size() == 1);
    CHECK_OR_RETURN(end.size() == 1);
    CHECK(int64_t(start[0]["id"]) == 7);
    CHECK(int64_t(end[0]["id"]) == 7);
    CHECK(int64_t(start[0]["tid"]) == main_tid);
    CHECK(int64_t(end[0]["tid"]) == consumer_tid);
    CHECK(String(end[0]["bp"]) == "e");
    CHECK(int64_t(start[0]["ts"]) >= int64_t(producer[0]["ts"]));
    CHECK(int64_t(start[0]["ts"]) <= int64_t(producer[0]["ts"]) + int64_t(producer[0]["dur"]));
}

TEST_CASE("Trace: spans and flow points name the span they run in") {
    Trace::start();
    {
        OV_TRACE("test", "outer");
        {
            // Siblings that start and end within the same microsecond: only the parent ids tell them apart.
            OV_TRACE("test", "first");
        }
        {
            OV_TRACE("test", "second");
            Trace::flow(Trace::Kind::FLOW_START, "nested", 1);
        }
    }
    CHECK(Trace::open_span == 0); // every span closed
    Trace::stop();

    const Array all = trace_events();
    const auto arg = [](const Dictionary &p_e, const char *p_key) { return int64_t(Dictionary(p_e["args"])[p_key]); };
    const std::vector<Dictionary> outer = events(all, "X", "outer");
    const std::vector<Dictionary> first = events(all, "X", "first");
    const std::vector<Dictionary> second = events(all, "X", "second");
    const std::vector<Dictionary> point = events(all, "s", "nested");
    CHECK_OR_RETURN(outer.size() == 1);
    CHECK_OR_RETURN(first.size() == 1);
    CHECK_OR_RETURN(second.size() == 1);
    CHECK_OR_RETURN(point.size() == 1);
    CHECK(arg(outer[0], "p") == 0);
    CHECK(arg(first[0], "p") == arg(outer[0], "s"));
    CHECK(arg(second[0], "p") == arg(outer[0], "s"));
    CHECK(arg(first[0], "s") != arg(second[0], "s"));
    CHECK(arg(point[0], "p") == arg(second[0], "s"));
}

TEST_CASE("Trace: GPU spans go to the GPU track") {
    Trace::start();
    const uint32_t frame = Trace::gpu_span("gpu", "frame", 900, 600);
    const uint32_t opaque = Trace::gpu_span("render", Trace::intern("Render \"Opaque\" Pass"), 1000, 250, 0, frame);
    Trace::gpu_flow(Trace::Kind::FLOW_STEP, "batch", 3, 1100, opaque);
    Trace::stop();
    CHECK(frame != 0);
    CHECK(opaque != frame);
    const Array all = trace_events();
    const int64_t gpu_tid = tid_named(all, "GPU");
    CHECK_OR_RETURN(gpu_tid >= 0);
    const std::vector<Dictionary> pass = events(all, "X", "Render \"Opaque\" Pass"); // escaped and read back
    CHECK_OR_RETURN(pass.size() == 1);
    CHECK(int64_t(pass[0]["tid"]) == gpu_tid);
    CHECK(int64_t(pass[0]["ts"]) == 1000);
    CHECK(int64_t(pass[0]["dur"]) == 250);
    CHECK(int64_t(Dictionary(pass[0]["args"])["p"]) == frame);
    const std::vector<Dictionary> step = events(all, "t", "batch");
    CHECK_OR_RETURN(step.size() == 1);
    CHECK(int64_t(step[0]["tid"]) == gpu_tid);
}

TEST_CASE("Trace: past the cap events are dropped and counted, start clears") {
    Trace::start(3);
    for (int i = 0; i < 5; i++) {
        OV_TRACE("test", "capped");
    }
    Trace::stop();
    CHECK(Trace::event_count() == 3);
    CHECK(Trace::dropped_count() == 2);
    CHECK(events(trace_events(), "X", "capped").size() == 3);

    Trace::start();
    Trace::stop();
    CHECK(Trace::event_count() == 0);
    CHECK(Trace::dropped_count() == 0);
    CHECK(events(trace_events(), "X", "capped").empty());
}

TEST_CASE("Trace: interned names are stable") {
    const char *a = Trace::intern("Render Shadows");
    const char *b = Trace::intern(std::string("Render ") + "Shadows");
    CHECK(a == b);
    CHECK(std::string(a) == "Render Shadows");
}
