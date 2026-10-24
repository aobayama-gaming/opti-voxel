#include "trace.hpp"

#include "debug_log.hpp"

#include <algorithm>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace Trace {

namespace {

// One per thread that ever recorded, kept for the whole run (a thread holds a pointer to its own): start()
// empties them instead of freeing them.
struct Buffer {
    std::mutex mutex;
    std::vector<Event> events;
    const char *name = nullptr;
    uint32_t tid = 0;
};

// The GPU track's tid, past any real thread's.
constexpr uint32_t GPU_TID = 1000;

struct Registry {
    std::mutex mutex;
    std::vector<std::unique_ptr<Buffer>> buffers;
    Buffer gpu;
    std::atomic<uint64_t> count{ 0 };
    std::atomic<uint64_t> dropped{ 0 };
    std::atomic<uint64_t> max_events{ 0 };
    std::mutex intern_mutex;
    std::unordered_set<std::string> interned; // nodes: their strings never move
    Registry() {
        gpu.name = "GPU";
        gpu.tid = GPU_TID;
    }
};

Registry &registry() {
    static Registry r;
    return r;
}

thread_local Buffer *tl_buffer = nullptr;
thread_local const char *tl_name = nullptr;

Buffer &own_buffer() {
    if (tl_buffer == nullptr) {
        Registry &r = registry();
        const std::lock_guard<std::mutex> lock(r.mutex);
        r.buffers.push_back(std::make_unique<Buffer>());
        tl_buffer = r.buffers.back().get();
        tl_buffer->tid = uint32_t(r.buffers.size());
        tl_buffer->name = tl_name;
    }
    return *tl_buffer;
}

// Reserves room for one event; false when the cap is reached (counted as dropped).
bool admit() {
    Registry &r = registry();
    if (r.count.fetch_add(1, std::memory_order_relaxed) >= r.max_events.load(std::memory_order_relaxed)) {
        r.dropped.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}

void push(Buffer &r_buffer, const Event &p_event) {
    const std::lock_guard<std::mutex> lock(r_buffer.mutex);
    r_buffer.events.push_back(p_event);
}

void append_escaped(std::string &r_out, const char *p_text) {
    for (const char *c = p_text; *c; c++) {
        if (*c == '"' || *c == '\\') {
            r_out += '\\';
        }
        if (uint8_t(*c) >= 0x20) {
            r_out += *c;
        }
    }
}

void append_event(std::string &r_out, const Event &p_e, uint32_t p_tid) {
    static const char *const PHASE[] = { "X", "s", "t", "f" };
    r_out += "{\"ph\":\"";
    r_out += PHASE[int(p_e.kind)];
    r_out += "\",\"cat\":\"";
    append_escaped(r_out, p_e.system);
    r_out += "\",\"name\":\"";
    append_escaped(r_out, p_e.name);
    r_out += "\",\"pid\":1,\"tid\":" + std::to_string(p_tid) + ",\"ts\":" + std::to_string(p_e.ts_usec);
    if (p_e.kind == Kind::SPAN) {
        r_out += ",\"dur\":" + std::to_string(p_e.dur_usec);
        if (p_e.id != 0) {
            r_out += ",\"args\":{\"n\":" + std::to_string(p_e.id) + "}";
        }
    } else {
        r_out += ",\"id\":" + std::to_string(p_e.id);
        if (p_e.kind == Kind::FLOW_END) {
            r_out += ",\"bp\":\"e\""; // binds to the span it lands in, not the next one
        }
    }
    r_out += "},\n";
}

void append_thread_name(std::string &r_out, uint32_t p_tid, const char *p_name) {
    r_out += "{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":1,\"tid\":" + std::to_string(p_tid) + ",\"args\":{\"name\":\"";
    append_escaped(r_out, p_name ? p_name : "worker");
    r_out += "\"}},\n";
}

} // namespace

void start(uint64_t p_max_events) {
    Registry &r = registry();
    recording.store(false);
    {
        const std::lock_guard<std::mutex> lock(r.mutex);
        for (const std::unique_ptr<Buffer> &b : r.buffers) {
            const std::lock_guard<std::mutex> block(b->mutex);
            b->events.clear();
        }
        const std::lock_guard<std::mutex> block(r.gpu.mutex);
        r.gpu.events.clear();
    }
    r.count.store(0);
    r.dropped.store(0);
    r.max_events.store(p_max_events);
    recording.store(true);
}

void stop() {
    recording.store(false);
}

uint64_t event_count() {
    Registry &r = registry();
    return std::min(r.count.load(), r.max_events.load());
}

uint64_t dropped_count() {
    return registry().dropped.load();
}

void name_thread(const char *p_name) {
    tl_name = p_name;
    if (tl_buffer != nullptr) {
        const std::lock_guard<std::mutex> lock(tl_buffer->mutex);
        tl_buffer->name = p_name;
    }
}

const char *intern(const std::string &p_name) {
    Registry &r = registry();
    const std::lock_guard<std::mutex> lock(r.intern_mutex);
    return r.interned.insert(p_name).first->c_str();
}

uint64_t now_usec() {
    return ::now_usec();
}

void record(const Event &p_event) {
    if (enabled() && admit()) {
        push(own_buffer(), p_event);
    }
}

void flow(Kind p_kind, const char *p_name, uint64_t p_id) {
    if (enabled()) {
        record(Event{ "flow", p_name, now_usec(), 0, p_id, p_kind });
    }
}

void gpu_span(const char *p_system, const char *p_name, uint64_t p_ts_usec, uint64_t p_dur_usec, uint64_t p_id) {
    if (enabled() && admit()) {
        push(registry().gpu, Event{ p_system, p_name, p_ts_usec, p_dur_usec, p_id, Kind::SPAN });
    }
}

void gpu_flow(Kind p_kind, const char *p_name, uint64_t p_id, uint64_t p_ts_usec) {
    if (enabled() && admit()) {
        push(registry().gpu, Event{ "flow", p_name, p_ts_usec, 0, p_id, p_kind });
    }
}

std::string to_json() {
    Registry &r = registry();
    std::string out = "{\"traceEvents\":[\n";
    const auto dump = [&out](Buffer &b) {
        const std::lock_guard<std::mutex> lock(b.mutex);
        if (b.events.empty()) {
            return;
        }
        append_thread_name(out, b.tid, b.name);
        for (const Event &e : b.events) {
            append_event(out, e, b.tid);
        }
    };
    {
        const std::lock_guard<std::mutex> lock(r.mutex);
        for (const std::unique_ptr<Buffer> &b : r.buffers) {
            dump(*b);
        }
    }
    dump(r.gpu);
    out += "{\"ph\":\"M\",\"name\":\"process_name\",\"pid\":1,\"args\":{\"name\":\"opti-voxel\"}}\n],\n";
    out += "\"displayTimeUnit\":\"ms\",\"otherData\":{\"events\":" + std::to_string(event_count()) +
            ",\"dropped\":" + std::to_string(dropped_count()) + ",\"gpu_tid\":" + std::to_string(GPU_TID) + "}}\n";
    return out;
}

} // namespace Trace
