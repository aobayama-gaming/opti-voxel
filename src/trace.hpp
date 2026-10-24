#pragma once

#include <atomic>
#include <cstdint>
#include <string>

// Engine trace: which thread runs what, when, and how work hands off between threads. Spans are named
// (system, stage) on the thread that runs them; flows link a piece of work across threads (a scan from the
// discoverer to the main thread, a mesh batch from the main thread to the render thread, the GPU and back).
// The GPU's passes are spans on a "GPU" track, placed from the timestamps the RenderingDevice captures.
// `bench.sh --trace=1` writes the result as a Chrome trace (<run>/trace.json, opens in ui.perfetto.dev) and
// `tools/engine_map.py <run>` draws the map from it. Off by default: then a span costs one atomic load.
namespace Trace {

enum class Kind : uint8_t {
    SPAN, // [ts, ts + dur) on its thread
    FLOW_START, // a flow (name, id) leaves the enclosing span
    FLOW_STEP, // passes through it
    FLOW_END, // arrives in it
};

// Names are static literals or interned (intern()): events keep the pointer.
struct Event {
    const char *system;
    const char *name;
    uint64_t ts_usec;
    uint64_t dur_usec;
    uint64_t id; // a flow's id, a span's argument (chunks in a batch, points in a request)
    Kind kind;
};

inline std::atomic<bool> recording{ false };
inline bool enabled() {
    return recording.load(std::memory_order_relaxed);
}

// Clears what was recorded and starts recording (at most p_max_events, the rest counted as dropped).
void start(uint64_t p_max_events = 2'000'000);
void stop();
uint64_t event_count();
uint64_t dropped_count();

// The calling thread's track name ("main", "render", "discoverer"...). Unnamed threads are "worker".
void name_thread(const char *p_name);
// A stable copy of a runtime string, for names coming from Godot (the renderer's pass names).
const char *intern(const std::string &p_name);

uint64_t now_usec();
void record(const Event &p_event);
void flow(Kind p_kind, const char *p_name, uint64_t p_id);
// A span on the GPU track (its timestamps converted to the CPU clock by the caller).
void gpu_span(const char *p_system, const char *p_name, uint64_t p_ts_usec, uint64_t p_dur_usec, uint64_t p_id = 0);
void gpu_flow(Kind p_kind, const char *p_name, uint64_t p_id, uint64_t p_ts_usec);

// Everything recorded, as a Chrome trace (JSON object format: traceEvents + thread names + otherData).
std::string to_json();

// RAII span: OV_TRACE("mesher", "dispatch_batch") times the rest of the block.
class Scope {
public:
    Scope(const char *p_system, const char *p_name) :
            system(p_system), name(p_name), begin(enabled() ? now_usec() : 0) {}
    ~Scope() {
        if (begin != 0 && enabled()) {
            record(Event{ system, name, begin, now_usec() - begin, arg, Kind::SPAN });
        }
    }
    void set_arg(uint64_t p_arg) { arg = p_arg; }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

private:
    const char *system;
    const char *name;
    uint64_t begin;
    uint64_t arg = 0;
};

} // namespace Trace

#define OV_TRACE_CAT2(a, b) a##b
#define OV_TRACE_CAT(a, b) OV_TRACE_CAT2(a, b)
#define OV_TRACE(p_system, p_name) const Trace::Scope OV_TRACE_CAT(ov_trace_, __LINE__)(p_system, p_name)
