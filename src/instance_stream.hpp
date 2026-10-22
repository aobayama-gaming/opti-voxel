#pragma once

#include "scatter_source.hpp"

#include <godot_cpp/variant/dictionary.hpp>

#include <atomic>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// The streamed sources' residency: which keys of the procedural sources (scatter_source.hpp) are computed, and where
// their records sit in the GPU table. A chunk is meshed only once every key it lists is placed (admit()): its records are
// the keys' (a pure function of each key) and its list folds them in key order, so a chunk's content is a pure function
// of its id whatever streamed when, and no chunk is meshed twice because a key arrived late.
//
// One entry per key asked for, in one table: computing (its inputs first: a coarser part tile reads the tiles under it, a
// part layer's blob tile the LOD-0 part tiles around it), then ready with its content, which a listed key copies into
// slots: the GPU table's records from slot_base on, sent through the placement log (uploads_since). Keys without records
// stay entries like the others. Ready entries not pinned by a parked chunk are on a least-recently-used list; the oldest
// go when room is needed (slots, inputs kept, entries), none asked for in the last GRACE_FRAMES (a batch in flight
// still reads its slots). A key's inputs are handed to it as shared pointers when they land: an input may go meanwhile.
//
// Main thread: admit(), take_ready(), poll(), uploads_since(). chunk_list() reads while the main thread waits (build_lists
// on the pool); point_records() from any thread. Keys are computed by drain tasks of the executor (Godot's pool at low
// priority), never under the stream's lock.
namespace Archetype {

// Runs the stream's drain tasks: Godot's worker pool at low priority (pool()), or at once on the calling thread (tests).
class StreamExecutor {
public:
    virtual ~StreamExecutor() = default;
    // How many drain tasks may run side by side.
    virtual int workers() const { return 1; }
    // Runs p_task, now or later, on any thread.
    virtual void run(std::function<void()> p_task) { p_task(); }
    // Releases the finished tasks; p_wait: waits for all of them first.
    virtual void reap(bool /* p_wait */) {}

    // Godot's WorkerThreadPool, low priority (threading/worker_pool/low_priority_thread_ratio of its threads).
    static std::unique_ptr<StreamExecutor> pool();
};

class InstanceStream {
public:
    static constexpr uint64_t GRACE_FRAMES = 16;
    static constexpr uint64_t POLL_USEC_MAX = 2000; // poll()'s landing time per frame, at most
    static constexpr uint32_t SLOTS_MIN = uint32_t(1) << 16;
    static constexpr uint32_t SLOTS_MAX = uint32_t(1) << 21;
    static constexpr int64_t INPUT_RECORDS = int64_t(1) << 21; // the records of ready keys not in slots (inputs), kept at most
    static constexpr size_t ENTRIES_MAX = size_t(1) << 18; // keys kept at most (the empty ones count)
    static constexpr size_t LOG_MAX = size_t(1) << 16; // placements uploads_since() can replay; a reader further behind gets them all

    struct Settings {
        uint32_t slot_base = 0; // the first slot's record id (past the table's own capacity)
        uint32_t slot_capacity = SLOTS_MIN;
        int64_t input_records = INPUT_RECORDS;
        size_t entries = ENTRIES_MAX;
        size_t log = LOG_MAX;
    };
    enum Admit {
        READY, // every key placed: mesh it now
        PARKED, // the stream keeps it until its keys land (take_ready)
        LATER, // a key finds no slot: stop popping this frame, ask again later
    };
    // A placed key's records, sent to the GPU table from record id first.
    struct Upload {
        uint32_t first = 0;
        std::shared_ptr<const SourceContent> content;
    };

    // p_sources[i] has the setup id i.
    InstanceStream(std::vector<std::shared_ptr<const ScatterSource>> p_sources, const ChunkParam &p_param, const Settings &p_settings,
            std::unique_ptr<StreamExecutor> p_executor);
    ~InstanceStream(); // waits for the drain tasks running; the keys not started are dropped

    // Σ over the sources of the sites their zone holds times the records per site, twice, within [SLOTS_MIN, SLOTS_MAX].
    static uint32_t slot_budget(const std::vector<std::shared_ptr<const ScatterSource>> &p_sources);

    uint32_t slot_base() const { return settings.slot_base; }
    uint32_t slot_capacity() const { return settings.slot_capacity; }

    // Main thread. Asks for every key the chunk lists (the first asked are computed first). PARKED: the stream pins the
    // chunk's keys and hands it back through take_ready() with p_tag once they are placed; asked again meanwhile, it stays
    // parked (with the new tag).
    Admit admit(const ChunkId &p_chunk, uint64_t p_tag);
    // Main thread: the parked chunks whose keys are all placed, or which lost a key's slot (asked again, they answer
    // LATER until there is room), with their tags, in the order they were parked.
    void take_ready(std::vector<std::pair<ChunkId, uint64_t>> &r_out);
    // Main thread, once per frame: lands what the drain tasks finished (POLL_USEC_MAX, the rest the next frame), placing
    // the listed keys and starting the keys whose last input landed, then evicts down to the limits.
    void poll();

    // The record ids of the chunk's streamed records whose box test can pass at its samples, in key order. Every key
    // placed (admit() said READY), else the missing ones are left out and counted (lists_missing).
    void chunk_list(const ChunkId &p_chunk, std::vector<uint32_t> &r_out) const;
    // The streamed records whose box test at p_lod can pass at p_voxel, in key order, appended (copies): a resident key's,
    // else computed here and dropped (counted). Any thread (on the pool's, its loops run inline).
    void point_records(const VoxelPos &p_voxel, int p_lod, std::vector<TableRecord> &r_out) const;

    // The placement log: every placement bumps placements(). uploads_since(p_seq) hands the keys placed after p_seq that
    // still hold their slots, in placement order (a later one may reuse an earlier one's slots); p_seq older than the log
    // replays every placed key. Main thread.
    uint64_t placements() const;
    void uploads_since(uint64_t p_seq, std::vector<Upload> &r_out) const;

    Dictionary get_stats() const;

private:
    struct KeyHash {
        size_t operator()(const SourceKey &p_k) const;
    };
    struct ChunkHash {
        size_t operator()(const ChunkId &p_c) const { return Vector3i64Hasher::hash(p_c); }
    };
    struct Job {
        const ScatterSource *source = nullptr;
        SourceKey key;
        std::vector<std::shared_ptr<const SourceContent>> inputs; // aligned with source->inputs(key)
        std::shared_ptr<SourceContent> out;
        uint64_t usec = 0;
        void run();
    };
    struct Entry {
        enum State {
            WAITING, // for its inputs
            RUNNING,
            READY,
        };
        SourceKey key;
        State state = WAITING;
        std::shared_ptr<const SourceContent> content; // READY
        std::vector<std::shared_ptr<const SourceContent>> inputs; // WAITING: those landed so far
        int missing = 0; // WAITING: its inputs not landed
        std::vector<std::pair<SourceKey, uint32_t>> dependents; // not READY: the keys waiting for it (and its input index)
        std::vector<std::pair<ChunkId, uint64_t>> waiters; // the parked chunks waiting for its placement (their park seq)
        uint32_t slot = 0;
        uint32_t count = 0;
        bool listed = false; // a chunk lists it: it goes in slots
        bool placed = false;
        uint64_t placement = 0; // its placement's sequence number
        uint64_t used = 0; // the frame it was last asked for
        int pins = 0; // parked chunks holding it
        Entry *older = nullptr; // the LRU list (READY, unpinned)
        Entry *newer = nullptr;
        bool in_lru = false;
    };
    struct Parked {
        uint64_t tag = 0;
        uint64_t seq = 0; // park order
        int missing = 0;
        bool done = false; // in ready_chunks
        std::vector<SourceKey> keys; // pinned
    };

    std::vector<std::shared_ptr<const ScatterSource>> sources;
    ChunkParam param;
    Settings settings;
    std::unique_ptr<StreamExecutor> executor;

    mutable std::mutex mutex; // the main thread's changes vs point_records
    std::unordered_map<SourceKey, Entry, KeyHash> entries; // references stay valid across rehashes
    Entry *newest = nullptr; // the LRU list's ends
    Entry *oldest = nullptr;
    uint64_t frame = 0;

    std::mutex job_mutex; // the drain tasks' queues
    std::deque<std::unique_ptr<Job>> jobs; // to run
    std::vector<std::unique_ptr<Job>> finished; // run, to land
    int draining = 0; // drain tasks started and not done
    std::deque<std::unique_ptr<Job>> landing; // taken from finished, landed by poll() within its budget

    std::unordered_map<ChunkId, Parked, ChunkHash> parked;
    std::vector<ChunkId> ready_chunks; // parked chunks to hand back
    uint64_t park_seq = 0;

    std::map<uint32_t, uint32_t> free_runs; // first slot -> count, below slots_end
    uint32_t slots_end = 0; // slots ever used
    int64_t free_slots = 0;
    int64_t input_records = 0; // READY, not placed
    uint64_t placement_seq = 0;
    std::deque<std::pair<uint64_t, SourceKey>> log; // (placement, key), the last Settings::log

    struct Stats {
        int64_t computed = 0;
        int64_t compute_usec = 0;
        int64_t computed_by[2][ChunkMath::MAX_LOD_NUMBERS] = {}; // per kind and LOD
        int64_t usec_by[2][ChunkMath::MAX_LOD_NUMBERS] = {};
        int64_t placed = 0;
        int64_t evicted = 0;
        int64_t slots_full = 0; // placements refused: no room even after evicting
        int64_t slots_peak = 0;
        int64_t admitted[3] = {}; // per Admit answer
        int64_t landed_late = 0; // landings left to the next poll (budget spent)
        uint64_t poll_usec = 0;
        uint64_t poll_usec_max = 0;
        uint64_t admit_usec = 0;
    };
    Stats stats;
    mutable std::atomic<int64_t> lists_missing{ 0 }; // keys a list needed that weren't placed (stays 0)
    mutable std::atomic<int64_t> point_computed{ 0 }; // point_records' keys computed (not resident)

    const ScatterSource &source(const SourceKey &p_key) const { return *sources[p_key.source]; }
    // Every source's keys of the chunk, in key order.
    void chunk_keys(const ChunkId &p_chunk, std::vector<SourceKey> &r_out) const;
    // The key's entry, asked for now (created and its inputs asked for if new; placed if listed and ready).
    Entry &request(const SourceKey &p_key, bool p_listed);
    void start(Entry &r_entry);
    void drain();
    void land(std::unique_ptr<Job> p_job);
    // Gives a ready listed entry its slots and logs it (false: no room even after evicting).
    bool place(Entry &r_entry);
    bool allocate(uint32_t p_count, uint32_t &r_slot);
    void release(uint32_t p_slot, uint32_t p_count);
    // Evicts the oldest entry if it is out of its grace (false: none can go).
    bool evict_oldest();
    void pin(Entry &r_entry);
    void unpin(Entry &r_entry);
    void touch(Entry &r_entry);
    void lru_link(Entry &r_entry);
    void lru_unlink(Entry &r_entry);
    // A parked chunk handed back by take_ready().
    void hand_back(const ChunkId &p_chunk);
    int64_t slots_used() const { return int64_t(slots_end) - free_slots; }
};

} // namespace Archetype
