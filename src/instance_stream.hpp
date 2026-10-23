#pragma once

#include "latency_hist.hpp"
#include "scatter_source.hpp"

#include <godot_cpp/variant/dictionary.hpp>

#include <atomic>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

// The streamed sources' residency: which keys of the procedural sources (scatter_source.hpp) are computed, and where
// their records sit in the GPU table. A chunk is meshed only once every key it lists is placed (admit()): its records are
// the keys' (a pure function of each key) and its list folds them in key order, so a chunk's content is a pure function
// of its id whatever streamed when, and no chunk is meshed twice because a key arrived late.
//
// One entry per key asked for, in one table: computing (its inputs first: a part tile reads the LOD-0 site tiles inside
// it, a part layer's blob tile the LOD-0 part tiles around it, a far layer's its site tile), then ready with its content,
// which a listed key copies into slots: the GPU table's records from slot_base on, sent through the placement log
// (uploads_since). Keys without records (the site tiles, empty tiles) stay entries like the others. Ready entries not pinned by a parked chunk are on a least-recently-used list; the oldest
// go when room is needed (slots, inputs kept, entries), none asked for in the last GRACE_FRAMES (a batch in flight
// still reads its slots). A key's inputs are handed to it as shared pointers when they land: an input may go meanwhile.
//
// Main thread: admit(), take_ready(), poll(), uploads_since(). chunk_list() reads under a shared lock (build_lists on the
// pool); point_records() from any thread. Keys are computed by drain tasks of the executor (Godot's pool at low priority),
// never under the stream's lock, and landed by them under it: a key's dependents start at once, on the same task.
namespace Archetype {

// Runs the stream's drain tasks: Godot's worker pool at low priority (pool()), or on the thread calling reap() (tests: a
// key asked for is computed and landed by the next poll()).
class StreamExecutor {
public:
    virtual ~StreamExecutor() = default;
    // How many drain tasks may run side by side.
    virtual int workers() const { return 1; }
    // Runs p_task later, on any thread. Called from any thread, the stream's lock maybe held.
    virtual void run(std::function<void()> p_task);
    // Releases the finished tasks (here: runs the queued ones); p_wait: waits for all of them first. The stream's lock not
    // held (a drain task takes it).
    virtual void reap(bool p_wait);

    // Godot's WorkerThreadPool, low priority (threading/worker_pool/low_priority_thread_ratio of its threads).
    static std::unique_ptr<StreamExecutor> pool();

private:
    std::mutex queue_mutex;
    std::vector<std::function<void()>> queued;
};

class InstanceStream {
public:
    static constexpr uint64_t GRACE_FRAMES = 16;
    static constexpr uint32_t SLOTS_MIN = uint32_t(1) << 16;
    static constexpr uint32_t SLOTS_MAX = uint32_t(1) << 21;
    static constexpr int64_t INPUT_RECORDS = int64_t(1) << 21; // the records of ready keys not in slots (inputs), kept at most
    static constexpr size_t ENTRIES_MAX = size_t(1) << 18; // landed keys kept at most (the empty ones count)
    // Keys asked for and not landed at most: past it admit() answers LATER for a chunk needing a new key, so the meshers
    // stop popping (else a frame parks the whole queue: millions of keys waiting, the ones in view behind them).
    static constexpr size_t PENDING_MAX = size_t(1) << 12;
    static constexpr size_t LOG_MAX = size_t(1) << 16; // placements uploads_since() can replay; a reader further behind gets them all

    struct Settings {
        uint32_t slot_base = 0; // the first slot's record id (past the table's own capacity)
        uint32_t slot_capacity = SLOTS_MIN;
        int64_t input_records = INPUT_RECORDS;
        size_t entries = ENTRIES_MAX;
        size_t pending = PENDING_MAX;
        size_t log = LOG_MAX;
    };
    enum Admit {
        READY, // every key placed: mesh it now
        PARKED, // the stream keeps it until its keys land (take_ready)
        LATER, // a key finds no slot, or too many keys are computing: stop popping this frame, ask again later
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
    // chunk's keys and hands it back to p_owner (a mesher: each has its own parked chunks) through take_ready() with p_tag
    // once they are placed; asked again meanwhile, it stays parked (with the new tag).
    Admit admit(const ChunkId &p_chunk, uint64_t p_owner, uint64_t p_tag);
    // Main thread: p_owner's parked chunks whose keys are all placed, or which lost a key's slot (asked again, they answer
    // LATER until there is room), with their tags, in the order they were parked.
    void take_ready(uint64_t p_owner, std::vector<std::pair<ChunkId, uint64_t>> &r_out);
    // Main thread: p_owner's chunks parked and not taken back yet.
    size_t parked_count(uint64_t p_owner) const;
    // Main thread, once per frame: releases the finished drain tasks (keys land as they finish: placed if listed, their
    // dependents started), then evicts down to the limits.
    void poll();

    // The record ids of the chunk's streamed records whose box test can pass at its samples, in key order, appended, and
    // their materials to r_materials (if given). Every key placed (admit() said READY), else the missing ones are left out
    // and counted (lists_missing).
    void chunk_list(const ChunkId &p_chunk, std::vector<uint32_t> &r_out, std::vector<uint32_t> *r_materials = nullptr) const;
    // The streamed records whose box test at p_lod can pass at p_voxel, in key order, appended (copies): a resident key's,
    // else computed here and dropped (counted). Any thread (on the pool's, its loops run inline).
    void point_records(const VoxelPos &p_voxel, int p_lod, std::vector<TableRecord> &r_out) const;
    // point_records without computing on the caller's thread, under the shared lock: only the landed keys' records are
    // appended; the next poll() asks for the keys (kept resident while viewed, the missing ones started on the pool). Main
    // thread (the camera's queries).
    void view_records(const VoxelPos &p_voxel, int p_lod, std::vector<TableRecord> &r_out);

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
    // A parked chunk: the same chunk parked by two meshers is two of them.
    struct ParkKey {
        ChunkId chunk;
        uint64_t owner = 0;
        bool operator==(const ParkKey &p_o) const { return chunk == p_o.chunk && owner == p_o.owner; }
    };
    struct ParkHash {
        size_t operator()(const ParkKey &p_k) const { return Vector3i64Hasher::hash(p_k.chunk) ^ size_t(p_k.owner * 0x9E3779B97F4A7C15ull); }
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
        std::vector<std::pair<ParkKey, uint64_t>> waiters; // the parked chunks waiting for its placement (their park seq)
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
        uint64_t frame = 0; // the poll() frame it was parked at (stats: park to hand-back)
        int missing = 0;
        bool done = false; // in ready_chunks
        std::vector<SourceKey> keys; // pinned
    };

    std::vector<std::shared_ptr<const ScatterSource>> sources;
    ChunkParam param;
    Settings settings;
    std::unique_ptr<StreamExecutor> executor;

    mutable std::shared_mutex mutex; // the stream's state: the main thread and the drain tasks change it, the lists read it
    std::unordered_map<SourceKey, Entry, KeyHash> entries; // references stay valid across rehashes
    Entry *newest = nullptr; // the LRU list's ends
    Entry *oldest = nullptr;
    uint64_t frame = 0;
    size_t pending = 0; // entries not READY (never evicted)

    std::mutex job_mutex; // the drain tasks' queues
    std::deque<std::unique_ptr<Job>> jobs; // to run
    int draining = 0; // drain tasks started and not done

    std::unordered_map<ParkKey, Parked, ParkHash> parked;
    std::vector<ParkKey> ready_chunks; // parked chunks to hand back
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
        int64_t computed_by[SourceKey::KIND_COUNT][ChunkMath::MAX_LOD_NUMBERS] = {}; // per kind and LOD
        int64_t usec_by[SourceKey::KIND_COUNT][ChunkMath::MAX_LOD_NUMBERS] = {};
        int64_t placed = 0;
        int64_t evicted = 0;
        int64_t evicted_by[SourceKey::KIND_COUNT][ChunkMath::MAX_LOD_NUMBERS] = {}; // per kind and LOD: computed / (computed - evicted) = recompute ratio
        int64_t keys_created = 0; // entries created by request(), inputs included
        int64_t admit_keys_created = 0; // of keys_created: by admit()
        int64_t admit_keys_created_max = 0; // by one admit()
        int64_t jobs_queued_max = 0; // the FIFO's depth (start)
        uint64_t land_usec = 0; // land() and its place(), under the lock
        uint64_t evict_usec = 0; // poll()'s eviction loop
        LatencyHist park_frames; // poll() frames from a chunk's park to its hand-back
        int64_t slots_full = 0; // placements refused: no room even after evicting
        int64_t slots_peak = 0;
        int64_t admitted[3] = {}; // per Admit answer
        uint64_t poll_usec = 0;
        uint64_t poll_usec_max = 0;
        uint64_t admit_usec = 0;
        uint64_t admit_wait_usec = 0; // of admit_usec: waiting for the lock (drain tasks landing)
    };
    Stats stats;
    mutable std::atomic<int64_t> lists_missing{ 0 }; // keys a list needed that weren't placed (stays 0)
    mutable std::atomic<int64_t> point_computed{ 0 }; // point_records' keys computed (not resident)
    int64_t view_missing = 0; // view_records' keys not landed yet (left out)
    std::vector<SourceKey> viewed; // view_records' keys since the last poll (main thread), asked for by it

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
    // Evicts the oldest entry if it is out of its grace (false: none can go). p_inputs: the oldest not placed and holding
    // records (making room for inputs: evicting a placed key or a site tile frees none, and asked for again it is computed
    // again).
    bool evict_oldest(bool p_inputs = false);
    void pin(Entry &r_entry);
    void unpin(Entry &r_entry);
    void touch(Entry &r_entry);
    void lru_link(Entry &r_entry);
    void lru_unlink(Entry &r_entry);
    // A parked chunk handed back by take_ready().
    void hand_back(const ParkKey &p_park);
    int64_t slots_used() const { return int64_t(slots_end) - free_slots; }
};

} // namespace Archetype
