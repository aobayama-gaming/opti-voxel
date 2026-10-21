#pragma once

#include "forest_field.hpp"

#include <godot_cpp/variant/dictionary.hpp>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

// The streamed forest's residency: which keys of a ForestField (forest_field.hpp) are computed, and where their records
// sit in the GPU table. A chunk is meshed only once every key it lists is resident (ready()): its records are the keys'
// (a pure function of each key) and its list folds them in key order, so a chunk's content is a pure function of its id
// whatever streamed when, and no chunk is meshed twice because a key arrived late.
//
// Keys are computed on the stream's own threads (Godot's pool gives low-priority tasks a few of its threads), their inputs first (a coarser tree tile reads its LOD-0 tile, a part layer's blob
// tile the LOD-0 tree tiles around it: kept in a CPU cache for their readers), and a listed key's records are copied into
// slots: the GPU table's records from slot_base on, uploaded as the pages changed since a mesher's version. Nothing is
// freed until room is needed; then the keys least recently asked for, none asked for in the last GRACE_FRAMES (a batch in
// flight still reads its slots).
namespace Archetype {

class ForestStream {
public:
    static constexpr uint64_t GRACE_FRAMES = 16;
    static constexpr int JOBS_PER_WORKER = 32; // keys handed to the workers at once, per worker
    // A chunk still waiting for its keys is checked in full again once they are all placed, or after RECHECK_FRAMES (the
    // safety net: a key evicted or refused meanwhile); between, ready() answers from the keys placed since. Its placed keys
    // may then go unasked past the grace: only a full pool evicts them, and the full check asks for them again. A full
    // check every GRACE_FRAMES / 2 was 49% of the main thread on the 1M load (a LOD-7 chunk lists hundreds of keys). A
    // wait not asked for in WAIT_DROP_FRAMES is dropped.
    static constexpr uint64_t RECHECK_FRAMES = 64;
    static constexpr uint64_t WAIT_DROP_FRAMES = 64;
    // Placed keys without records (air, empty ground) hold no slot, so no eviction drops them: poll() sweeps those not
    // asked for in GRACE_FRAMES once the keys doubled since the last sweep (at least SWEEP_MIN_KEYS): amortized O(1).
    static constexpr size_t SWEEP_MIN_KEYS = 128;
    static constexpr int64_t INPUT_CACHE_RECORDS = int64_t(4) << 20; // the LOD-0 tree tiles kept for their readers, in records
    static constexpr int64_t DEMAND_CACHE_RECORDS = int64_t(1) << 20; // the contents CPU evaluations computed, in records

    ForestStream(std::shared_ptr<const ForestField> p_field, const ChunkParam &p_param, uint32_t p_slot_base, uint32_t p_slot_capacity,
            int64_t p_input_cache_records = INPUT_CACHE_RECORDS);
    ~ForestStream(); // waits for the keys being computed, the others are dropped

    const ForestField &field() const { return *field_ptr; }
    uint32_t slot_base() const { return base; }
    uint32_t slot_capacity() const { return capacity; }

    // Main thread. Whether every key the chunk lists is resident; asks for the ones that aren't (the meshers ask in queue
    // order: the first asked are computed first). A chunk found waiting is answered from the count of its keys placed
    // since (no key looked at) until RECHECK_FRAMES have passed, then checked in full again.
    bool ready(const ChunkId &p_chunk);
    // Main thread, once per frame: lands the keys the workers finished (their dependents queued once their last input is
    // there), starts the queued ones. Its cost follows what landed, not what the stream holds.
    void poll();

    // The ids (slot_base + slot) of the chunk's streamed records whose box test can pass at its samples, in key order.
    // Every key resident (ready() was true), else the missing ones are left out and counted. Reads only: build_lists runs
    // it on the pool while the main thread waits.
    void chunk_list(const ChunkId &p_chunk, std::vector<uint32_t> &r_out) const;
    // The slots (record slot_base + i is slots()[i]) and their version: bumped by every key placed.
    const RecordStore &slots() const { return slot_store; }
    uint32_t version() const { return slot_version; }

    // CPU evaluation, any thread but the worker pool's (a key not resident is computed here, on the pool; a small cache
    // keeps it): the streamed records whose box test at p_lod can pass at p_voxel, in key order, appended (copies).
    void point_records(const VoxelPos &p_voxel, int p_lod, std::vector<TableRecord> &r_out) const;

    Dictionary get_stats() const;

private:
    struct KeyHash {
        size_t operator()(const FieldKey &p_k) const;
    };
    struct ChunkHash {
        size_t operator()(const ChunkId &p_c) const { return Vector3i64Hasher::hash(p_c); }
    };
    struct Job {
        const ForestField *field = nullptr;
        FieldKey key;
        std::vector<std::pair<FieldKey, std::shared_ptr<const FieldContent>>> inputs;
        std::shared_ptr<FieldContent> out;
        uint64_t usec = 0;
        void run();
    };
    struct Entry {
        enum State {
            WAITING, // for its inputs
            QUEUED, // to start
            RUNNING,
            READY,
        };
        State state = WAITING;
        std::shared_ptr<const FieldContent> content; // kept while it can be read (an input) or placed
        std::vector<FieldContent::Group> groups; // once placed: relative to its slot
        uint32_t slot = 0;
        uint32_t count = 0;
        bool placed = false;
        bool listed = false; // a chunk lists it: it goes in slots
        uint64_t used = 0; // the frame it was last asked for
        std::unique_ptr<Job> job;
        int missing = 0; // WAITING: its inputs not there yet
        std::vector<FieldKey> dependents; // the keys waiting for it (as an input), told when it lands
        std::vector<std::pair<ChunkId, uint32_t>> waiters; // the chunks waiting for it (with their wait's gen), told when it is placed
    };
    // A chunk ready() found waiting: how many of its keys weren't placed at its last full check, minus those placed since.
    struct ChunkWait {
        int missing = 0;
        uint32_t gen = 0; // bumped by every full check: older waiters are ignored
        uint64_t checked = 0; // the frame of the last full check
        uint64_t asked = 0; // the frame ready() last asked
    };

    std::shared_ptr<const ForestField> field_ptr;
    ChunkParam param;
    uint32_t base = 0;
    uint32_t capacity = 0;
    std::unordered_map<FieldKey, Entry, KeyHash> entries;
    // The workers (half the hardware threads, at least 2), each computing one key at a time, inline.
    std::vector<std::thread> workers;
    std::mutex job_mutex;
    std::condition_variable job_cv;
    std::deque<Job *> jobs; // handed out, not started
    std::vector<FieldKey> finished; // computed, to land (under job_mutex)
    bool stopping = false;
    void work();
    std::deque<FieldKey> queued; // to start, in the order their inputs arrived
    int running = 0;
    int max_running = 0;
    int64_t cache_limit = INPUT_CACHE_RECORDS;
    std::unordered_map<ChunkId, ChunkWait, ChunkHash> chunk_waits;
    size_t sweep_at = SWEEP_MIN_KEYS;
    uint64_t frame = 0;
    RecordStore slot_store;
    uint32_t slot_version = 0;
    std::map<uint32_t, uint32_t> free_runs; // first slot -> count, below the store's size
    int64_t free_slots = 0; // the sum of free_runs
    int64_t cached_records = 0; // the contents kept (inputs and not yet placed)
    mutable std::mutex mutex; // main-thread changes vs point_records on other threads
    struct Demand {
        std::shared_ptr<const FieldContent> content;
        uint64_t used = 0; // demand_clock when last read
    };
    mutable std::map<FieldKey, Demand> demand; // least recently read dropped past DEMAND_CACHE_RECORDS
    mutable uint64_t demand_clock = 0;
    mutable int64_t demand_records = 0;

    struct Stats {
        int64_t computed = 0;
        int64_t compute_usec = 0;
        // Per kind (FieldKey::Kind) and LOD: keys computed, their time and records.
        int64_t by_count[2][ChunkMath::MAX_LOD_NUMBERS] = {};
        int64_t by_usec[2][ChunkMath::MAX_LOD_NUMBERS] = {};
        int64_t by_records[2][ChunkMath::MAX_LOD_NUMBERS] = {};
        int64_t placed = 0;
        int64_t evicted = 0;
        int64_t slots_full = 0; // placements refused: no room even after evicting
        int64_t slots_peak = 0;
        int64_t ready_asked = 0;
        int64_t ready_waits = 0;
        int64_t asked_by_lod[ChunkMath::MAX_LOD_NUMBERS] = {};
        int64_t waited_by_lod[ChunkMath::MAX_LOD_NUMBERS] = {};
        int64_t full_checks = 0; // ready() calls that looked at every key of their chunk
        int64_t trimmed = 0; // inputs dropped from the cache
        int64_t swept = 0; // empty keys dropped
        uint64_t poll_usec_max = 0;
        uint64_t poll_usec = 0;
        uint64_t ready_usec = 0;
    };
    Stats stats;
    mutable std::atomic<int64_t> lists_missing{ 0 }; // keys a list needed that weren't resident (should stay 0)
    mutable std::atomic<int64_t> demand_computed{ 0 };

    // The key's entry, asked for now (created and its inputs asked for if new).
    Entry &request(const FieldKey &p_key, bool p_listed);
    // Counts the key's inputs not there yet (asking for them) and registers it as their dependent; queues it if none.
    void wait_for_inputs(const FieldKey &p_key, Entry &r_entry);
    void start(const FieldKey &p_key, Entry &r_entry);
    void land(Entry &r_entry);
    // ready()'s full check: every key of the chunk asked for; the ones not placed get the chunk as a waiter.
    bool check_chunk(const ChunkId &p_chunk, ChunkWait &r_wait);
    // Copies a ready entry's records into slots (false: no room).
    bool place(Entry &r_entry);
    bool allocate(uint32_t p_count, uint32_t &r_slot);
    void release(uint32_t p_slot, uint32_t p_count);
    // Frees the slots of the least recently used placed entries not asked for in GRACE_FRAMES until p_count fit.
    bool make_room(uint32_t p_count, uint32_t &r_slot);
    void trim_cache();
    void sweep_empty();
    // A key's content for a CPU evaluation (under the mutex): kept (an input), back from its slots or computed, its
    // inputs the same way; cached in demand.
    const FieldContent &on_demand(const FieldKey &p_key) const;
    // Past DEMAND_CACHE_RECORDS, drops the least recently read contents down to half of it (between evaluations: no
    // reference into it is held then).
    void trim_demand() const;
    int64_t slots_used() const;
};

} // namespace Archetype
