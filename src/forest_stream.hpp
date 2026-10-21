#pragma once

#include "forest_field.hpp"

#include <godot_cpp/classes/worker_thread_pool.hpp>
#include <godot_cpp/variant/dictionary.hpp>

#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>

// The streamed forest's residency: which keys of a ForestField (forest_field.hpp) are computed, and where their records
// sit in the GPU table. A chunk is meshed only once every key it lists is resident (ready()): its records are the keys'
// (a pure function of each key) and its list folds them in key order, so a chunk's content is a pure function of its id
// whatever streamed when, and no chunk is meshed twice because a key arrived late.
//
// Keys are computed on the worker pool, their inputs first (a coarser tree tile reads its LOD-0 tile, a part layer's blob
// tile the LOD-0 tree tiles around it: kept in a CPU cache for their readers), and a listed key's records are copied into
// slots: the GPU table's records from slot_base on, uploaded as the pages changed since a mesher's version. Nothing is
// freed until room is needed; then the keys least recently asked for, none asked for in the last GRACE_FRAMES (a batch in
// flight still reads its slots).
namespace Archetype {

class ForestStream {
public:
    static constexpr uint64_t GRACE_FRAMES = 16;
    static constexpr int MAX_RUNNING = 48; // keys computed at once
    static constexpr int64_t INPUT_CACHE_RECORDS = int64_t(4) << 20; // the LOD-0 tree tiles kept for their readers, in records
    static constexpr size_t DEMAND_CACHE_KEYS = 256; // contents computed for CPU evaluations of keys not resident

    ForestStream(std::shared_ptr<const ForestField> p_field, const ChunkParam &p_param, uint32_t p_slot_base, uint32_t p_slot_capacity);
    ~ForestStream(); // waits for the keys being computed

    const ForestField &field() const { return *field_ptr; }
    uint32_t slot_base() const { return base; }
    uint32_t slot_capacity() const { return capacity; }

    // Main thread. Whether every key the chunk lists is resident; asks for the ones that aren't (the meshers ask in queue
    // order: the first asked are computed first).
    bool ready(const ChunkId &p_chunk);
    // Main thread, once per frame: lands the computed keys, starts the next ones.
    void poll();

    // The ids (slot_base + slot) of the chunk's streamed records whose box test can pass at its samples, in key order.
    // Every key resident (ready() was true), else the missing ones are left out and counted. Reads only: build_lists runs
    // it on the pool while the main thread waits.
    void chunk_list(const ChunkId &p_chunk, std::vector<uint32_t> &r_out) const;
    // The slots (record slot_base + i is slots()[i]) and their version: bumped by every key placed.
    const RecordStore &slots() const { return slot_store; }
    uint32_t version() const { return slot_version; }

    // CPU evaluation, any thread: the streamed records whose box test at p_lod can pass at p_voxel, in key order, appended
    // (copies). A key not resident is computed here (a small cache keeps it): the same records.
    void point_records(const VoxelPos &p_voxel, int p_lod, std::vector<TableRecord> &r_out) const;

    Dictionary get_stats() const;

private:
    struct KeyHash {
        size_t operator()(const FieldKey &p_k) const;
    };
    struct Job {
        const ForestField *field = nullptr;
        FieldKey key;
        std::vector<std::pair<FieldKey, std::shared_ptr<const FieldContent>>> inputs;
        std::shared_ptr<FieldContent> out;
        uint64_t usec = 0;
        static void run(void *p_job);
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
        WorkerThreadPool::TaskID task = -1;
    };

    std::shared_ptr<const ForestField> field_ptr;
    ChunkParam param;
    uint32_t base = 0;
    uint32_t capacity = 0;
    std::unordered_map<FieldKey, Entry, KeyHash> entries;
    std::deque<FieldKey> queued; // to start, in the order asked
    std::vector<FieldKey> waiting; // for inputs
    int running = 0;
    uint64_t frame = 0;
    RecordStore slot_store;
    uint32_t slot_version = 0;
    std::map<uint32_t, uint32_t> free_runs; // first slot -> count, below the store's size
    int64_t cached_records = 0; // the contents kept (inputs and not yet placed)
    mutable std::mutex mutex; // main-thread changes vs point_records on other threads
    mutable std::map<FieldKey, std::shared_ptr<const FieldContent>> demand;

    struct Stats {
        int64_t computed = 0;
        int64_t compute_usec = 0;
        int64_t placed = 0;
        int64_t evicted = 0;
        int64_t slots_full = 0; // placements refused: no room even after evicting
        int64_t slots_peak = 0;
        int64_t ready_asked = 0;
        int64_t ready_waits = 0;
    };
    Stats stats;
    mutable std::atomic<int64_t> lists_missing{ 0 }; // keys a list needed that weren't resident (should stay 0)
    mutable std::atomic<int64_t> demand_computed{ 0 };

    // The key's entry, asked for now (created and its inputs asked for if new).
    Entry &request(const FieldKey &p_key, bool p_listed);
    void start(const FieldKey &p_key, Entry &r_entry);
    void land(Entry &r_entry);
    // Copies a ready entry's records into slots (false: no room).
    bool place(Entry &r_entry);
    bool allocate(uint32_t p_count, uint32_t &r_slot);
    void release(uint32_t p_slot, uint32_t p_count);
    // Frees the slots of the least recently used placed entries not asked for in GRACE_FRAMES until p_count fit.
    bool make_room(uint32_t p_count, uint32_t &r_slot);
    void trim_cache();
    int64_t slots_used() const;
};

} // namespace Archetype
