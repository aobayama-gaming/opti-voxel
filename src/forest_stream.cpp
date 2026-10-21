// The streamed forest's residency (forest_stream.hpp).
#include "forest_stream.hpp"

#include "debug_log.hpp"
#include "parallel_for.hpp"

#include <godot_cpp/core/error_macros.hpp>

#include <algorithm>

namespace Archetype {

namespace {

// A placed entry's content back from its slots (an input read after its records went to the GPU table).
std::shared_ptr<const FieldContent> from_slots(const RecordStore &p_slots, uint32_t p_slot, uint32_t p_count, const std::vector<FieldContent::Group> &p_groups) {
    auto c = std::make_shared<FieldContent>();
    c->records.reserve(p_count);
    for (uint32_t i = 0; i < p_count; i++) {
        c->records.push_back(p_slots[p_slot + i]);
    }
    c->groups = p_groups;
    return c;
}

} // namespace

size_t ForestStream::KeyHash::operator()(const FieldKey &p_k) const {
    return ExactNoise::hash_cell(p_k.tile, uint32_t(p_k.kind * 64 + p_k.lod + 1) * 0x9E3779B1u);
}

void ForestStream::Job::run(void *p_job) {
    Job &job = *static_cast<Job *>(p_job);
    const InlineParallelFor inline_loops; // a pool task: the walks' and layers' loops run here
    const uint64_t t0 = now_usec();
    job.out = std::make_shared<FieldContent>();
    job.field->content(job.key, *job.out, [&job](const FieldKey &p_k) -> const FieldContent * {
        for (const std::pair<FieldKey, std::shared_ptr<const FieldContent>> &in : job.inputs) {
            if (in.first == p_k) {
                return in.second.get();
            }
        }
        return nullptr; // computed there
    });
    job.usec = now_usec() - t0;
}

ForestStream::ForestStream(std::shared_ptr<const ForestField> p_field, const ChunkParam &p_param, uint32_t p_slot_base, uint32_t p_slot_capacity) :
        field_ptr(std::move(p_field)), param(p_param), base(p_slot_base), capacity(p_slot_capacity) {}

ForestStream::~ForestStream() {
    for (std::pair<const FieldKey, Entry> &kv : entries) {
        if (kv.second.state == Entry::RUNNING) {
            WorkerThreadPool::get_singleton()->wait_for_task_completion(kv.second.task);
        }
    }
}

ForestStream::Entry &ForestStream::request(const FieldKey &p_key, bool p_listed) {
    const auto [it, fresh] = entries.try_emplace(p_key);
    Entry &e = it->second; // references survive the rehashes of the requests below
    e.used = frame;
    if (fresh) {
        std::vector<FieldKey> inputs;
        field().inputs(p_key, inputs);
        for (const FieldKey &k : inputs) {
            request(k, false);
        }
        waiting.push_back(p_key); // poll() queues it once its inputs are there
    }
    if (p_listed) {
        e.listed = true;
    }
    if (e.listed && e.state == Entry::READY && !e.placed) {
        place(e);
    }
    return e;
}

bool ForestStream::ready(const ChunkId &p_chunk) {
    const std::lock_guard<std::mutex> lock(mutex);
    thread_local std::vector<FieldKey> keys;
    field().chunk_keys(p_chunk, param, keys);
    bool all = true;
    for (const FieldKey &k : keys) {
        const Entry &e = request(k, true);
        all = all && e.state == Entry::READY && e.placed;
    }
    stats.ready_asked++;
    stats.ready_waits += !all;
    return all;
}

void ForestStream::start(const FieldKey &p_key, Entry &r_entry) {
    auto job_ptr = std::make_unique<Job>();
    Job &job = *job_ptr;
    job.field = field_ptr.get();
    job.key = p_key;
    std::vector<FieldKey> inputs;
    field().inputs(p_key, inputs);
    for (const FieldKey &k : inputs) {
        const Entry &in = request(k, false);
        if (in.state != Entry::READY || (!in.content && !in.placed)) {
            r_entry.state = Entry::WAITING; // an input made room meanwhile: computed again first
            waiting.push_back(p_key);
            return;
        }
        job.inputs.push_back({ k, in.content ? in.content : from_slots(slot_store, in.slot, in.count, in.groups) });
    }
    r_entry.job = std::move(job_ptr);
    r_entry.task = WorkerThreadPool::get_singleton()->add_native_task(&Job::run, &job, false, "ForestStream");
    r_entry.state = Entry::RUNNING;
    running++;
}

void ForestStream::land(Entry &r_entry) {
    WorkerThreadPool::get_singleton()->wait_for_task_completion(r_entry.task);
    running--;
    stats.computed++;
    stats.compute_usec += int64_t(r_entry.job->usec);
    r_entry.content = r_entry.job->out;
    r_entry.count = uint32_t(r_entry.content->records.size());
    r_entry.job.reset();
    r_entry.state = Entry::READY;
    cached_records += r_entry.count;
    if (r_entry.listed) {
        place(r_entry);
    }
}

void ForestStream::poll() {
    const std::lock_guard<std::mutex> lock(mutex);
    frame++;
    for (std::pair<const FieldKey, Entry> &kv : entries) {
        if (kv.second.state == Entry::RUNNING && WorkerThreadPool::get_singleton()->is_task_completed(kv.second.task)) {
            land(kv.second);
        }
    }
    // The waiting keys whose inputs are all there go to the queue (an input asked for again if it was dropped meanwhile).
    std::vector<FieldKey> check; // request() may add to waiting (an input made room meanwhile): not iterated
    check.swap(waiting);
    for (const FieldKey &k : check) {
        std::vector<FieldKey> inputs;
        field().inputs(k, inputs);
        bool all = true;
        for (const FieldKey &i : inputs) {
            const Entry &in = request(i, false);
            all = all && in.state == Entry::READY && (in.content || in.placed);
        }
        if (all) {
            entries.at(k).state = Entry::QUEUED;
            queued.push_back(k);
        } else {
            waiting.push_back(k);
        }
    }
    while (running < MAX_RUNNING && !queued.empty()) {
        const FieldKey k = queued.front();
        queued.pop_front();
        const auto it = entries.find(k);
        if (it != entries.end() && it->second.state == Entry::QUEUED) {
            start(k, it->second);
        }
    }
    trim_cache();
}

bool ForestStream::place(Entry &r_entry) {
    uint32_t slot = 0;
    if (r_entry.count > 0 && !allocate(r_entry.count, slot) && !make_room(r_entry.count, slot)) {
        if (stats.slots_full++ == 0) {
            ERR_PRINT(vformat("ForestStream: %d records don't fit the %d slots, even past the keys asked for in the last %d frames: the chunks "
                              "needing them wait (raise SDFTable::set_forest_slots).",
                    r_entry.count, capacity, GRACE_FRAMES));
        }
        return false;
    }
    if (r_entry.count > 0) {
        slot_version++;
        while (slot_store.size() < slot + r_entry.count) {
            TableRecord none;
            none.removed = true;
            slot_store.push_back(none, slot_version);
        }
        for (uint32_t i = 0; i < r_entry.count; i++) {
            slot_store.set(slot + i, r_entry.content->records[i], slot_version);
        }
    }
    r_entry.slot = slot;
    r_entry.groups = r_entry.content->groups;
    r_entry.placed = true;
    // Its records live in the slots now (an input reads them back from there).
    cached_records -= r_entry.count;
    r_entry.content.reset();
    stats.placed++;
    stats.slots_peak = MAX(stats.slots_peak, slots_used());
    return true;
}

bool ForestStream::allocate(uint32_t p_count, uint32_t &r_slot) {
    for (auto it = free_runs.begin(); it != free_runs.end(); ++it) {
        if (it->second >= p_count) {
            r_slot = it->first;
            const uint32_t rest = it->second - p_count;
            free_runs.erase(it);
            if (rest > 0) {
                free_runs[r_slot + p_count] = rest;
            }
            return true;
        }
    }
    if (uint64_t(slot_store.size()) + p_count <= capacity) {
        r_slot = slot_store.size();
        return true;
    }
    return false;
}

void ForestStream::release(uint32_t p_slot, uint32_t p_count) {
    if (p_count == 0) {
        return;
    }
    auto it = free_runs.emplace(p_slot, p_count).first;
    const auto next = std::next(it);
    if (next != free_runs.end() && it->first + it->second == next->first) {
        it->second += next->second;
        free_runs.erase(next);
    }
    if (it != free_runs.begin()) {
        const auto prev = std::prev(it);
        if (prev->first + prev->second == it->first) {
            prev->second += it->second;
            free_runs.erase(it);
        }
    }
}

bool ForestStream::make_room(uint32_t p_count, uint32_t &r_slot) {
    std::vector<std::pair<uint64_t, FieldKey>> old;
    for (const std::pair<const FieldKey, Entry> &kv : entries) {
        if (kv.second.placed && kv.second.count > 0 && kv.second.used + GRACE_FRAMES < frame) {
            old.push_back({ kv.second.used, kv.first });
        }
    }
    std::sort(old.begin(), old.end(), [](const std::pair<uint64_t, FieldKey> &p_a, const std::pair<uint64_t, FieldKey> &p_b) {
        return p_a.first != p_b.first ? p_a.first < p_b.first : p_a.second < p_b.second;
    });
    for (const std::pair<uint64_t, FieldKey> &o : old) {
        const auto it = entries.find(o.second);
        release(it->second.slot, it->second.count);
        entries.erase(it); // asked for again, it is computed again (the same records)
        stats.evicted++;
        if (allocate(p_count, r_slot)) {
            return true;
        }
    }
    return false;
}

void ForestStream::trim_cache() {
    if (cached_records <= INPUT_CACHE_RECORDS) {
        return;
    }
    std::vector<std::pair<uint64_t, FieldKey>> old;
    for (const std::pair<const FieldKey, Entry> &kv : entries) {
        if (kv.second.state == Entry::READY && kv.second.content && !kv.second.listed && kv.second.used + GRACE_FRAMES < frame) {
            old.push_back({ kv.second.used, kv.first });
        }
    }
    std::sort(old.begin(), old.end(), [](const std::pair<uint64_t, FieldKey> &p_a, const std::pair<uint64_t, FieldKey> &p_b) {
        return p_a.first != p_b.first ? p_a.first < p_b.first : p_a.second < p_b.second;
    });
    for (const std::pair<uint64_t, FieldKey> &o : old) {
        if (cached_records <= INPUT_CACHE_RECORDS) {
            break;
        }
        const auto it = entries.find(o.second);
        cached_records -= it->second.count;
        entries.erase(it);
    }
}

int64_t ForestStream::slots_used() const {
    int64_t free = 0;
    for (const std::pair<const uint32_t, uint32_t> &r : free_runs) {
        free += r.second;
    }
    return int64_t(slot_store.size()) - free;
}

void ForestStream::chunk_list(const ChunkId &p_chunk, std::vector<uint32_t> &r_out) const {
    thread_local std::vector<FieldKey> keys;
    field().chunk_keys(p_chunk, param, keys);
    VoxelPos lo, hi;
    chunk_sample_box(p_chunk, param, lo, hi);
    const int lod = ChunkMath::get_lod(p_chunk);
    for (const FieldKey &k : keys) {
        const auto it = entries.find(k);
        if (it == entries.end() || !it->second.placed) {
            lists_missing++;
            continue;
        }
        const size_t from = r_out.size();
        FieldContent::meeting(it->second.groups, RecordView(slot_store), it->second.slot, lo, hi, lod, r_out);
        for (size_t i = from; i < r_out.size(); i++) {
            r_out[i] += base;
        }
    }
}

void ForestStream::point_records(const VoxelPos &p_voxel, int p_lod, std::vector<TableRecord> &r_out) const {
    const std::lock_guard<std::mutex> lock(mutex);
    trim_demand(); // before any reference into it is held
    std::vector<FieldKey> keys;
    field().keys(p_voxel, p_voxel, p_lod, keys);
    std::vector<uint32_t> hit;
    for (const FieldKey &k : keys) {
        hit.clear();
        const auto it = entries.find(k);
        if (it != entries.end() && it->second.placed) {
            FieldContent::meeting(it->second.groups, RecordView(slot_store), it->second.slot, p_voxel, p_voxel, p_lod, hit);
            for (const uint32_t i : hit) {
                r_out.push_back(slot_store[i]);
            }
            continue;
        }
        const FieldContent &c = on_demand(k);
        c.meeting(p_voxel, p_voxel, p_lod, hit);
        for (const uint32_t i : hit) {
            r_out.push_back(c.records[i]);
        }
    }
}

const FieldContent &ForestStream::on_demand(const FieldKey &p_key) const {
    const auto it = entries.find(p_key);
    if (it != entries.end() && it->second.content) {
        return *it->second.content;
    }
    const auto d = demand.find(p_key);
    if (d != demand.end()) {
        d->second.used = ++demand_clock;
        return *d->second.content;
    }
    std::shared_ptr<const FieldContent> c;
    if (it != entries.end() && it->second.placed) {
        c = from_slots(slot_store, it->second.slot, it->second.count, it->second.groups);
    } else {
        auto computed = std::make_shared<FieldContent>();
        field().content(p_key, *computed, [this](const FieldKey &p_k) { return &on_demand(p_k); });
        demand_computed++;
        c = std::move(computed);
    }
    demand_records += int64_t(c->records.size());
    return *demand.emplace(p_key, Demand{ std::move(c), ++demand_clock }).first->second.content; // std::map: references stay valid
}

void ForestStream::trim_demand() const {
    if (demand_records <= DEMAND_CACHE_RECORDS) {
        return;
    }
    std::vector<std::pair<uint64_t, FieldKey>> old;
    for (const std::pair<const FieldKey, Demand> &kv : demand) {
        old.push_back({ kv.second.used, kv.first });
    }
    std::sort(old.begin(), old.end(), [](const std::pair<uint64_t, FieldKey> &p_a, const std::pair<uint64_t, FieldKey> &p_b) { return p_a.first < p_b.first; });
    for (const std::pair<uint64_t, FieldKey> &o : old) {
        if (demand_records <= DEMAND_CACHE_RECORDS / 2) {
            break;
        }
        const auto it = demand.find(o.second);
        demand_records -= int64_t(it->second.content->records.size());
        demand.erase(it);
    }
}

Dictionary ForestStream::get_stats() const {
    const std::lock_guard<std::mutex> lock(mutex);
    int64_t placed = 0, ready = 0;
    for (const std::pair<const FieldKey, Entry> &kv : entries) {
        placed += kv.second.placed;
        ready += kv.second.state == Entry::READY;
    }
    Dictionary d;
    d["keys"] = int64_t(entries.size());
    d["keys_ready"] = ready;
    d["keys_placed"] = placed;
    d["keys_running"] = int64_t(running);
    d["keys_queued"] = int64_t(queued.size());
    d["keys_waiting"] = int64_t(waiting.size());
    d["computed"] = stats.computed;
    d["compute_ms_avg"] = stats.computed > 0 ? double(stats.compute_usec) / 1000.0 / double(stats.computed) : 0.0;
    d["compute_ms_total"] = double(stats.compute_usec) / 1000.0;
    d["placed"] = stats.placed;
    d["evicted"] = stats.evicted;
    d["slots_used"] = slots_used();
    d["slots_peak"] = stats.slots_peak;
    d["slots_capacity"] = int64_t(capacity);
    d["slots_full"] = stats.slots_full;
    d["cached_records"] = cached_records;
    d["chunks_asked"] = stats.ready_asked;
    d["chunks_waited"] = stats.ready_waits;
    d["lists_missing"] = lists_missing.load();
    d["evaluate_computed"] = demand_computed.load();
    return d;
}

} // namespace Archetype
