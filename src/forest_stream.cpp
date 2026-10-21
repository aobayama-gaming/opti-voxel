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

void ForestStream::Job::run() {
    const InlineParallelFor inline_loops; // one key per worker: the walks' and layers' loops run here
    const uint64_t t0 = now_usec();
    out = std::make_shared<FieldContent>();
    field->content(key, *out, [this](const FieldKey &p_k) -> const FieldContent * {
        for (const std::pair<FieldKey, std::shared_ptr<const FieldContent>> &in : inputs) {
            if (in.first == p_k) {
                return in.second.get();
            }
        }
        return nullptr; // computed there
    });
    usec = now_usec() - t0;
}

void ForestStream::work() {
    for (;;) {
        Job *job = nullptr;
        {
            std::unique_lock<std::mutex> lock(job_mutex);
            job_cv.wait(lock, [this] { return stopping || !jobs.empty(); });
            if (stopping) {
                return;
            }
            job = jobs.front();
            jobs.pop_front();
        }
        job->run();
        const std::lock_guard<std::mutex> lock(job_mutex);
        finished.push_back(job->key); // poll() lands it (the job is the entry's: it lives until then)
    }
}

ForestStream::ForestStream(std::shared_ptr<const ForestField> p_field, const ChunkParam &p_param, uint32_t p_slot_base, uint32_t p_slot_capacity,
        int64_t p_input_cache_records) :
        field_ptr(std::move(p_field)), param(p_param), base(p_slot_base), capacity(p_slot_capacity), cache_limit(p_input_cache_records) {
    const unsigned n = MAX(2u, std::thread::hardware_concurrency() / 2);
    max_running = int(n) * JOBS_PER_WORKER;
    for (unsigned i = 0; i < n; i++) {
        workers.emplace_back(&ForestStream::work, this);
    }
}

ForestStream::~ForestStream() {
    {
        const std::lock_guard<std::mutex> lock(job_mutex);
        stopping = true; // the jobs not started are dropped with their entries
    }
    job_cv.notify_all();
    for (std::thread &w : workers) {
        w.join();
    }
}

ForestStream::Entry &ForestStream::request(const FieldKey &p_key, bool p_listed) {
    const auto [it, fresh] = entries.try_emplace(p_key);
    Entry &e = it->second; // references survive the rehashes of the requests below
    e.used = frame;
    if (fresh) {
        wait_for_inputs(p_key, e);
    }
    if (p_listed) {
        e.listed = true;
    }
    if (e.listed && e.state == Entry::READY && !e.placed) {
        place(e);
    }
    return e;
}

void ForestStream::wait_for_inputs(const FieldKey &p_key, Entry &r_entry) {
    std::vector<FieldKey> inputs;
    field().inputs(p_key, inputs);
    r_entry.state = Entry::WAITING;
    r_entry.missing = 0;
    for (const FieldKey &k : inputs) {
        Entry &in = request(k, false);
        if (in.state != Entry::READY) { // a ready entry holds its content or is placed
            r_entry.missing++;
            in.dependents.push_back(p_key); // land() counts it down
        }
    }
    if (r_entry.missing == 0) {
        r_entry.state = Entry::QUEUED;
        queued.push_back(p_key);
    }
}

bool ForestStream::ready(const ChunkId &p_chunk) {
    const uint64_t t0 = now_usec();
    const std::lock_guard<std::mutex> lock(mutex);
    const auto [it, fresh] = chunk_waits.try_emplace(p_chunk);
    ChunkWait &w = it->second;
    w.asked = frame;
    // A chunk still missing keys since its last full check answers at once (its keys were asked for then).
    const bool all = !(!fresh && w.missing > 0 && frame - w.checked < RECHECK_FRAMES) && check_chunk(p_chunk, w);
    if (all) {
        chunk_waits.erase(it);
    }
    stats.ready_asked++;
    stats.ready_waits += !all;
    const int lod = MIN(MAX(ChunkMath::get_lod(p_chunk), 0), ChunkMath::MAX_LOD_NUMBERS - 1);
    stats.asked_by_lod[lod]++;
    stats.waited_by_lod[lod] += !all;
    stats.ready_usec += now_usec() - t0;
    return all;
}

bool ForestStream::check_chunk(const ChunkId &p_chunk, ChunkWait &r_wait) {
    stats.full_checks++;
    r_wait.gen++; // the waiters of older checks no longer count
    r_wait.checked = frame;
    r_wait.missing = 0;
    thread_local std::vector<FieldKey> keys;
    field().chunk_keys(p_chunk, param, keys);
    for (const FieldKey &k : keys) {
        Entry &e = request(k, true);
        if (e.placed) {
            continue;
        }
        r_wait.missing++;
        if (!e.waiters.empty() && e.waiters.back().first == p_chunk) {
            e.waiters.back().second = r_wait.gen;
        } else {
            e.waiters.push_back({ p_chunk, r_wait.gen }); // place() counts it down
        }
    }
    return r_wait.missing == 0;
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
        if (in.state != Entry::READY) {
            wait_for_inputs(p_key, r_entry); // an input made room meanwhile: computed again first
            return;
        }
        job.inputs.push_back({ k, in.content ? in.content : from_slots(slot_store, in.slot, in.count, in.groups) });
    }
    r_entry.job = std::move(job_ptr);
    {
        const std::lock_guard<std::mutex> lock(job_mutex);
        jobs.push_back(&job);
    }
    job_cv.notify_one();
    r_entry.state = Entry::RUNNING;
    running++;
}

void ForestStream::land(Entry &r_entry) {
    running--;
    stats.computed++;
    stats.compute_usec += int64_t(r_entry.job->usec);
    const FieldKey &key = r_entry.job->key;
    const int lod = MIN(MAX(key.lod, 0), ChunkMath::MAX_LOD_NUMBERS - 1);
    stats.by_count[key.kind][lod]++;
    stats.by_usec[key.kind][lod] += int64_t(r_entry.job->usec);
    stats.by_records[key.kind][lod] += int64_t(r_entry.job->out->records.size());
    r_entry.content = r_entry.job->out;
    r_entry.count = uint32_t(r_entry.content->records.size());
    r_entry.job.reset();
    r_entry.state = Entry::READY;
    cached_records += r_entry.count;
    // The keys waiting for it as an input: queued once it was their last.
    std::vector<FieldKey> dependents;
    dependents.swap(r_entry.dependents);
    for (const FieldKey &d : dependents) {
        const auto it = entries.find(d);
        if (it != entries.end() && it->second.state == Entry::WAITING && --it->second.missing == 0) {
            it->second.state = Entry::QUEUED;
            queued.push_back(d);
        }
    }
    if (r_entry.listed) {
        place(r_entry);
    }
}

void ForestStream::poll() {
    const uint64_t t0 = now_usec();
    const std::lock_guard<std::mutex> lock(mutex);
    frame++;
    std::vector<FieldKey> done;
    {
        const std::lock_guard<std::mutex> job_lock(job_mutex);
        done.swap(finished);
    }
    for (const FieldKey &k : done) {
        const auto it = entries.find(k); // a running entry is never dropped
        if (it != entries.end() && it->second.state == Entry::RUNNING) {
            land(it->second);
        }
    }
    while (running < max_running && !queued.empty()) {
        const FieldKey k = queued.front();
        queued.pop_front();
        const auto it = entries.find(k);
        if (it != entries.end() && it->second.state == Entry::QUEUED) {
            start(k, it->second);
        }
    }
    trim_cache();
    if (frame % WAIT_DROP_FRAMES == 0) {
        for (auto it = chunk_waits.begin(); it != chunk_waits.end();) {
            it = it->second.asked + WAIT_DROP_FRAMES < frame ? chunk_waits.erase(it) : std::next(it); // left the queues
        }
    }
    const uint64_t dt = now_usec() - t0;
    stats.poll_usec += dt;
    stats.poll_usec_max = MAX(stats.poll_usec_max, dt);
}

bool ForestStream::place(Entry &r_entry) {
    uint32_t slot = 0;
    if (r_entry.count > 0 && !allocate(r_entry.count, slot) && !make_room(r_entry.count, slot)) {
        if (stats.slots_full++ == 0) {
            ERR_PRINT(vformat("ForestStream: %d records don't fit the %d slots, even past the keys asked for in the last %d frames: the chunks "
                              "needing them wait (raise SDFTable::FOREST_SLOTS).",
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
    // The chunks waiting for it, counted down (a waiter from an older full check of its chunk no longer counts).
    for (const std::pair<ChunkId, uint32_t> &w : r_entry.waiters) {
        const auto it = chunk_waits.find(w.first);
        if (it != chunk_waits.end() && it->second.gen == w.second && it->second.missing > 0) {
            it->second.missing--;
        }
    }
    r_entry.waiters.clear();
    return true;
}

bool ForestStream::allocate(uint32_t p_count, uint32_t &r_slot) {
    for (auto it = free_runs.begin(); it != free_runs.end(); ++it) {
        if (it->second >= p_count) {
            r_slot = it->first;
            const uint32_t rest = it->second - p_count;
            free_slots -= p_count;
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
    free_slots += p_count;
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
    std::sort(old.begin(), old.end()); // least recently asked first, then key order
    // Past the request, a sixteenth of the pool more is freed: the scan above runs once per that much churn.
    const int64_t slack = int64_t(capacity / 16);
    int64_t freed = 0;
    bool got = false;
    for (const std::pair<uint64_t, FieldKey> &o : old) {
        const auto it = entries.find(o.second);
        release(it->second.slot, it->second.count);
        freed += it->second.count;
        entries.erase(it); // asked for again, it is computed again (the same records)
        stats.evicted++;
        got = got || allocate(p_count, r_slot);
        if (got && freed >= slack) {
            break;
        }
    }
    return got;
}

void ForestStream::trim_cache() {
    if (cached_records <= cache_limit) {
        return;
    }
    std::vector<std::pair<uint64_t, FieldKey>> old;
    for (const std::pair<const FieldKey, Entry> &kv : entries) {
        if (kv.second.state == Entry::READY && kv.second.content && !kv.second.listed && kv.second.used + GRACE_FRAMES < frame) {
            old.push_back({ kv.second.used, kv.first });
        }
    }
    std::sort(old.begin(), old.end()); // least recently asked first, then key order
    for (const std::pair<uint64_t, FieldKey> &o : old) {
        if (cached_records <= cache_limit * 3 / 4) { // down to 3/4: the scan above runs once per quarter of the cache
            break;
        }
        const auto it = entries.find(o.second);
        cached_records -= it->second.count;
        stats.trimmed++;
        entries.erase(it);
    }
}

int64_t ForestStream::slots_used() const {
    return int64_t(slot_store.size()) - free_slots;
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
    std::sort(old.begin(), old.end()); // least recently read first (the clock is unique)
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
    int64_t placed = 0, ready = 0, waiting = 0;
    for (const std::pair<const FieldKey, Entry> &kv : entries) {
        placed += kv.second.placed;
        ready += kv.second.state == Entry::READY;
        waiting += kv.second.state == Entry::WAITING;
    }
    Dictionary d;
    d["keys"] = int64_t(entries.size());
    d["keys_ready"] = ready;
    d["keys_placed"] = placed;
    d["keys_running"] = int64_t(running);
    d["keys_queued"] = int64_t(queued.size());
    d["keys_waiting"] = waiting;
    d["chunk_waits"] = int64_t(chunk_waits.size());
    d["chunk_full_checks"] = stats.full_checks;
    d["cache_trimmed"] = stats.trimmed;
    d["poll_ms_total"] = double(stats.poll_usec) / 1000.0;
    d["poll_ms_max"] = double(stats.poll_usec_max) / 1000.0;
    d["ready_ms_total"] = double(stats.ready_usec) / 1000.0;
    d["computed"] = stats.computed;
    d["compute_ms_avg"] = stats.computed > 0 ? double(stats.compute_usec) / 1000.0 / double(stats.computed) : 0.0;
    d["compute_ms_total"] = double(stats.compute_usec) / 1000.0;
    Dictionary by;
    for (int kind = 0; kind < 2; kind++) {
        for (int lod = 0; lod < ChunkMath::MAX_LOD_NUMBERS; lod++) {
            const int64_t n = stats.by_count[kind][lod];
            if (n > 0) {
                Dictionary k;
                k["count"] = n;
                k["ms_avg"] = double(stats.by_usec[kind][lod]) / 1000.0 / double(n);
                k["records_avg"] = double(stats.by_records[kind][lod]) / double(n);
                by[vformat("%s %d", kind == FieldKey::TREES ? "trees" : "blobs", lod)] = k;
            }
        }
    }
    d["computed_by_key"] = by;
    d["placed"] = stats.placed;
    d["evicted"] = stats.evicted;
    d["slots_used"] = slots_used();
    d["slots_peak"] = stats.slots_peak;
    d["slots_capacity"] = int64_t(capacity);
    d["slots_full"] = stats.slots_full;
    d["cached_records"] = cached_records;
    d["chunks_asked"] = stats.ready_asked;
    d["chunks_waited"] = stats.ready_waits;
    String per_lod; // asked / waited per LOD
    for (int l = 0; l < ChunkMath::MAX_LOD_NUMBERS; l++) {
        if (stats.asked_by_lod[l] > 0) {
            per_lod += vformat("%sL%d %d/%d", per_lod.is_empty() ? "" : " ", l, stats.asked_by_lod[l], stats.waited_by_lod[l]);
        }
    }
    d["chunks_asked_waited_per_lod"] = per_lod;
    d["lists_missing"] = lists_missing.load();
    d["evaluate_computed"] = demand_computed.load();
    return d;
}

} // namespace Archetype
