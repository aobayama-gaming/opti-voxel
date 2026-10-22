// The streamed sources' residency (instance_stream.hpp).
#include "instance_stream.hpp"

#include "debug_log.hpp"
#include "parallel_for.hpp"

#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <algorithm>

namespace Archetype {

namespace {

class PoolExecutor : public StreamExecutor {
public:
    ~PoolExecutor() override { reap(true); }
    // The pool gives low-priority tasks threading/worker_pool/low_priority_thread_ratio (0.5) of its threads.
    int workers() const override { return MAX(1, OS::get_singleton()->get_processor_count() / 2); }
    void run(std::function<void()> p_task) override {
        auto task = std::make_unique<Task>(Task{ std::move(p_task) });
        const int64_t id = WorkerThreadPool::get_singleton()->add_native_task(&Task::call, task.get(), false, "InstanceStream");
        const std::lock_guard<std::mutex> lock(task_mutex);
        tasks.emplace_back(id, std::move(task));
    }
    void reap(bool p_wait) override {
        WorkerThreadPool *pool = WorkerThreadPool::get_singleton();
        for (;;) {
            std::vector<std::pair<int64_t, std::unique_ptr<Task>>> done;
            {
                const std::lock_guard<std::mutex> lock(task_mutex);
                for (size_t i = 0; i < tasks.size();) {
                    if (p_wait || pool->is_task_completed(tasks[i].first)) {
                        done.push_back(std::move(tasks[i]));
                        tasks[i] = std::move(tasks.back());
                        tasks.pop_back();
                    } else {
                        i++;
                    }
                }
            }
            // Released without the lock: a running task (p_wait) may start others meanwhile, taken by the next round.
            for (const std::pair<int64_t, std::unique_ptr<Task>> &t : done) {
                pool->wait_for_task_completion(t.first); // releases its id
            }
            if (!p_wait || done.empty()) {
                return;
            }
        }
    }

private:
    struct Task {
        std::function<void()> fn;
        static void call(void *p_task) { static_cast<Task *>(p_task)->fn(); }
    };
    std::mutex task_mutex; // run() is called from the drain tasks too
    std::vector<std::pair<int64_t, std::unique_ptr<Task>>> tasks;
};

} // namespace

std::unique_ptr<StreamExecutor> StreamExecutor::pool() {
    return std::make_unique<PoolExecutor>();
}

void StreamExecutor::run(std::function<void()> p_task) {
    const std::lock_guard<std::mutex> lock(queue_mutex);
    queued.push_back(std::move(p_task));
}

void StreamExecutor::reap(bool /* p_wait */) {
    for (;;) {
        std::vector<std::function<void()>> now;
        {
            const std::lock_guard<std::mutex> lock(queue_mutex);
            now.swap(queued);
        }
        if (now.empty()) {
            return;
        }
        for (const std::function<void()> &task : now) {
            task();
        }
    }
}

size_t InstanceStream::KeyHash::operator()(const SourceKey &p_k) const {
    return ExactNoise::hash_cell(p_k.tile, ((p_k.source * 2u + uint32_t(p_k.kind)) * 64u + uint32_t(p_k.lod) + 1u) * 0x9E3779B1u);
}

void InstanceStream::Job::run() {
    const uint64_t t0 = now_usec();
    ScatterSource::Inputs raw;
    for (const std::shared_ptr<const SourceContent> &in : inputs) {
        raw.push_back(in.get());
    }
    out = std::make_shared<SourceContent>();
    source->content(key, raw, *out);
    usec = now_usec() - t0;
}

InstanceStream::InstanceStream(std::vector<std::shared_ptr<const ScatterSource>> p_sources, const ChunkParam &p_param, const Settings &p_settings,
        std::unique_ptr<StreamExecutor> p_executor) :
        sources(std::move(p_sources)), param(p_param), settings(p_settings), executor(std::move(p_executor)) {
    for (size_t i = 0; i < sources.size(); i++) {
        ERR_FAIL_COND_MSG(sources[i]->setup().id != i, "InstanceStream: source ids must be their index.");
    }
}

InstanceStream::~InstanceStream() {
    {
        const std::lock_guard<std::mutex> lock(job_mutex);
        jobs.clear(); // the running drain tasks find nothing more
    }
    executor->reap(true);
}

uint32_t InstanceStream::slot_budget(const std::vector<std::shared_ptr<const ScatterSource>> &p_sources) {
    double records = 0.0;
    for (const std::shared_ptr<const ScatterSource> &s : p_sources) {
        const CellScatter &c = s->setup().scatter;
        if (!c.valid()) {
            continue;
        }
        // A surface crosses the zone: about one site per cell of its widest face.
        const VoxelPos e = c.zone.hi - c.zone.lo + VoxelPos(1, 1, 1);
        const double face = std::max({ double(e.x) * double(e.y), double(e.y) * double(e.z), double(e.z) * double(e.x) });
        records += face / (double(c.side) * double(c.side)) * s->profile().records_per_site;
    }
    return uint32_t(std::clamp(2.0 * records, double(SLOTS_MIN), double(SLOTS_MAX)));
}

void InstanceStream::chunk_keys(const ChunkId &p_chunk, std::vector<SourceKey> &r_out) const {
    r_out.clear();
    thread_local std::vector<SourceKey> one;
    for (const std::shared_ptr<const ScatterSource> &s : sources) {
        s->chunk_keys(p_chunk, param, one);
        r_out.insert(r_out.end(), one.begin(), one.end());
    }
}

InstanceStream::Admit InstanceStream::admit(const ChunkId &p_chunk, uint64_t p_owner, uint64_t p_tag) {
    const uint64_t t0 = now_usec();
    const std::unique_lock<std::shared_mutex> lock(mutex);
    stats.admit_wait_usec += now_usec() - t0;
    Admit answer = READY;
    const ParkKey park{ p_chunk, p_owner };
    const auto it = parked.find(park);
    if (it != parked.end()) {
        it->second.tag = p_tag;
        answer = PARKED;
    } else {
        thread_local std::vector<SourceKey> keys;
        chunk_keys(p_chunk, keys);
        int missing = 0;
        for (const SourceKey &k : keys) {
            const Entry &e = request(k, true);
            if (!e.placed && e.state == Entry::READY) {
                answer = LATER; // no slot for it (the keys asked for so far stay in their grace)
                break;
            }
            missing += !e.placed;
        }
        if (answer == READY && missing > 0) {
            // Every key asked for this frame is in its grace or not ready: none went meanwhile.
            Parked &p = parked[park];
            p.tag = p_tag;
            p.seq = ++park_seq;
            p.missing = missing;
            p.keys = keys;
            for (const SourceKey &k : keys) {
                Entry &e = entries.find(k)->second;
                pin(e);
                if (!e.placed) {
                    e.waiters.push_back({ park, p.seq }); // place() counts it down
                }
            }
            answer = PARKED;
        }
    }
    stats.admitted[answer]++;
    stats.admit_usec += now_usec() - t0;
    return answer;
}

void InstanceStream::take_ready(uint64_t p_owner, std::vector<std::pair<ChunkId, uint64_t>> &r_out) {
    const std::unique_lock<std::shared_mutex> lock(mutex);
    r_out.clear();
    // p_owner's first, in park order; the other owners' stay.
    const auto others = std::stable_partition(ready_chunks.begin(), ready_chunks.end(), [p_owner](const ParkKey &p_k) { return p_k.owner == p_owner; });
    std::sort(ready_chunks.begin(), others, [this](const ParkKey &p_a, const ParkKey &p_b) { return parked[p_a].seq < parked[p_b].seq; });
    for (auto c = ready_chunks.begin(); c != others; ++c) {
        const auto it = parked.find(*c);
        for (const SourceKey &k : it->second.keys) {
            unpin(entries.find(k)->second); // pinned: still there
        }
        r_out.push_back({ c->chunk, it->second.tag });
        parked.erase(it);
    }
    ready_chunks.erase(ready_chunks.begin(), others);
}

size_t InstanceStream::parked_count(uint64_t p_owner) const {
    const std::shared_lock<std::shared_mutex> lock(mutex);
    size_t n = 0;
    for (const std::pair<const ParkKey, Parked> &p : parked) {
        n += p.first.owner == p_owner;
    }
    return n;
}

void InstanceStream::hand_back(const ParkKey &p_park) {
    const auto it = parked.find(p_park);
    if (it != parked.end() && !it->second.done) {
        it->second.done = true;
        ready_chunks.push_back(p_park);
    }
}

InstanceStream::Entry &InstanceStream::request(const SourceKey &p_key, bool p_listed) {
    const auto [it, fresh] = entries.try_emplace(p_key);
    Entry &e = it->second;
    if (fresh) {
        e.key = p_key;
        thread_local std::vector<SourceKey> in;
        source(p_key).inputs(p_key, in);
        const std::vector<SourceKey> inputs = in; // the requests below reuse the thread_local
        e.inputs.resize(inputs.size());
        for (uint32_t i = 0; i < inputs.size(); i++) {
            Entry &input = request(inputs[i], false);
            if (input.state == Entry::READY) {
                e.inputs[i] = input.content;
            } else {
                e.missing++;
                input.dependents.push_back({ p_key, i }); // land() hands it over
            }
        }
        if (e.missing == 0) {
            start(e);
        }
    }
    touch(e);
    e.listed = e.listed || p_listed;
    if (e.listed && e.state == Entry::READY && !e.placed) {
        place(e);
    }
    return e;
}

void InstanceStream::start(Entry &r_entry) {
    auto job = std::make_unique<Job>();
    job->source = &source(r_entry.key);
    job->key = r_entry.key;
    job->inputs = std::move(r_entry.inputs);
    r_entry.inputs.clear();
    r_entry.state = Entry::RUNNING;
    bool spawn = false;
    {
        const std::lock_guard<std::mutex> lock(job_mutex);
        jobs.push_back(std::move(job));
        spawn = draining < executor->workers();
        draining += spawn;
    }
    if (spawn) {
        executor->run([this] { drain(); });
    }
}

void InstanceStream::drain() {
    const InlineParallelFor inline_loops; // a pool task: the site walks' and layers' loops run here
    for (;;) {
        std::unique_ptr<Job> job;
        {
            const std::lock_guard<std::mutex> lock(job_mutex);
            if (jobs.empty()) {
                draining--;
                return;
            }
            job = std::move(jobs.front());
            jobs.pop_front();
        }
        job->run();
        const std::unique_lock<std::shared_mutex> lock(mutex);
        land(std::move(job)); // the keys it completes join jobs: this loop takes them
    }
}

void InstanceStream::poll() {
    const uint64_t t0 = now_usec();
    executor->reap(false); // before the lock: the tests' executor runs its drain tasks here
    const std::unique_lock<std::shared_mutex> lock(mutex);
    frame++;
    for (const SourceKey &k : viewed) {
        request(k, false); // kept resident while viewed, started if missing
    }
    viewed.clear();
    while ((entries.size() > settings.entries || input_records > settings.input_records) && evict_oldest()) {
    }
    const uint64_t dt = now_usec() - t0;
    stats.poll_usec += dt;
    stats.poll_usec_max = MAX(stats.poll_usec_max, dt);
}

void InstanceStream::land(std::unique_ptr<Job> p_job) {
    Entry &e = entries.find(p_job->key)->second; // running: never evicted
    const int lod = CLAMP(p_job->key.lod, 0, ChunkMath::MAX_LOD_NUMBERS - 1);
    stats.computed++;
    stats.compute_usec += int64_t(p_job->usec);
    stats.computed_by[p_job->key.kind][lod]++;
    stats.usec_by[p_job->key.kind][lod] += int64_t(p_job->usec);
    e.content = std::move(p_job->out);
    e.count = uint32_t(e.content->records.size());
    e.state = Entry::READY;
    input_records += e.count;
    std::vector<std::pair<SourceKey, uint32_t>> dependents;
    dependents.swap(e.dependents);
    for (const std::pair<SourceKey, uint32_t> &d : dependents) {
        Entry &dep = entries.find(d.first)->second; // waiting: never evicted
        dep.inputs[d.second] = e.content;
        if (--dep.missing == 0) {
            start(dep);
        }
    }
    if (e.listed) {
        place(e); // before it is on the LRU list: making room never evicts it
    }
    if (e.pins == 0) {
        e.used = MAX(e.used, frame);
        lru_link(e);
    }
}

bool InstanceStream::place(Entry &r_entry) {
    uint32_t slot = 0;
    if (r_entry.count > 0) {
        bool room = allocate(r_entry.count, slot);
        while (!room && evict_oldest()) {
            room = allocate(r_entry.count, slot);
        }
        if (!room) {
            if (stats.slots_full++ == 0) {
                ERR_PRINT(vformat("InstanceStream: %d records don't fit the %d slots, even past the keys asked for in the last %d frames: the "
                                  "chunks needing them wait (InstanceStream::slot_budget).",
                        r_entry.count, settings.slot_capacity, GRACE_FRAMES));
            }
            // Its parked chunks go back to their mesher (their pins released, room comes back): asked again, LATER.
            for (const std::pair<ParkKey, uint64_t> &w : r_entry.waiters) {
                const auto it = parked.find(w.first);
                if (it != parked.end() && it->second.seq == w.second) {
                    hand_back(w.first);
                }
            }
            r_entry.waiters.clear();
            return false;
        }
        r_entry.placement = ++placement_seq;
        log.push_back({ r_entry.placement, r_entry.key });
        if (log.size() > settings.log) {
            log.pop_front();
        }
    }
    r_entry.slot = slot;
    r_entry.placed = true;
    input_records -= r_entry.count;
    stats.placed++;
    stats.slots_peak = MAX(stats.slots_peak, slots_used());
    for (const std::pair<ParkKey, uint64_t> &w : r_entry.waiters) {
        const auto it = parked.find(w.first);
        if (it != parked.end() && it->second.seq == w.second && --it->second.missing == 0) {
            hand_back(w.first);
        }
    }
    r_entry.waiters.clear();
    return true;
}

bool InstanceStream::allocate(uint32_t p_count, uint32_t &r_slot) {
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
    if (uint64_t(slots_end) + p_count <= settings.slot_capacity) {
        r_slot = slots_end;
        slots_end += p_count;
        return true;
    }
    return false;
}

void InstanceStream::release(uint32_t p_slot, uint32_t p_count) {
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
            it = prev;
        }
    }
    if (it->first + it->second == slots_end) { // back to the tail: one run with the slots never used
        slots_end = it->first;
        free_slots -= it->second;
        free_runs.erase(it);
    }
}

bool InstanceStream::evict_oldest() {
    Entry *e = oldest;
    if (e == nullptr || e->used + GRACE_FRAMES >= frame) {
        return false; // the rest is newer
    }
    lru_unlink(*e);
    if (e->placed && e->count > 0) {
        release(e->slot, e->count);
    }
    if (!e->placed) {
        input_records -= e->count;
    }
    stats.evicted++;
    const SourceKey key = e->key; // asked for again, it is computed again (the same records)
    entries.erase(key);
    return true;
}

void InstanceStream::touch(Entry &r_entry) {
    r_entry.used = MAX(r_entry.used, frame);
    if (r_entry.in_lru && newest != &r_entry) {
        lru_unlink(r_entry);
        lru_link(r_entry);
    }
}

void InstanceStream::pin(Entry &r_entry) {
    if (r_entry.pins++ == 0 && r_entry.in_lru) {
        lru_unlink(r_entry);
    }
}

void InstanceStream::unpin(Entry &r_entry) {
    if (--r_entry.pins == 0 && r_entry.state == Entry::READY) {
        r_entry.used = MAX(r_entry.used, frame); // the chunk's batch reads it: its grace starts now
        lru_link(r_entry);
    }
}

void InstanceStream::lru_link(Entry &r_entry) {
    r_entry.older = newest;
    r_entry.newer = nullptr;
    if (newest != nullptr) {
        newest->newer = &r_entry;
    } else {
        oldest = &r_entry;
    }
    newest = &r_entry;
    r_entry.in_lru = true;
}

void InstanceStream::lru_unlink(Entry &r_entry) {
    (r_entry.older != nullptr ? r_entry.older->newer : oldest) = r_entry.newer;
    (r_entry.newer != nullptr ? r_entry.newer->older : newest) = r_entry.older;
    r_entry.older = r_entry.newer = nullptr;
    r_entry.in_lru = false;
}

void InstanceStream::chunk_list(const ChunkId &p_chunk, std::vector<uint32_t> &r_out, std::vector<uint32_t> *r_materials) const {
    thread_local std::vector<SourceKey> keys;
    chunk_keys(p_chunk, keys);
    VoxelPos lo, hi;
    chunk_sample_box(p_chunk, param, lo, hi);
    const int lod = ChunkMath::get_lod(p_chunk);
    const std::shared_lock<std::shared_mutex> lock(mutex); // drain tasks land meanwhile
    for (const SourceKey &k : keys) {
        const auto it = entries.find(k);
        if (it == entries.end() || !it->second.placed) {
            lists_missing++;
            continue;
        }
        const size_t from = r_out.size();
        it->second.content->meeting(lo, hi, lod, r_out);
        for (size_t i = from; i < r_out.size(); i++) {
            if (r_materials) {
                r_materials->push_back(it->second.content->records[r_out[i]].material);
            }
            r_out[i] += settings.slot_base + it->second.slot;
        }
    }
}

void InstanceStream::view_records(const VoxelPos &p_voxel, int p_lod, std::vector<TableRecord> &r_out) {
    std::vector<SourceKey> keys;
    std::vector<uint32_t> hit;
    for (const std::shared_ptr<const ScatterSource> &s : sources) {
        s->keys(p_voxel, p_voxel, p_lod, keys);
        // Shared: the drain tasks land under the lock, a writer here would queue behind them. poll() asks for the keys.
        const std::shared_lock<std::shared_mutex> lock(mutex);
        for (const SourceKey &k : keys) {
            viewed.push_back(k);
            const auto it = entries.find(k);
            if (it == entries.end() || it->second.state != Entry::READY) {
                view_missing++;
                continue;
            }
            hit.clear();
            it->second.content->meeting(p_voxel, p_voxel, p_lod, hit);
            for (const uint32_t i : hit) {
                r_out.push_back(it->second.content->records[i]);
            }
        }
    }
}

void InstanceStream::point_records(const VoxelPos &p_voxel, int p_lod, std::vector<TableRecord> &r_out) const {
    std::vector<SourceKey> keys, in;
    std::vector<uint32_t> hit;
    for (const std::shared_ptr<const ScatterSource> &s : sources) {
        s->keys(p_voxel, p_voxel, p_lod, keys);
        for (const SourceKey &k : keys) {
            std::shared_ptr<const SourceContent> content;
            std::vector<std::shared_ptr<const SourceContent>> inputs; // the resident ones (the others computed with it)
            {
                const std::shared_lock<std::shared_mutex> lock(mutex);
                const auto it = entries.find(k);
                if (it != entries.end() && it->second.state == Entry::READY) {
                    content = it->second.content;
                } else {
                    s->inputs(k, in);
                    for (const SourceKey &i : in) {
                        const auto input = entries.find(i);
                        inputs.push_back(input != entries.end() && input->second.state == Entry::READY ? input->second.content : nullptr);
                    }
                }
            }
            if (!content) {
                const InlineParallelFor inline_loops; // maybe a pool task (the discoverer's batches)
                ScatterSource::Inputs raw;
                for (const std::shared_ptr<const SourceContent> &i : inputs) {
                    raw.push_back(i.get());
                }
                auto computed = std::make_shared<SourceContent>();
                s->content(k, raw, *computed);
                content = std::move(computed);
                point_computed++;
            }
            hit.clear();
            content->meeting(p_voxel, p_voxel, p_lod, hit);
            for (const uint32_t i : hit) {
                r_out.push_back(content->records[i]);
            }
        }
    }
}

uint64_t InstanceStream::placements() const {
    const std::shared_lock<std::shared_mutex> lock(mutex);
    return placement_seq;
}

void InstanceStream::uploads_since(uint64_t p_seq, std::vector<Upload> &r_out) const {
    const std::shared_lock<std::shared_mutex> lock(mutex);
    r_out.clear();
    if (p_seq >= placement_seq) {
        return;
    }
    const auto upload = [&](const Entry &p_e) { r_out.push_back({ settings.slot_base + p_e.slot, p_e.content }); };
    if (log.empty() || p_seq + 1 < log.front().first) {
        // Past the log: every key holding slots, in placement order.
        std::vector<const Entry *> placed;
        for (const std::pair<const SourceKey, Entry> &kv : entries) {
            if (kv.second.placed && kv.second.count > 0) {
                placed.push_back(&kv.second);
            }
        }
        std::sort(placed.begin(), placed.end(), [](const Entry *p_a, const Entry *p_b) { return p_a->placement < p_b->placement; });
        for (const Entry *e : placed) {
            upload(*e);
        }
        return;
    }
    const auto from = std::upper_bound(log.begin(), log.end(), p_seq, [](uint64_t p_s, const std::pair<uint64_t, SourceKey> &p_l) { return p_s < p_l.first; });
    for (auto l = from; l != log.end(); ++l) {
        const auto it = entries.find(l->second);
        // Evicted since (its slots free, or a later placement's), or placed again later: nothing to send for this one.
        if (it != entries.end() && it->second.placed && it->second.placement == l->first) {
            upload(it->second);
        }
    }
}

Dictionary InstanceStream::get_stats() const {
    const std::shared_lock<std::shared_mutex> lock(mutex);
    int64_t placed = 0, ready = 0, running = 0, waiting = 0;
    for (const std::pair<const SourceKey, Entry> &kv : entries) {
        placed += kv.second.placed;
        ready += kv.second.state == Entry::READY;
        running += kv.second.state == Entry::RUNNING;
        waiting += kv.second.state == Entry::WAITING;
    }
    Dictionary d;
    d["keys"] = int64_t(entries.size());
    d["keys_ready"] = ready;
    d["keys_placed"] = placed;
    d["keys_running"] = running;
    d["keys_waiting"] = waiting;
    d["chunks_parked"] = int64_t(parked.size());
    d["admit_ready"] = stats.admitted[READY];
    d["admit_parked"] = stats.admitted[PARKED];
    d["admit_later"] = stats.admitted[LATER];
    d["admit_ms_total"] = double(stats.admit_usec) / 1000.0;
    d["admit_wait_ms_total"] = double(stats.admit_wait_usec) / 1000.0;
    d["poll_ms_total"] = double(stats.poll_usec) / 1000.0;
    d["poll_ms_max"] = double(stats.poll_usec_max) / 1000.0;
    d["computed"] = stats.computed;
    d["compute_ms_total"] = double(stats.compute_usec) / 1000.0;
    Dictionary by;
    for (int kind = 0; kind < 2; kind++) {
        for (int lod = 0; lod < ChunkMath::MAX_LOD_NUMBERS; lod++) {
            const int64_t n = stats.computed_by[kind][lod];
            if (n > 0) {
                Dictionary k;
                k["count"] = n;
                k["ms_avg"] = double(stats.usec_by[kind][lod]) / 1000.0 / double(n);
                by[vformat("%s %d", kind == SourceKey::PARTS ? "parts" : "blobs", lod)] = k;
            }
        }
    }
    d["computed_by_key"] = by;
    d["placed"] = stats.placed;
    d["evicted"] = stats.evicted;
    d["slots_used"] = slots_used();
    d["slots_peak"] = stats.slots_peak;
    d["slots_capacity"] = int64_t(settings.slot_capacity);
    d["slots_full"] = stats.slots_full;
    d["input_records"] = input_records;
    d["lists_missing"] = lists_missing.load();
    d["point_computed"] = point_computed.load();
    d["view_missing"] = view_missing;
    return d;
}

} // namespace Archetype
