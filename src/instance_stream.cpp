// The streamed sources' residency (instance_stream.hpp).
#include "instance_stream.hpp"

#include "debug_log.hpp"
#include "parallel_for.hpp"

#include <godot_cpp/core/error_macros.hpp>

#include <algorithm>

namespace Archetype {

size_t InstanceStream::KeyHash::operator()(const SourceKey &p_k) const {
    return ExactNoise::hash_cell(p_k.tile, ((p_k.source * uint32_t(SourceKey::KIND_COUNT) + uint32_t(p_k.kind)) * 64u + uint32_t(p_k.lod) + 1u) * 0x9E3779B1u);
}

void InstanceStream::Job::run() {
    const uint64_t t0 = now_usec();
    if (expand) {
        source->inputs(key, input_keys);
        if (!input_keys.empty()) {
            usec = now_usec() - t0;
            return; // land() asks for them, then the key runs again with them
        }
    }
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
    stop();
    executor->reap(true);
}

void InstanceStream::stop() {
    stopping = true; // a job computing lands nothing (no expansion, no dependent started)
    const std::lock_guard<std::mutex> lock(job_mutex);
    jobs.clear(); // the running drain tasks find nothing more
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

InstanceStream::Admit InstanceStream::admit(const ChunkId &p_chunk, uint64_t p_owner, uint64_t p_tag, uint64_t p_rank, uint64_t *r_words) {
    const uint64_t t0 = now_usec();
    const std::unique_lock<std::shared_mutex> lock(mutex);
    stats.admit_wait_usec += now_usec() - t0;
    Admit answer = READY;
    const int64_t created_before = stats.keys_created;
    const ParkKey park{ p_chunk, p_owner };
    const auto it = parked.find(park);
    if (it != parked.end()) {
        Parked &p = it->second;
        p.tag = p_tag;
        if (p_rank < p.rank) {
            p.rank = p_rank;
            for (const SourceKey &k : p.keys) {
                improve(entries.find(k)->second, p_rank); // pinned: still there
            }
        }
        answer = PARKED;
    } else {
        thread_local std::vector<SourceKey> keys;
        chunk_keys(p_chunk, keys);
        int missing = 0;
        uint64_t words = 0;
        for (const SourceKey &k : keys) {
            if (pending_asked >= settings.pending && entries.find(k) == entries.end()) {
                answer = LATER; // its keys asked for so far stay, computed meanwhile
                stats.later_pending++;
                break;
            }
            const Entry &e = request(k, true, p_rank);
            if (!e.placed && e.state == Entry::READY) {
                answer = LATER; // no slot for it (the keys asked for so far stay in their grace)
                break;
            }
            missing += !e.placed;
            words += e.count;
        }
        if (answer == READY && missing == 0 && r_words) {
            *r_words = words;
        }
        if (answer == READY && missing > 0) {
            // Every key asked for this frame is in its grace or not ready: none went meanwhile.
            Parked &p = parked[park];
            p.tag = p_tag;
            p.rank = p_rank;
            p.seq = ++park_seq;
            p.frame = frame;
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
    const int64_t created = stats.keys_created - created_before;
    stats.admit_keys_created += created;
    stats.admit_keys_created_max = MAX(stats.admit_keys_created_max, created);
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
        release_pins(it->second);
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

void InstanceStream::set_tag(uint64_t p_owner, uint64_t p_tag, const std::function<int64_t(const ChunkId &)> &p_rank,
        const std::function<void(const ChunkId &)> &p_kept) {
    const std::unique_lock<std::shared_mutex> lock(mutex);
    std::vector<SourceKey> released;
    for (auto it = parked.begin(); it != parked.end();) {
        Parked &p = it->second;
        if (it->first.owner != p_owner || p.tag >= p_tag) {
            ++it;
            continue;
        }
        const int64_t rank = p_rank(it->first.chunk);
        if (rank >= 0) {
            p.tag = p_tag;
            // A better rank moves its queued jobs ahead; a worse one leaves them where they are (the order is only a
            // priority: a job runs early, never wrong), and the next better ask compares against it.
            if (uint64_t(rank) < p.rank) {
                for (const SourceKey &k : p.keys) {
                    improve(entries.find(k)->second, uint64_t(rank)); // pinned: still there
                }
            }
            p.rank = uint64_t(rank);
            p_kept(it->first.chunk);
            ++it;
        } else {
            released.insert(released.end(), p.keys.begin(), p.keys.end());
            release_pins(p);
            stats.stale_handed_back++;
            it = parked.erase(it); // its waiters' seq finds nothing
        }
    }
    // take_ready() reads parked for each of them.
    ready_chunks.erase(std::remove_if(ready_chunks.begin(), ready_chunks.end(), [this](const ParkKey &p_k) { return parked.find(p_k) == parked.end(); }),
            ready_chunks.end());
    for (const SourceKey &k : released) {
        const auto it = entries.find(k);
        if (it != entries.end()) {
            drop_unneeded(it->second);
        }
    }
}

void InstanceStream::release_pins(Parked &r_parked) {
    for (const SourceKey &k : r_parked.keys) {
        unpin(entries.find(k)->second); // pinned: still there
    }
    r_parked.keys.clear();
}

void InstanceStream::hand_back(const ParkKey &p_park) {
    const auto it = parked.find(p_park);
    if (it != parked.end() && !it->second.done) {
        it->second.done = true;
        ready_chunks.push_back(p_park);
        stats.park_frames.add(frame - it->second.frame);
    }
}

InstanceStream::Entry &InstanceStream::request(const SourceKey &p_key, bool p_listed, uint64_t p_rank, std::vector<Entry *> *r_input_jobs) {
    const auto [it, fresh] = entries.try_emplace(p_key);
    Entry &e = it->second;
    if (fresh) {
        pending++;
        stats.keys_created++;
        e.key = p_key;
        e.rank = p_rank;
        if (r_input_jobs) {
            r_input_jobs->push_back(&e); // the expansion queues them together
        } else {
            start(e); // its job lists its inputs first, on a drain task (expand())
        }
    } else {
        e.orphan = false;
        improve(e, p_rank);
    }
    if (!r_input_jobs && !e.asked && e.state != Entry::READY) {
        e.asked = true; // asked for by a chunk or the camera: counts against the pending cap
        pending_asked++;
    }
    touch(e);
    e.listed = e.listed || p_listed;
    if (e.listed && e.state == Entry::READY && !e.placed) {
        place(e);
    }
    return e;
}

void InstanceStream::expand(Entry &r_entry, std::vector<SourceKey> &&p_inputs) {
    r_entry.expanded = true;
    r_entry.state = Entry::WAITING;
    r_entry.input_keys = std::move(p_inputs);
    r_entry.inputs.assign(r_entry.input_keys.size(), nullptr);
    r_entry.missing = 0;
    stats.keys_expanded++;
    thread_local std::vector<Entry *> fresh; // the new inputs: their jobs queued at once (one job_mutex take, one wake)
    fresh.clear();
    for (uint32_t i = 0; i < r_entry.input_keys.size(); i++) {
        Entry &input = request(r_entry.input_keys[i], false, r_entry.rank, &fresh); // try_emplace: r_entry stays valid
        if (input.state == Entry::READY) {
            r_entry.inputs[i] = input.content;
        } else {
            r_entry.missing++;
            input.dependents.push_back({ r_entry.key, i }); // land() hands it over
        }
    }
    if (r_entry.missing == 0) {
        fresh.push_back(&r_entry);
    }
    start(fresh.data(), fresh.size()); // none evicted meanwhile: not READY
}

void InstanceStream::start(Entry &r_entry) {
    Entry *one = &r_entry;
    start(&one, 1);
}

void InstanceStream::start(Entry *const *p_entries, size_t p_count) {
    if (p_count == 0) {
        return;
    }
    // The jobs are made before job_mutex (the drain tasks pop under it), then merged in one pass (no allocation there).
    std::map<JobKey, std::unique_ptr<Job>> made;
    for (size_t i = 0; i < p_count; i++) {
        Entry &e = *p_entries[i];
        auto job = std::make_unique<Job>();
        job->source = &source(e.key);
        job->key = e.key;
        job->expand = !e.expanded;
        job->inputs = std::move(e.inputs);
        e.inputs.clear();
        e.input_keys = std::vector<SourceKey>(); // landed: no rank or drop walks through them any more
        e.state = Entry::RUNNING;
        e.job_seq = ++job_seq;
        made.emplace_hint(made.end(), JobKey(e.rank, e.job_seq), std::move(job));
    }
    bool spawn = false;
    {
        const std::lock_guard<std::mutex> lock(job_mutex);
        jobs.merge(made);
        stats.jobs_queued_max = MAX(stats.jobs_queued_max, int64_t(jobs.size())); // under the stream's lock too
        spawn = claim_drain();
    }
    if (spawn) {
        executor->run([this] { drain(); });
    }
}

bool InstanceStream::claim_drain() {
    const bool spawn = !jobs.empty() && draining < executor->workers();
    draining += spawn;
    return spawn;
}

void InstanceStream::wake() {
    bool spawn = false;
    {
        const std::lock_guard<std::mutex> lock(job_mutex);
        spawn = claim_drain();
    }
    if (spawn) {
        executor->run([this] { drain(); });
    }
}

void InstanceStream::improve(Entry &r_entry, uint64_t p_rank) {
    if (r_entry.state == Entry::READY || p_rank >= r_entry.rank) {
        return;
    }
    const uint64_t old = r_entry.rank;
    r_entry.rank = p_rank;
    if (r_entry.state == Entry::WAITING) {
        for (const SourceKey &k : r_entry.input_keys) {
            const auto it = entries.find(k);
            if (it != entries.end()) { // else landed, handed over and evicted since
                improve(it->second, p_rank);
            }
        }
        return;
    }
    const std::lock_guard<std::mutex> lock(job_mutex);
    auto node = jobs.extract(JobKey(old, r_entry.job_seq));
    if (node.empty()) {
        return; // a drain task took it: it is computing
    }
    node.key() = JobKey(p_rank, r_entry.job_seq);
    jobs.insert(std::move(node));
    stats.jobs_reranked++;
}

bool InstanceStream::drop_unneeded(Entry &r_entry) {
    if (r_entry.state == Entry::READY || r_entry.pins > 0 || !r_entry.dependents.empty()) {
        return false;
    }
    // Asked for this frame (the keys of an admit answered LATER, the camera's queries): kept, computed meanwhile. Not an
    // orphan landing: dropped since it was asked (asked again, it is no orphan).
    const bool asked_now = r_entry.used == frame && !r_entry.orphan;
    if (r_entry.state == Entry::RUNNING) {
        const std::lock_guard<std::mutex> lock(job_mutex);
        const auto job = jobs.find(JobKey(r_entry.rank, r_entry.job_seq));
        if (job == jobs.end()) {
            r_entry.orphan = true; // a drain task took it: it lands into its grace (listing its inputs: dropped then)
            return false;
        }
        if (asked_now) {
            return false;
        }
        jobs.erase(job);
        stats.jobs_dropped++;
    } else if (asked_now) {
        return false;
    }
    const SourceKey key = r_entry.key;
    const std::vector<SourceKey> inputs = std::move(r_entry.input_keys); // WAITING: its inputs, landed or not
    pending_asked -= r_entry.asked;
    entries.erase(key); // never on an LRU list (not READY)
    pending--;
    stats.keys_dropped++;
    for (uint32_t i = 0; i < inputs.size(); i++) {
        const auto it = entries.find(inputs[i]);
        if (it == entries.end() || it->second.state == Entry::READY) {
            continue; // landed: handed over already
        }
        std::vector<std::pair<SourceKey, uint32_t>> &deps = it->second.dependents;
        deps.erase(std::remove(deps.begin(), deps.end(), std::make_pair(key, i)), deps.end());
        drop_unneeded(it->second);
    }
    return true;
}

void InstanceStream::drain() {
    const InlineParallelFor inline_loops; // a pool task: the site walks' and layers' loops run here
    std::vector<std::unique_ptr<Job>> batch;
    for (;;) {
        batch.clear();
        {
            const std::lock_guard<std::mutex> lock(job_mutex);
            if (jobs.empty() || stopping || !executor->more()) {
                draining--;
                return;
            }
            batch.push_back(std::move(jobs.begin()->second));
            jobs.erase(jobs.begin());
            if (settings.walk_batch > 1 && batch[0]->key.walk()) {
                take_walks(batch);
            }
        }
        if (batch.size() == 1) {
            batch[0]->run();
        } else {
            run_walks(batch);
        }
        for (size_t i = 0; i < batch.size(); i++) {
            executor->landing();
        }
        if (stopping) {
            continue; // nobody reads them: no landing (an expansion would queue its subtree)
        }
        const std::unique_lock<std::shared_mutex> lock(mutex);
        if (batch.size() > 1) {
            stats.walk_batches++;
            stats.walk_batch_keys += int64_t(batch.size());
        }
        for (std::unique_ptr<Job> &job : batch) {
            land(std::move(job)); // the keys it completes join jobs: this loop takes them
        }
    }
}

void InstanceStream::take_walks(std::vector<std::unique_ptr<Job>> &r_batch) {
    // The queue's next walk jobs of the same source, in rank order, looking a few batches deep (the others stay for the
    // other tasks). A walk key has no input: its first job computes it.
    const ScatterSource *source = r_batch[0]->source;
    size_t looked = 0;
    for (auto it = jobs.begin(); it != jobs.end() && r_batch.size() < settings.walk_batch && looked < WALK_LOOK * settings.walk_batch; looked++) {
        if (it->second->key.walk() && it->second->source == source) {
            r_batch.push_back(std::move(it->second));
            it = jobs.erase(it);
        } else {
            ++it;
        }
    }
}

void InstanceStream::run_walks(std::vector<std::unique_ptr<Job>> &r_batch) {
    const uint64_t t0 = now_usec();
    std::vector<SourceKey> keys;
    std::vector<SourceContent *> outs;
    for (std::unique_ptr<Job> &job : r_batch) {
        job->out = std::make_shared<SourceContent>();
        keys.push_back(job->key);
        outs.push_back(job->out.get());
    }
    r_batch[0]->source->walk_tiles(keys.data(), keys.size(), outs.data());
    const uint64_t each = (now_usec() - t0) / r_batch.size();
    for (std::unique_ptr<Job> &job : r_batch) {
        job->usec = each; // their compute time shared
    }
}

void InstanceStream::poll() {
    const uint64_t t0 = now_usec();
    executor->reap(false); // before the lock: the tests' executor runs its drain tasks here
    const std::unique_lock<std::shared_mutex> lock(mutex);
    frame++;
    for (const SourceKey &k : viewed) {
        request(k, false, 0); // kept resident while viewed, started if missing, first
    }
    viewed.clear();
    const uint64_t t_evict = now_usec();
    for (;;) {
        const bool too_many = int64_t(entries.size()) - int64_t(pending) - int64_t(walks_ready) > int64_t(settings.entries);
        if (!(too_many || input_records > settings.input_records) || !evict_oldest(too_many ? chain_all : chain_inputs)) {
            break;
        }
    }
    while (walks_ready > settings.walks && evict_oldest(chain_walks)) {
    }
    wake(); // the jobs a drain task left (StreamExecutor::more())
    const uint64_t t_end = now_usec();
    stats.evict_usec += t_end - t_evict;
    const uint64_t dt = t_end - t0;
    stats.poll_usec += dt;
    stats.poll_usec_max = MAX(stats.poll_usec_max, dt);
}

void InstanceStream::land(std::unique_ptr<Job> p_job) {
    const uint64_t t0 = now_usec();
    Entry &e = entries.find(p_job->key)->second; // running: never evicted
    if (p_job->expand && !p_job->input_keys.empty()) {
        stats.compute_usec += int64_t(p_job->usec); // the workers' time
        stats.expand_usec += int64_t(p_job->usec); // of which listing inputs
        // Dropped while it listed them (an orphan): erased if still unneeded, no input asked for.
        if (e.orphan) {
            e.state = Entry::WAITING;
        }
        if (!e.orphan || !drop_unneeded(e)) {
            e.orphan = false; // kept
            expand(e, std::move(p_job->input_keys));
        }
        stats.land_usec += now_usec() - t0;
        return;
    }
    const int lod = CLAMP(p_job->key.lod, 0, ChunkMath::MAX_LOD_NUMBERS - 1);
    stats.computed++;
    stats.compute_usec += int64_t(p_job->usec);
    stats.computed_by[p_job->key.kind][lod]++;
    stats.usec_by[p_job->key.kind][lod] += int64_t(p_job->usec);
    e.content = std::move(p_job->out);
    e.count = uint32_t(e.content->records.size());
    e.state = Entry::READY;
    e.orphan = false; // lands into its grace
    pending--;
    pending_asked -= e.asked;
    e.asked = false;
    walks_ready += e.key.walk();
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
    stats.land_usec += now_usec() - t0;
}

bool InstanceStream::place(Entry &r_entry) {
    uint32_t slot = 0;
    if (r_entry.count > 0) {
        bool room = allocate(r_entry.count, slot);
        while (!room && evict_oldest(chain_all)) {
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
    if (r_entry.lru_inputs.in) {
        chain_inputs.unlink(r_entry); // its records are in slots now
    }
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

bool InstanceStream::evict_oldest(const Chain &p_chain) {
    Entry *e = p_chain.oldest;
    if (e == nullptr || e->used + GRACE_FRAMES >= frame) {
        return false; // the rest is newer
    }
    lru_unlink(*e);
    walks_ready -= e->key.walk();
    if (e->placed && e->count > 0) {
        release(e->slot, e->count);
    }
    if (!e->placed) {
        input_records -= e->count;
    }
    stats.evicted++;
    stats.evicted_by[e->key.kind][CLAMP(e->key.lod, 0, ChunkMath::MAX_LOD_NUMBERS - 1)]++;
    const SourceKey key = e->key; // asked for again, it is computed again (the same records)
    entries.erase(key);
    return true;
}

void InstanceStream::touch(Entry &r_entry) {
    r_entry.used = MAX(r_entry.used, frame);
    if (r_entry.lru_all.in && lru_chain(r_entry).newest != &r_entry) {
        lru_unlink(r_entry);
        lru_link(r_entry);
    }
}

void InstanceStream::pin(Entry &r_entry) {
    if (r_entry.pins++ == 0 && r_entry.lru_all.in) {
        lru_unlink(r_entry);
    }
}

void InstanceStream::unpin(Entry &r_entry) {
    if (--r_entry.pins == 0 && r_entry.state == Entry::READY) {
        r_entry.used = MAX(r_entry.used, frame); // the chunk's batch reads it: its grace starts now
        lru_link(r_entry);
    }
}

void InstanceStream::Chain::link(Entry &r_entry) {
    Links &l = r_entry.*links;
    l.older = newest;
    l.newer = nullptr;
    if (newest != nullptr) {
        (newest->*links).newer = &r_entry;
    } else {
        oldest = &r_entry;
    }
    newest = &r_entry;
    l.in = true;
}

void InstanceStream::Chain::unlink(Entry &r_entry) {
    Links &l = r_entry.*links;
    (l.older != nullptr ? (l.older->*links).newer : oldest) = l.newer;
    (l.newer != nullptr ? (l.newer->*links).older : newest) = l.older;
    l = Links();
}

void InstanceStream::lru_link(Entry &r_entry) {
    lru_chain(r_entry).link(r_entry);
    if (!r_entry.placed && r_entry.count > 0) {
        chain_inputs.link(r_entry);
    }
}

void InstanceStream::lru_unlink(Entry &r_entry) {
    if (r_entry.lru_all.in) {
        lru_chain(r_entry).unlink(r_entry);
    }
    if (r_entry.lru_inputs.in) {
        chain_inputs.unlink(r_entry);
    }
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
            {
                const std::shared_lock<std::shared_mutex> lock(mutex);
                const auto it = entries.find(k);
                if (it != entries.end() && it->second.state == Entry::READY) {
                    content = it->second.content;
                }
            }
            if (!content) {
                const InlineParallelFor inline_loops; // maybe a pool task (the discoverer's batches)
                s->inputs(k, in); // a coarse part tile's walks to its site tiles: out of the lock
                std::vector<std::shared_ptr<const SourceContent>> inputs; // the resident ones (the others computed with it)
                {
                    const std::shared_lock<std::shared_mutex> lock(mutex);
                    for (const SourceKey &i : in) {
                        const auto input = entries.find(i);
                        inputs.push_back(input != entries.end() && input->second.state == Entry::READY ? input->second.content : nullptr);
                    }
                }
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
    const auto upload = [&](const Entry &p_e) { r_out.push_back({ settings.slot_base + p_e.slot, p_e.content, p_e.key }); };
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

} // namespace Archetype
