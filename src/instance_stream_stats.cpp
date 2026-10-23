// The stream's counters (instance_stream.hpp).
#include "instance_stream.hpp"

#include "godot_conv.hpp"

namespace Archetype {

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
    d["evict_ms_total"] = double(stats.evict_usec) / 1000.0;
    d["land_ms_total"] = double(stats.land_usec) / 1000.0;
    d["keys_created"] = stats.keys_created;
    d["admit_keys_created"] = stats.admit_keys_created;
    d["admit_keys_created_max"] = stats.admit_keys_created_max;
    d["keys_expanded"] = stats.keys_expanded;
    d["keys_pending_asked"] = int64_t(pending_asked);
    d["keys_pending_inputs"] = int64_t(pending - pending_asked);
    d["admit_later_pending"] = stats.later_pending;
    d["expand_ms_total"] = double(stats.expand_usec) / 1000.0;
    d["jobs_queued_max"] = stats.jobs_queued_max;
    d["jobs_reranked"] = stats.jobs_reranked;
    d["walk_batches"] = stats.walk_batches;
    d["walk_batch_keys"] = stats.walk_batch_keys;
    d["stale_handed_back"] = stats.stale_handed_back;
    d["keys_dropped"] = stats.keys_dropped;
    d["jobs_dropped"] = stats.jobs_dropped;
    d["park_frames"] = godot_dict(stats.park_frames);
    d["workers"] = int64_t(executor->workers());
    d["computed"] = stats.computed;
    d["compute_ms_total"] = double(stats.compute_usec) / 1000.0;
    Dictionary by;
    for (int kind = 0; kind < SourceKey::KIND_COUNT; kind++) {
        for (int lod = 0; lod < ChunkMath::MAX_LOD_NUMBERS; lod++) {
            const int64_t n = stats.computed_by[kind][lod];
            if (n > 0) {
                Dictionary k;
                k["count"] = n;
                k["ms_avg"] = double(stats.usec_by[kind][lod]) / 1000.0 / double(n);
                k["evicted"] = stats.evicted_by[kind][lod];
                by[vformat("%s %d", SourceKey::kind_name(kind), lod)] = k;
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
    d["walks_ready"] = int64_t(walks_ready);
    d["lists_missing"] = lists_missing.load();
    d["point_computed"] = point_computed.load();
    d["view_missing"] = view_missing;
    return d;
}

} // namespace Archetype
