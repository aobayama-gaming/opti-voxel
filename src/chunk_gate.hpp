#pragma once

#include "debug_log.hpp"
#include "sdf_base.hpp"

#include <cstdint>
#include <utility>
#include <vector>

// A mesher's main-thread dispatch sub-timers, cumulative us: VoxelTerrain records their per-frame deltas as _process
// sections (parts of mesher_dispatch).
struct DispatchTimes {
    uint64_t gate_admit = 0; // ChunkGate::admit: the SDF's admit (the stream's key requests)
    uint64_t build_lists = 0; // SDFGpuTable::build_lists
    uint64_t record_bytes = 0; // SDFGpuTable::record_bytes
};

// A mesher's side of the SDF's gating (SDFBase::admit): each chunk it pops is asked for, and a parked one comes back
// through take_ready() into the queue it came from. One owner per queue kind (the mesher's address plus the kind: a
// pointer is aligned, so it stays apart from any other mesher's), the scan generation as the tag: a first mesh parked
// before the last scan goes back only if it was parked under that scan (send_chunk queued again those the scan wants).
class ChunkGate {
public:
    enum Kind {
        CHUNK, // chunk_queue: a first mesh
        REMESH, // remesh_queue: an edit's re-mesh
        AUDIT, // the discovery audit's counts
        KINDS,
    };

    void bind(SDFBase *p_sdf, const void *p_mesher) {
        sdf = p_sdf;
        base = uint64_t(uintptr_t(p_mesher));
    }
    // A new scan replaced the chunk queue (send_chunk).
    void next_scan() { scan++; }

    SDFBase::Admit admit(const ChunkId &p_chunk, Kind p_kind) {
        const uint64_t t0 = now_usec();
        const SDFBase::Admit a = sdf->admit(p_chunk, base + p_kind, scan);
        admit_usec += now_usec() - t0;
        parked += a == SDFBase::Admit::PARKED;
        later += a == SDFBase::Admit::LATER;
        return a;
    }
    // The chunks handed back since, to p_requeue(chunk, kind), kind by kind in park order.
    template <typename F>
    void take_ready(F p_requeue) {
        for (int k = 0; k < KINDS; k++) {
            sdf->take_ready(base + k, ready);
            for (const std::pair<ChunkId, uint64_t> &r : ready) {
                if (k != CHUNK || r.second == scan) {
                    p_requeue(r.first, Kind(k));
                } else {
                    stale++;
                }
            }
        }
    }
    // No chunk of this kind parked.
    bool idle(Kind p_kind) const { return sdf->parked_count(base + p_kind) == 0; }

    uint64_t parked = 0; // admits answered PARKED
    uint64_t later = 0; // admits answered LATER (no room: popping stopped)
    uint64_t stale = 0; // first meshes handed back parked under an older scan (dropped: send_chunk queued them again if wanted)
    uint64_t admit_usec = 0; // in admit()

private:
    SDFBase *sdf = nullptr;
    uint64_t base = 0;
    uint64_t scan = 0;
    std::vector<std::pair<ChunkId, uint64_t>> ready;
};
