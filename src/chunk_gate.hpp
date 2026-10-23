#pragma once

#include "debug_log.hpp"
#include "lod_structures.hpp"
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
// pointer is aligned, so it stays apart from any other mesher's), the scan generation as the tag: at a new scan the first
// meshes parked that it still wants stay parked under it at their new rank, the others are forgotten by the SDF;
// re-meshes and audits keep their tag. The rank is the mesher's pop order: re-meshes, then the first meshes by load queue, audits last.
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
    // A new scan replaces the chunk queue (send_chunk): the first meshes parked that p_scan holds stay parked at the rank
    // of their new load queue p_queue(chunk) (keeps(): not to queue again, take_ready() brings them back), the others are
    // forgotten.
    template <typename F>
    void next_scan(const LODHashSet &p_scan, F p_queue) {
        scan++;
        kept.clear();
        if (sdf) {
            sdf->set_tag(
                    base + CHUNK, scan, [&](const ChunkId &p_chunk) { return p_scan.has(p_chunk) ? int64_t(rank(CHUNK, p_queue(p_chunk))) : int64_t(-1); },
                    [this](const ChunkId &p_chunk) { kept.insert(p_chunk); });
        }
    }
    // A chunk of the last scan still parked (next_scan).
    bool keeps(const ChunkId &p_chunk) const { return kept.has(p_chunk); }

    // After any first mesh (rank 1 + its load queue).
    static constexpr uint64_t AUDIT_RANK = uint64_t(1) << 62;
    // The SDF's rank of a chunk popped from p_kind's queue (p_queue: a first mesh's ChunkMath::load_queue).
    static uint64_t rank(Kind p_kind, int p_queue) {
        if (p_kind == REMESH) {
            return 0;
        }
        return p_kind == AUDIT ? AUDIT_RANK : 1 + uint64_t(p_queue);
    }

    // A frame's admits from now on: p_max_admits answered PARKED or LATER at most (0: no cap; a READY one asks for no key
    // and the batches bound them). A frame that parks thousands of chunks asks for thousands of keys at once, past the
    // ones the view needs first.
    void begin_frame(uint32_t p_max_admits) {
        max_admits = p_max_admits;
        frame_admits = 0;
        closed = false;
    }
    // Whether this frame may pop and admit another chunk: no LATER yet, the cap not reached.
    bool open() const { return !closed; }

    // READY: r_words gets a bound of the words the chunk adds to its batch's instance list (SDFBase::admit).
    SDFBase::Admit admit(const ChunkId &p_chunk, Kind p_kind, int p_queue, uint64_t &r_words) {
        const uint64_t t0 = now_usec();
        const SDFBase::Admit a = sdf->admit(p_chunk, base + p_kind, scan, rank(p_kind, p_queue), r_words);
        admit_usec += now_usec() - t0;
        if (a != SDFBase::Admit::READY) {
            frame_admits++;
            closed = closed || a == SDFBase::Admit::LATER; // later counts it
            if (!closed && max_admits > 0 && frame_admits >= max_admits) {
                capped_frames++;
                closed = true;
            }
        }
        parked += a == SDFBase::Admit::PARKED;
        later += a == SDFBase::Admit::LATER;
        return a;
    }
    // The next chunk to mesh: chunks come from p_pop(chunk, kind, queue) -> bool (queue: CHUNK's load queue) and are
    // admitted until one is READY (true; r_words if given: its list's bound, admit()); a LATER one goes back through
    // p_requeue(chunk, kind), a parked one comes back through take_ready(). False once the frame's admits close or p_pop
    // has none.
    template <typename Pop, typename Requeue>
    bool next(ChunkId &r_chunk, Kind &r_kind, Pop p_pop, Requeue p_requeue, uint64_t *r_words = nullptr) {
        int queue = 0;
        uint64_t words = 0;
        while (open() && p_pop(r_chunk, r_kind, queue)) {
            const SDFBase::Admit a = admit(r_chunk, r_kind, queue, r_words ? *r_words : words);
            if (a == SDFBase::Admit::READY) {
                return true;
            }
            if (a == SDFBase::Admit::LATER) {
                p_requeue(r_chunk, r_kind);
            }
        }
        return false;
    }
    // The chunks handed back since, to p_requeue(chunk, kind), kind by kind in park order.
    template <typename F>
    void take_ready(F p_requeue) {
        for (int k = 0; k < KINDS; k++) {
            sdf->take_ready(base + k, ready);
            for (const std::pair<ChunkId, uint64_t> &r : ready) {
                p_requeue(r.first, Kind(k)); // next_scan retagged or forgot every older first mesh
            }
        }
    }
    // No chunk of this kind parked.
    bool idle(Kind p_kind) const { return sdf->parked_count(base + p_kind) == 0; }

    uint64_t parked = 0; // admits answered PARKED
    uint64_t later = 0; // admits answered LATER (no room: popping stopped)
    uint64_t admit_usec = 0; // in admit()
    uint64_t capped_frames = 0; // frames whose admits stopped at begin_frame's cap

private:
    SDFBase *sdf = nullptr;
    uint64_t base = 0;
    uint64_t scan = 0;
    uint32_t max_admits = 0;
    uint32_t frame_admits = 0;
    bool closed = true; // this frame: a LATER, or the cap reached (begin_frame opens it)
    std::vector<std::pair<ChunkId, uint64_t>> ready;
    LODHashSet kept; // the last scan's parked chunks (next_scan)
};
