#include "test_common.hpp"

#include "chunk_memory.hpp"

#include <cmath>
#include <random>
#include <vector>

namespace {

constexpr int MAX_LOD = 5;

const ChunkId PARENT(2, 2, 2); // LOD 1

std::vector<ChunkId> children_of(const ChunkId &p_parent) {
    const int64_t h = ChunkMath::get_lod_size(p_parent) / 2;
    std::vector<ChunkId> out;
    for (int i = 0; i < 8; i++) {
        out.push_back(p_parent + ChunkId((i & 1) ? h : -h, (i & 2) ? h : -h, (i & 4) ? h : -h));
    }
    return out;
}

void scan(VoxelState &r_state, const std::vector<ChunkId> &p_ids) {
    std::unique_ptr<LODHashSet> scanned = std::make_unique<LODHashSet>();
    for (const ChunkId &id : p_ids) {
        scanned->insert(id);
    }
    ScanDelta delta;
    ScanDelta::between(*r_state.scanned_chunk, *scanned, delta);
    r_state.add_scan_chunk(std::move(scanned), delta);
}

// A chunk as a mesher hands it back. No RIDs and no owner, so free() has nothing to release.
void load(VoxelState &r_state, const std::vector<ChunkId> &p_ids) {
    for (const ChunkId &id : p_ids) {
        ChunkOutput out;
        out.chunk_id = id;
        r_state.add_new_loaded(out);
    }
}

// The incremental destroy check, compared with the full recompute.
void update(VoxelState &r_state) {
    r_state.update_destroy();
    CHECK(r_state.verify_against_full() == 0);
}

VoxelState make_state() {
    VoxelState state;
    state.set_max_lod(MAX_LOD);
    state.verify = true;
    return state;
}

} // namespace

TEST_CASE("VoxelState: a scan marks the chunks not loaded yet as dirty") {
    VoxelState state = make_state();
    load(state, { ChunkId(1, 1, 1) });
    scan(state, { ChunkId(1, 1, 1), ChunkId(3, 1, 1) });
    CHECK(state.dirty_chunk.size() == 1);
    CHECK(state.dirty_chunk.has(ChunkId(3, 1, 1)));
}

TEST_CASE("VoxelState: a parent stays until all its children are loaded (split)") {
    VoxelState state = make_state();
    load(state, { PARENT });
    const std::vector<ChunkId> children = children_of(PARENT);
    scan(state, children);

    update(state);
    CHECK_FALSE(state.to_destroy_chunk.has(PARENT));

    load(state, std::vector<ChunkId>(children.begin(), children.end() - 1));
    update(state);
    CHECK_FALSE(state.to_destroy_chunk.has(PARENT)); // one child still missing

    load(state, { children.back() });
    update(state);
    CHECK(state.to_destroy_chunk.has(PARENT));
    CHECK(state.to_destroy_chunk.size() == 1);
}

TEST_CASE("VoxelState: children stay until their parent is loaded (merge)") {
    VoxelState state = make_state();
    const std::vector<ChunkId> children = children_of(PARENT);
    load(state, children);
    scan(state, { PARENT });

    update(state);
    CHECK(state.to_destroy_chunk.size() == 0);

    load(state, { PARENT });
    update(state);
    CHECK(state.to_destroy_chunk.size() == 8);
    for (const ChunkId &child : children) {
        CHECK(state.to_destroy_chunk.has(child));
    }
}

TEST_CASE("VoxelState: an unrelated chunk outside the scan is destroyed at once") {
    VoxelState state = make_state();
    load(state, { ChunkId(1, 1, 1), ChunkId(-31, 1, 1) });
    scan(state, { ChunkId(1, 1, 1) });
    update(state);
    CHECK(state.to_destroy_chunk.size() == 1);
    CHECK(state.to_destroy_chunk.has(ChunkId(-31, 1, 1)));
}

namespace {

// Randomized runs of the scan -> mesher -> destroy loop on a small world: LODs [0, SIM_MAX_LOD), 2x2x2
// roots at the top LOD (ids in [-2^SIM_MAX_LOD, 2^SIM_MAX_LOD]).
constexpr int SIM_MAX_LOD = 5;

uint64_t id_hash(const ChunkId &p_id) {
    uint64_t h = 0x9E3779B97F4A7C15ull;
    for (const int64_t v : { p_id.x, p_id.y, p_id.z }) {
        h ^= uint64_t(v) + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
        h *= 0xBF58476D1CE4E5B9ull;
        h ^= h >> 31;
    }
    return h;
}

// An octree cut like the discoverer's: from the roots down, a node is dropped (no surface: a verdict fixed
// per id, like the SDF's), split while the camera is close, kept otherwise.
void sim_scan_node(const ChunkId &p_node, int p_lod, const double p_camera[3], double p_split, LODHashSet &r_scan) {
    if (id_hash(p_node) % 8 == 0) {
        return;
    }
    const double size = double(int64_t(1) << p_lod);
    const double dx = double(p_node.x) - p_camera[0];
    const double dy = double(p_node.y) - p_camera[1];
    const double dz = double(p_node.z) - p_camera[2];
    if (p_lod == 0 || std::sqrt(dx * dx + dy * dy + dz * dz) - std::sqrt(3.0) * size >= p_split * size) {
        r_scan.insert(p_node);
        return;
    }
    const int64_t h = int64_t(1) << (p_lod - 1);
    for (int i = 0; i < 8; i++) {
        sim_scan_node(p_node + ChunkId((i & 1) ? h : -h, (i & 2) ? h : -h, (i & 4) ? h : -h), p_lod - 1, p_camera, p_split, r_scan);
    }
}

struct Sim {
    std::mt19937_64 rng;
    VoxelState state;
    double camera[3] = { 0.0, 0.0, 0.0 };
    double split = 2.0;

    // Discoverer: the latest scan and its delta, until the main thread takes them (a skipped scan's delta
    // is merged into the next one, like ChunkDiscoverer::_publish).
    LODHashSet worker_scan;
    std::unique_ptr<LODHashSet> published;
    ScanDelta published_delta;
    // Mesher: queue rebuilt from the dirty chunks on every scan (minus those in flight), results back in any order.
    std::vector<ChunkId> queue;
    std::vector<ChunkId> in_flight;
    LODHashSet in_flight_set;

    int64_t events = 0;
    int64_t scans_published = 0;
    int64_t scans_skipped = 0;
    int64_t results = 0;
    int64_t stale_results = 0;
    int64_t destroyed = 0;
    // Broken invariants, counted (doctest's REQUIRE needs exceptions); a run stops at the first.
    int64_t failures = 0;

    explicit Sim(uint64_t p_seed) :
            rng(p_seed) {
        state.set_max_lod(SIM_MAX_LOD);
        state.verify = true;
    }

    int uniform(int p_n) { return int(rng() % uint64_t(p_n)); }

    void publish_scan() {
        // Random walk, with jumps and LOD distance changes now and then.
        const double world = double(int64_t(1) << SIM_MAX_LOD);
        if (uniform(20) == 0) {
            for (double &c : camera) {
                c = (double(uniform(2001)) / 1000.0 - 1.0) * world;
            }
        } else {
            for (double &c : camera) {
                c = CLAMP(c + double(uniform(7) - 3), -world, world);
            }
        }
        if (uniform(30) == 0) {
            split = 1.0 + double(uniform(4));
        }
        std::unique_ptr<LODHashSet> scan = std::make_unique<LODHashSet>();
        const int64_t root = int64_t(1) << (SIM_MAX_LOD - 1);
        for (int i = 0; i < 8; i++) {
            sim_scan_node(ChunkId((i & 1) ? root : -root, (i & 2) ? root : -root, (i & 4) ? root : -root), SIM_MAX_LOD - 1, camera, split, *scan);
        }
        ScanDelta delta;
        ScanDelta::between(worker_scan, *scan, delta);
        delta.apply(worker_scan);
        if (published) {
            published_delta.merge(delta);
            scans_skipped++;
        } else {
            published_delta = delta;
        }
        published = std::move(scan);
        scans_published++;
    }

    void take_scan() {
        state.add_scan_chunk(std::move(published), published_delta);
        queue.clear();
        for (const ChunkId &id : state.dirty_chunk) {
            if (!in_flight_set.has(id)) {
                queue.push_back(id);
            }
        }
    }

    void dispatch() {
        // Bounded like the meshers' batches in flight.
        for (int n = 1 + uniform(16); n > 0 && !queue.empty() && in_flight.size() < 48; n--) {
            const size_t i = size_t(uniform(int(queue.size())));
            const ChunkId id = queue[i];
            queue[i] = queue.back();
            queue.pop_back();
            in_flight.push_back(id);
            in_flight_set.insert(id);
        }
    }

    void result() {
        const size_t i = size_t(uniform(int(in_flight.size())));
        ChunkOutput out;
        out.chunk_id = in_flight[i];
        in_flight[i] = in_flight.back();
        in_flight.pop_back();
        in_flight_set.erase(out.chunk_id);
        results++;
        stale_results += state.scanned_chunk->has(out.chunk_id) ? 0 : 1;
        state.add_new_loaded(out);
    }

    // The frame's destroy check, then (like _process) the frame's results, then the destruction.
    void frame() {
        update(state);
        for (const KeyValue<ChunkId, ChunkOutput> &kv : state.to_destroy_chunk) {
            failures += state.loaded_chunk.has(kv.key) && !state.scanned_chunk->has(kv.key) ? 0 : 1;
        }
        for (int n = uniform(3); n > 0 && !in_flight.empty(); n--) {
            result();
        }
        destroyed += state.to_destroy_chunk.size();
        state.destroy_marked();
    }

    // The incremental destroy check against the full recompute (which also checks dirty = scan - loaded
    // and leaving = loaded - scan), as if a frame ran now.
    void check_incremental() {
        state.update_destroy();
        failures += state.verify_against_full();
        state.to_destroy_chunk.clear();
        failures += state.stats.verify_mismatches + state.stats.delta_mismatches + state.stats.octree_violations;
    }

    // The dirty set is always the scan minus the loaded chunks.
    void check_dirty() {
        int64_t expected = 0;
        for (const ChunkId &id : *state.scanned_chunk) {
            const bool dirty = !state.loaded_chunk.has(id);
            expected += dirty ? 1 : 0;
            failures += state.dirty_chunk.has(id) == dirty ? 0 : 1;
        }
        failures += int64_t(state.dirty_chunk.size()) == expected ? 0 : 1;
    }

    void step() {
        const int r = uniform(100);
        if (r < 6) {
            publish_scan();
        } else if (r < 16) {
            if (published) {
                take_scan();
            }
        } else if (r < 36) {
            dispatch();
        } else if (r < 75) {
            if (!in_flight.empty()) {
                result();
            }
        } else {
            frame();
        }
        events++;
        check_dirty();
        check_incremental();
    }

    // No new scan: the mesher drains, then the loaded set must be exactly the scan.
    void settle() {
        if (published) {
            take_scan();
        }
        for (int guard = 0; guard < 100000 && (!queue.empty() || !in_flight.empty() || !state.to_destroy_chunk.is_empty()); guard++) {
            dispatch();
            while (!in_flight.empty()) {
                result();
            }
            frame();
            frame();
        }
        failures += state.dirty_chunk.is_empty() ? 0 : 1;
        failures += state.loaded_chunk.size() == state.scanned_chunk->size() ? 0 : 1;
        for (const ChunkId &id : *state.scanned_chunk) {
            failures += state.loaded_chunk.has(id) ? 0 : 1;
        }
    }
};

} // namespace

// I1 of the incremental destroy: after every event the incremental state gives the same to_destroy set
// and stall counts as the full recompute.
TEST_CASE("VoxelState: randomized scans and results, incremental destroy = full recompute") {
    int64_t events = 0;
    for (uint64_t seed = 1; seed <= 4; seed++) {
        Sim sim(seed);
        for (int i = 0; i < 25000 && sim.failures == 0; i++) {
            sim.step();
            if (i % 5000 == 4999) {
                sim.settle();
            }
        }
        CHECK(sim.failures == 0);
        events += sim.events;
        MESSAGE("seed ", seed, ": ", sim.events, " events, ", sim.state.stats.verify_checks, " comparisons, ", sim.state.stats.rescans, " rescans, ", sim.scans_published, " scans (", sim.scans_skipped, " skipped), ", sim.results,
                " results (", sim.stale_results, " stale), ", sim.destroyed, " destroyed, last scan ", sim.state.scanned_chunk->size(), " chunks");
    }
    CHECK(events >= 100000);
}
