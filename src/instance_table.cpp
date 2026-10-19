#include "instance_table.hpp"

#include <algorithm>
#include <cmath>

namespace Archetype {

// The chunk id at LOD p_lod holding LOD0 voxel p_v: chunk ids are centres in chunk-id units (half a
// LOD0 chunk), the LOD-L id has its lowest set bit at L.
static ChunkId chunk_at_voxel(const VoxelPos &p_v, int p_lod, const ChunkParam &p_param) {
    const int64_t span = int64_t(p_param.chunk_voxel_dimension()) * (int64_t(1) << p_lod);
    const int64_t step = int64_t(1) << (p_lod + 1);
    const int64_t half = int64_t(1) << p_lod;
    return ChunkId(WorldCoord::floor_div(p_v.x, span) * step + half, WorldCoord::floor_div(p_v.y, span) * step + half,
            WorldCoord::floor_div(p_v.z, span) * step + half);
}

// The finest LOD whose chunk span holds the box grown by the locality margin at that LOD.
static int instance_lod(double p_bare_size, float p_k, float p_margin_c, const ChunkParam &p_param, int p_max_lod) {
    const double c0 = double(p_param.chunk_voxel_dimension());
    for (int lod = 0; lod <= p_max_lod; lod++) {
        const double s = std::ldexp(1.0, lod);
        if (p_bare_size + 2.0 * (double(p_k) + double(p_margin_c) * s) <= c0 * s) {
            return lod;
        }
    }
    return p_max_lod;
}

void chunk_sample_box(const ChunkId &p_chunk, const ChunkParam &p_param, VoxelPos &r_lo, VoxelPos &r_hi) {
    const int64_t cell = ChunkMath::get_lod_size(p_chunk);
    const VoxelPos min = ChunkMath::chunk_min_voxel(p_chunk, p_param);
    const int64_t lo = -int64_t(p_param.skirt_size) * cell;
    const int64_t hi = int64_t(p_param.chunk_size - 1 - p_param.skirt_size) * cell;
    r_lo = min + VoxelPos(lo, lo, lo);
    r_hi = min + VoxelPos(hi, hi, hi);
}

bool influence_meets(const TableRecord &p_record, int p_lod, const VoxelPos &p_lo, const VoxelPos &p_hi) {
    const int64_t m = influence_margin(p_record, p_lod);
    for (int a = 0; a < 3; a++) {
        const int64_t lo = p_record.anchor[a] + p_record.box_lo[a] - m;
        const int64_t hi = p_record.anchor[a] + p_record.box_hi[a] + m;
        if (p_hi[a] < lo || p_lo[a] > hi) {
            return false;
        }
    }
    return true;
}

void NodeRangeMap::reserve(size_t p_count) {
    size_t cap = 16;
    while (cap < p_count * 2) {
        cap <<= 1;
    }
    if (cap <= slots.size()) {
        return;
    }
    std::vector<Slot> old;
    old.swap(slots);
    slots.assign(cap, Slot());
    mask = cap - 1;
    count = 0;
    for (const Slot &s : old) {
        if (s.used) {
            insert(s.node, s.range);
        }
    }
}

void NodeRangeMap::insert(const ChunkId &p_node, const IndexRange &p_range) {
    if ((count + 1) * 2 > slots.size()) {
        reserve(count + 1 > 8 ? (count + 1) * 2 : 16);
    }
    size_t i = size_t(hash(p_node) >> 20) & mask;
    while (slots[i].used) {
        i = (i + 1) & mask;
    }
    slots[i] = { p_node, p_range, true };
    count++;
}

const IndexRange *NodeRangeMap::getptr(const ChunkId &p_node) const {
    if (slots.empty()) {
        return nullptr;
    }
    size_t i = size_t(hash(p_node) >> 20) & mask;
    while (slots[i].used) {
        if (slots[i].node == p_node) {
            return &slots[i].range;
        }
        i = (i + 1) & mask;
    }
    return nullptr;
}

namespace {

struct Placement {
    ChunkId node;
    uint32_t id;
    bool operator==(const Placement &p_o) const { return node == p_o.node && id == p_o.id; }
};

// Morton order of the nodes of one LOD: on their indices at that LOD, biased by 2^(60 - lod) so a parent's key is
// exactly its child's key >> 1. A parent's order is then its children's: a level's entries mapped to their
// parents stay sorted (by node), and each LOD merges instead of sorting again.
// Negative, zero or positive as a is before, equal to or after b in Morton order (keys of the three axes): the
// axis with the highest differing bit decides.
int morton_compare(const uint64_t *p_a, const uint64_t *p_b) {
    int d = 0;
    uint64_t top = 0;
    for (int k = 0; k < 3; k++) {
        const uint64_t x = p_a[k] ^ p_b[k];
        if (top < x && top < (top ^ x)) { // x's highest bit is above top's
            d = k;
            top = x;
        }
    }
    return top == 0 ? 0 : (p_a[d] < p_b[d] ? -1 : 1);
}

struct MortonLess {
    int lod = 0;
    uint64_t key(int64_t p_c) const { return uint64_t((p_c >> (lod + 1)) + (int64_t(1) << (60 - lod))); }
    bool operator()(const ChunkId &p_a, const ChunkId &p_b) const {
        const uint64_t a[3] = { key(p_a.x), key(p_a.y), key(p_a.z) };
        const uint64_t b[3] = { key(p_b.x), key(p_b.y), key(p_b.z) };
        return morton_compare(a, b) < 0;
    }
    bool operator()(const Placement &p_a, const Placement &p_b) const {
        return (*this)(p_a.node, p_b.node) || (p_a.node == p_b.node && p_a.id < p_b.id);
    }
};

// Sorted placements -> node ranges into r_ids (one map entry per node, reserved up front).
void fill_ranges(const std::vector<Placement> &p_sorted, NodeRangeMap &r_map, std::vector<uint32_t> &r_ids) {
    for (size_t i = 0; i < p_sorted.size();) {
        size_t j = i;
        IndexRange range;
        range.begin = uint32_t(r_ids.size());
        while (j < p_sorted.size() && p_sorted[j].node == p_sorted[i].node) {
            r_ids.push_back(p_sorted[j].id);
            j++;
        }
        range.count = uint32_t(j - i);
        r_map.insert(p_sorted[i].node, range);
        i = j;
    }
}

// std::sort by MortonLess with the keys computed once per placement, not per comparison.
void sort_morton(std::vector<Placement> &r_v, const MortonLess &p_less) {
    struct Keyed {
        uint64_t k[3];
        uint32_t i;
    };
    std::vector<Keyed> keyed(r_v.size());
    for (size_t i = 0; i < r_v.size(); i++) {
        keyed[i] = { { p_less.key(r_v[i].node.x), p_less.key(r_v[i].node.y), p_less.key(r_v[i].node.z) }, uint32_t(i) };
    }
    std::sort(keyed.begin(), keyed.end(), [&r_v](const Keyed &p_a, const Keyed &p_b) {
        const int c = morton_compare(p_a.k, p_b.k);
        return c != 0 ? c < 0 : r_v[p_a.i].id < r_v[p_b.i].id;
    });
    std::vector<Placement> sorted(r_v.size());
    for (size_t i = 0; i < keyed.size(); i++) {
        sorted[i] = r_v[keyed[i].i];
    }
    r_v.swap(sorted);
}

size_t unique_nodes(const std::vector<Placement> &p_sorted) {
    size_t n = 0;
    for (size_t i = 0; i < p_sorted.size(); i++) {
        n += (i == 0 || !(p_sorted[i].node == p_sorted[i - 1].node)) ? 1 : 0;
    }
    return n;
}

// Where each record of p_ids is stored (r_by_lod, one bucket per storage LOD) or, unbounded, listed everywhere.
void place_instances(const DefLibrary &p_library, const RecordStore &p_records, const std::vector<uint32_t> &p_ids, const ChunkParam &p_param,
        int p_max_lod, std::vector<std::vector<Placement>> &r_by_lod, std::vector<uint32_t> &r_unbounded) {
    r_by_lod.assign(p_max_lod + 1, std::vector<Placement>());
    for (const uint32_t i : p_ids) {
        const TableRecord &rec = p_records[i];
        if (rec.removed || rec.max_lod == 0) {
            continue; // removed: kept for stable ids; dropped at every LOD: never listed
        }
        const Bounds wb = record_world_bounds(p_library[rec.def], rec);
        if (!wb.bounded) {
            r_unbounded.push_back(i); // listed by every chunk
            continue;
        }
        double size = 0.0;
        for (int a = 0; a < 3; a++) {
            size = MAX(size, double(wb.hi[a] - wb.lo[a]));
        }
        // Once, in the node holding its centre: its grown box is at most a node wide there, so the chunks
        // whose samples it meets are near that node at every LOD (gather). Wider than a node even at the top
        // LOD: in every node its grown box touches.
        const int lod = instance_lod(size, rec.k, rec.margin_c, p_param, p_max_lod);
        const double m = double(rec.k) + double(rec.margin_c) * std::ldexp(1.0, lod);
        const int64_t half = int64_t(1) << lod;
        const int64_t span = int64_t(p_param.chunk_voxel_dimension()) * half;
        if (size + 2.0 * m <= double(span)) {
            VoxelPos centre;
            for (int a = 0; a < 3; a++) {
                centre[a] = wb.lo[a] + (wb.hi[a] - wb.lo[a]) / 2;
            }
            r_by_lod[lod].push_back({ chunk_at_voxel(centre, lod, p_param), i });
            continue;
        }
        const int64_t step = int64_t(1) << (lod + 1);
        int64_t id[3];
        int64_t count[3];
        for (int a = 0; a < 3; a++) {
            id[a] = WorldCoord::floor_div(int64_t(std::floor(double(wb.lo[a]) - m)), span) * step + half;
            count[a] = (WorldCoord::floor_div(int64_t(std::ceil(double(wb.hi[a]) + m)), span) * step + half - id[a]) / step + 1;
        }
        for (int64_t x = 0; x < count[0]; x++) {
            for (int64_t y = 0; y < count[1]; y++) {
                for (int64_t z = 0; z < count[2]; z++) {
                    r_by_lod[lod].push_back({ ChunkId(id[0] + x * step, id[1] + y * step, id[2] + z * step), i });
                }
            }
        }
    }
}

// A LOD's subtree entries mapped to their parents, those still listed one LOD up (the band rule). The parents stay
// in Morton order; a parent's ids come from up to 8 children: each run is sorted.
void carry_up(const std::vector<Placement> &p_level, int p_lod, const RecordStore &p_records, std::vector<Placement> &r_carry) {
    r_carry.clear();
    for (const Placement &e : p_level) {
        if (p_lod + 1 < p_records[e.id].max_lod) {
            r_carry.push_back({ ChunkMath::get_parent_from_child(e.node), e.id });
        }
    }
    for (size_t i = 0; i < r_carry.size();) {
        size_t j = i + 1;
        while (j < r_carry.size() && r_carry[j].node == r_carry[i].node) {
            j++;
        }
        if (j - i > 1) {
            std::sort(r_carry.begin() + i, r_carry.begin() + j, [](const Placement &p_a, const Placement &p_b) { return p_a.id < p_b.id; });
        }
        i = j;
    }
}

// A LOD's occupancy: its own placements' nodes merged with the occupied children's parents (r_carry in, the
// next LOD's carry out).
void occupancy_level(const std::vector<Placement> &p_own, const MortonLess &p_less, bool p_top, std::vector<ChunkId> &r_carry, std::vector<ChunkId> &r_occ) {
    std::vector<ChunkId> nodes;
    nodes.reserve(p_own.size());
    for (const Placement &e : p_own) {
        if (nodes.empty() || !(nodes.back() == e.node)) {
            nodes.push_back(e.node);
        }
    }
    r_occ.resize(r_carry.size() + nodes.size());
    std::merge(r_carry.begin(), r_carry.end(), nodes.begin(), nodes.end(), r_occ.begin(), p_less);
    r_occ.erase(std::unique(r_occ.begin(), r_occ.end()), r_occ.end());
    r_carry.clear();
    if (!p_top) {
        for (const ChunkId &n : r_occ) {
            const ChunkId parent = ChunkMath::get_parent_from_child(n);
            if (r_carry.empty() || !(r_carry.back() == parent)) {
                r_carry.push_back(parent);
            }
        }
    }
}

// The layer over the records p_ids (ascending): one placement per instance, then per LOD from the finest the
// stored entries, the subtree entries (that LOD's own placements plus its children's entries still listed at
// it: a coarse node holds the instances big enough for it, not every instance under it) and the occupancy
// (every placement climbs to the top, band or not). Each LOD merges instead of sorting (MortonLess).
std::shared_ptr<const IndexLayer> build_layer(const DefLibrary &p_library, const RecordStore &p_records, const std::vector<uint32_t> &p_ids,
        const ChunkParam &p_param, int p_max_lod) {
    std::shared_ptr<IndexLayer> layer = std::make_shared<IndexLayer>();
    std::vector<std::vector<Placement>> by_lod;
    place_instances(p_library, p_records, p_ids, p_param, p_max_lod, by_lod, layer->unbounded);
    size_t placements = 0;
    for (const std::vector<Placement> &b : by_lod) {
        placements += b.size();
    }
    layer->stored_at.reserve(uint32_t(placements)); // an upper bound of the nodes
    layer->stored_ids.reserve(placements);
    std::vector<std::vector<Placement>> sub(p_max_lod + 1);
    std::vector<Placement> carry;
    std::vector<ChunkId> occ_carry;
    layer->occupied.resize(p_max_lod + 1);
    size_t sub_total = 0, sub_nodes = 0;
    for (int lod = 0; lod <= p_max_lod; lod++) {
        const MortonLess less{ lod };
        std::vector<Placement> &own = by_lod[lod];
        sort_morton(own, less);
        fill_ranges(own, layer->stored_at, layer->stored_ids);
        std::vector<Placement> &level = sub[lod];
        level.resize(carry.size() + own.size());
        std::merge(carry.begin(), carry.end(), own.begin(), own.end(), level.begin(), less);
        level.erase(std::unique(level.begin(), level.end()), level.end());
        sub_total += level.size();
        sub_nodes += unique_nodes(level);
        if (lod < p_max_lod) {
            carry_up(level, lod, p_records, carry);
        }
        occupancy_level(own, less, lod == p_max_lod, occ_carry, layer->occupied[lod]);
        own = std::vector<Placement>();
    }
    layer->subtree.reserve(uint32_t(sub_nodes));
    layer->subtree_ids.reserve(sub_total);
    for (std::vector<Placement> &level : sub) {
        fill_ranges(level, layer->subtree, layer->subtree_ids);
        level = std::vector<Placement>();
    }
    return layer;
}

void append_range(const NodeRangeMap &p_map, const std::vector<uint32_t> &p_ids, const ChunkId &p_node,
        std::vector<uint32_t> &r_out) {
    if (const IndexRange *r = p_map.getptr(p_node)) {
        r_out.insert(r_out.end(), p_ids.begin() + r->begin, p_ids.begin() + r->begin + r->count);
    }
}

} // namespace

bool IndexLayer::is_occupied(const ChunkId &p_node, int p_lod) const {
    if (p_lod < 0 || p_lod >= int(occupied.size())) {
        return false;
    }
    const std::vector<ChunkId> &v = occupied[p_lod];
    return std::binary_search(v.begin(), v.end(), p_node, MortonLess{ p_lod });
}

void InstanceSnapshot::gather(const ChunkId &p_centre, int p_lod, const VoxelPos &p_lo, const VoxelPos &p_hi, std::vector<uint32_t> &r_out) const {
    r_out.clear();
    thread_local std::vector<uint32_t> candidates;
    candidates.clear();
    for (const IndexLayer *layer : { base.get(), overlay.get() }) {
        if (!layer) {
            continue;
        }
        const size_t from = candidates.size();
        // Stored at this LOD or finer: in the subtree of one of the 125 chunks within 2 (its influence box grows
        // with the chunk's LOD: up to ~1.5 nodes from its centre). Stored coarser (a bigger instance): in one of
        // the 27 nodes around the chunk's ancestor at its storage LOD (its box is at most half a node there).
        const int64_t s0 = int64_t(2) << p_lod;
        for (int i = 0; i < 125; i++) {
            const ChunkId n = p_centre + ChunkId((i % 5 - 2) * s0, (i / 5 % 5 - 2) * s0, (i / 25 - 2) * s0);
            append_range(layer->subtree, layer->subtree_ids, n, candidates);
        }
        ChunkId c = p_centre;
        for (int lod = p_lod + 1; lod <= max_lod; lod++) {
            c = ChunkMath::get_parent_from_child(c);
            const int64_t s = int64_t(2) << lod;
            for (int i = 0; i < 27; i++) {
                append_range(layer->stored_at, layer->stored_ids, c + ChunkId((i % 3 - 1) * s, (i / 3 % 3 - 1) * s, (i / 9 - 1) * s), candidates);
            }
        }
        r_out.insert(r_out.end(), layer->unbounded.begin(), layer->unbounded.end());
        if (layer == base.get() && !masked.is_empty()) {
            // The base entries of edited records are stale: the overlay holds their current state.
            candidates.erase(std::remove_if(candidates.begin() + from, candidates.end(), [this](uint32_t p_i) { return masked.has(p_i); }),
                    candidates.end());
            r_out.erase(std::remove_if(r_out.begin(), r_out.end(), [this](uint32_t p_i) { return masked.has(p_i); }), r_out.end());
        }
    }
    std::sort(candidates.begin(), candidates.end());
    // An instance stored in several chunks of the neighbourhood is gathered once per node: dedup, or a
    // smooth op would fold it several times.
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    for (const uint32_t i : candidates) {
        if (records[i].listed_at(p_lod) && influence_meets(records[i], p_lod, p_lo, p_hi)) {
            r_out.push_back(i);
        }
    }
    std::sort(r_out.begin(), r_out.end());
}

void InstanceSnapshot::chunk_list(const ChunkId &p_chunk, std::vector<uint32_t> &r_out) const {
    VoxelPos lo, hi;
    chunk_sample_box(p_chunk, param, lo, hi);
    gather(p_chunk, ChunkMath::get_lod(p_chunk), lo, hi, r_out);
}

void InstanceSnapshot::point_candidates(const VoxelPos &p_voxel, int p_lod, std::vector<uint32_t> &r_out) const {
    const int lod = MIN(MAX(p_lod, 0), max_lod);
    gather(chunk_at_voxel(p_voxel, lod, param), lod, p_voxel, p_voxel, r_out);
}

bool InstanceSnapshot::near_layer(const IndexLayer &p_layer, const VoxelPos &p_lo, const VoxelPos &p_hi, int p_lod) const {
    for (const uint32_t i : p_layer.unbounded) {
        if (&p_layer != base.get() || !masked.has(i)) {
            return true; // an unbounded instance (edited away from the base: its stale entry doesn't count)
        }
    }
    // An instance stored at this LOD or finer: some node of this LOD near the grown box holds it in its
    // subtree. A bigger one is stored at a coarser LOD: the nodes near the grown box there, exact test.
    for (int l = p_lod; l <= max_lod; l++) {
        // An instance stored at LOD l has its centre within half a node of its bare box.
        const int64_t half = (int64_t(param.chunk_voxel_dimension()) << l) / 2 + 1;
        const ChunkId a = chunk_at_voxel(p_lo - VoxelPos(half, half, half), l, param);
        const ChunkId b = chunk_at_voxel(p_hi + VoxelPos(half, half, half), l, param);
        const int64_t step = int64_t(2) << l;
        for (int64_t x = a.x; x <= b.x; x += step) {
            for (int64_t y = a.y; y <= b.y; y += step) {
                for (int64_t z = a.z; z <= b.z; z += step) {
                    const ChunkId n(x, y, z);
                    if (l == p_lod) {
                        if (p_layer.is_occupied(n, l)) {
                            return true;
                        }
                        continue;
                    }
                    const IndexRange *r = p_layer.stored_at.getptr(n);
                    if (!r) {
                        continue;
                    }
                    for (uint32_t k = r->begin; k < r->begin + r->count; k++) {
                        const TableRecord &rec = records[p_layer.stored_ids[k]];
                        bool meets = true;
                        for (int ax = 0; ax < 3 && meets; ax++) {
                            meets = p_hi[ax] >= rec.anchor[ax] + rec.box_lo[ax] && p_lo[ax] <= rec.anchor[ax] + rec.box_hi[ax];
                        }
                        if (meets) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}

bool InstanceSnapshot::near_instance(const ChunkId &p_node, int p_lod) const {
    const int lod = MIN(MAX(p_lod, 0), max_lod);
    const int64_t span = int64_t(param.chunk_voxel_dimension()) << lod;
    const VoxelPos min = ChunkMath::chunk_min_voxel(p_node, param);
    const VoxelPos lo = min - VoxelPos(2 * span, 2 * span, 2 * span);
    const VoxelPos hi = min + VoxelPos(3 * span, 3 * span, 3 * span);
    return (base && near_layer(*base, lo, hi, lod)) || (overlay && near_layer(*overlay, lo, hi, lod));
}

std::shared_ptr<const InstanceSnapshot> build_snapshot(std::shared_ptr<const DefLibrary> p_library, const Program &p_ground,
        float p_out_scale, bool p_normalize, RecordStore p_records, const ChunkParam &p_param, int p_max_lod, uint32_t p_version) {
    std::shared_ptr<InstanceSnapshot> snap = std::make_shared<InstanceSnapshot>();
    snap->library = p_library;
    snap->ground = p_ground;
    snap->out_scale = p_out_scale;
    snap->normalize = p_normalize;
    snap->version = p_version;
    snap->max_lod = p_max_lod;
    snap->param = p_param;
    snap->records = std::move(p_records);
    std::vector<uint32_t> all(snap->records.size());
    for (uint32_t i = 0; i < all.size(); i++) {
        all[i] = i;
    }
    snap->base = build_layer(*p_library, snap->records, all, p_param, p_max_lod);
    snap->base_count = snap->records.size();
    return snap;
}

std::shared_ptr<const InstanceSnapshot> edit_snapshot(const InstanceSnapshot &p_previous, RecordStore p_records,
        const std::vector<uint32_t> &p_changed, uint32_t p_version) {
    std::vector<uint32_t> ids = p_previous.overlay_ids;
    ids.insert(ids.end(), p_changed.begin(), p_changed.end());
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    if (ids.size() > InstanceSnapshot::REBASE || !p_previous.base) {
        return build_snapshot(p_previous.library, p_previous.ground, p_previous.out_scale, p_previous.normalize, std::move(p_records),
                p_previous.param, p_previous.max_lod, p_version);
    }
    std::shared_ptr<InstanceSnapshot> snap = std::make_shared<InstanceSnapshot>();
    snap->library = p_previous.library;
    snap->ground = p_previous.ground;
    snap->out_scale = p_previous.out_scale;
    snap->normalize = p_previous.normalize;
    snap->version = p_version;
    snap->max_lod = p_previous.max_lod;
    snap->param = p_previous.param;
    snap->records = std::move(p_records);
    snap->base = p_previous.base;
    snap->base_count = p_previous.base_count;
    snap->overlay_ids = std::move(ids);
    for (const uint32_t i : snap->overlay_ids) {
        if (i < snap->base_count) {
            snap->masked.insert(i);
        }
    }
    snap->overlay = build_layer(*snap->library, snap->records, snap->overlay_ids, snap->param, snap->max_lod);
    return snap;
}

} // namespace Archetype
