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

void InstanceSnapshot::gather(const ChunkId &p_centre, int p_lod, const VoxelPos &p_lo, const VoxelPos &p_hi, std::vector<uint32_t> &r_out) const {
    r_out = unbounded;
    thread_local std::vector<uint32_t> candidates;
    candidates.clear();
    const int64_t step = int64_t(2) << p_lod;
    for (int i = 0; i < 27; i++) {
        const ChunkId n = p_centre + ChunkId((i % 3 - 1) * step, (i / 3 % 3 - 1) * step, (i / 9 - 1) * step);
        if (const std::vector<uint32_t> *s = subtree.getptr(n)) {
            candidates.insert(candidates.end(), s->begin(), s->end());
        }
        ChunkId a = n;
        for (int lod = p_lod; lod < max_lod; lod++) {
            a = ChunkMath::get_parent_from_child(a);
            if (const std::vector<uint32_t> *v = stored_at.getptr(a)) {
                candidates.insert(candidates.end(), v->begin(), v->end());
            }
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

std::shared_ptr<const InstanceSnapshot> build_snapshot(std::shared_ptr<const DefLibrary> p_library, const Program &p_ground,
        float p_out_scale, bool p_normalize, std::vector<TableRecord> p_records, const ChunkParam &p_param, int p_max_lod,
        uint32_t p_version) {
    std::shared_ptr<InstanceSnapshot> snap = std::make_shared<InstanceSnapshot>();
    snap->library = p_library;
    snap->ground = p_ground;
    snap->out_scale = p_out_scale;
    snap->normalize = p_normalize;
    snap->version = p_version;
    snap->max_lod = p_max_lod;
    snap->param = p_param;
    snap->records = std::move(p_records);

    for (uint32_t i = 0; i < snap->records.size(); i++) {
        const TableRecord &rec = snap->records[i];
        if (rec.removed || rec.max_lod <= rec.min_lod) {
            continue; // removed: kept for stable ids; dropped at every LOD: never listed
        }
        const Bounds wb = record_world_bounds((*p_library)[rec.def], rec);
        if (!wb.bounded) {
            snap->unbounded.push_back(i); // listed by every chunk
            continue;
        }
        double size = 0.0;
        for (int a = 0; a < 3; a++) {
            size = MAX(size, double(wb.hi[a] - wb.lo[a]));
        }
        const int lod = instance_lod(size, rec.k, rec.margin_c, p_param, p_max_lod);
        const double s = std::ldexp(1.0, lod);
        const double m = double(rec.k) + double(rec.margin_c) * s;
        const int64_t step = int64_t(1) << (lod + 1);
        const int64_t half = int64_t(1) << lod;
        const int64_t span = int64_t(p_param.chunk_voxel_dimension()) * half;
        int64_t id[3][2];
        int64_t count[3];
        for (int a = 0; a < 3; a++) {
            const int64_t lo = int64_t(std::floor(double(wb.lo[a]) - m));
            const int64_t hi = int64_t(std::ceil(double(wb.hi[a]) + m));
            const int64_t i0 = WorldCoord::floor_div(lo, span) * step + half;
            const int64_t i1 = WorldCoord::floor_div(hi, span) * step + half;
            id[a][0] = i0;
            id[a][1] = i1;
            count[a] = (i1 - i0) / step + 1;
        }
        for (int64_t x = 0; x < count[0]; x++) {
            for (int64_t y = 0; y < count[1]; y++) {
                for (int64_t z = 0; z < count[2]; z++) {
                    snap->stored_at[ChunkId(id[0][0] + x * step, id[1][0] + y * step, id[2][0] + z * step)].push_back(i);
                }
            }
        }
    }

    // The subtree lists, one LOD at a time from the finest: every node's list is final when its LOD is
    // reached, and each one goes into its parent, whether or not the parent stores anything itself (an
    // instance stored at LOD 0 must reach its LOD 2+ ancestors through LOD 1 nodes that store nothing).
    std::vector<std::vector<ChunkId>> by_lod(p_max_lod + 1);
    for (const KeyValue<ChunkId, std::vector<uint32_t>> &kv : snap->stored_at) {
        snap->subtree[kv.key] = kv.value;
        by_lod[MIN(ChunkMath::get_lod(kv.key), p_max_lod)].push_back(kv.key);
    }
    for (int lod = 0; lod < p_max_lod; lod++) {
        for (const ChunkId &n : by_lod[lod]) {
            // The band rule: the parent's chunks only need the instances still listed at its LOD, so a coarse
            // node's subtree holds the instances big enough for it, not every instance under it.
            std::vector<uint32_t> up;
            for (const uint32_t i : snap->subtree[n]) {
                if (lod + 1 < snap->records[i].max_lod) {
                    up.push_back(i);
                }
            }
            if (up.empty()) {
                continue;
            }
            const ChunkId parent = ChunkMath::get_parent_from_child(n);
            if (!snap->subtree.has(parent)) {
                by_lod[lod + 1].push_back(parent);
            }
            std::vector<uint32_t> &pacc = snap->subtree[parent];
            pacc.insert(pacc.end(), up.begin(), up.end());
        }
    }
    // Dedup every subtree: an instance stored in sibling chunks is appended once per sibling.
    for (KeyValue<ChunkId, std::vector<uint32_t>> &kv : snap->subtree) {
        std::vector<uint32_t> &v = kv.value;
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }
    return snap;
}

} // namespace Archetype
