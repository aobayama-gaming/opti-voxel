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

void InstanceSnapshot::chunk_list(const ChunkId &p_chunk, std::vector<uint32_t> &r_out) const {
    r_out = unbounded;
    if (const std::vector<uint32_t> *s = subtree.getptr(p_chunk)) {
        r_out.insert(r_out.end(), s->begin(), s->end());
    }
    // The strict ancestors' own instances.
    int lod = ChunkMath::get_lod(p_chunk);
    ChunkId a = p_chunk;
    while (lod < max_lod) {
        a = ChunkMath::get_parent_from_child(a);
        lod++;
        if (const std::vector<uint32_t> *v = stored_at.getptr(a)) {
            r_out.insert(r_out.end(), v->begin(), v->end());
        }
    }
    std::sort(r_out.begin(), r_out.end());
}

void InstanceSnapshot::point_candidates(const VoxelPos &p_voxel, std::vector<uint32_t> &r_out) const {
    r_out = unbounded;
    // One node per LOD on the point's path; the instance that owns a sample point sits on its node's
    // path at its own LOD.
    for (int lod = 0; lod <= max_lod; lod++) {
        const ChunkId id = chunk_at_voxel(p_voxel, lod, param);
        if (const std::vector<uint32_t> *v = stored_at.getptr(id)) {
            r_out.insert(r_out.end(), v->begin(), v->end());
        }
    }
    std::sort(r_out.begin(), r_out.end());
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

    // The subtree lists: process nodes finest first and append each into its parent (children are
    // always finer, so a node is final when it is reached).
    std::vector<ChunkId> nodes;
    nodes.reserve(snap->stored_at.size());
    for (const KeyValue<ChunkId, std::vector<uint32_t>> &kv : snap->stored_at) {
        nodes.push_back(kv.key);
    }
    std::sort(nodes.begin(), nodes.end(), [](const ChunkId &p_a, const ChunkId &p_b) { return ChunkMath::get_lod(p_a) > ChunkMath::get_lod(p_b); });
    for (const ChunkId &n : nodes) {
        std::vector<uint32_t> acc;
        if (const std::vector<uint32_t> *own = snap->stored_at.getptr(n)) {
            acc = *own;
        }
        if (const std::vector<uint32_t> *sub = snap->subtree.getptr(n)) {
            acc.insert(acc.end(), sub->begin(), sub->end());
        }
        if (ChunkMath::get_lod(n) >= p_max_lod) {
            snap->subtree[n] = std::move(acc);
            continue;
        }
        snap->subtree[n] = acc;
        const ChunkId parent = ChunkMath::get_parent_from_child(n);
        std::vector<uint32_t> &pacc = snap->subtree[parent];
        pacc.insert(pacc.end(), snap->subtree[n].begin(), snap->subtree[n].end());
    }
    return snap;
}

} // namespace Archetype
