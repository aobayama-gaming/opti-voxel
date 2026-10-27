#include "base_regions.hpp"

#include "cascade.hpp"
#include "gpu_frame.hpp"

#include "ov/on_error.hpp"

namespace BaseRegions {

namespace {

// The slot of a region cell: its parity per axis (GLSL base_at).
int slot(const Vector3i64 &p_cell) {
    return int(p_cell.x & 1) | int(p_cell.y & 1) << 1 | int(p_cell.z & 1) << 2;
}

} // namespace

void encode(uint8_t *r_at, const BaseSurface::Region &p_region) {
    ov::put_f32(r_at + REGION_D_AT, p_region.d);
    for (int a = 0; a < 3; a++) {
        ov::put_f32(r_at + REGION_N_AT + 4 * a, p_region.n[a]);
        ov::put_f32(r_at + REGION_H_DIAG_AT + 4 * a, p_region.h[a]);
        ov::put_f32(r_at + REGION_H_OFF_AT + 4 * a, p_region.h[3 + a]);
    }
}

void build(const BaseSurface &p_base, const uint8_t *p_items, uint32_t p_count, const ChunkParam &p_param, bool p_points, Tables &r_tables) {
    r_tables.index.assign(size_t(p_count) * SLOTS, 0);
    r_tables.regions.assign(REGION_BYTES, 0);
    OV_ERR_FAIL_COND_MSG(p_base.kind() != BaseSurface::Kind::SPHERE, "BaseRegions: only a sphere has regions.");
    OV_ERR_FAIL_COND_MSG(p_param.chunk_size > (1 << BaseSurface::REGION_SHIFT), "BaseRegions: a chunk wider than a region.");
    // A point item's box is its point alone (point_param keeps the chunk size: its scratch strides).
    ChunkParam param = p_param;
    if (p_points) {
        param.chunk_size = 1;
        param.skirt_size = 0;
    }
    const int switch_lod = p_base.get_switch_lod();
    r_tables.seen.resize(size_t(switch_lod));
    for (auto &s : r_tables.seen) {
        s.clear();
    }
    uint32_t count = 0;
    auto add = [&](const Vector3i64 &p_cell, int p_lod) {
        if (const uint32_t *known = r_tables.seen[size_t(p_lod)].getptr(p_cell)) {
            return *known;
        }
        r_tables.seen[size_t(p_lod)].insert(p_cell, count);
        r_tables.regions.resize(size_t(count + 1) * REGION_BYTES);
        encode(r_tables.regions.data() + size_t(count) * REGION_BYTES, p_base.region(p_cell, p_lod));
        return count++;
    };
    for (uint32_t i = 0; i < p_count; i++) {
        VoxelPos anchor;
        int lod = 0;
        GPUFrame::decode_item(p_items + size_t(i) * GPUFrame::CHUNK_INPUT_BYTES, param, p_points, anchor, lod);
        if (lod >= switch_lod) {
            continue; // the radius form
        }
        uint32_t *slots = r_tables.index.data() + size_t(i) * SLOTS;
        // The item's box on Λ_lod (a point item's: its point), a cell of which is 2^REGION_SHIFT points per axis.
        const Cascade::Box box = Cascade::item_box(anchor, lod, param);
        const int s = BaseSurface::REGION_SHIFT;
        for (int64_t z = box.lo[2] >> s; z <= box.hi[2] >> s; z++) {
            for (int64_t y = box.lo[1] >> s; y <= box.hi[1] >> s; y++) {
                for (int64_t x = box.lo[0] >> s; x <= box.hi[0] >> s; x++) {
                    const Vector3i64 cell(x, y, z);
                    slots[slot(cell)] = add(cell, lod);
                }
            }
        }
    }
}

ov::str glsl(const BaseSurface &p_base) {
    OV_ERR_FAIL_COND_V_MSG(p_base.kind() != BaseSurface::Kind::SPHERE, ov::str(), "BaseRegions: only a sphere has regions.");
    return ov::replace_all(ov::str(R"glsl(
layout(set = 0, binding = REGION_BINDING, std430) restrict readonly buffer BaseRegions { BaseRegion base_regions[]; };
layout(set = 0, binding = INDEX_BINDING, std430) restrict readonly buffer BaseRegionIndex { uint base_region_index[]; };
uint g_base_item; // the item being sampled: its SLOTS region indices (BaseRegions::build)

// BaseSurface::at on a sphere: the radius form from the switch LOD, below it the curvature form of the point's region,
// the item's slot of its cell (BaseRegions::slot, the cell's parity per axis).
vec4 base_at(WorldPos p, int lod) {
    if (lod >= SWITCH_LOD) {
        return base_radius(p, lod);
    }
    uvec3 b = (p.lo >> uint(lod + REGION_SHIFT)) & 1u;
    uint slot = b.x | (b.y << 1u) | (b.z << 2u);
    return base_curvature(p, lod, base_regions[base_region_index[g_base_item * SLOTSu + slot]]);
}
)glsl"),
            { { "REGION_BINDING", ov::itos(REGION_BINDING) }, { "INDEX_BINDING", ov::itos(INDEX_BINDING) }, { "SWITCH_LOD", ov::itos(p_base.get_switch_lod()) },
                    { "REGION_SHIFT", ov::itos(BaseSurface::REGION_SHIFT) }, { "SLOTS", ov::itos(SLOTS) } });
}

} // namespace BaseRegions
