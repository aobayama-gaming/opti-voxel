#include "base_regions.hpp"

#include "exact_noise.hpp"
#include "gpu_frame.hpp"

#include "ov/containers.hpp"
#include "ov/on_error.hpp"

namespace BaseRegions {

int slot(const Vector3i64 &p_cell) {
    return int(p_cell.x & 1) | int(p_cell.y & 1) << 1 | int(p_cell.z & 1) << 2;
}

void encode(uint8_t *r_at, const BaseSurface::Region &p_region) {
    for (uint32_t b = 0; b < REGION_BYTES; b += 4) {
        ov::put_u32(r_at + b, 0);
    }
    ov::put_f32(r_at + 0, p_region.d);
    for (int a = 0; a < 3; a++) {
        ov::put_f32(r_at + 16 + 4 * a, p_region.n[a]);
        ov::put_f32(r_at + 32 + 4 * a, p_region.h[a]);
        ov::put_f32(r_at + 48 + 4 * a, p_region.h[3 + a]);
    }
}

void chunk_cells(const VoxelPos &p_anchor, int p_lod, const ChunkParam &p_param, Vector3i64 &r_first, Vector3i64 &r_last) {
    // Cascade::chunk_box: lo_L = anchor / 2^L - skirt, chunk_size points; a cell of Λ_L is 2^REGION_SHIFT of them.
    const int64_t skirt = p_param.skirt_size;
    const int64_t last = p_param.chunk_size - 1;
    const Vector3i64 lo(p_anchor.x >> p_lod, p_anchor.y >> p_lod, p_anchor.z >> p_lod);
    const int s = BaseSurface::REGION_SHIFT;
    r_first = Vector3i64((lo.x - skirt) >> s, (lo.y - skirt) >> s, (lo.z - skirt) >> s);
    r_last = Vector3i64((lo.x - skirt + last) >> s, (lo.y - skirt + last) >> s, (lo.z - skirt + last) >> s);
}

void build(const BaseSurface &p_base, const uint8_t *p_items, uint32_t p_count, const ChunkParam &p_param, bool p_points, Tables &r_tables) {
    r_tables.index.assign(size_t(p_count) * SLOTS, 0);
    r_tables.regions.assign(REGION_BYTES, 0);
    OV_ERR_FAIL_COND_MSG(p_base.kind() != BaseSurface::Kind::SPHERE, "BaseRegions: only a sphere has regions.");
    OV_ERR_FAIL_COND_MSG(!p_points && p_param.chunk_size > (1 << BaseSurface::REGION_SHIFT), "BaseRegions: a chunk wider than a region.");
    const int switch_lod = p_base.get_switch_lod();
    std::vector<ov::HashMap<Vector3i64, uint32_t, Vector3i64Hasher>> seen(static_cast<size_t>(switch_lod));
    uint32_t count = 0;
    auto add = [&](const Vector3i64 &p_cell, int p_lod) {
        if (const uint32_t *known = seen[size_t(p_lod)].getptr(p_cell)) {
            return *known;
        }
        seen[size_t(p_lod)].insert(p_cell, count);
        r_tables.regions.resize(size_t(count + 1) * REGION_BYTES);
        encode(r_tables.regions.data() + size_t(count) * REGION_BYTES, p_base.region(p_cell, p_lod));
        return count++;
    };
    for (uint32_t i = 0; i < p_count; i++) {
        VoxelPos anchor;
        int lod = 0;
        GPUFrame::decode_item(p_items + size_t(i) * GPUFrame::CHUNK_INPUT_BYTES, p_param, p_points, anchor, lod);
        if (lod >= switch_lod) {
            continue; // the radius form
        }
        uint32_t *slots = r_tables.index.data() + size_t(i) * SLOTS;
        if (p_points) {
            const Vector3i64 cell = BaseSurface::region_cell(anchor, lod);
            slots[slot(cell)] = add(cell, lod);
            continue;
        }
        Vector3i64 first, last;
        chunk_cells(anchor, lod, p_param, first, last);
        for (int64_t z = first.z; z <= last.z; z++) {
            for (int64_t y = first.y; y <= last.y; y++) {
                for (int64_t x = first.x; x <= last.x; x++) {
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
