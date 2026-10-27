#pragma once

#include "base_surface.hpp"
#include "chunk_math.hpp"

#include "ov/bytes.hpp"
#include "ov/str.hpp"

#include <vector>

// A sphere's curvature-form regions on the GPU (BaseSurface: below its switch LOD a point's base reads the Region of its
// cell, computed on the CPU in double). The cascade passes upload, per batch, the regions its items touch and SLOTS
// region indices per item; the level shader's base_at picks the slot from the point's cell.
//
// A slot is the parity of the region cell on each axis (slot()). A chunk's sample box on Λ_L is chunk_size points, at
// most 2^REGION_SHIFT, so it meets at most two consecutive cells per axis, whose parities differ: the 8 slots name its
// cells without a base cell. A point item fills only its own cell's slot. Items at or above the switch LOD read the
// radius form and get no region. The regions are deduplicated per batch (the cell and the LOD); index 0 is a valid
// region whenever the batch has one (the unused slots point at it).
namespace BaseRegions {

static constexpr int SLOTS = 8;
// std430 BaseRegion (BaseSurface::glsl): d at 0, n at 16, h_diag at 32, h_off at 48.
static constexpr uint32_t REGION_BYTES = 64;
// The cascade level shader's bindings (after CascadeGLSL's 0-9).
static constexpr int REGION_BINDING = 10;
static constexpr int INDEX_BINDING = 11;

// The slot of a region cell: its parity per axis (GLSL base_at).
int slot(const Vector3i64 &p_cell);

// One region in its GPU layout at r_at (REGION_BYTES).
void encode(uint8_t *r_at, const BaseSurface::Region &p_region);

struct Tables {
    std::vector<uint32_t> index; // SLOTS per item
    ov::bytes regions; // REGION_BYTES per region, at least one (zero) region
    uint32_t region_count() const { return uint32_t(regions.size() / REGION_BYTES); }
};

// The regions of p_count items given as their ChunkInput bytes (GPUFrame::encode_chunk_input, or encode_point_item
// with p_points): what the level shader's base_at reads for them. p_param: the chunks' (chunk size, skirt).
void build(const BaseSurface &p_base, const uint8_t *p_items, uint32_t p_count, const ChunkParam &p_param, bool p_points, Tables &r_tables);

// The region cells of a chunk item's sample box on Λ_lod (anchor its minimum voxel), per axis the first and last.
void chunk_cells(const VoxelPos &p_anchor, int p_lod, const ChunkParam &p_param, Vector3i64 &r_first, Vector3i64 &r_last);

// GLSL after BaseSurface::glsl() of a SPHERE: the region buffers, the item global g_base_item (the shader sets it to
// the item being sampled) and `vec4 base_at(WorldPos p, int lod)`, BaseSurface::at bit for bit.
ov::str glsl(const BaseSurface &p_base);

} // namespace BaseRegions
