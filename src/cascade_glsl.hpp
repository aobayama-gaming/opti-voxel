#pragma once

#include "cascade.hpp"
#include "chunk_math.hpp"
#include "sdf_base.hpp"

#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/string.hpp>

// The Hermite cascade on the GPU (shared by GPUDirectMesher, GPUMesher and CascadeSpike), bit for bit what
// Cascade::run computes on the CPU.
//
// One compute dispatch per level, every chunk of the batch in the same dispatch: level m = L + j of each
// chunk (its LOD L), j = J ... 0 with J = T - (the batch's smallest LOD). A chunk whose level m is above
// the spec's top T has nothing there (all zero: H(0) = +0 and 0 + o = o, so starting a chunk at T instead
// of the batch's highest level changes no bit) and returns. The level-m box of a chunk comes from its
// anchor in closed form, lo_m = floor(lo_L / 2^j), hi_m = ceil(hi_L / 2^j) (Cascade::level_box), on (lo, hi)
// words (GLSL_FRAME_INT64); a box has at most Cascade::level_bound(32, j) points per axis (17, 10, 6, ...).
//
// A child point is computed from its 2x2x2 parent corners, replaying the CPU's three axis passes on the
// same inputs in the same order: the x pass at the (up to) 4 (y, z) parent positions, the y pass at the
// 2 z positions, the z pass once (copy at even indices, midpoint at odd ones). Same ops on the same values,
// so the same bits as the CPU's box-wide passes, with no intermediate box stored. Then the level's band is
// inserted (perlin3_cross times the band factors, ch = ch + o) and every channel flushed below 2^-64, like
// Cascade::run.
//
// Ping-pong scratch: odd j in buffer A (17^3 points per chunk), even j >= 2 in buffer B (10^3), 8 channels =
// 2 vec4 per point (4 channels: 1). j = 0 (Λ_L) stores no channel: it converts straight to the mesher's
// sdf_sample (SDF cascade_terrain_sdf -> metres -> skirt), the same 16 B per sample as the sdf pass it
// replaces. The bands reach the GPU as a table indexed by level (band_table), so the shader does not change
// with the spec. Debug variant (CascadeSpike): the last level also stores its channels; the meshers' variant
// doesn't, so the channels the sample doesn't use are dead code there.
//
// Items: the meshers' are chunks (32 points per axis, box from the chunk's anchor minus the skirt). The
// discoverer's are points (create_points): a point item is a ChunkInput whose anchor is the point itself
// (encode_point_input), with the point instance's ChunkParam (point_param: skirt 0), so its box is that one
// point and every level above it at most 2 points per axis (level_bound(1, j)); the items are flattened over
// the workgroups' x (no 65,535 limit on y), and Λ_L writes the bare cascade_terrain_sdf(...).w, one float per
// point. The meshers' shader text is the same as before point items.
namespace CascadeGLSL {

static constexpr uint32_t BAND_BYTES = 64;
static constexpr int BAND_ENTRIES = Cascade::MAX_LEVEL + 1;
static constexpr uint32_t PUSH_BYTES = 32;
static constexpr uint32_t LOCAL_SIZE = 64;

// std430 CascadeBand[BAND_ENTRIES], indexed by insertion level m: factor (amp * 2^(-shift |S|) per derivative
// order), offset low words, offset high words, info (present, seed, k). At most one band per level.
PackedByteArray band_table(const Cascade::Spec &p_spec);

// Scratch points per chunk of buffer A (odd levels) and B (even levels >= 2), and vec4s per point.
uint32_t scratch_points_a(int p_item_size);
uint32_t scratch_points_b(int p_item_size);
uint32_t vec4s_per_point(Cascade::ChannelSet p_set);

// The level pass compute shader (one source for every level) for p_sdf's cascade; p_debug: the variant
// whose level 0 also stores its channels in debug_channels; p_points: point items.
String level_shader(const SDFBase *p_sdf, bool p_debug, bool p_points = false);

// The point instance's ChunkParam: p_param with skirt 0 and chunk_size = the chunk's voxel dimension, so the
// GPU's anchor (id - lod size) * half chunk stays the meshers' and an item's box is its anchor alone.
ChunkParam point_param(const ChunkParam &p_param);

// Writes point item p_index (r_bytes must hold it): the LOD-p_lod chunk whose anchor is p_point, which must be
// on that LOD's corner grid (a multiple of 2^(lod + 1) half chunks, as the discoverer's node corners are).
// False (and an error) otherwise.
bool encode_point_input(PackedByteArray &r_bytes, int64_t p_index, const VoxelPos &p_point, int p_lod, const ChunkParam &p_point_param);

// Fills /*GLSL_SKIRT_SDF*/ and /*GLSL_ALGORITHM_PASS*/ of a mesher shader (both meshers). With SDFDummy
// exactly as before; with a cascade terrain it adds cascade_vertex_gradient and, in the vertex pass
// (p_vertex_pass), takes the normal from it instead of skirt_sdf: the cascade has no off-lattice evaluation.
// An unmatched normal call leaves the placeholder, so the shader fails to compile (no silent NaN stub).
String fill_mesher_pass(const String &p_shader, const SDFBase *p_sdf, const String &p_pass, bool p_vertex_pass);

} // namespace CascadeGLSL

// The cascade passes of one mesher on one RenderingDevice: shader, pipeline, scratch and band buffers,
// uniform set. Not thread-safe; use it on the thread that owns the device (render thread for the main one).
class CascadePasses {
public:
    // p_input (ChunkInput[]), p_param_ubo (ChunkParam std140), p_sdf_sample (vec4 per sample): the mesher's.
    // p_max_chunks: chunks per dispatch (scratch size). p_debug: valid for the debug variant, a buffer for
    // the last level's channels (vec4s_per_point vec4s per sample, CascadeSpike); invalid for the meshers' (a
    // 16 B placeholder is bound). p_share_scratch: use that object's scratch buffers instead of new ones.
    bool create(RenderingDevice *p_rd, const SDFBase *p_sdf, const ChunkParam &p_param, uint32_t p_max_chunks, RID p_input, RID p_param_ubo,
            RID p_sdf_sample, RID p_debug = RID(), const CascadePasses *p_share_scratch = nullptr);
    // Point items (the discoverer's): p_input holds encode_point_input items, p_param_ubo the std140
    // point_param, p_values one float per item.
    bool create_points(RenderingDevice *p_rd, const SDFBase *p_sdf, const ChunkParam &p_point_param, uint32_t p_max_points, RID p_input, RID p_param_ubo,
            RID p_values);
    void free();
    bool is_valid() const { return uniform_set.is_valid(); }

    // Number of levels a batch whose smallest LOD is p_min_lod runs (j = count - 1 ... 0).
    int level_count(int p_min_lod) const;
    // Records level j for chunks [p_base, p_base + p_count) of the input (p_count <= max chunks: the scratch
    // holds chunk p_base + i at slot i) as one compute list.
    void record_level(int p_j, uint32_t p_base, uint32_t p_count);
    // Every level, sub-batches of at most max chunks.
    void record_all(int p_min_lod, uint32_t p_count);

    uint64_t scratch_bytes() const { return scratch_bytes_total; }
    uint32_t get_max_chunks() const { return max_chunks; }
    bool is_points() const { return item_size == 1; }
    RID get_scratch_a() const { return scratch_a; }
    RID get_scratch_b() const { return scratch_b; }

private:
    RenderingDevice *rd = nullptr;
    RID shader;
    RID pipeline;
    RID scratch_a;
    RID scratch_b;
    RID bands;
    RID debug_placeholder;
    RID uniform_set;
    int top = 0;
    int item_size = 32; // points per axis of an item: the chunk size, or 1 for point items
    uint32_t max_chunks = 0;
    uint64_t scratch_bytes_total = 0;
    bool owns_scratch = true;

    bool _create(RenderingDevice *p_rd, const SDFBase *p_sdf, int p_item_size, uint32_t p_max_items, RID p_input, RID p_param_ubo, RID p_output,
            RID p_debug, const CascadePasses *p_share_scratch);
};
