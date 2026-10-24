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
// Items: the meshers' are chunks (32 points per axis, box from the chunk's anchor minus the skirt). Point items
// (create_points: the discoverer's corners, the site walk's readings) are any points of Λ_L at any LODs in one batch
// (GPUFrame::encode_point_item: the point and its LOD, point_frame), with the point instance's ChunkParam (point_param:
// skirt 0), so an item's box is that one point and every level above it at most 2 points per axis (level_bound(1, j));
// the items are flattened over the workgroups' x (no 65,535 limit on y), and Λ_L writes the bare cascade's unit register
// (cascade_terrain_unit, 16 B per point). The meshers' shader text is the same as before point items.
namespace CascadeGLSL {

static constexpr uint32_t BAND_BYTES = 64;
static constexpr int BAND_ENTRIES = Cascade::MAX_LEVEL + 1;
static constexpr uint32_t PUSH_BYTES = 32;
static constexpr uint32_t LOCAL_SIZE = 64;
// The instance table's buffers in the level shader (after the cascade's own bindings 0-6).
static constexpr int TABLE_RECORD_BINDING = 7;
static constexpr int TABLE_LIST_BINDING = 8;
// The per-sample material (chunk palette index) the meshers give create(): uint per sample, next to sdf_sample.
static constexpr int MATERIAL_BINDING = 9;

// std430 CascadeBand[group_count * BAND_ENTRIES], group g's band of insertion level m at g * BAND_ENTRIES + m:
// factor (amp * 2^(-shift |S|) per derivative order; unused when modulated), offset low words, offset high words,
// info (present, seed, k). At most one band per group and level.
PackedByteArray band_table(const Cascade::Spec &p_spec);

// Scratch points per chunk of buffer A (odd levels) and B (even levels >= 2), and vec4s per point (the ground's
// channels, plus 4 per field group where p_range, the passes' field_range, carries any).
uint32_t scratch_points_a(int p_item_size);
uint32_t scratch_points_b(int p_item_size);
uint32_t vec4s_per_point(const Cascade::Spec &p_spec, const Cascade::FieldRange &p_range);

// Whether p_sdf's samples carry a ground material (its base palette is longer than {GROUND}: the climate's biomes).
bool ground_materials(const SDFBase *p_sdf);
// Where the passes of p_sdf carry the field groups: the spec's FieldRange (what its modulation reads), or with ground
// materials, for chunk items (p_points false), every level of every LOD, so level 0 has the fields its pick reads (the
// ground's bits are the same whether they are carried or not). Empty without field groups.
Cascade::FieldRange field_range(const SDFBase *p_sdf, bool p_points);

// The level pass compute shader (one source for every level) for p_sdf's cascade; p_debug: the variant
// whose level 0 also stores its channels in debug_channels; p_points: point items; p_fold: level 0 folds the
// chunks' instance lists (cascade_sample_sdf). Without, it is the bare cascade_terrain_sdf: the same bits as the
// fold over empty lists, for the batches that have none. p_material: level 0 also writes each sample's material
// (the fold's winner's palette index, 0 = ground) to sdf_material at MATERIAL_BINDING; with ground materials
// (ground_materials), for every chunk, a ground sample's being cascade_ground_material's (its biome's).
// p_fields false: the ground alone (a spec with fields: the variant for the levels below its climate floor).
String level_shader(const SDFBase *p_sdf, bool p_debug, bool p_points = false, bool p_fold = false, bool p_material = false, bool p_fields = true);

// The point items' ChunkParam: p_param with skirt 0 (with the items' extent 0, an item's box is its point alone).
ChunkParam point_param(const ChunkParam &p_param);

// The sdf_material declaration (uint per sample, indexed like sdf_sample) at p_binding, with p_access (readonly,
// writeonly): the cascade writes it, the meshers' vertex passes read it.
String glsl_material_buffer(int p_binding, const char *p_access);

// Fills /*GLSL_VERTEX_GRADIENT*/ and /*GLSL_ALGORITHM_PASS*/ of a mesher shader (both meshers): the vertex
// passes take their normal from cascade_vertex_gradient, since the cascade has no off-lattice evaluation, and
// their material from cascade_vertex_material (the mesher declares sdf_material).
String fill_mesher_pass(const String &p_shader, const String &p_pass);

} // namespace CascadeGLSL

// The cascade passes of one mesher on one RenderingDevice: shader, pipeline, scratch and band buffers,
// uniform set. Not thread-safe; use it on the thread that owns the device (render thread for the main one).
class CascadePasses {
public:
    // p_input (ChunkInput[]), p_param_ubo (ChunkParam std140), p_sdf_sample (vec4 per sample): the mesher's.
    // p_max_chunks: chunks per dispatch (scratch size). p_debug: valid for the debug variant, a buffer for
    // the last level's channels (vec4s_per_point vec4s per sample, CascadeSpike); invalid for the meshers' (a
    // 16 B placeholder is bound). p_share_scratch: use that object's scratch buffers instead of new ones.
    // p_table_records / p_table_list: the mesher's instance table buffers (a terrain with a table): with them, a
    // second variant whose level 0 folds the chunks' instance lists (record_level's p_fold). p_sdf_material: the
    // mesher's per-sample material buffer (uint per sample, indexed like sdf_sample), written by level 0 when given.
    bool create(RenderingDevice *p_rd, const SDFBase *p_sdf, const ChunkParam &p_param, uint32_t p_max_chunks, RID p_input, RID p_param_ubo,
            RID p_sdf_sample, RID p_debug = RID(), const CascadePasses *p_share_scratch = nullptr, RID p_table_records = RID(),
            RID p_table_list = RID(), RID p_sdf_material = RID());
    // Point items (CascadePointsGPU): p_input holds GPUFrame::encode_point_item items (any LODs), p_param_ubo the std140
    // point_param, p_values one vec4 per item (SDFCascade::ground_unit's bits).
    bool create_points(RenderingDevice *p_rd, const SDFBase *p_sdf, const ChunkParam &p_point_param, uint32_t p_max_points, RID p_input, RID p_param_ubo,
            RID p_values);
    void free();
    bool is_valid() const { return uniform_set[0].is_valid() || uniform_set[2].is_valid(); }

    // Number of levels a batch whose smallest LOD is p_min_lod runs (j = count - 1 ... 0).
    int level_count(int p_min_lod) const;
    // Records level j for chunks [p_base, p_base + p_count) of the input (p_count <= max chunks: the scratch
    // holds chunk p_base + i at slot i) as one compute list. p_fold: level 0 folds the chunks' instance lists (a
    // batch with any instance listed; needs the table's buffers). Off, the bare cascade: the same bits over empty
    // lists, without the fold's cost. p_lods: the batch's smallest and largest LOD (ChunkMath::lod_range): the
    // fields' variant where Cascade::FieldRange::batch says.
    void record_level(int p_j, uint32_t p_base, uint32_t p_count, const Vector2i &p_lods, bool p_fold = false);
    // Every level, sub-batches of at most max chunks.
    void record_all(const Vector2i &p_lods, uint32_t p_count, bool p_fold = false);

    uint64_t scratch_bytes() const { return scratch_bytes_total; }
    uint32_t get_max_chunks() const { return max_chunks; }
    bool is_points() const { return item_size == 1; }
    RID get_scratch_a() const { return scratch_a; }
    RID get_scratch_b() const { return scratch_b; }

private:
    RenderingDevice *rd = nullptr;
    // [(fields ? 2 : 0) + (fold ? 1 : 0)]: the bare level-0 output or the instance fold (given the table's buffers), the
    // ground alone or with the fields (where the spec's FieldRange carries any).
    static constexpr int VARIANTS = 4;
    RID shader[VARIANTS];
    RID pipeline[VARIANTS];
    RID uniform_set[VARIANTS];
    Cascade::FieldRange fields; // empty: no fields variant
    RID scratch_a;
    RID scratch_b;
    RID bands;
    RID debug_placeholder;
    int top = 0;
    int item_size = 32; // points per axis of an item: the chunk size, or 1 for point items
    uint32_t max_chunks = 0;
    uint64_t scratch_bytes_total = 0;
    bool owns_scratch = true;

    bool _compile(int p_variant, const String &p_name, const String &p_code);
    bool _create(RenderingDevice *p_rd, const SDFBase *p_sdf, int p_item_size, uint32_t p_max_items, RID p_input, RID p_param_ubo, RID p_output,
            RID p_debug, const CascadePasses *p_share_scratch, RID p_table_records, RID p_table_list, RID p_material);
};
