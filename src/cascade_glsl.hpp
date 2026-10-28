#pragma once

#include "cascade.hpp"
#include "cascade_source.hpp"
#include "chunk_math.hpp"
#include "ov/bytes.hpp"
#include "ov/str.hpp"

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
// The fold variant's workgroup: a cube of FOLD_CUBE^3 samples of level 0 (given a chunk size that is a multiple of it,
// else the fold runs per sample), the box arch_fold tests the chunk's list against once per workgroup.
static constexpr uint32_t FOLD_CUBE = 4;
static_assert(FOLD_CUBE * FOLD_CUBE * FOLD_CUBE == LOCAL_SIZE, "CascadeGLSL: a fold workgroup is one cube of samples.");
// The instance table's buffers in the level shader (after the cascade's own bindings 0-6).
static constexpr int TABLE_RECORD_BINDING = 7;
static constexpr int TABLE_LIST_BINDING = 8;
// The per-sample material (chunk palette index) the meshers give create(): uint per sample, next to sdf_sample.
static constexpr int MATERIAL_BINDING = 9;
// A climate with region guards, chunk items: per item a uvec4 (mode, on, fail, 0) the passes own
// (CascadePasses::upload_items), after the base regions' bindings. mode: Cascade::CLIMATE_*; on: the guarded regions on
// at some point of the item's level 1 (bit r, ORed from 0); fail: non-zero where a CLIMATE_REDUCE item had a region on
// outside its mode.
static constexpr int CLIMATE_BINDING = 12;
static constexpr uint32_t CLIMATE_ITEM_BYTES = 16;

// std430 CascadeBand[group_count * BAND_ENTRIES], group g's band of insertion level m at g * BAND_ENTRIES + m:
// factor (amp * 2^(-shift |S|) per derivative order; unused when modulated), offset low words, offset high words,
// info (present, seed, k). At most one band per group and level.
ov::bytes band_table(const Cascade::Spec &p_spec);

// Scratch points per chunk of buffer A (odd levels) and B (even levels >= 2), and vec4s per point (the ground's
// channels, plus 4 per field group where p_range, the passes' field_range, carries any).
uint32_t scratch_points_a(int p_item_size);
uint32_t scratch_points_b(int p_item_size);
uint32_t vec4s_per_point(const Cascade::Spec &p_spec, const Cascade::FieldRange &p_range);
// The scratch buffers A and B of p_sdf's passes for p_items items of p_item_size points per axis (1: point items): what
// CascadePasses allocates, and what sizes a batch of point items.
void scratch_bytes(const CascadeSource *p_sdf, int p_item_size, uint64_t p_items, uint64_t &r_a, uint64_t &r_b);

// Whether p_sdf's samples carry a ground material (its base palette is longer than {GROUND}: the climate's biomes).
bool ground_materials(const CascadeSource *p_sdf);
// Where the passes of p_sdf carry the field groups and the carry: the spec's FieldRange (what its modulation and ground
// terms read), with ground materials, for chunk items (p_points false), the samples' terms too (the pick, derived on the
// level above every sample; the ground's bits are the same whether they are carried or not). Empty without a climate.
Cascade::FieldRange field_range(const CascadeSource *p_sdf, bool p_points);

// The channels a level shader carries: the ground alone (the levels no climate reaches), the ground, the fields and
// the carry (the levels that derive the climate for the level below), or the ground and the carry (a run's last level,
// its samples: what they read is derived on the level above).
enum class View {
    GROUND,
    FIELDS,
    SAMPLES,
};

// The level pass compute shader (one source for every level) for p_sdf's cascade; p_debug: the variant
// whose level 0 also stores its channels in debug_channels; p_points: point items; p_fold: level 0 folds the
// chunks' instance lists (cascade_sample_sdf), its workgroups cubes of FOLD_CUBE^3 samples (level 0 only; p_cubes false,
// for a chunk size that is not a multiple of FOLD_CUBE: per sample, the chunk's own workgroups). Without, it
// is the bare cascade_terrain_sdf: the same bits as the fold over empty lists, for the batches that have none. p_material: level 0 also writes each sample's material
// (the fold's winner's palette index, 0 = ground) to sdf_material at MATERIAL_BINDING; with ground materials
// (ground_materials), for every chunk, a ground sample's being cascade_ground_material's (its biome's).
// p_view: the channels it carries (View; a spec without a climate has only GROUND, which FIELDS equals). p_words: a
// climate with region guards reads and reports the chunks' climate words at CLIMATE_BINDING (chunk items only,
// CascadePasses' p_climate_words).
ov::str level_shader(const CascadeSource *p_sdf, bool p_debug, bool p_points = false, bool p_fold = false, bool p_material = false, View p_view = View::FIELDS,
        bool p_cubes = true, bool p_words = false);

// The point items' ChunkParam: p_param with skirt 0 (with the items' extent 0, an item's box is its point alone).
ChunkParam point_param(const ChunkParam &p_param);

// The sdf_material declaration (uint per sample, indexed like sdf_sample) at p_binding, with p_access (readonly,
// writeonly): the cascade writes it, the meshers' vertex passes read it.
ov::str glsl_material_buffer(int p_binding, const char *p_access);

// Fills /*GLSL_VERTEX_GRADIENT*/ and /*GLSL_ALGORITHM_PASS*/ of a mesher shader (both meshers): the vertex
// passes take their normal from cascade_vertex_gradient, since the cascade has no off-lattice evaluation, and
// their material from cascade_vertex_material (the mesher declares sdf_material).
ov::str fill_mesher_pass(const ov::str &p_shader, const ov::str &p_pass);

} // namespace CascadeGLSL
