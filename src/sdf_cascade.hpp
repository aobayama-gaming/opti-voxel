#pragma once

#include "base_surface.hpp"
#include "cascade.hpp"
#include "cascade_terrain.hpp"
#include "climate.hpp"
#include "exact_noise.hpp"
#include "sdf_table.hpp"

#include <cmath>
#include <optional>

// The cascade test terrain (CascadeTerrain, PR 2) as the meshed terrain (VoxelTerrain.terrain_mode 1,
// --terrain=cascade): a 3D height field of perlin3 octaves carried down the Hermite cascade, octaves from
// k = 3 (8 voxels, inserted on Λ_0) up to k = lod_count + 4, steepness 1/8, amplitudes capped at 3.3 km,
// 4 channels (value + gradient, see channel_set),
// surface at altitude h over the base surface (BaseSurface::flat at BaseSurface::DATUM_M). A function of the
// absolute position only.
//
// The cascade only exists on lattice points: a LOD-L chunk's samples are the points of Λ_L, computed by
// the meshers' cascade passes (cascade_glsl.hpp) bit for bit like Cascade::run, and evaluate() takes the
// same points only (frac 0, voxel a multiple of 2^lod), through Cascade::point. There is no off-lattice
// evaluation: the vertex normals between samples come from the samples' gradients (cascade_glsl.hpp).
//
// The instance table (SDFTable: --instances, --trees, runtime edits) is folded onto each sample: the cascade's
// register (unit gradient times 2^L, distance in LOD0 voxels: exact rescales) through Archetype::fold_onto, then
// finish_table, at the lattice points only. A sample no instance's box test passes is the bare cascade's, bit for
// bit (sample_sdf).
class SDFCascade : public SDFTable {
public:
    static constexpr uint32_t SEED = 0xEA27u;
    static constexpr double STEEPNESS = 0.125;
    static constexpr int INSERT_SHIFT = 3;
    // CascadeTerrain::EARTH_MAX_AMP (2^16 voxels) at 5 cm voxels.
    static constexpr double MAX_AMP_M = CascadeTerrain::EARTH_MAX_AMP * 0.05;

    // Set before configure(). LINEAR4 (value + gradient) is the meshed set: by the design's criterion G7
    // (keep 8 channels unless they cost > 15% more GPU time than 4), measured with CascadeSpike: the
    // 8-channel passes cost 1.30x the 4-channel ones (1.29 vs 1.00 ms per batch of 128 LOD-0 chunks).
    // HERMITE8 stays implemented and proven (tools/cascade_test.sh checks both).
    Cascade::ChannelSet channel_set = Cascade::ChannelSet::LINEAR4;

    // Set before configure(): a climate table (Climate::parse's format), empty for none. configure() parses it against
    // this ground and carries its fields in the cascade (Climate::combined_spec), the ground's octaves modulated by the
    // biomes' multipliers; a table it refuses is an ERR and no climate.
    Dictionary climate_data;

    // Set before configure(). DEFAULT: every octave at STEEPNESS (capped at MAX_AMP_M). CONTINENTS:
    // the same octaves with an amplitude per wavelength (continent_amp_m), so continents, ocean basins and
    // mountain ranges show at their own scales. Only the amplitudes differ: lattices, seeds and the
    // cascade are the same, so both presets are the same proven code path. FLAT (the shapes gallery): no
    // band and the ground at FLAT_GROUND_M, a plane y = datum (the same passes, a zero height).
    enum class Preset { DEFAULT,
        CONTINENTS,
        FLAT };
    Preset preset = Preset::DEFAULT;
    // The flat preset's ground: 50 m below the datum, halfway between the page shader's 100 m contour lines. On one
    // (300 m), a flat ground has fwidth ~ 0 and float noise picks the contour shade per triangle: it flickers.
    static constexpr double FLAT_GROUND_M = BaseSurface::DATUM_M - 50.0;

    // Amplitude (metres) of the octave of wavelength p_wavelength_m in the continents preset.
    static double continent_amp_m(double p_wavelength_m) {
        const double km = p_wavelength_m / 1000.0;
        if (km >= 20000.0) {
            return 500.0;
        }
        if (km >= 2000.0) {
            return 2500.0;
        }
        if (km >= 400.0) {
            return 1200.0;
        }
        if (km >= 100.0) {
            return 600.0;
        }
        if (km >= 16.0) {
            return 2200.0; // mountain ranges
        }
        if (km >= 4.0) {
            return 1100.0;
        }
        if (km >= 1.0) {
            return p_wavelength_m / 5.0;
        }
        return p_wavelength_m / 8.0;
    }

    void configure(const ChunkParam &p_param, int p_lod_count) override {
        SDFBase::configure(p_param, p_lod_count);
        const double vs = p_param.voxel_size;
        k_top = p_lod_count + 4;
        spec = CascadeTerrain::make_spec(SEED, k_top, STEEPNESS, channel_set, INSERT_SHIFT);
        const double cap = MAX_AMP_M / vs;
        amp_cap = float(cap);
        for (Cascade::Band &b : spec.bands) {
            b.amp = preset == Preset::CONTINENTS ? float(continent_amp_m(std::ldexp(vs, b.k)) / vs) : float(MIN(double(b.amp), cap));
        }
        if (preset == Preset::FLAT) {
            spec.bands.clear(); // no top lattice either: a chunk's run is its own level
        }
        base = BaseSurface::flat(std::llround((preset == Preset::FLAT ? FLAT_GROUND_M : BaseSurface::DATUM_M) / vs));
        spec.base = base; // a climate's stretched octaves stretch along it
        voxel_size = p_param.voxel_size;
        attach_climate();
        build_table(p_lod_count);
        // Skirt depth per LOD (metres). A face neighbour is at most one LOD away, so its surface differs
        // from this chunk's by one octave: k = L + 2 (a finer neighbour's extra one) or k = L + 3 (the one
        // a coarser neighbour drops). Twice the larger amplitude (the 2D seam model needs 1.3x), never deeper than
        // the old skirt_depth voxels of the LOD: the groove at the border is depth / 4. With instances placed, plus
        // their erosion step (SDFTable::min_skirt_depth cells): next to a rock smooth-unioned into the ground both
        // steps happen at the same seam, so the sum. Without, the bare cascade's
        // skirts, bit for bit: a runtime edit on a cascade configured with no instance gets no extra depth. Flat: the
        // hard-CSG gallery's creases differ between LODs by up to a cell, so skirt_depth cells of the LOD.
        // The moment blobs' own step (SDFTable::MOMENT_SKIRT_CELLS) is added only at the samples a blob touches
        // (moment_skirt_table, sample_sdf): everywhere else the skirt walls stay the band rule's.
        const double erode_cells = current()->empty() ? 0.0 : SDFTable::instance_erode_cells();
        skirt_depth_table.assign(p_lod_count, 0.0f);
        moment_skirt_table.assign(p_lod_count, 0.0f);
        for (int l = 0; l < p_lod_count; l++) {
            double amp = octave_step(l);
            if (has_ground_terms()) {
                amp = Climate::ground_step(*climate, l, amp);
            }
            const double cell_m = vs * std::ldexp(1.0, l);
            const double ground_m = preset == Preset::FLAT ? double(p_param.skirt_depth) * cell_m : MIN(2.0 * amp * vs, double(p_param.skirt_depth) * cell_m);
            skirt_depth_table[l] = float(ground_m + erode_cells * cell_m);
            moment_skirt_table[l] = float(SDFTable::moment_skirt_cells() * cell_m);
        }
    }

    // The ground's band step between LOD p_lod and p_lod + 1 (LOD0 voxels): its largest L+2 / L+3 octave, the climate's
    // largest multiplier of it. The skirts take it, plus the climate's ground terms (Climate::ground_step).
    double octave_step(int p_lod) const {
        double amp = 0.0;
        for (const Cascade::Band &b : spec.bands) {
            if (b.k == p_lod + 2 || b.k == p_lod + 3) {
                amp = MAX(amp, double(climate ? Climate::max_amp(*climate, Climate::GROUND, b) : b.amp));
            }
        }
        return amp;
    }

    const Cascade::Spec *get_cascade() const override { return &spec; }
    // The parsed climate (configure), null without one; why climate_data was refused, empty if it wasn't.
    const Climate::Table *get_climate() const { return climate ? &*climate : nullptr; }
    const String &get_climate_error() const { return climate_error; }

    // Whether ground samples take their biome's material: a climate with fields (the pick reads them; a table without
    // any has its one default biome, drawn as the procedural ground).
    bool has_ground_materials() const { return climate && !climate->fields.empty(); }
    // The palette index of a ground sample (one no record colours; 0 without ground materials): 1 + its biome (the base
    // palette holds biome b's material, Climate::biome_material, at 1 + b), Climate::pick_slots of p_slots, the slots of
    // lattice point p_p of Λ_m with the fields carried. The GPU's cascade_ground_material, bit for bit.
    uint32_t ground_material(const float p_slots[Cascade::SLOTS], const Vector3i64 &p_p, int p_m) const {
        return has_ground_materials() ? uint32_t(1 + Climate::pick_slots(*climate, p_slots, p_p, p_m)) : 0u;
    }
    // The same at LOD0 voxel p_voxel, a multiple of 2^lod.
    uint32_t ground_material_at(const VoxelPos &p_voxel, int p_lod) const {
        const Vector3i64 p(p_voxel.x >> p_lod, p_voxel.y >> p_lod, p_voxel.z >> p_lod);
        float slots[Cascade::SLOTS];
        Cascade::point_slots(spec, p, p_lod, slots, true);
        return ground_material(slots, p, p_lod);
    }

    String get_ground_material_glsl() const override {
        if (!has_ground_materials()) {
            return String();
        }
        return Climate::pick_glsl(*climate, spec) + R"glsl(
// SDFCascade::ground_material: 1 + the sample's biome.
uint cascade_ground_material(Ch ch, WorldPos p) {
    return 1u + climate_biome(ch, p);
}
)glsl";
    }
    // The ground a climate table is checked against and reads (Climate::parse / validate / fields_at), after configure().
    Climate::Ground climate_ground() const { return Climate::Ground{ &spec, double(voxel_size), k_top, amp_cap }; }
    float skirt_depth_m(int p_lod) const override { return skirt_depth_table[p_lod]; }
    const Cascade::Spec &get_spec() const { return spec; }
    int64_t get_datum_y() const { return base.datum_y(); }
    const BaseSurface &get_base() const { return base; }
    UpAxis up_at(const VoxelPos &p_voxel) const override { return base.up_dir(p_voxel); }

    // Lattice points only: frac 0 and every coordinate a multiple of 2^lod (ERR otherwise, and 0).
    float evaluate(const WorldPos &p_pos, int p_lod) const override {
        if (!on_lattice(p_pos, p_lod)) {
            return 0.0f;
        }
        float ch[Cascade::CHANNELS];
        channels_at(p_pos.voxel, p_lod, ch);
        thread_local std::vector<uint32_t> list;
        const std::shared_ptr<const Archetype::InstanceSnapshot> snap = current();
        snap->point_candidates(p_pos.voxel, p_lod, list);
        return sample_sdf(*snap, list.data(), uint32_t(list.size()), ch, p_pos, p_lod).w;
    }

    // The discoverer's corners: the bare cascade, as its GPU point items (a node near an instance is never culled).
    float evaluate_cull(const WorldPos &p_pos, int p_lod) const override {
        if (!on_lattice(p_pos, p_lod)) {
            return 0.0f;
        }
        float ch[Cascade::CHANNELS];
        channels_at(p_pos.voxel, p_lod, ch);
        return sdf(ch, p_pos.voxel, p_lod).w;
    }

    // The ground channels of Λ_lod at LOD0 voxel p_voxel, a multiple of 2^lod: Cascade::point_into, through
    // ground_channels where the climate has ground terms (the fields carried).
    void channels_at(const VoxelPos &p_voxel, int p_lod, float r_ch[Cascade::CHANNELS]) const {
        const Vector3i64 p(p_voxel.x >> p_lod, p_voxel.y >> p_lod, p_voxel.z >> p_lod);
        if (!has_ground_terms()) {
            Cascade::point_into(spec, p, p_lod, r_ch);
            return;
        }
        float slots[Cascade::SLOTS];
        Cascade::point_slots(spec, p, p_lod, slots, true);
        ground_channels(slots, r_ch);
    }

    // Whether the ground at a sample is more than the cascade's channels (Climate::Table::has_ground_terms): the
    // samples then read the fields (carried at every level, CPU and GPU, point items too).
    bool has_ground_terms() const override { return ground_terms; }
    // The ground channels of a point's slots p_slots (the fields carried): Climate::ground_value, or the cascade's
    // ground channels as they are (GLSL: climate_ground, get_ground_glsl).
    void ground_channels(const float p_slots[Cascade::SLOTS], float r_ch[Cascade::CHANNELS]) const {
        for (int c = 0; c < Cascade::CHANNELS; c++) {
            r_ch[c] = c < Cascade::channel_count(spec.set) ? p_slots[c] : 0.0f;
        }
        if (has_ground_terms()) {
            const ExactNoise::Float4 g = Climate::ground_value(*climate, p_slots);
            r_ch[Cascade::VALUE] = g.w;
            r_ch[Cascade::DX] = g.x;
            r_ch[Cascade::DY] = g.y;
            r_ch[Cascade::DZ] = g.z;
        }
    }

    String get_ground_glsl() const override { return has_ground_terms() ? Climate::ground_glsl(*climate, spec) : String(); }
    String get_climate_helpers_glsl() const override {
        return climate && Climate::splices_helpers(*climate, spec) ? Climate::shared_helpers_glsl() : String();
    }

    // CascadeTerrain::height_sdf of the channels of Λ_lod over the base at LOD0 voxel p_voxel: the unit gradient
    // and the distance in voxels of LOD p_lod (GLSL cascade_terrain_unit, bit for bit).
    ExactNoise::Float4 unit(const float p_ch[Cascade::CHANNELS], const VoxelPos &p_voxel, int p_lod) const {
        return CascadeTerrain::height_sdf(p_ch, p_lod, base.at(p_voxel, p_lod));
    }

    // The bare cascade's SDF, the distance in metres (GLSL cascade_terrain_sdf, bit for bit).
    ExactNoise::Float4 sdf(const float p_ch[Cascade::CHANNELS], const VoxelPos &p_voxel, int p_lod) const {
        return metres(unit(p_ch, p_voxel, p_lod), p_lod);
    }

    // sdf() with the listed instances p_list folded in (GLSL cascade_sample_sdf, bit for bit): the bare sdf() where
    // no instance's box test passes. *r_skirt_extra (if given): the moment skirt (metres) where a moment blob's box test
    // passed, else 0.
    ExactNoise::Float4 sample_sdf(const Archetype::InstanceSnapshot &p_snap, const uint32_t *p_list, uint32_t p_count,
            const float p_ch[Cascade::CHANNELS], const WorldPos &p_pos, int p_lod, float *r_skirt_extra = nullptr) const {
        const ExactNoise::Float4 u = unit(p_ch, p_pos.voxel, p_lod);
        if (r_skirt_extra) {
            *r_skirt_extra = 0.0f;
        }
        if (p_count > 0) {
            const float s = ExactNoise::exact_pow2(p_lod);
            bool touched = false, blobs = false;
            const ExactNoise::Float4 r = Archetype::fold_onto(ExactNoise::Float4{ u.x * s, u.y * s, u.z * s, u.w * s }, *p_snap.library, p_snap.records,
                    p_list, p_count, p_pos, p_lod, true, &touched, nullptr, nullptr, &blobs);
            if (r_skirt_extra && blobs) {
                *r_skirt_extra = moment_skirt_table[p_lod];
            }
            if (touched) {
                return Archetype::finish_table(r, p_lod, p_snap.out_scale, p_snap.normalize);
            }
        }
        return metres(u, p_lod);
    }

    // The meshers' sdf_sample at vertex-grid point p_grid of a LOD-p_lod chunk: p_r (sdf() or sample_sdf()) plus the
    // skirt (the cascade level pass's formula, cascade_glsl.cpp, in the same order, with the depth from skirt_depth_m
    // plus p_extra, sample_sdf's moment skirt).
    ExactNoise::Float4 with_skirt(ExactNoise::Float4 r, int p_lod, const Float3 &p_grid, float p_extra = 0.0f) const {
        if (param.skirt_size > 0 && param.skirt_depth > 0.0f) {
            const float lo = float(param.skirt_size);
            const float hi = lo + float(param.chunk_voxel_dimension());
            auto inside = [lo, hi](float p_c) { return MIN(p_c - lo, hi - p_c); };
            const float in = MIN(inside(p_grid.x), MIN(inside(p_grid.y), inside(p_grid.z)));
            const float depth_m = skirt_depth_table[p_lod] + p_extra;
            r.w = r.w + CLAMP(-in, 0.0f, 1.0f) * depth_m;
        }
        return r;
    }

    // The skirt depth (metres) a LOD-p_lod chunk's rim gets at lattice point p_pos: skirt_depth_m plus the moment skirt
    // where a moment blob touches it (tests: the seam step's bound).
    float skirt_depth_at(const WorldPos &p_pos, int p_lod) const {
        const std::shared_ptr<const Archetype::InstanceSnapshot> snap = current();
        thread_local std::vector<uint32_t> list;
        snap->point_candidates(p_pos.voxel, p_lod, list);
        float ch[Cascade::CHANNELS];
        channels_at(p_pos.voxel, p_lod, ch);
        float extra = 0.0f;
        sample_sdf(*snap, list.data(), uint32_t(list.size()), ch, p_pos, p_lod, &extra);
        return skirt_depth_table[p_lod] + extra;
    }

    ExactNoise::Float4 ground_field(const VoxelPos &p_voxel, int p_lod) const override {
        const int64_t half = p_lod > 0 ? int64_t(1) << (p_lod - 1) : 0;
        const VoxelPos at((p_voxel.x + half) >> p_lod << p_lod, (p_voxel.y + half) >> p_lod << p_lod, (p_voxel.z + half) >> p_lod << p_lod);
        float ch[Cascade::CHANNELS];
        channels_at(at, p_lod, ch);
        ExactNoise::Float4 f = unit(ch, at, p_lod);
        const VoxelPos d = p_voxel - at;
        f.w = f.w * ExactNoise::exact_pow2(p_lod) + f.x * float(d.x) + f.y * float(d.y) + f.z * float(d.z);
        return f;
    }

    // The bare cascade's top solid voxel on the line from p_start down the base's up there (the voxel t steps along it,
    // each axis rounded: on the flat base the column): sphere tracing on the LOD0 distance (whole steps, at least one)
    // until the side flips, then the boundary between the last two points by bisection.
    VoxelPos ground_along(const VoxelPos &p_start) const {
        const UpAxis up = base.up_dir(p_start);
        auto at = [&](int64_t p_t) {
            return VoxelPos(p_start.x - std::llround(double(p_t) * up[0]), p_start.y - std::llround(double(p_t) * up[1]),
                    p_start.z - std::llround(double(p_t) * up[2]));
        };
        auto distance = [&](int64_t p_t) { return ground_field(at(p_t), 0).w; };
        int64_t t = 0;
        float d = distance(t);
        const bool start_air = d > 0.0f;
        int64_t other = t;
        bool flipped = false;
        for (int i = 0; i < 4096 && !flipped; i++) {
            const float step = std::floor(std::fabs(d));
            other = t + (start_air ? 1 : -1) * MAX<int64_t>(1, int64_t(MIN(step, float(BaseSurface::ALT_LIMIT))));
            d = distance(other);
            flipped = (d > 0.0f) != start_air;
            if (!flipped) {
                t = other;
            }
        }
        ERR_FAIL_COND_V_MSG(!flipped, at(t), vformat("SDFCascade::ground_along: no surface found from %s within 4096 steps.", WorldCoord::to_string(p_start)));
        // solid = !(d > 0), the meshers' side rule; solid deeper (larger t) than air.
        int64_t solid = start_air ? other : t;
        int64_t air = start_air ? t : other;
        while (solid - air > 1) {
            const int64_t mid = solid - (solid - air) / 2;
            if (distance(mid) > 0.0f) {
                air = mid;
            } else {
                solid = mid;
            }
        }
        return at(solid);
    }

    // The flat world's column (x, z): ground_along from the datum. TODO: the table's scenes still place by columns
    // (SDFTable::ground_y); a sphere needs them on ground_along.
    int64_t ground_y(int64_t p_x, int64_t p_z) const override { return ground_along(VoxelPos(p_x, base.datum_y(), p_z)).y; }

    // cascade_terrain_sdf is sdf(), for the discoverer's point items and the batches with no instance listed.
    String get_glsl_code() const override {
        using namespace ExactNoise;
        String code = base.glsl() + CascadeTerrain::GLSL_HEIGHT_SDF + R"glsl(
// CascadeTerrain::height_sdf over the base (SDFCascade::unit). c0 = (value, d/dx, d/dy, d/dz) on Λ_lod at p.
vec4 cascade_terrain_unit(vec4 c0, WorldPos p, int lod) {
    return height_sdf(c0, base_at(p, lod), lod);
}

// SDFCascade::metres: the distance of a unit register in metres.
vec4 cascade_metres(vec4 u, int lod) {
    precise vec4 r = u;
    precise float to_m = chunk_param.voxel_size * exact_pow2(lod);
    r.w = r.w * to_m;
    return r;
}

// SDFCascade::sdf: the distance in metres.
vec4 cascade_terrain_sdf(vec4 c0, WorldPos p, int lod) {
    return cascade_metres(cascade_terrain_unit(c0, p, lod), lod);
}

// SDFCascade::skirt_depth_m: the skirt depth (metres) of a LOD-lod chunk.
float cascade_skirt_depth_m(int lod) {
    const uint t[SKIRT_N] = uint[SKIRT_N](SKIRT_TABLE);
    return uintBitsToFloat(t[lod]);
}

// The moment skirt (metres) of a LOD-lod chunk, at the samples a moment blob touches (sample_sdf).
float cascade_moment_skirt_m(int lod) {
    const uint t[SKIRT_N] = uint[SKIRT_N](MOMENT_SKIRT_TABLE);
    return uintBitsToFloat(t[lod]);
}
)glsl";
        auto table_text = [](const std::vector<float> &p_table) {
            String table;
            for (size_t l = 0; l < p_table.size(); l++) {
                table += String(l ? ", " : "") + glsl_uint(float_bits(p_table[l]));
            }
            return table;
        };
        return code.replace("SKIRT_N", itos(int64_t(skirt_depth_table.size())))
                .replace("MOMENT_SKIRT_TABLE", table_text(moment_skirt_table))
                .replace("SKIRT_TABLE", table_text(skirt_depth_table));
    }

    // The level passes' fold variant (after the table buffers): sample_sdf on the GPU, the instances of input chunk
    // `chunk`'s list folded onto the register.
    String get_sample_fold_glsl() const override {
        const std::shared_ptr<const Archetype::InstanceSnapshot> snap = current();
        return Archetype::generate_fold_glsl(*snap->library, snap->out_scale, snap->normalize) + R"glsl(
vec4 cascade_sample_sdf(vec4 c0, WorldPos p, int lod, uint chunk, out uint material, out float skirt_extra) {
    material = 0u;
    skirt_extra = 0.0;
    precise vec4 u = cascade_terrain_unit(c0, p, lod);
    g_sdf_chunk = input_chunk[chunk];
    if (g_sdf_chunk.anchor_hi.w > 0) {
        precise float s = exact_pow2(lod);
        bool touched = false;
        bool moments = false;
        precise vec4 r = arch_fold(u * s, p, lod, touched, material, moments);
        if (moments) {
            skirt_extra = cascade_moment_skirt_m(lod);
        }
        if (touched) {
            return arch_finish(r, lod);
        }
    }
    return cascade_metres(u, lod);
}
)glsl";
    }

private:
    // ERR and false unless p_pos is a point of Λ_lod (frac 0, voxel a multiple of 2^lod).
    static bool on_lattice(const WorldPos &p_pos, int p_lod) {
        const int64_t mask = (int64_t(1) << p_lod) - 1;
        ERR_FAIL_COND_V_MSG(p_pos.frac.x != 0.0f || p_pos.frac.y != 0.0f || p_pos.frac.z != 0.0f || ((p_pos.voxel.x | p_pos.voxel.y | p_pos.voxel.z) & mask) != 0,
                false, "SDFCascade: the cascade only exists on lattice points (frac 0, voxel a multiple of 2^lod).");
        return true;
    }

    // The climate_data table on the ground spec (configure, after the ground's bands and voxel size).
    void attach_climate() {
        climate.reset();
        ground_terms = false;
        climate_error = String();
        set_ground_materials({});
        if (climate_data.is_empty()) {
            return;
        }
        Climate::Table table;
        climate_error = Climate::parse(climate_data, climate_ground(), table);
        ERR_FAIL_COND_MSG(!climate_error.is_empty(), "SDFCascade: " + climate_error);
        spec = Climate::combined_spec(table, spec);
        climate = std::move(table);
        ground_terms = climate->has_ground_terms();
        std::vector<std::pair<String, Material::Params>> grounds;
        for (size_t b = 0; has_ground_materials() && b < climate->biomes.size(); b++) {
            grounds.emplace_back("biome:" + climate->biomes[b].name, Climate::biome_material(climate->biomes[b]));
        }
        set_ground_materials(grounds);
    }

    // A unit register (unit()) with its distance in metres: sdf(), and sample_sdf() where no instance folds.
    ExactNoise::Float4 metres(const ExactNoise::Float4 &p_u, int p_lod) const {
        const float to_m = voxel_size * ExactNoise::exact_pow2(p_lod);
        return ExactNoise::Float4{ p_u.x, p_u.y, p_u.z, p_u.w * to_m };
    }

    Cascade::Spec spec;
    std::vector<float> skirt_depth_table; // skirt_depth_m per LOD, see configure()
    std::vector<float> moment_skirt_table; // the moment skirt per LOD (metres, 0 without moment blobs), see configure()
    BaseSurface base; // the ground's reference: flat at the datum (configure)
    float voxel_size = 0.5f;
    int k_top = 0; // the coarsest octave (lod_count + 4)
    std::optional<Climate::Table> climate;
    bool ground_terms = false; // climate->has_ground_terms(), read per sample
    String climate_error;
    float amp_cap = 0.0f; // MAX_AMP_M in LOD0 voxels
};
