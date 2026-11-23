#pragma once

#include "base_regions.hpp"
#include "base_surface.hpp"
#include "cascade.hpp"
#include "cascade_terrain.hpp"
#include "climate.hpp"
#include "exact_noise.hpp"
#include "graph_compile.hpp"
#include "level_glsl.hpp"
#include "level_program.hpp"
#include "ov/format.hpp"
#include "ov/json.hpp"
#include "ov/math.hpp"
#include "ov/on_error.hpp"
#include "ov/str.hpp"
#include "ov/value.hpp"
#include "sdf_table.hpp"
#include "world_file.hpp"

#include <algorithm>
#include <cmath>

// The meshed terrain: the world's climate compiled into one cascade v2 program (Terrain::compile, docs/cascade-v2.md;
// a world names its ground: no biomes, no program), its output the height h (value + gradient) of a
// surface over the base surface (BaseSurface::flat at BaseSurface::DATUM_M). A function of the absolute position only.
//
// The cascade only exists on lattice points: a LOD-L chunk's samples are the points of Λ_L, computed by
// the meshers' cascade passes (cascade_glsl.hpp) bit for bit like Terrain::run, and evaluate() takes the
// same points only (frac 0, voxel a multiple of 2^lod), through Terrain::point. There is no off-lattice
// evaluation: the vertex normals between samples come from the samples' gradients (cascade_glsl.hpp).
//
// The instance table (SDFTable: the world's instances and trees, runtime edits) is folded onto each sample: the cascade's
// register (unit gradient times 2^L, distance in LOD0 voxels: exact rescales) through Archetype::fold_onto, then
// finish_table, at the lattice points only. A sample no instance's box test passes is the bare cascade's, bit for
// bit (sample_sdf).
class SDFCascade : public SDFTable {
public:
    // The stream's tasks read ground_field: stopped before this class's members go.
    ~SDFCascade() override { stop_streaming(); }

    // Set before configure(): a climate table (the world file's format). configure() compiles it (Terrain::compile).
    // Without biomes, or one it refuses, it is a world error (get_world_error: no program).
    ov::Value climate_data = ov::Value::object();
    // Set before configure(): the program specialised to the climate's values (Terrain::Mode::SPECIALISED), or generic
    // (false, the biome mixer's: its GPU text interprets the program as data, so an edit compiles nothing).
    bool specialize = true;

    // > 0: the ground's base is a sphere of this radius (metres) centred on the voxel (0, 0, 0) (BaseSurface::sphere: a
    // centre on its 2^31-voxel grid, exactly), its +y pole at y = radius; 0: the flat datum.
    double planet_radius_m = 0.0;
    // > 0 with a planet: a torus instead, a ring of this radius (metres) round y, its tube planet_radius_m.
    double planet_ring_m = 0.0;
    // Set before configure(): the planet's seed (Climate::Ground::seed: every note and field band mixed with it; 0 the
    // identity) and its sea level (metres above the base; NAN: no sea, a "sea" node refused).
    uint32_t planet_seed = 0;
    double sea_level_m = NAN;

    // A note's or a field's coarsest octave at p_lod_count LODs (the cascade's top lattice's).
    static int top_octave(int p_lod_count) { return p_lod_count + 4; }

    void configure(const ChunkParam &p_param, int p_lod_count) override {
        SDFBase::configure(p_param, p_lod_count);
        const double vs = p_param.voxel_size;
        k_max = top_octave(p_lod_count);
        base = BaseSurface::flat(std::llround(BaseSurface::DATUM_M / vs));
        voxel_size = p_param.voxel_size;
        skirt_depth_table.clear();
        moment_skirt_table.clear();
        world_error = ov::str();
        if (planet_radius_m > 0.0) {
            ov::str refusal;
            base = BaseSurface::planet(planet_radius_m, vs, planet_ring_m, &refusal);
            if (!refusal.empty()) {
                world_error = ov::format("the planet (radius %.0f m, ring %.0f m): ", planet_radius_m, planet_ring_m) + refusal;
            }
        }
        world_error = world_error.empty() ? attach_climate() : world_error;
        world_error = world_error.empty() ? species_error() : world_error;
        world_error = world_error.empty() ? build_table(p_lod_count) : world_error;
        if (!world_error.empty()) {
            prog = Terrain::LevelProgram(); // no world: every consumer refuses it (get_world_error)
            table = Climate::Table();
            given = false;
            ground_materials = false;
            set_ground_materials({});
            return;
        }
        // Skirt depth per LOD (metres). A face neighbour is at most one LOD away, so its surface differs from this chunk's
        // by the program's step (LevelProgram::step: the output's largest change when the notes of Λ_L are dropped), never
        // deeper than skirt_depth cells of the LOD: the groove at the border is depth / 4. With instances placed, plus their
        // own step: under the band rule its erosion (SDFTable::min_skirt_depth cells: next to a rock smooth-unioned into
        // the ground both steps happen at the same seam, so the sum); kept whole at every LOD (a world's "instance_band":
        // false) no erosion, but hard-CSG edits (has_hard_edits: the shapes gallery on its constant ground, step 0) have
        // creases which differ between LODs by up to a cell: skirt_depth cells of the LOD. The moment blobs' own step
        // (SDFTable::MOMENT_SKIRT_CELLS) is added only at the samples a blob touches (moment_skirt_table, sample_sdf).
        const bool placed = !current()->empty() || get_stream();
        const double instance_cells = !placed ? 0.0
                : get_instance_band()         ? double(SDFTable::instance_erode_cells())
                : SDFTable::has_hard_edits()  ? double(p_param.skirt_depth)
                                              : 0.0;
        skirt_depth_table.assign(p_lod_count, 0.0f);
        moment_skirt_table.assign(p_lod_count, 0.0f);
        for (int l = 0; l < p_lod_count; l++) {
            const double step = size_t(l) < prog.step.size() ? prog.step[size_t(l)] : 0.0;
            const double cell_m = vs * std::ldexp(1.0, l);
            const double ground_m = ov::min(step * vs, double(p_param.skirt_depth) * cell_m);
            skirt_depth_table[l] = float(ground_m + instance_cells * cell_m);
            moment_skirt_table[l] = float(SDFTable::moment_skirt_cells() * cell_m);
        }
    }

    const Terrain::LevelProgram *get_program() const override { return &prog; }
    // The world's parsed climate (configure), null without one (a world error).
    const Climate::Table *get_climate() const { return given ? &table : nullptr; }
    // The scenes' sites read their biome from the world's climate (SDFTable's sources, BiomePick).
    const Climate::Table *placing_climate() const override { return get_climate(); }
    const Terrain::LevelProgram *placing_program() const override { return &prog; }
    // The world's lists, or a biome of climate_data listing instances or features (not compiled yet: min_skirt_depth is
    // asked before configure).
    bool places_instances() const override {
        if (SDFTable::places_instances()) {
            return true;
        }
        const ov::Value biomes = climate_data.get("biomes", ov::Value::array());
        for (size_t i = 0; biomes.is_array() && i < biomes.size(); i++) {
            const ov::Value &b = biomes[i];
            for (const char *key : { "instances", "features" }) {
                if (b.is_object() && b.has(key) && !b[key].is_empty()) {
                    return true;
                }
            }
        }
        return false;
    }
    // Why configure() built no world, empty if it built the one it was given: a refused planet, climate (or none) or
    // species, or edit (SDFTable::set_edits). Then there is no program, no table and no skirts: nothing may sample it
    // (World::start, ClimateReport, CascadeCheck refuse it).
    const ov::str &get_world_error() const { return world_error; }
    // p_world's shape, seed, sea, climate, species, instances (streamed), features and their region, band rule and edits,
    // for the next configure() (World::start sets these, then its own overrides: the start's edits, the stream sources).
    void set_world(const WorldFile &p_world) {
        planet_radius_m = p_world.planet_radius_m;
        planet_ring_m = p_world.planet_ring_m;
        planet_seed = p_world.seed;
        sea_level_m = p_world.sea.on ? p_world.sea.level_m : NAN;
        set_site_floor_m(p_world.sea.on ? p_world.sea.level_m + p_world.sea.shore_above_m : -INFINITY);
        climate_data = p_world.climate_table;
        set_species(p_world.species);
        set_instances(p_world.instances);
        set_features(p_world.features);
        set_feature_region(p_world.feature_region);
        set_instance_band(p_world.instance_band);
        set_edits(p_world.edits);
    }
    // Configured on p_world (set_world), its voxel size and LOD count (r_param.voxel_size set): get_world_error.
    const ov::str &configure_world(const WorldFile &p_world, ChunkParam &r_param) {
        set_world(p_world);
        r_param.voxel_size = float(p_world.voxel_m);
        configure(r_param, p_world.lods);
        return world_error;
    }
    // The relief the sky stays above (metres: AtmosphereMedium, CloudLayer): the program's top, its output's upper
    // bound over the base (a bound, not the highest point drawn); 0 for a ground that never rises above its base.
    double ground_top_m() const { return prog.hi.empty() ? 0.0 : ov::max(prog.hi[size_t(prog.output)] * double(voxel_size), 0.0); }

    // Whether ground samples take their biome's material: a climate with a look of its own (several biomes, or one with
    // a `surface`; a fieldless one's too). One biome on the default surface (the template ground) is the procedural
    // ground already: drawn as it, no palette or pick channel.
    bool has_ground_materials() const { return ground_materials; }
    // The palette index of a ground sample (one no record colours; 0 without ground materials): 1 + its biome p_pick (the
    // base palette holds biome b's material, Climate::biome_material, at 1 + b), the program's pick at the sample. The
    // GPU's cascade_ground_material, bit for bit.
    uint32_t ground_material(int p_pick) const { return has_ground_materials() ? uint32_t(1 + p_pick) : 0u; }

    ov::str get_ground_material_glsl() const override {
        if (!has_ground_materials()) {
            return ov::str();
        }
        return R"glsl(
// SDFCascade::ground_material: 1 + the sample's biome (Terrain::level_run's pick).
uint cascade_ground_material(float pick) {
    return 1u + uint(pick);
}
)glsl";
    }

    // The climate at point p_p of Λ_m (a run down to m): the ground, the pick, each field's and biome weight's value
    // (r_fields[f], r_w[b], 0 past the table's) (ClimateReport).
    void climate_at(const Vector3i64 &p_p, int p_m, ExactNoise::Float4 &r_ground, int &r_pick, ExactNoise::Float4 r_fields[Climate::MAX_FIELDS],
            ExactNoise::Float4 r_w[Climate::MAX_BIOMES]) const {
        std::vector<int> targets = { prog.output, prog.pick };
        targets.insert(targets.end(), prog.fields.begin(), prog.fields.end());
        targets.insert(targets.end(), prog.weights.begin(), prog.weights.end());
        std::vector<ExactNoise::Float4> v(targets.size());
        Terrain::point_nodes(prog, p_p, p_m, targets.data(), int(targets.size()), v.data());
        r_ground = v[0];
        r_pick = int(v[1].w);
        for (int f = 0; f < Climate::MAX_FIELDS; f++) {
            r_fields[f] = size_t(f) < prog.fields.size() ? v[2 + size_t(f)] : ExactNoise::Float4{};
        }
        for (int b = 0; b < Climate::MAX_BIOMES; b++) {
            r_w[b] = size_t(b) < prog.weights.size() ? v[2 + prog.fields.size() + size_t(b)] : ExactNoise::Float4{};
        }
    }
    // What a climate table is compiled for (Terrain::compile), after configure(): the voxel, the LOD count's octaves, the
    // base's limits (a curved base's centre under the ground) and poles, the planet's seed and sea level.
    Climate::Ground climate_ground() const {
        Climate::Ground g;
        g.voxel_size = double(voxel_size);
        g.k_max = k_max;
        g.floor = base.curved() ? -base.get_radius() : -INFINITY;
        g.seed = planet_seed;
        g.sea_level_m = sea_level_m;
        g.pole_level = base.pole_lod();
        return g;
    }
    float skirt_depth_m(int p_lod) const override { return skirt_depth_table[p_lod]; }
    const Terrain::LevelProgram &program() const { return prog; }
    int64_t get_datum_y() const { return base.datum_y(); }
    const BaseSurface &get_base() const override { return base; }
    const BaseSurface *get_curved_base() const override { return base.curved() ? &base : nullptr; }
    UpAxis up_at(const VoxelPos &p_voxel) const override { return base.up_dir(p_voxel); }

    // Lattice points only: frac 0 and every coordinate a multiple of 2^lod (ERR otherwise, and 0).
    float evaluate(const WorldPos &p_pos, int p_lod) const override { return evaluate_at(p_pos, p_lod, false); }
    float evaluate_view(const WorldPos &p_pos, int p_lod) const override { return evaluate_at(p_pos, p_lod, true); }

    // The discoverer's corners: the bare cascade, as its GPU point items (a node near an instance is never culled).
    float evaluate_cull(const WorldPos &p_pos, int p_lod) const override {
        if (!on_lattice(p_pos, p_lod)) {
            return 0.0f;
        }
        float ch[Cascade::CHANNELS];
        channels_at(p_pos.voxel, p_lod, ch);
        return sdf(ch, p_pos.voxel, p_lod).w;
    }

    // The ground channels of Λ_lod at LOD0 voxel p_voxel, a multiple of 2^lod: the program's output there
    // (Terrain::point), in Cascade::Channel order.
    void channels_at(const VoxelPos &p_voxel, int p_lod, float r_ch[Cascade::CHANNELS]) const {
        const ExactNoise::Float4 g = Terrain::point(prog, Vector3i64(p_voxel.x >> p_lod, p_voxel.y >> p_lod, p_voxel.z >> p_lod), p_lod).ground;
        r_ch[Cascade::VALUE] = g.w;
        r_ch[Cascade::DX] = g.x;
        r_ch[Cascade::DY] = g.y;
        r_ch[Cascade::DZ] = g.z;
    }

    // CascadeTerrain::height_sdf of the channels of Λ_lod over the base at LOD0 voxel p_voxel: the unit gradient
    // and the distance in voxels of LOD p_lod (GLSL cascade_terrain_unit, bit for bit).
    ExactNoise::Float4 unit(const float p_ch[Cascade::CHANNELS], const VoxelPos &p_voxel, int p_lod) const {
        return CascadeTerrain::height_sdf(p_ch, p_lod, base.at(p_voxel, p_lod));
    }

    // A unit register (unit()) with its distance in metres at voxel size p_voxel_size (GLSL cascade_metres): sdf(),
    // sample_sdf() where no instance folds, and the GPU discoverer's corners (CascadePointsGPU::cull_values).
    static ExactNoise::Float4 to_metres(const ExactNoise::Float4 &p_u, int p_lod, float p_voxel_size) {
        const float to_m = p_voxel_size * ExactNoise::exact_pow2(p_lod);
        return ExactNoise::Float4{ p_u.x, p_u.y, p_u.z, p_u.w * to_m };
    }

    // The bare cascade's SDF, the distance in metres (GLSL cascade_terrain_sdf, bit for bit).
    ExactNoise::Float4 sdf(const float p_ch[Cascade::CHANNELS], const VoxelPos &p_voxel, int p_lod) const {
        return metres(unit(p_ch, p_voxel, p_lod), p_lod);
    }

    // sdf() with the listed instances p_list of p_records folded in (GLSL cascade_sample_sdf, bit for bit): the bare
    // sdf() where no instance's box test passes. *r_skirt_extra (if given): the moment skirt (metres) where a moment
    // blob's box test passed, else 0.
    ExactNoise::Float4 sample_sdf(const Archetype::InstanceSnapshot &p_snap, const uint32_t *p_list, uint32_t p_count,
            const float p_ch[Cascade::CHANNELS], const WorldPos &p_pos, int p_lod, float *r_skirt_extra = nullptr) const {
        return sample_sdf(p_snap, p_snap.records, p_list, p_count, p_ch, p_pos, p_lod, r_skirt_extra);
    }
    ExactNoise::Float4 sample_sdf(const Archetype::InstanceSnapshot &p_snap, Archetype::RecordView p_records, const uint32_t *p_list, uint32_t p_count,
            const float p_ch[Cascade::CHANNELS], const WorldPos &p_pos, int p_lod, float *r_skirt_extra = nullptr) const {
        const ExactNoise::Float4 u = unit(p_ch, p_pos.voxel, p_lod);
        if (r_skirt_extra) {
            *r_skirt_extra = 0.0f;
        }
        if (p_count > 0) {
            const float s = ExactNoise::exact_pow2(p_lod);
            bool touched = false, blobs = false;
            const ExactNoise::Float4 r = Archetype::fold_onto(ExactNoise::Float4{ u.x * s, u.y * s, u.z * s, u.w * s }, *p_snap.library, p_records,
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
            auto inside = [lo, hi](float p_c) { return ov::min(p_c - lo, hi - p_c); };
            const float in = ov::min(inside(p_grid.x), ov::min(inside(p_grid.y), inside(p_grid.z)));
            const float depth_m = skirt_depth_table[p_lod] + p_extra;
            r.w = r.w + ov::clamp(-in, 0.0f, 1.0f) * depth_m;
        }
        return r;
    }

    // The skirt depth (metres) a LOD-p_lod chunk's rim gets at lattice point p_pos: skirt_depth_m plus the moment skirt
    // where a moment blob touches it (tests: the seam step's bound).
    float skirt_depth_at(const WorldPos &p_pos, int p_lod) const {
        const std::shared_ptr<const Archetype::InstanceSnapshot> snap = current();
        thread_local std::vector<uint32_t> list;
        const Archetype::RecordView records = point_list(*snap, p_pos.voxel, p_lod, list);
        float ch[Cascade::CHANNELS];
        channels_at(p_pos.voxel, p_lod, ch);
        float extra = 0.0f;
        sample_sdf(*snap, records, list.data(), uint32_t(list.size()), ch, p_pos, p_lod, &extra);
        return skirt_depth_table[p_lod] + extra;
    }

    // ground_field's middle step (SDFBase::ground_lattice, extend_field), the GPU's field points' bits
    // (CascadePasses::create_points, CascadePointsGPU): the unit register at a lattice point.
    ExactNoise::Float4 ground_unit(const VoxelPos &p_at, int p_lod) const {
        float ch[Cascade::CHANNELS];
        channels_at(p_at, p_lod, ch);
        return unit(ch, p_at, p_lod);
    }

    ExactNoise::Float4 ground_field(const VoxelPos &p_voxel, int p_lod) const override {
        const VoxelPos at = ground_lattice(p_voxel, p_lod);
        return extend_field(ground_unit(at, p_lod), at, p_voxel, p_lod);
    }

    // The bare cascade's top solid voxel on the line from p_start down the base's up there (the voxel t steps along it,
    // each axis rounded: on the flat base the column): sphere tracing on the distance as LOD p_lod sees it (ground_field;
    // whole steps, at least one) until the side flips, then the boundary between the last two points by bisection: to
    // the voxel at LOD 0; above, to within 2^p_lod voxels, then their distances' linear crossing (a map of Λ_m, the
    // biome mixer's, needs no finer ground than Λ_m's).
    VoxelPos ground_along(const VoxelPos &p_start, int p_lod = 0) const {
        const UpAxis up = base.up_dir(p_start);
        auto at = [&](int64_t p_t) {
            return VoxelPos(p_start.x - std::llround(double(p_t) * up[0]), p_start.y - std::llround(double(p_t) * up[1]),
                    p_start.z - std::llround(double(p_t) * up[2]));
        };
        auto distance = [&](int64_t p_t) { return ground_field(at(p_t), p_lod).w; };
        int64_t t = 0;
        float d = distance(t);
        const bool start_air = d > 0.0f;
        int64_t other = t;
        bool flipped = false;
        for (int i = 0; i < 4096 && !flipped; i++) {
            const float step = std::floor(std::fabs(d));
            other = t + (start_air ? 1 : -1) * ov::max<int64_t>(1, int64_t(ov::min(step, float(BaseSurface::ALT_LIMIT))));
            d = distance(other);
            flipped = (d > 0.0f) != start_air;
            if (!flipped) {
                t = other;
            }
        }
        OV_ERR_FAIL_COND_V_MSG(!flipped, at(t), ov::format("SDFCascade::ground_along: no surface found from %s within 4096 steps.", WorldCoord::to_string(p_start)));
        // solid = !(d > 0), the meshers' side rule; solid deeper (larger t) than air.
        int64_t solid = start_air ? other : t;
        int64_t air = start_air ? t : other;
        while (solid - air > (int64_t(1) << p_lod)) {
            const int64_t mid = solid - (solid - air) / 2;
            if (distance(mid) > 0.0f) {
                air = mid;
            } else {
                solid = mid;
            }
        }
        if (p_lod > 0 && solid - air > 1) { // the crossing between the last two, linear in their distances
            const double d_air = distance(air), d_solid = distance(solid);
            return at(air + std::llround(double(solid - air) * d_air / ov::max(d_air - d_solid, 1e-6)));
        }
        return at(solid);
    }

    VoxelPos ground_under(const VoxelPos &p_start) const override { return ground_along(p_start); }

    // cascade_terrain_sdf is sdf(), for the batches with no instance listed (point items write cascade_terrain_unit; their
    // metres come from to_metres on the CPU).
    ov::str get_glsl_code(Cascade::ShaderParams &r_params) const override {
        using namespace ExactNoise;
        const ov::str regions = base.curved() ? BaseRegions::glsl(base) : ov::str();
        // A stretched note reads base_stretched (flat base only: attach_climate refuses one on a curved base).
        const ov::str stretch = base.curved() ? ov::str(STRETCH_STUBS) : base.glsl_stretch();
        ov::str code = base.glsl() + base.glsl_pole() + stretch + regions + CascadeTerrain::GLSL_HEIGHT_SDF + R"glsl(
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

// SDFCascade::skirt_depth_m: the skirt depth (metres) of a LOD-lod chunk (shader params: it follows the amplitudes).
float cascade_skirt_depth_m(int lod) {
    return SKIRT_PARAM;
}

// The moment skirt (metres) of a LOD-lod chunk, at the samples a moment blob touches (sample_sdf).
float cascade_moment_skirt_m(int lod) {
    const uint t[SKIRT_N] = uint[SKIRT_N](MOMENT_SKIRT_TABLE);
    return uintBitsToFloat(t[lod]);
}
)glsl";
        auto table_text = [](const std::vector<float> &p_table) {
            ov::str table;
            for (size_t l = 0; l < p_table.size(); l++) {
                table += ov::str(l ? ", " : "") + glsl_uint(float_bits(p_table[l]));
            }
            return table;
        };
        return ov::replace_all(code, { { "SKIRT_N", ov::itos(int64_t(skirt_depth_table.size())) }, { "MOMENT_SKIRT_TABLE", table_text(moment_skirt_table) }, { "SKIRT_PARAM", r_params.table(skirt_depth_table, "lod") } });
    }

    // The level passes' fold variant (after the table buffers): sample_sdf on the GPU, the instances of input chunk
    // `chunk`'s list folded onto the register (per workgroup box, or per sample with p_group_size 0).
    ov::str get_sample_fold_glsl(uint32_t p_group_size) const override {
        const std::shared_ptr<const Archetype::InstanceSnapshot> snap = current();
        const ov::str fold = Archetype::generate_fold_glsl(*snap->library, snap->out_scale, snap->normalize, p_group_size) + R"glsl(
vec4 cascade_sample_sdf(vec4 c0, WorldPos p, int lod, WorldPos group_lo, WorldPos group_hi, uint chunk, out uint material, out float skirt_extra) {
    material = 0u;
    skirt_extra = 0.0;
    precise vec4 u = cascade_terrain_unit(c0, p, lod);
    g_sdf_chunk = input_chunk[chunk];
    // The branch and the fold's list read input_chunk itself (chunk is the workgroup's): uniform control flow for the
    // group fold's barriers, also under WGSL's uniformity analysis.
    if (input_chunk[chunk].anchor_hi.w > 0) {
        precise float s = exact_pow2(lod);
        bool touched = false;
        bool moments = false;
        precise vec4 r = arch_fold(u * s, p, FOLD_GROUP lod, FOLD_LIST touched, material, moments);
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
        return ov::replace_all(fold, { { "FOLD_GROUP ", p_group_size ? "group_lo, group_hi, " : "" }, { "FOLD_LIST ", p_group_size ? "input_chunk[chunk].id.w, input_chunk[chunk].anchor_hi.w, " : "" } });
    }

private:
    // evaluate / evaluate_view (p_view: SDFTable::point_list's).
    float evaluate_at(const WorldPos &p_pos, int p_lod, bool p_view) const {
        if (!on_lattice(p_pos, p_lod)) {
            return 0.0f;
        }
        float ch[Cascade::CHANNELS];
        channels_at(p_pos.voxel, p_lod, ch);
        thread_local std::vector<uint32_t> list;
        const std::shared_ptr<const Archetype::InstanceSnapshot> snap = current();
        const Archetype::RecordView records = point_list(*snap, p_pos.voxel, p_lod, list, p_view);
        return sample_sdf(*snap, records, list.data(), uint32_t(list.size()), ch, p_pos, p_lod).w;
    }

    // ERR and false unless p_pos is a point of Λ_lod (frac 0, voxel a multiple of 2^lod).
    static bool on_lattice(const WorldPos &p_pos, int p_lod) {
        const int64_t mask = (int64_t(1) << p_lod) - 1;
        OV_ERR_FAIL_COND_V_MSG(p_pos.frac.x != 0.0f || p_pos.frac.y != 0.0f || p_pos.frac.z != 0.0f || ((p_pos.voxel.x | p_pos.voxel.y | p_pos.voxel.z) & mask) != 0,
                false, "SDFCascade: the cascade only exists on lattice points (frac 0, voxel a multiple of 2^lod).");
        return true;
    }

    // The curved base's stand-ins for base_stretched (no stretched note there: attach_climate).
    static constexpr const char STRETCH_STUBS[] = R"glsl(
WorldPos base_stretched(WorldPos p, int j) {
    return p;
}

vec4 base_stretch_gradient(vec4 g, int j) {
    return g;
}
)glsl";
    // p_data compiled into the program (its table, given or not): empty, or why it can't run here.
    ov::str compile_climate(const ov::Value &p_data, Terrain::Mode p_mode) {
        ov::str err = Terrain::compile(p_data, climate_ground(), p_mode, table, prog);
        const bool stretched = std::any_of(prog.nodes.begin(), prog.nodes.end(), [](const Terrain::Node &p_n) { return p_n.op == Terrain::Op::NOTE && p_n.band.stretch > 0; });
        // TODO: a stretched note reads the base at its own level, whose regions the GPU doesn't have (BaseRegions
        // uploads the chunk's LOD only); upload those too to lift this.
        if (err.empty() && base.curved() && stretched) {
            err = "a stretched note (stretch_y) on a curved base has no GPU path yet";
        }
        ov::str layout_error;
        if (err.empty()) {
            Terrain::gpu_layout(prog, layout_error);
        }
        return err.empty() ? layout_error : err;
    }

    // The program (configure, after the base and voxel size): climate_data's. Empty, or why it can't run here (no biomes:
    // a world names its ground): never another ground instead.
    ov::str attach_climate() {
        given = false;
        ground_materials = false;
        set_ground_materials({});
        ov::str err;
        if (climate_data.get("biomes", ov::Value::array()).is_empty()) {
            err = ov::str("a world names its ground: give it biomes, e.g. \"biomes\": \"") + WorldFile::GROUND_BIOMES + "\"";
        } else {
            err = compile_climate(climate_data, specialize ? Terrain::Mode::SPECIALISED : Terrain::Mode::GENERIC);
            given = err.empty();
        }
        if (!err.empty()) {
            return err;
        }
        prog.base = base; // compiled on the flat datum 0: the program's stretch and `up` read this one
        const Climate::Surface plain;
        const Climate::Surface &first = table.biomes[0].surface;
        ground_materials = given && (table.biomes.size() > 1 || !(first.flat == plain.flat && first.steep == plain.steep && first.roughness == plain.roughness));
        std::vector<std::pair<ov::str, Material::Params>> grounds;
        for (size_t b = 0; has_ground_materials() && b < table.biomes.size(); b++) {
            grounds.emplace_back("biome:" + table.biomes[b].name, Climate::biome_material(table.biomes[b]));
        }
        set_ground_materials(grounds);
        return ov::str();
    }

    ExactNoise::Float4 metres(const ExactNoise::Float4 &p_u, int p_lod) const { return to_metres(p_u, p_lod, voxel_size); }

    Terrain::LevelProgram prog;
    std::vector<float> skirt_depth_table; // skirt_depth_m per LOD, see configure()
    std::vector<float> moment_skirt_table; // the moment skirt per LOD (metres, 0 without moment blobs), see configure()
    BaseSurface base; // the ground's reference: flat at the datum, or the planet (configure)
    float voxel_size = 0.5f;
    int k_max = 0; // a note's or a field's coarsest octave (lod_count + 4)
    Climate::Table table; // the climate's fields and biomes (places, looks, contents): climate_data's
    bool given = false; // table is climate_data's (get_climate)
    bool ground_materials = false; // has_ground_materials (attach_climate)
    ov::str world_error; // get_world_error
};
