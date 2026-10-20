#pragma once

#include "cascade.hpp"
#include "cascade_terrain.hpp"
#include "exact_noise.hpp"
#include "sdf_table.hpp"

#include <cmath>

// The cascade test terrain (CascadeTerrain, PR 2) as the meshed terrain (VoxelTerrain.terrain_mode 1,
// --terrain=cascade): a 3D height field of perlin3 octaves carried down the Hermite cascade, octaves from
// k = 3 (8 voxels, inserted on Λ_0) up to k = lod_count + 4, steepness 1/8, amplitudes capped at 3.3 km,
// 4 channels (value + gradient, see channel_set),
// surface at datum + h with the datum at SDFBase::DATUM_M. A function of the absolute position only.
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
    // Saturation of the height above the datum (LOD0 voxels). A saturated |SDF| only makes chunk
    // discovery keep more chunks, never drop one, but it must stay several cells of the largest LOD.
    static constexpr int32_t DY_LIMIT = 1 << 30;

    // Set before configure(). LINEAR4 (value + gradient) is the meshed set: by the design's criterion G7
    // (keep 8 channels unless they cost > 15% more GPU time than 4), measured with CascadeSpike: the
    // 8-channel passes cost 1.30x the 4-channel ones (1.29 vs 1.00 ms per batch of 128 LOD-0 chunks).
    // HERMITE8 stays implemented and proven (tools/cascade_test.sh checks both).
    Cascade::ChannelSet channel_set = Cascade::ChannelSet::LINEAR4;

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
    static constexpr double FLAT_GROUND_M = DATUM_M - 50.0;

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
        spec = CascadeTerrain::make_spec(SEED, p_lod_count + 4, STEEPNESS, channel_set, INSERT_SHIFT);
        const double cap = MAX_AMP_M / vs;
        for (Cascade::Band &b : spec.bands) {
            b.amp = preset == Preset::CONTINENTS ? float(continent_amp_m(std::ldexp(vs, b.k)) / vs) : float(MIN(double(b.amp), cap));
        }
        if (preset == Preset::FLAT) {
            spec.bands.clear(); // no top lattice either: a chunk's run is its own level
        }
        datum = VoxelPos(0, std::llround((preset == Preset::FLAT ? FLAT_GROUND_M : DATUM_M) / vs), 0);
        voxel_size = p_param.voxel_size;
        build_table(p_lod_count);
        // Skirt depth per LOD (metres). A face neighbour is at most one LOD away, so its surface differs
        // from this chunk's by one octave: k = L + 2 (a finer neighbour's extra one) or k = L + 3 (the one
        // a coarser neighbour drops). Twice the larger amplitude (the 2D seam model needs 1.3x), never deeper than
        // the old skirt_depth voxels of the LOD: the groove at the border is depth / 4. With instances placed, plus
        // their erosion step (SDFTable::min_skirt_depth cells): next to a rock smooth-unioned into the ground both
        // steps happen at the same seam, so the sum. Without, the bare cascade's
        // skirts, bit for bit: a runtime edit on a cascade configured with no instance gets no extra depth. Flat: the
        // hard-CSG gallery's creases differ between LODs by up to a cell, so skirt_depth cells of the LOD.
        const double erode_cells = current()->empty() ? 0.0 : SDFTable::min_skirt_depth(p_param, p_lod_count);
        skirt_depth_table.assign(p_lod_count, 0.0f);
        for (int l = 0; l < p_lod_count; l++) {
            double amp = 0.0;
            for (const Cascade::Band &b : spec.bands) {
                if (b.k == l + 2 || b.k == l + 3) {
                    amp = MAX(amp, double(b.amp));
                }
            }
            const double cell_m = vs * std::ldexp(1.0, l);
            const double ground_m = preset == Preset::FLAT ? double(p_param.skirt_depth) * cell_m : MIN(2.0 * amp * vs, double(p_param.skirt_depth) * cell_m);
            skirt_depth_table[l] = float(ground_m + erode_cells * cell_m);
        }
    }

    const Cascade::Spec *get_cascade() const override { return &spec; }
    float skirt_depth_m(int p_lod) const override { return skirt_depth_table[p_lod]; }
    const Cascade::Spec &get_spec() const { return spec; }
    int64_t get_datum_y() const { return datum.y; }

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
        return sdf(ch, p_pos.voxel.y, p_lod).w;
    }

    // The channels of Λ_lod at LOD0 voxel p_voxel, a multiple of 2^lod (Cascade::point_into).
    void channels_at(const VoxelPos &p_voxel, int p_lod, float r_ch[Cascade::CHANNELS]) const {
        Cascade::point_into(spec, Vector3i64(p_voxel.x >> p_lod, p_voxel.y >> p_lod, p_voxel.z >> p_lod), p_lod, r_ch);
    }

    // CascadeTerrain::height_sdf of the channels of Λ_lod at a point of absolute height p_voxel_y: the unit
    // gradient and the distance in voxels of LOD p_lod (GLSL cascade_terrain_unit, bit for bit).
    ExactNoise::Float4 unit(const float p_ch[Cascade::CHANNELS], int64_t p_voxel_y, int p_lod) const {
        const float s = ExactNoise::exact_pow2(-p_lod);
        const float dy = float(WorldCoord::sat_diff(p_voxel_y, datum.y, DY_LIMIT)) * s;
        return CascadeTerrain::height_sdf(p_ch, p_lod, dy);
    }

    // The bare cascade's SDF, the distance in metres (GLSL cascade_terrain_sdf, bit for bit).
    ExactNoise::Float4 sdf(const float p_ch[Cascade::CHANNELS], int64_t p_voxel_y, int p_lod) const {
        return metres(unit(p_ch, p_voxel_y, p_lod), p_lod);
    }

    // sdf() with the listed instances p_list folded in (GLSL cascade_sample_sdf, bit for bit): the bare sdf() where
    // no instance's box test passes. *r_material (if given, with p_palette: the list entries' palette indices): the
    // sample's palette index, 0 (ground) unless a record wins the fold.
    ExactNoise::Float4 sample_sdf(const Archetype::InstanceSnapshot &p_snap, const uint32_t *p_list, uint32_t p_count,
            const float p_ch[Cascade::CHANNELS], const WorldPos &p_pos, int p_lod, const uint16_t *p_palette = nullptr,
            uint32_t *r_material = nullptr) const {
        const ExactNoise::Float4 u = unit(p_ch, p_pos.voxel.y, p_lod);
        if (r_material) {
            *r_material = 0;
        }
        if (p_count > 0) {
            const float s = ExactNoise::exact_pow2(p_lod);
            bool touched = false;
            const ExactNoise::Float4 r = Archetype::fold_onto(ExactNoise::Float4{ u.x * s, u.y * s, u.z * s, u.w * s }, *p_snap.library, p_snap.records,
                    p_list, p_count, p_pos, p_lod, true, &touched, p_palette, r_material);
            if (touched) {
                return Archetype::finish_table(r, p_lod, p_snap.out_scale, p_snap.normalize);
            }
        }
        return metres(u, p_lod);
    }

    // The meshers' sdf_sample at vertex-grid point p_grid of a LOD-p_lod chunk: p_r (sdf() or sample_sdf()) plus the
    // skirt (the cascade level pass's formula, cascade_glsl.cpp, in the same order, with the depth from skirt_depth_m).
    ExactNoise::Float4 with_skirt(ExactNoise::Float4 r, int p_lod, const Float3 &p_grid) const {
        if (param.skirt_size > 0 && param.skirt_depth > 0.0f) {
            const float lo = float(param.skirt_size);
            const float hi = lo + float(param.chunk_voxel_dimension());
            auto inside = [lo, hi](float p_c) { return MIN(p_c - lo, hi - p_c); };
            const float in = MIN(inside(p_grid.x), MIN(inside(p_grid.y), inside(p_grid.z)));
            const float depth_m = skirt_depth_table[p_lod];
            r.w = r.w + CLAMP(-in, 0.0f, 1.0f) * depth_m;
        }
        return r;
    }

    // The bare cascade's top solid voxel in the column: sphere tracing on the LOD0 distance (whole voxels, at least
    // one per step) until the side flips, then the boundary between the last two points by bisection.
    int64_t ground_y(int64_t p_x, int64_t p_z) const override {
        auto distance = [&](int64_t p_y) {
            float ch[Cascade::CHANNELS];
            channels_at(VoxelPos(p_x, p_y, p_z), 0, ch);
            return unit(ch, p_y, 0).w;
        };
        int64_t y = datum.y;
        float d = distance(y);
        const bool start_air = d > 0.0f;
        int64_t other = y;
        bool flipped = false;
        for (int i = 0; i < 4096 && !flipped; i++) {
            const float step = std::floor(std::fabs(d));
            other = y + (start_air ? -1 : 1) * MAX<int64_t>(1, int64_t(MIN(step, float(DY_LIMIT))));
            d = distance(other);
            flipped = (d > 0.0f) != start_air;
            if (!flipped) {
                y = other;
            }
        }
        ERR_FAIL_COND_V_MSG(!flipped, y, vformat("SDFCascade::ground_y: no surface found in column (%d, %d) within 4096 steps.", p_x, p_z));
        // solid = !(d > 0), the meshers' side rule; lo solid, hi air.
        int64_t lo = start_air ? other : y;
        int64_t hi = start_air ? y : other;
        while (hi - lo > 1) {
            const int64_t mid = lo + (hi - lo) / 2;
            if (distance(mid) > 0.0f) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        return lo;
    }

    // cascade_terrain_sdf is sdf(), for the discoverer's point items and the batches with no instance listed.
    String get_glsl_code() const override {
        using namespace ExactNoise;
        String code = R"glsl(
// CascadeTerrain::height_sdf (SDFCascade::unit). c0 = (value, d/dx, d/dy, d/dz) on Λ_lod at p.
vec4 cascade_terrain_unit(vec4 c0, WorldPos p, int lod) {
    precise float s = exact_pow2(-lod);
    int sy = sat_diff(p, DATUM_LO, DATUM_HI, DY_LIMIT).y;
    precise float dy = float(sy) * s;
    precise float h = c0.x * s;
    precise float tx = -(c0.y * s);
    precise float ty = 1.0 - c0.z * s;
    precise float tz = -(c0.w * s);
    precise float tr = det_rsqrt(tx * tx + ty * ty + tz * tz);
    return vec4(tx * tr, ty * tr, tz * tr, (dy - h) * tr);
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
)glsl";
        String table;
        for (size_t l = 0; l < skirt_depth_table.size(); l++) {
            if (l) {
                table += ", ";
            }
            table += glsl_uint(float_bits(skirt_depth_table[l]));
        }
        return code.replace("DATUM_LO", glsl_lo(datum)).replace("DATUM_HI", glsl_hi(datum)).replace("DY_LIMIT", itos(DY_LIMIT)).replace("SKIRT_N", itos(int64_t(skirt_depth_table.size()))).replace("SKIRT_TABLE", table);
    }

    // The level passes' fold variant (after the table buffers): sample_sdf on the GPU, the instances of input chunk
    // `chunk`'s list folded onto the register.
    String get_sample_fold_glsl() const override {
        const std::shared_ptr<const Archetype::InstanceSnapshot> snap = current();
        return Archetype::generate_fold_glsl(*snap->library, snap->out_scale, snap->normalize) + R"glsl(
vec4 cascade_sample_sdf(vec4 c0, WorldPos p, int lod, uint chunk, out uint material) {
    material = 0u;
    precise vec4 u = cascade_terrain_unit(c0, p, lod);
    g_sdf_chunk = input_chunk[chunk];
    if (g_sdf_chunk.anchor_hi.w > 0) {
        precise float s = exact_pow2(lod);
        bool touched = false;
        precise vec4 r = arch_fold(u * s, p, lod, touched, material);
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

    // A unit register (unit()) with its distance in metres: sdf(), and sample_sdf() where no instance folds.
    ExactNoise::Float4 metres(const ExactNoise::Float4 &p_u, int p_lod) const {
        const float to_m = voxel_size * ExactNoise::exact_pow2(p_lod);
        return ExactNoise::Float4{ p_u.x, p_u.y, p_u.z, p_u.w * to_m };
    }

    Cascade::Spec spec;
    std::vector<float> skirt_depth_table; // skirt_depth_m per LOD, see configure()
    VoxelPos datum;
    float voxel_size = 0.5f;
};
