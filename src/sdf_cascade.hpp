#pragma once

#include "cascade.hpp"
#include "cascade_terrain.hpp"
#include "exact_noise.hpp"
#include "sdf_base.hpp"

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
class SDFCascade : public SDFBase {
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

    // Set before configure(). 0: every octave at STEEPNESS (capped at MAX_AMP_M). 1 (continents, demo):
    // the same octaves with an amplitude per wavelength (continent_amp_m), so continents, ocean basins and
    // mountain ranges show at their own scales. Only the amplitudes differ: lattices, seeds and the
    // cascade are the same, so both presets are the same proven code path.
    int preset = 0;

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
            b.amp = preset == 1 ? float(continent_amp_m(std::ldexp(vs, b.k)) / vs) : float(MIN(double(b.amp), cap));
        }
        datum = VoxelPos(0, std::llround(DATUM_M / vs), 0);
        voxel_size = p_param.voxel_size;
        // Skirt depth per LOD (metres). A face neighbour is at most one LOD away, so its surface differs
        // from this chunk's by one octave: k = L + 2 (a finer neighbour's extra one) or k = L + 3 (the one
        // a coarser neighbour drops). Twice the larger amplitude (the 2D seam model needs 1.3x), never
        // deeper than the old skirt_depth voxels of the LOD: the groove at the border is depth / 4.
        skirt_depth_table.assign(p_lod_count, 0.0f);
        for (int l = 0; l < p_lod_count; l++) {
            double amp = 0.0;
            for (const Cascade::Band &b : spec.bands) {
                if (b.k == l + 2 || b.k == l + 3) {
                    amp = MAX(amp, double(b.amp));
                }
            }
            const double full_m = double(p_param.skirt_depth) * vs * std::ldexp(1.0, l);
            skirt_depth_table[l] = float(MIN(2.0 * amp * vs, full_m));
        }
    }

    const Cascade::Spec *get_cascade() const override { return &spec; }
    float skirt_depth_m(int p_lod) const override { return skirt_depth_table[p_lod]; }
    const Cascade::Spec &get_spec() const { return spec; }
    int64_t get_datum_y() const { return datum.y; }

    // Lattice points only: frac 0 and every coordinate a multiple of 2^lod (ERR otherwise, and 0).
    float evaluate(const WorldPos &p_pos, int p_lod) const override {
        const int64_t mask = (int64_t(1) << p_lod) - 1;
        ERR_FAIL_COND_V_MSG(p_pos.frac.x != 0.0f || p_pos.frac.y != 0.0f || p_pos.frac.z != 0.0f || ((p_pos.voxel.x | p_pos.voxel.y | p_pos.voxel.z) & mask) != 0,
                0.0f, "SDFCascade: the cascade only exists on lattice points (frac 0, voxel a multiple of 2^lod).");
        float ch[Cascade::CHANNELS];
        Cascade::point_into(spec, Vector3i64(p_pos.voxel.x >> p_lod, p_pos.voxel.y >> p_lod, p_pos.voxel.z >> p_lod), p_lod, ch);
        return sdf(ch, p_pos.voxel.y, p_lod).w;
    }

    // evaluate() over the WorkerThreadPool: the CPU discoverer's corner batches (and the GPU discoverer's oracle
    // and fallback).
    void evaluate_batch(const VoxelPos *p_points, int p_count, int p_lod, float *r_out) const override {
        evaluate_batch_parallel(p_points, p_count, p_lod, r_out);
    }

    // CascadeTerrain::height_sdf of the channels of Λ_lod at a point of absolute height p_voxel_y, the
    // distance in metres (GLSL cascade_terrain_sdf, bit for bit).
    ExactNoise::Float4 sdf(const float p_ch[Cascade::CHANNELS], int64_t p_voxel_y, int p_lod) const {
        using namespace ExactNoise;
        const float s = exact_pow2(-p_lod);
        const float dy = float(WorldCoord::sat_diff(p_voxel_y, datum.y, DY_LIMIT)) * s;
        const Float4 h = CascadeTerrain::height_sdf(p_ch, p_lod, dy);
        const float to_m = voxel_size * exact_pow2(p_lod);
        return Float4{ h.x, h.y, h.z, h.w * to_m };
    }

    // The meshers' sdf_sample at vertex-grid point p_grid (integer coordinates) of a LOD-p_lod chunk: sdf()
    // plus the skirt (gpu_glsl.hpp's skirt_sdf formula, written out in the same order, with the depth from
    // skirt_depth_m; cascade_glsl.hpp).
    ExactNoise::Float4 sample(const float p_ch[Cascade::CHANNELS], int64_t p_voxel_y, int p_lod, const Float3 &p_grid) const {
        ExactNoise::Float4 r = sdf(p_ch, p_voxel_y, p_lod);
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

    // terrain_sdf is a NaN stub: the meshers' templates reference it, but in cascade mode the sdf pass
    // and the vertex normals are replaced by the cascade passes (a stray call would show as NaN vertices).
    // cascade_terrain_sdf is sdf() for the cascade passes' last level.
    String get_glsl_code() const override {
        using namespace ExactNoise;
        String code = R"glsl(
// SDFCascade: lattice points only, see cascade_terrain_sdf and the cascade passes.
vec4 terrain_sdf(WorldPos p, int lod) {
    return vec4(uintBitsToFloat(0x7FC00000u));
}

// CascadeTerrain::height_sdf -> metres (SDFCascade::sdf). c0 = (value, d/dx, d/dy, d/dz) on Λ_lod at p.
vec4 cascade_terrain_sdf(vec4 c0, WorldPos p, int lod) {
    precise float s = exact_pow2(-lod);
    int sy = sat_diff(p, DATUM_LO, DATUM_HI, DY_LIMIT).y;
    precise float dy = float(sy) * s;
    precise float h = c0.x * s;
    precise float tx = -(c0.y * s);
    precise float ty = 1.0 - c0.z * s;
    precise float tz = -(c0.w * s);
    precise float tr = det_rsqrt(tx * tx + ty * ty + tz * tz);
    precise vec4 r = vec4(tx * tr, ty * tr, tz * tr, (dy - h) * tr);
    precise float to_m = chunk_param.voxel_size * exact_pow2(lod);
    r.w = r.w * to_m;
    return r;
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

private:
    Cascade::Spec spec;
    std::vector<float> skirt_depth_table; // skirt_depth_m per LOD, see configure()
    VoxelPos datum;
    float voxel_size = 0.5f;
};
