#pragma once

#include "exact_noise.hpp"
#include "sdf_base.hpp"
#include "sdf_test_rig.hpp"

#include <cmath>

// Placeholder terrain on the exact noise (exact_noise.hpp): mountains (2D fBm heights around an
// absolute datum) carved by sparse 3D cave blobs in a depth band below the surface, with a smooth
// subtraction. It is a function of the absolute position only: the same world point gives the same
// value from any world origin, and the CPU evaluation (sdf()) is bit-identical to the GLSL
// (terrain_sdf), gradient included. Everything is computed in LOD0 voxels and the distance is turned
// into metres last.
//
// Parameters are written in metres and rounded to wavelengths of 2^k voxels in configure(), so
// another voxel size gives another world (the hashed cells are voxels), at about the same scale.
//
// The debug shapes (test rig, zero pillar) are placed relative to the world origin, in front of the
// main_scene spawn: with them on, the SDF follows the origin.
class SDFDummy : public SDFBase {
public:
    // Adds the centimetre-scale test shapes of sdf_test_rig.hpp (set before meshing starts).
    bool test_rig = false;

    // Debug: a square column in front of the main_scene spawn whose flat top lies exactly on a lattice
    // row (multiple of 64 LOD0 voxels above the origin), so the SDF is exactly 0.0 there at LODs 0-6.
    // 1 = exact top, 2 = top 1 cm higher (no exact zero, same signs: must mesh the same quad count).
    int zero_pillar = 0;
    static constexpr float ZERO_PILLAR_X = -52.75f;
    static constexpr float ZERO_PILLAR_Z = -110.0f;
    static constexpr float ZERO_PILLAR_HALF = 4.0f;
    static constexpr float ZERO_PILLAR_TOP_M = 416.0f;

    // Terrain parameters (metres).
    static constexpr double DATUM_M = 300.0;
    static constexpr double HEIGHT_WAVELENGTH_M = 1.0 / 0.0035;
    static constexpr double HEIGHT_AMPLITUDE_M = 50.0;
    static constexpr int HEIGHT_OCTAVES = 5;
    static constexpr double CAVE_WAVELENGTH_M = 1.0 / 0.008;
    static constexpr int CAVE_OCTAVES = 3;
    // Caves only between these depths below the surface. Outside the band the cave field is the
    // distance to the band, so deep rock reads its real depth (the cave noise's own estimate c / |grad c|
    // underestimates it badly, and skirts would turn rock closer than one cell to them into walls).
    static constexpr double CAVE_TOP_M = 8.0;
    static constexpr double CAVE_FLOOR_M = 240.0;
    static constexpr double CAVE_DEPTH_GAIN_PER_M = 0.0025;
    static constexpr double BLEND_M = 5.0; // smoothing radius of the cave subtraction
    // Scale the octaves to the spread of the simplex noise this terrain was tuned with (std 0.36 for
    // one octave, 0.41 for the cave fBm): perlin2 has std 0.48, the perlin3 cave fBm 0.31.
    static constexpr double NORM_2D = 0.75;
    static constexpr double NORM_3D = 4.0 / 3.0;
    static constexpr uint32_t HEIGHT_SEED = 0x7E7A11u;
    static constexpr uint32_t CAVE_SEED = 0xCA7E5u;
    // Saturation of the vertical distance to the datum (voxels): exact float conversion. A saturated
    // |SDF| only makes chunk discovery keep more chunks, never drop one.
    static constexpr int32_t DY_LIMIT = 1 << 24;
    // Saturation of the debug shapes' origin-relative position (voxels).
    static constexpr int32_t DEBUG_LIMIT = 1 << 20;

    SDFDummy() = default;
    ~SDFDummy() override = default;

    void configure(const ChunkParam &p_param) override {
        SDFBase::configure(p_param);
        const double vs = p_param.voxel_size;
        auto k_of = [vs](double p_wavelength_m) { return int(std::lround(std::log2(p_wavelength_m / vs))); };
        height = ExactNoise::make_fbm(true, HEIGHT_SEED, k_of(HEIGHT_WAVELENGTH_M), HEIGHT_OCTAVES, HEIGHT_AMPLITUDE_M / vs * NORM_2D, 0.5);
        caves = ExactNoise::make_fbm(false, CAVE_SEED, k_of(CAVE_WAVELENGTH_M), CAVE_OCTAVES, NORM_3D, 0.5);
        datum = VoxelPos(0, std::llround(DATUM_M / vs), 0);
        cave_top = float(CAVE_TOP_M / vs);
        cave_floor = float(CAVE_FLOOR_M / vs);
        cave_gain = float(CAVE_DEPTH_GAIN_PER_M * vs);
        cave_eps2 = float((1e-4 * vs) * (1e-4 * vs));
        blend = float(BLEND_M / vs);
        inv_blend = float(vs / BLEND_M);
        voxel_size = p_param.voxel_size;
        debug_anchor = p_param.origin_voxel();
        pillar_top = float(std::llround(ZERO_PILLAR_TOP_M / vs / 64.0) * 64) + (zero_pillar == 2 ? float(0.01 / vs) : 0.0f);
        pillar_x = float(ZERO_PILLAR_X / vs);
        pillar_z = float(ZERO_PILLAR_Z / vs);
        pillar_half = float(ZERO_PILLAR_HALF / vs);
    }

    // The two fBms, GLSL "terrain_height" / "terrain_caves" (FrameSpike probes them).
    const ExactNoise::OctaveSet &get_height() const { return height; }
    const ExactNoise::OctaveSet &get_caves() const { return caves; }

    float evaluate(const WorldPos &p_pos) const override {
        const ExactNoise::Float4 d = sdf(p_pos);
        return test_rig ? MIN(d.w, SDFTestRig::evaluate(debug_metres(p_pos))) : d.w;
    }

    // Gradient (normalised) and distance in metres: the CPU side of terrain_sdf without the test rig,
    // bit for bit.
    ExactNoise::Float4 sdf(const WorldPos &p_pos) const {
        using namespace ExactNoise;
        // Height field: surface at datum + h(x, z), the terrain distance normalised by its gradient.
        const Float4 hn = fbm(height, p_pos);
        const float dy = float(WorldCoord::sat_diff(p_pos.voxel.y, datum.y, DY_LIMIT)) + p_pos.frac.y;
        Float4 t{ -hn.x, 1.0f, -hn.z, dy - hn.w };
        const float tr = det_rsqrt(t.x * t.x + t.y * t.y + t.z * t.z);
        const Float4 terrain{ t.x * tr, t.y * tr, t.z * tr, t.w * tr };

        // Caves: carve where the blob noise exceeds a threshold that drops with the distance into the
        // band (tent: positive inside, distance to the band outside, in voxels along the terrain
        // gradient). Positive = carved out.
        const Float4 blob = fbm(caves, p_pos);
        const float depth = hn.w - dy;
        const float below_top = depth - cave_top;
        const float above_floor = cave_floor - depth;
        const bool near_top = below_top <= above_floor;
        const float tent = near_top ? below_top : above_floor;
        // d depth / d p = -(t.x, t.y, t.z).
        const float ts = near_top ? -1.0f : 1.0f;
        const float gs = ts * cave_gain;
        Float4 c{ 0.8f * blob.x + gs * t.x, 0.8f * blob.y + gs * t.y, 0.8f * blob.z + gs * t.z,
            0.8f * blob.w - 0.66f + cave_gain * tent };
        const float cr = det_rsqrt(MAX(c.x * c.x + c.y * c.y + c.z * c.z, cave_eps2));
        const Float4 blobs{ c.x * cr, c.y * cr, c.z * cr, c.w * cr };
        const Float4 band{ ts * terrain.x, ts * terrain.y, ts * terrain.z, tent * tr };
        const Float4 carve = blobs.w < band.w ? blobs : band;

        // Smooth max(terrain, carve) over `blend` voxels.
        const float dw = terrain.w - carve.w;
        const float h = CLAMP(0.5f + 0.5f * (dw * inv_blend), 0.0f, 1.0f);
        const float dist = carve.w + dw * h + blend * h * (1.0f - h);
        const float gx = carve.x + (terrain.x - carve.x) * h;
        const float gy = carve.y + (terrain.y - carve.y) * h;
        const float gz = carve.z + (terrain.z - carve.z) * h;
        const float gr = det_rsqrt(MAX(gx * gx + gy * gy + gz * gz, exact_pow2(-40)));
        Float4 r{ gx * gr, gy * gr, gz * gr, dist };

        if (zero_pillar) {
            r = pillar_min(r, p_pos);
        }
        return Float4{ r.x, r.y, r.z, r.w * voxel_size };
    }

    String get_glsl_code() const override {
        using namespace ExactNoise;
        String code = glsl_fbm("terrain_height", height) + glsl_fbm("terrain_caves", caves);
        String body = R"glsl(
vec4 terrain_sdf(WorldPos p) {
    precise vec4 hn = terrain_height(p);
    int sy = sat_diff(p, DATUM_LO, DATUM_HI, DY_LIMIT).y;
    precise float dy = float(sy) + p.frac.y;
    precise vec4 t = vec4(-hn.x, 1.0, -hn.z, dy - hn.w);
    precise float tr = det_rsqrt(t.x * t.x + t.y * t.y + t.z * t.z);
    precise vec4 terrain = t * tr;

    precise vec4 blob = terrain_caves(p);
    precise float depth = hn.w - dy;
    precise float below_top = depth - CAVE_TOP;
    precise float above_floor = CAVE_FLOOR - depth;
    bool near_top = below_top <= above_floor;
    precise float tent = near_top ? below_top : above_floor;
    precise float ts = near_top ? -1.0 : 1.0;
    precise float gs = ts * CAVE_GAIN;
    precise vec4 c = vec4(0.8 * blob.xyz + gs * t.xyz, 0.8 * blob.w - 0.66 + CAVE_GAIN * tent);
    precise float cr = det_rsqrt(max(c.x * c.x + c.y * c.y + c.z * c.z, CAVE_EPS2));
    precise vec4 blobs = c * cr;
    precise vec4 band = vec4(ts * terrain.xyz, tent * tr);
    precise vec4 carve = blobs.w < band.w ? blobs : band;

    precise float dw = terrain.w - carve.w;
    precise float h = clamp(0.5 + 0.5 * (dw * INV_BLEND), 0.0, 1.0);
    precise float dist = carve.w + dw * h + BLEND * h * (1.0 - h);
    precise vec3 g = carve.xyz + (terrain.xyz - carve.xyz) * h;
    precise float gr = det_rsqrt(max(g.x * g.x + g.y * g.y + g.z * g.z, uintBitsToFloat(0x2B800000u)));
    precise vec4 r = vec4(g * gr, dist);
    /*ZERO_PILLAR*/
    r.w = r.w * VOXEL_SIZE;
    /*TEST_RIG*/
    return r;
}
)glsl";
        body = body.replace("DATUM_LO", glsl_lo(datum)).replace("DATUM_HI", glsl_hi(datum)).replace("DY_LIMIT", itos(DY_LIMIT));
        body = body.replace("CAVE_TOP", glsl_float(cave_top)).replace("CAVE_FLOOR", glsl_float(cave_floor)).replace("CAVE_GAIN", glsl_float(cave_gain));
        body = body.replace("CAVE_EPS2", glsl_float(cave_eps2)).replace("INV_BLEND", glsl_float(inv_blend));
        body = body.replace("BLEND", glsl_float(blend)).replace("VOXEL_SIZE", glsl_float(voxel_size));
        body = body.replace("0.8 *", glsl_float(0.8f) + " *").replace("- 0.66", "- " + glsl_float(0.66f));
        code += String(R"glsl(
// Debug shapes: position relative to the world origin, in voxels.
vec3 debug_voxels(WorldPos p) {
    precise vec3 v = vec3(sat_diff(p, DEBUG_LO, DEBUG_HI, DEBUG_LIMIT)) + p.frac;
    return v;
}
)glsl")
                        .replace("DEBUG_LIMIT", itos(DEBUG_LIMIT))
                        .replace("DEBUG_LO", glsl_lo(debug_anchor))
                        .replace("DEBUG_HI", glsl_hi(debug_anchor));
        if (zero_pillar) {
            code += String(R"glsl(
vec4 zero_pillar_min(vec4 r, WorldPos p) {
    precise vec3 v = debug_voxels(p);
    precise vec4 d = vec4(0.0, 1.0, 0.0, v.y - TOP);
    precise float ox = v.x - PX;
    precise float oz = v.z - PZ;
    precise float dx = abs(ox) - HALF;
    precise float dz = abs(oz) - HALF;
    if (dx > d.w) d = vec4(ox < 0.0 ? -1.0 : 1.0, 0.0, 0.0, dx);
    if (dz > d.w) d = vec4(0.0, 0.0, oz < 0.0 ? -1.0 : 1.0, dz);
    return d.w < r.w ? d : r;
}
)glsl")
                            .replace("TOP", glsl_float(pillar_top))
                            .replace("PX", glsl_float(pillar_x))
                            .replace("PZ", glsl_float(pillar_z))
                            .replace("HALF", glsl_float(pillar_half));
            body = body.replace("/*ZERO_PILLAR*/", "r = zero_pillar_min(r, p);");
        }
        if (test_rig) {
            code = String(SDFTestRig::GLSL) + code;
            body = body.replace("/*TEST_RIG*/", "r = sdf_min(r, test_rig_sdf(debug_voxels(p) * VOXEL_SIZE));")
                           .replace("VOXEL_SIZE", glsl_float(voxel_size));
        }
        return code + body;
    }

private:
    ExactNoise::OctaveSet height;
    ExactNoise::OctaveSet caves;
    VoxelPos datum;
    VoxelPos debug_anchor;
    float cave_top = 0.0f;
    float cave_floor = 0.0f;
    float cave_gain = 0.0f;
    float cave_eps2 = 0.0f;
    float blend = 1.0f;
    float inv_blend = 1.0f;
    float voxel_size = 0.5f;
    float pillar_top = 0.0f;
    float pillar_x = 0.0f;
    float pillar_z = 0.0f;
    float pillar_half = 0.0f;

    Float3 debug_voxels(const WorldPos &p_pos) const {
        return Float3{ float(WorldCoord::sat_diff(p_pos.voxel.x, debug_anchor.x, DEBUG_LIMIT)) + p_pos.frac.x,
            float(WorldCoord::sat_diff(p_pos.voxel.y, debug_anchor.y, DEBUG_LIMIT)) + p_pos.frac.y,
            float(WorldCoord::sat_diff(p_pos.voxel.z, debug_anchor.z, DEBUG_LIMIT)) + p_pos.frac.z };
    }

    // Origin-relative metres, for the test rig.
    Vector3 debug_metres(const WorldPos &p_pos) const {
        const Float3 v = debug_voxels(p_pos);
        return Vector3(v.x * voxel_size, v.y * voxel_size, v.z * voxel_size);
    }

    ExactNoise::Float4 pillar_min(const ExactNoise::Float4 &p_r, const WorldPos &p_pos) const {
        const Float3 v = debug_voxels(p_pos);
        ExactNoise::Float4 d{ 0.0f, 1.0f, 0.0f, v.y - pillar_top };
        const float ox = v.x - pillar_x;
        const float oz = v.z - pillar_z;
        const float dx = std::abs(ox) - pillar_half;
        const float dz = std::abs(oz) - pillar_half;
        if (dx > d.w) {
            d = ExactNoise::Float4{ ox < 0.0f ? -1.0f : 1.0f, 0.0f, 0.0f, dx };
        }
        if (dz > d.w) {
            d = ExactNoise::Float4{ 0.0f, 0.0f, oz < 0.0f ? -1.0f : 1.0f, dz };
        }
        return d.w < p_r.w ? d : p_r;
    }
};
