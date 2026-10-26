#include "cascade_measure.hpp"

#include "base_surface.hpp"
#include "cascade_terrain.hpp"
#include "ov/math.hpp"

#include <chrono>
#include <cmath>

using namespace Cascade;
using ExactNoise::exact_pow2;
using ExactNoise::pcg3d;
using ExactNoise::UVec3;

namespace CascadeMeasure {

double angle_deg(const ExactNoise::Float4 &p_a, const ExactNoise::Float4 &p_b) {
    const double ax = p_a.x, ay = p_a.y, az = p_a.z;
    const double bx = p_b.x, by = p_b.y, bz = p_b.z;
    const double cx = ay * bz - az * by;
    const double cy = az * bx - ax * bz;
    const double cz = ax * by - ay * bx;
    return ov::rad_to_deg(std::atan2(std::sqrt(cx * cx + cy * cy + cz * cz), ax * bx + ay * by + az * bz));
}

void height_normals(const Level &p_level, const Vector3i64 &p_p, const ExactNoise::Float4 &p_perlin, int p_k, int p_steepness_log2,
        ExactNoise::Float4 &r_cascaded, ExactNoise::Float4 &r_analytic) {
    // Amplitude r * 2^k voxels: values * 2^(r + k); perlin slopes per cell * 2^r per voxel, cascade slopes
    // (per voxel, amplitude 1) * 2^(r + k).
    const float amp = exact_pow2(p_steepness_log2 + p_k);
    const float rs = exact_pow2(p_steepness_log2);
    float v[CHANNELS];
    p_level.channels_at(p_p, v);
    const float cc[CHANNELS] = { v[VALUE] * amp, v[DX] * amp, v[DY] * amp, v[DZ] * amp };
    const float ca[CHANNELS] = { p_perlin.w * amp, p_perlin.x * rs, p_perlin.y * rs, p_perlin.z * rs };
    // Over a flat base at altitude 0.
    const ExactNoise::Float4 base = BaseSurface::flat(0).at(VoxelPos(), 0);
    r_cascaded = CascadeTerrain::height_sdf(cc, 0, base);
    r_analytic = CascadeTerrain::height_sdf(ca, 0, base);
}

ChunkId random_chunk(int p_lod, uint32_t p_seed, int p_index) {
    const UVec3 h = pcg3d(UVec3{ p_seed, uint32_t(p_index), uint32_t(p_lod) });
    const UVec3 h2 = pcg3d(h);
    const int64_t range = int64_t(1) << (39 - p_lod);
    auto axis = [range, p_lod](uint32_t p_a, uint32_t p_b) {
        const int64_t n = int64_t(((uint64_t(p_a) << 32) | p_b) % uint64_t(range)) - range / 2;
        return (n * 2 + 1) * (int64_t(1) << p_lod);
    };
    return ChunkId(axis(h.x, h2.x), axis(h.y, h2.y), axis(h.z, h2.z));
}

Fidelity measure_fidelity(const Spec &p_spec, int p_k, int p_boxes, int p_size, uint32_t p_seed) {
    Fidelity f;
    double sq = 0.0;
    const float to_cell = exact_pow2(p_k); // Λ_0 slope (per voxel) -> per cell
    for (int b = 0; b < p_boxes; b++) {
        Spec spec = p_spec;
        spec.top = -1;
        spec.bands = ExactNoise::make_fbm(false, p_seed + uint32_t(b), p_k, 1, 1.0, 0.5).octaves;
        const Band &band = spec.bands[0];
        const UVec3 h = pcg3d(UVec3{ p_seed, uint32_t(b), 0xF1DEu });
        const int64_t base = (b % 4 == 3) ? (int64_t(1) << 50) : 0;
        Box box;
        const uint32_t r[3] = { h.x, h.y, h.z };
        for (int a = 0; a < 3; a++) {
            box.lo[a] = base + int64_t(r[a] % (1u << 30)) - (int64_t(1) << 29);
            box.hi[a] = box.lo[a] + p_size - 1;
        }
        const Level v = run(spec, box, 0);
        for_each_point(box, [&](const Vector3i64 &p_p, int64_t p_i) {
            WorldPos pos;
            pos.voxel = p_p;
            const ExactNoise::Float4 a = ExactNoise::perlin3(pos, p_k, band.seed, band.offset);
            const double dv = double(v.ch[VALUE][p_i]) - double(a.w);
            f.value_max = std::max(f.value_max, std::fabs(dv));
            sq += dv * dv;
            const double ex = double(v.ch[DX][p_i] * to_cell) - a.x;
            const double ey = double(v.ch[DY][p_i] * to_cell) - a.y;
            const double ez = double(v.ch[DZ][p_i] * to_cell) - a.z;
            f.slope_max = std::max(f.slope_max, std::sqrt(ex * ex + ey * ey + ez * ez));
            for (int s = 0; s < STEEPNESS_COUNT; s++) {
                ExactNoise::Float4 tc, ta;
                height_normals(v, p_p, a, p_k, STEEPNESS_LOG2[s], tc, ta);
                f.normal_max_deg[s] = std::max(f.normal_max_deg[s], angle_deg(tc, ta));
            }
            f.points++;
        });
    }
    f.value_rms = f.points > 0 ? std::sqrt(sq / double(f.points)) : 0.0;
    return f;
}

Cost measure_cost(const Spec &p_spec, int p_lod, int p_chunks, uint32_t p_seed, bool p_evaluate) {
    Cost c;
    const ChunkParam param;
    double total = 0.0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < p_chunks; i++) {
        const Box box = chunk_box(random_chunk(p_lod, p_seed, i), param);
        Stats stats;
        count_points(p_spec, box, p_lod, stats);
        if (p_evaluate) {
            run(p_spec, box, p_lod);
        }
        const int64_t t = stats.total();
        total += double(t);
        c.points_max = std::max(c.points_max, t);
        if (i == 0) {
            for (int m = p_lod; m <= MAX_LEVEL && stats.points[m] > 0; m++) {
                c.first_levels.push_back(stats.points[m]);
            }
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    c.points_mean = p_chunks > 0 ? total / p_chunks : 0.0;
    c.ms_per_chunk = p_evaluate && p_chunks > 0 ? ms / p_chunks : 0.0;
    return c;
}

} // namespace CascadeMeasure
