#include "base_surface.hpp"

#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <cmath>

using ExactNoise::exact_pow2;
using ExactNoise::Float4;

namespace {

constexpr int32_t REGION_HALF = 1 << (BaseSurface::REGION_SHIFT - 1);

// p_v - p_c >> p_lod saturated to RADIUS_LIMIT (GLSL: i64_sub, i64_asr, sat_words).
float radius_axis(int64_t p_v, int64_t p_c, int p_lod) {
    return float(WorldCoord::sat_diff((p_v - p_c) >> p_lod, 0, BaseSurface::RADIUS_LIMIT));
}

// p - p_c in double (an exact int64 difference, then rounded) and its length.
double offset(const VoxelPos &p_p, const VoxelPos &p_c, double r_v[3]) {
    r_v[0] = double(p_p.x - p_c.x);
    r_v[1] = double(p_p.y - p_c.y);
    r_v[2] = double(p_p.z - p_c.z);
    return std::sqrt(r_v[0] * r_v[0] + r_v[1] * r_v[1] + r_v[2] * r_v[2]);
}

// δ of a point in its region, voxels of LOD p_lod in [-32, 32): its low bits within the region, from the centre.
float region_delta(int64_t p_v, int p_lod) {
    const uint64_t mask = (uint64_t(1) << (p_lod + BaseSurface::REGION_SHIFT)) - 1;
    return float(int32_t((uint64_t(p_v) & mask) >> p_lod) - REGION_HALF);
}

} // namespace

BaseSurface BaseSurface::flat(int64_t p_datum_y) {
    BaseSurface b;
    b.base_kind = Kind::FLAT;
    b.anchor = VoxelPos(0, p_datum_y, 0);
    return b;
}

BaseSurface BaseSurface::sphere(const VoxelPos &p_centre, double p_radius, int p_switch_lod) {
    ERR_FAIL_COND_V_MSG(p_switch_lod < 0 || p_switch_lod > MAX_SWITCH_LOD, BaseSurface(), vformat("BaseSurface: the switch LOD must be in [0, %d].", MAX_SWITCH_LOD));
    ERR_FAIL_COND_V_MSG(!(p_radius > 0.0) || p_radius > std::ldexp(1.0, 60), BaseSurface(), "BaseSurface: the radius must be in (0, 2^60] voxels.");
    ERR_FAIL_COND_V_MSG(std::ldexp(p_radius, -p_switch_lod) > double(MAX_RADIUS_CELLS), BaseSurface(),
            vformat("BaseSurface: a radius of %f voxels needs a switch LOD of at least %d (the radius form's float error).", p_radius,
                    int(std::ceil(std::log2(p_radius / double(MAX_RADIUS_CELLS))))));
    BaseSurface b;
    b.base_kind = Kind::SPHERE;
    const int64_t snap = int64_t(1) << CENTRE_SHIFT;
    b.anchor = VoxelPos(WorldCoord::floor_div(p_centre.x, snap), WorldCoord::floor_div(p_centre.y, snap), WorldCoord::floor_div(p_centre.z, snap)) * snap;
    b.radius = p_radius;
    b.switch_lod = p_switch_lod;
    return b;
}

Float4 BaseSurface::at(const VoxelPos &p_voxel, int p_lod) const {
    if (base_kind == Kind::FLAT) {
        return flat_form(p_voxel, p_lod);
    }
    if (p_lod < switch_lod) {
        return curvature_form(p_voxel, p_lod, region(region_cell(p_voxel, p_lod), p_lod));
    }
    return radius_form(p_voxel, p_lod);
}

int64_t BaseSurface::datum_y() const {
    ERR_FAIL_COND_V_MSG(base_kind != Kind::FLAT, 0, "BaseSurface: only a flat base has a datum.");
    return anchor.y;
}

UpAxis BaseSurface::up_dir(const VoxelPos &p_voxel) const {
    if (base_kind == Kind::FLAT) {
        return UP_Y;
    }
    double v[3];
    const double len = offset(p_voxel, anchor, v);
    if (len == 0.0) {
        return UP_Y;
    }
    return { v[0] / len, v[1] / len, v[2] / len };
}

double BaseSurface::altitude(const VoxelPos &p_voxel) const {
    if (base_kind == Kind::FLAT) {
        return double(p_voxel.y - anchor.y);
    }
    double v[3];
    return offset(p_voxel, anchor, v) - radius;
}

VoxelPos BaseSurface::tangent_point(int64_t p_u, int64_t p_v) const {
    if (base_kind == Kind::FLAT) {
        return VoxelPos(p_u, anchor.y, p_v);
    }
    return VoxelPos(anchor.x + p_u, anchor.y + std::llround(radius), anchor.z + p_v);
}

Vector3i64 BaseSurface::region_cell(const VoxelPos &p_voxel, int p_lod) {
    const int s = p_lod + REGION_SHIFT;
    return Vector3i64(p_voxel.x >> s, p_voxel.y >> s, p_voxel.z >> s);
}

VoxelPos BaseSurface::region_centre(const Vector3i64 &p_cell, int p_lod) {
    const int s = p_lod + REGION_SHIFT;
    const int64_t half = int64_t(1) << (s - 1);
    return VoxelPos(p_cell.x * (int64_t(1) << s) + half, p_cell.y * (int64_t(1) << s) + half, p_cell.z * (int64_t(1) << s) + half);
}

BaseSurface::Region BaseSurface::region(const Vector3i64 &p_cell, int p_lod) const {
    // The level surface through c: d = |v| - R, n = v / |v|, H = (I - n n^T) / |v|. v is an exact int64 difference.
    const VoxelPos c = region_centre(p_cell, p_lod);
    double v[3];
    const double len = MAX(offset(c, anchor, v), 1.0);
    const double n[3] = { v[0] / len, v[1] / len, v[2] / len };
    const double k = 1.0 / len;
    const double h[6] = { k * (1.0 - n[0] * n[0]), k * (1.0 - n[1] * n[1]), k * (1.0 - n[2] * n[2]), -k * n[0] * n[1], -k * n[0] * n[2], -k * n[1] * n[2] };
    return make_region(len - radius, n, h, p_lod);
}

BaseSurface::Region BaseSurface::make_region(double p_d, const double p_g[3], const double p_h[6], int p_lod) {
    // In voxels of LOD L: d scales by 2^-L, the gradient stays, the Hessian scales by 2^L.
    Region r;
    r.d = float(std::ldexp(p_d, -p_lod));
    for (int a = 0; a < 3; a++) {
        r.n[a] = float(p_g[a]);
    }
    for (int i = 0; i < 6; i++) {
        r.h[i] = float(std::ldexp(p_h[i], p_lod));
    }
    return r;
}

Float4 BaseSurface::curvature_form(const VoxelPos &p_voxel, int p_lod, const Region &p_region) {
    const float dx = region_delta(p_voxel.x, p_lod);
    const float dy = region_delta(p_voxel.y, p_lod);
    const float dz = region_delta(p_voxel.z, p_lod);
    const float *h = p_region.h;
    const float *n = p_region.n;
    const float hx = h[0] * dx + h[3] * dy + h[4] * dz;
    const float hy = h[3] * dx + h[1] * dy + h[5] * dz;
    const float hz = h[4] * dx + h[5] * dy + h[2] * dz;
    const float nd = n[0] * dx + n[1] * dy + n[2] * dz;
    const float q = dx * hx + dy * hy + dz * hz;
    const float alt = p_region.d + nd + 0.5f * q;
    return Float4{ n[0] + hx, n[1] + hy, n[2] + hz, alt };
}

Float4 BaseSurface::radius_form(const VoxelPos &p_voxel, int p_lod) const {
    const float vx = radius_axis(p_voxel.x, anchor.x, p_lod);
    const float vy = radius_axis(p_voxel.y, anchor.y, p_lod);
    const float vz = radius_axis(p_voxel.z, anchor.z, p_lod);
    const float ss = MAX(vx * vx + vy * vy + vz * vz, 1.0f);
    const float r = ExactNoise::det_rsqrt(ss);
    const float rl = float(radius) * exact_pow2(-p_lod);
    return Float4{ vx * r, vy * r, vz * r, ss * r - rl };
}

Float4 BaseSurface::flat_form(const VoxelPos &p_voxel, int p_lod) const {
    // up = (-0, 1, -0): the ground's t = up - grad h then reads -(dh/dx) bit for bit (-0 - x is -x for every x).
    const float s = exact_pow2(-p_lod);
    const float alt = float(WorldCoord::sat_diff(p_voxel.y, anchor.y, ALT_LIMIT)) * s;
    return Float4{ -0.0f, 1.0f, -0.0f, alt };
}

String BaseSurface::glsl() const {
    using namespace ExactNoise;
    if (base_kind == Kind::FLAT) {
        return String(R"glsl(
// BaseSurface::flat_form: (up, altitude) of the plane y = datum, in voxels of LOD lod. up = (-0, 1, -0).
vec4 base_at(WorldPos p, int lod) {
    precise float s = exact_pow2(-lod);
    int sy = sat_diff(p, DATUM_LO, DATUM_HI, ALT_LIMIT).y;
    precise float alt = float(sy) * s;
    return vec4(-0.0, 1.0, -0.0, alt);
}
)glsl")
                .replace("DATUM_LO", glsl_lo(anchor))
                .replace("DATUM_HI", glsl_hi(anchor))
                .replace("ALT_LIMIT", itos(ALT_LIMIT));
    }
    return String(R"glsl(
// BaseSurface::Region: h_diag = (xx, yy, zz), h_off = (xy, xz, yz).
struct BaseRegion { float d; vec3 n; vec3 h_diag; vec3 h_off; };

// BaseSurface::curvature_form: d(c) + n.δ + 1/2 δ^T H δ around the centre of p's region at LOD lod.
vec4 base_curvature(WorldPos p, int lod, BaseRegion r) {
    uvec3 mask = uvec3((1u << uint(lod + REGION_SHIFT)) - 1u);
    ivec3 di = ivec3((p.lo & mask) >> uint(lod)) - ivec3(REGION_HALF);
    precise float dx = float(di.x);
    precise float dy = float(di.y);
    precise float dz = float(di.z);
    precise float hx = r.h_diag.x * dx + r.h_off.x * dy + r.h_off.y * dz;
    precise float hy = r.h_off.x * dx + r.h_diag.y * dy + r.h_off.z * dz;
    precise float hz = r.h_off.y * dx + r.h_off.z * dy + r.h_diag.z * dz;
    precise float nd = r.n.x * dx + r.n.y * dy + r.n.z * dz;
    precise float q = dx * hx + dy * hy + dz * hz;
    precise float alt = r.d + nd + 0.5 * q;
    return vec4(r.n.x + hx, r.n.y + hy, r.n.z + hz, alt);
}

// BaseSurface::radius_form: |p - centre| - radius in voxels of LOD lod, (p - centre) >> lod saturated per axis.
vec4 base_radius(WorldPos p, int lod) {
    uvec3 lo = p.lo;
    ivec3 hi = p.hi;
    i64_sub(lo, hi, CENTRE_LO, CENTRE_HI);
    i64_asr(lo, hi, lod);
    ivec3 v = sat_words(lo, hi, RADIUS_LIMIT);
    precise float vx = float(v.x);
    precise float vy = float(v.y);
    precise float vz = float(v.z);
    precise float ss = max(vx * vx + vy * vy + vz * vz, 1.0);
    precise float r = det_rsqrt(ss);
    precise float rl = RADIUS * exact_pow2(-lod);
    return vec4(vx * r, vy * r, vz * r, ss * r - rl);
}
)glsl")
            .replace("REGION_SHIFT", itos(REGION_SHIFT))
            .replace("REGION_HALF", itos(REGION_HALF))
            .replace("CENTRE_LO", glsl_lo(anchor))
            .replace("CENTRE_HI", glsl_hi(anchor))
            .replace("RADIUS_LIMIT", itos(RADIUS_LIMIT))
            .replace("RADIUS", glsl_float(float(radius)));
}
