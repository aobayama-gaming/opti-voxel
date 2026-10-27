#include "base_surface.hpp"

#include "ov/math.hpp"

#include "ov/str.hpp"

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
    OV_ERR_FAIL_COND_V_MSG(p_switch_lod < 0 || p_switch_lod > MAX_SWITCH_LOD, BaseSurface(), ov::format("BaseSurface: the switch LOD must be in [0, %d].", MAX_SWITCH_LOD));
    OV_ERR_FAIL_COND_V_MSG(!(p_radius > 0.0) || p_radius > std::ldexp(1.0, 60), BaseSurface(), "BaseSurface: the radius must be in (0, 2^60] voxels.");
    OV_ERR_FAIL_COND_V_MSG(std::ldexp(p_radius, -p_switch_lod) > double(MAX_RADIUS_CELLS), BaseSurface(),
            ov::format("BaseSurface: a radius of %f voxels needs a switch LOD of at least %d (the radius form's float error).", p_radius,
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
    OV_ERR_FAIL_COND_V_MSG(base_kind != Kind::FLAT, 0, "BaseSurface: only a flat base has a datum.");
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

VoxelPos BaseSurface::on_base(const VoxelPos &p_voxel) const {
    if (base_kind == Kind::FLAT) {
        return VoxelPos(p_voxel.x, anchor.y, p_voxel.z);
    }
    double v[3];
    const double len = offset(p_voxel, anchor, v);
    if (len == 0.0) {
        return tangent_point(0, 0);
    }
    const double s = radius / len;
    return anchor + VoxelPos(std::llround(v[0] * s), std::llround(v[1] * s), std::llround(v[2] * s));
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
    const double len = ov::max(offset(c, anchor, v), 1.0);
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
    const float ss = ov::max(vx * vx + vy * vy + vz * vz, 1.0f);
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

Float4 BaseSurface::stretch_base(const VoxelPos &p_voxel, int p_m) const {
    return base_kind == Kind::FLAT ? Float4{} : at(p_voxel, p_m);
}

WorldPos BaseSurface::stretched(const WorldPos &p_pos, int p_j, const Float4 &p_b, int p_m) const {
    WorldPos pos = p_pos;
    if (p_j == 0) {
        return pos;
    }
    if (base_kind == Kind::FLAT) {
        const int64_t rel = p_pos.voxel.y - anchor.y;
        const int64_t q = WorldCoord::floor_div(rel, int64_t(1) << p_j);
        const float rest = float(rel - q * (int64_t(1) << p_j));
        float f = (rest + p_pos.frac.y) * exact_pow2(-p_j);
        pos.voxel.y = anchor.y + q;
        if (f >= 1.0f) {
            pos.voxel.y += 1;
            f = 0.0f;
        }
        pos.frac.y = f;
        return pos;
    }
    const float k = (1.0f - exact_pow2(-p_j)) * p_b.w * exact_pow2(p_m);
    const float off[3] = { ov::clamp(k * p_b.x, -STRETCH_LIMIT, STRETCH_LIMIT), ov::clamp(k * p_b.y, -STRETCH_LIMIT, STRETCH_LIMIT),
        ov::clamp(k * p_b.z, -STRETCH_LIMIT, STRETCH_LIMIT) };
    float *frac[3] = { &pos.frac.x, &pos.frac.y, &pos.frac.z };
    for (int a = 0; a < 3; a++) {
        const float fl = std::floor(off[a]);
        const float r = off[a] - fl;
        float f = *frac[a] - r;
        int64_t d = -int64_t(fl);
        if (f < 0.0f) {
            f = f + 1.0f;
            d -= 1;
        }
        if (f >= 1.0f) {
            f = 0.0f;
            d += 1;
        }
        pos.voxel[a] += d;
        *frac[a] = f;
    }
    return pos;
}

Float4 BaseSurface::stretch_gradient(const Float4 &p_g, int p_j, const Float4 &p_b) const {
    if (base_kind == Kind::FLAT) {
        return Float4{ p_g.x, p_g.y * exact_pow2(-p_j), p_g.z, p_g.w };
    }
    const float c = (1.0f - exact_pow2(-p_j)) * (p_g.x * p_b.x + p_g.y * p_b.y + p_g.z * p_b.z);
    return Float4{ p_g.x - c * p_b.x, p_g.y - c * p_b.y, p_g.z - c * p_b.z, p_g.w };
}

ov::str BaseSurface::glsl_stretch() const {
    using namespace ExactNoise;
    if (base_kind == Kind::FLAT) {
        return ov::replace_all(ov::str(R"glsl(
// BaseSurface::stretched on the flat base: datum + (y - datum) / 2^j, the remainder to the fraction (carried at 1).
WorldPos base_stretched(WorldPos p, int j) {
    if (j == 0) {
        return p;
    }
    WorldPos q = p;
    uvec3 lo = uvec3(0u, p.lo.y, 0u);
    ivec3 hi = ivec3(0, p.hi.y, 0);
    i64_sub(lo, hi, DATUM_LO, DATUM_HI);
    uint rem = lo.y & ((1u << uint(j)) - 1u);
    i64_asr(lo, hi, j);
    i64_sub(lo, hi, NEG_DATUM_LO, NEG_DATUM_HI);
    precise float f = (float(rem) + p.frac.y) * exact_pow2(-j);
    if (f >= 1.0) {
        i64_add(lo, hi, ivec3(0, 1, 0));
        f = 0.0;
    }
    q.lo.y = lo.y;
    q.hi.y = hi.y;
    q.frac.y = f;
    return q;
}

// BaseSurface::stretch_gradient on the flat base.
vec4 base_stretch_gradient(vec4 g, int j) {
    precise float gy = g.y * exact_pow2(-j);
    return vec4(g.x, gy, g.z, g.w);
}
)glsl"),
                { { "NEG_DATUM_LO", glsl_lo(VoxelPos(0, -anchor.y, 0)) }, { "NEG_DATUM_HI", glsl_hi(VoxelPos(0, -anchor.y, 0)) }, { "DATUM_LO", glsl_lo(VoxelPos(0, anchor.y, 0)) }, { "DATUM_HI", glsl_hi(VoxelPos(0, anchor.y, 0)) } });
    }
    return ov::replace_all(ov::str(R"glsl(
// BaseSurface::stretched on a sphere, b its sample at LOD m (stretch_base).
WorldPos base_stretched_at(WorldPos p, int j, vec4 b, int m) {
    if (j == 0) {
        return p;
    }
    precise float k = (1.0 - exact_pow2(-j)) * b.w * exact_pow2(m);
    precise vec3 off = clamp(vec3(k * b.x, k * b.y, k * b.z), vec3(-STRETCH_LIMIT), vec3(STRETCH_LIMIT));
    WorldPos q = p;
    ivec3 d;
    for (int a = 0; a < 3; a++) {
        float fl = floor(off[a]);
        precise float r = off[a] - fl;
        precise float f = p.frac[a] - r;
        d[a] = -int(fl);
        if (f < 0.0) {
            f = f + 1.0;
            d[a] -= 1;
        }
        if (f >= 1.0) {
            f = 0.0;
            d[a] += 1;
        }
        q.frac[a] = f;
    }
    i64_add(q.lo, q.hi, d);
    return q;
}

// BaseSurface::stretch_gradient on a sphere.
vec4 base_stretch_gradient_at(vec4 g, int j, vec4 b) {
    precise float c = (1.0 - exact_pow2(-j)) * (g.x * b.x + g.y * b.y + g.z * b.z);
    precise vec3 o = vec3(g.x - c * b.x, g.y - c * b.y, g.z - c * b.z);
    return vec4(o, g.w);
}
)glsl"),
            "STRETCH_LIMIT", glsl_float(STRETCH_LIMIT));
}

ov::str BaseSurface::glsl() const {
    using namespace ExactNoise;
    if (base_kind == Kind::FLAT) {
        return ov::replace_all(ov::str(R"glsl(
// BaseSurface::flat_form: (up, altitude) of the plane y = datum, in voxels of LOD lod. up = (-0, 1, -0).
vec4 base_at(WorldPos p, int lod) {
    precise float s = exact_pow2(-lod);
    int sy = sat_diff(p, DATUM_LO, DATUM_HI, ALT_LIMIT).y;
    precise float alt = float(sy) * s;
    return vec4(-0.0, 1.0, -0.0, alt);
}
)glsl"),
                { { "DATUM_LO", glsl_lo(anchor) }, { "DATUM_HI", glsl_hi(anchor) }, { "ALT_LIMIT", ov::itos(ALT_LIMIT) } });
    }
    return ov::replace_all(ov::str(R"glsl(
// BaseSurface::Region: h_diag = (xx, yy, zz), h_off = (xy, xz, yz).
struct BaseRegion { float d; vec3 n; vec3 h_diag; vec3 h_off; };

// BaseSurface::curvature_form: d(c) + n.delta + 1/2 delta^T H delta around the centre of p's region at LOD lod.
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
    precise float alt = ss * r - rl;
    return vec4(vx * r, vy * r, vz * r, alt);
}
)glsl"),
            { { "REGION_SHIFT", ov::itos(REGION_SHIFT) }, { "REGION_HALF", ov::itos(REGION_HALF) }, { "CENTRE_LO", glsl_lo(anchor) }, { "CENTRE_HI", glsl_hi(anchor) }, { "RADIUS_LIMIT", ov::itos(RADIUS_LIMIT) }, { "RADIUS", glsl_float(float(radius)) } });
}

BaseSurface::Render BaseSurface::render(const VoxelPos &p_origin, double p_voxel_m) const {
    Render r;
    r.kind = int(base_kind);
    double v[3];
    const double len = offset(anchor, p_origin, v) * p_voxel_m;
    r.point_m = Float3{ float(v[0] * p_voxel_m), float(v[1] * p_voxel_m), float(v[2] * p_voxel_m) };
    if (base_kind == Kind::SPHERE) {
        const double rm = radius * p_voxel_m;
        r.radius_m = float(rm);
        r.k_m2 = float((len - rm) * (len + rm));
    }
    return r;
}

ov::str BaseSurface::glsl_render() {
    return ov::replace_all(ov::str(R"glsl(
const int BASE_SPHERE = SPHERE_KIND;

// SPHERE: |p - centre|^2 - (radius + h)^2 (m^2) from K, so its low bits survive (p - centre alone loses them on a large
// planet).
float base_q(vec3 p, float h) {
    return dot(p, p) - 2.0 * dot(p, base_point_m) + (base_k_m2 - h * (2.0 * base_radius_m + h));
}

// The altitude of render position p above the base (metres, negative below).
float base_altitude(vec3 p) {
    if (base_kind == BASE_SPHERE) {
        return base_q(p, 0.0) / (length(p - base_point_m) + base_radius_m);
    }
    return p.y - base_point_m.y;
}

// The base's up at p: base_altitude's unit gradient.
vec3 base_up(vec3 p) {
    return base_kind == BASE_SPHERE ? normalize(p - base_point_m) : vec3(0.0, 1.0, 0.0);
}

// Where the ray p + t dir (dir unit) is below altitude h: t in [t.x, t.y] (either may be behind p; +-1e30 unbounded);
// false where it never is. SPHERE: the roots of |p + t dir - c| = R + h, the far one from b and the near one as q / far,
// so both keep their bits.
bool base_level_span(vec3 p, vec3 dir, float h, out vec2 t) {
    if (base_kind == BASE_SPHERE) {
        float q = base_q(p, h);
        float b = dot(p - base_point_m, dir);
        float disc = b * b - q;
        float s = sqrt(max(disc, 0.0));
        float far = b > 0.0 ? -q / (b + s) : s - b;
        t = vec2(far != 0.0 ? q / far : -s - b, far);
        return disc >= 0.0;
    }
    float below = h - (p.y - base_point_m.y);
    float level = below / dir.y;
    t = dir.y > 0.0 ? vec2(-1e30, level) : (dir.y < 0.0 ? vec2(level, 1e30) : vec2(-1e30, 1e30));
    return dir.y != 0.0 || below > 0.0;
}
)glsl"),
            { { "SPHERE_KIND", ov::itos(int(Kind::SPHERE)) } });
}
