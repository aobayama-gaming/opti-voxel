#include "terrain_sdf.hpp"

#include "cascade_terrain.hpp"
#include "exact_noise.hpp"

#include <algorithm>
#include <cmath>

namespace ov::dawn {

namespace {

float smooth_min(float p_a, float p_b, float p_k) {
    if (p_k <= 0.0f) {
        return std::min(p_a, p_b);
    }
    const float h = std::max(p_k - std::fabs(p_a - p_b), 0.0f) / p_k;
    return std::min(p_a, p_b) - h * h * p_k * 0.25f;
}

float smooth_max(float p_a, float p_b, float p_k) {
    return -smooth_min(-p_a, -p_b, p_k);
}

// The primitive's distance (LOD0 voxels) and its outward gradient (unit) at p.
void primitive(const Edit &p_edit, const VoxelPos &p_voxel, float &r_distance, Float3 &r_gradient) {
    const double dx = double(p_voxel.x - p_edit.center.x);
    const double dy = double(p_voxel.y - p_edit.center.y);
    const double dz = double(p_voxel.z - p_edit.center.z);
    if (p_edit.kind == Edit::Kind::SPHERE) {
        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        r_distance = float(r) - p_edit.radius;
        if (r > 1e-9) {
            r_gradient = Float3{ float(dx / r), float(dy / r), float(dz / r) };
        } else {
            r_gradient = Float3{ 0.0f, 1.0f, 0.0f };
        }
        return;
    }
    // Capsule: local y is the axis; the basis rows are the local axes in world space.
    const double lx = double(p_edit.basis[0]) * dx + double(p_edit.basis[1]) * dy + double(p_edit.basis[2]) * dz;
    const double ly = double(p_edit.basis[3]) * dx + double(p_edit.basis[4]) * dy + double(p_edit.basis[5]) * dz;
    const double lz = double(p_edit.basis[6]) * dx + double(p_edit.basis[7]) * dy + double(p_edit.basis[8]) * dz;
    const double cy = std::clamp(ly, -double(p_edit.half_length), double(p_edit.half_length));
    const double qx = lx;
    const double qy = ly - cy;
    const double qz = lz;
    const double r = std::sqrt(qx * qx + qy * qy + qz * qz);
    r_distance = float(r) - p_edit.radius;
    double gx = 0.0;
    double gy = 1.0;
    double gz = 0.0;
    if (r > 1e-9) {
        gx = qx / r;
        gy = qy / r;
        gz = qz / r;
    }
    r_gradient = Float3{
        float(double(p_edit.basis[0]) * gx + double(p_edit.basis[3]) * gy + double(p_edit.basis[6]) * gz),
        float(double(p_edit.basis[1]) * gx + double(p_edit.basis[4]) * gy + double(p_edit.basis[7]) * gz),
        float(double(p_edit.basis[2]) * gx + double(p_edit.basis[5]) * gy + double(p_edit.basis[8]) * gz),
    };
}

} // namespace

void DawnSDF::configure(uint32_t p_seed, int p_k_top, double p_steepness, float p_voxel_size, const BaseSurface &p_base) {
    spec = CascadeTerrain::make_spec(p_seed, p_k_top, p_steepness, Cascade::ChannelSet::LINEAR4, 3);
    base_surface = p_base;
    voxel_m = p_voxel_size;
    param = ChunkParam();
    param.chunk_size = 16;
    param.voxel_size = p_voxel_size;
    param.skirt_size = 0;
    param.skirt_depth = 1.0f;
    param.origin_chunk = ChunkId();
}

float DawnSDF::distance_m(const VoxelPos &p_voxel, int p_lod) const {
    float ch[Cascade::CHANNELS];
    Cascade::point_into(spec, p_voxel, p_lod, ch);
    const ExactNoise::Float4 base = base_surface.at(p_voxel, p_lod);
    const ExactNoise::Float4 sdf = CascadeTerrain::height_sdf(ch, p_lod, base);
    // height_sdf's distance is in voxels of LOD p_lod; the edits are LOD0 voxels.
    float d = sdf.w * float(int64_t(1) << p_lod);
    for (const Edit &edit : edits) {
        float prim = 0.0f;
        Float3 gradient{};
        primitive(edit, p_voxel, prim, gradient);
        const float k = edit.smooth / std::max(voxel_m, 1e-6f);
        d = edit.subtract ? smooth_max(d, -prim, k) : smooth_min(d, prim, k);
    }
    return d * voxel_m;
}

Float3 DawnSDF::normal(const VoxelPos &p_voxel) const {
    const float e = 1.0f;
    const float gx = distance_m(VoxelPos(p_voxel.x + int64_t(e), p_voxel.y, p_voxel.z)) - distance_m(VoxelPos(p_voxel.x - int64_t(e), p_voxel.y, p_voxel.z));
    const float gy = distance_m(VoxelPos(p_voxel.x, p_voxel.y + int64_t(e), p_voxel.z)) - distance_m(VoxelPos(p_voxel.x, p_voxel.y - int64_t(e), p_voxel.z));
    const float gz = distance_m(VoxelPos(p_voxel.x, p_voxel.y, p_voxel.z + int64_t(e))) - distance_m(VoxelPos(p_voxel.x, p_voxel.y, p_voxel.z - int64_t(e)));
    const float len = std::sqrt(gx * gx + gy * gy + gz * gz);
    if (len <= 1e-9f) {
        return Float3{ 0.0f, 1.0f, 0.0f };
    }
    return Float3{ gx / len, gy / len, gz / len };
}

ov::bytes DawnSDF::edit_bytes() const {
    ov::bytes bytes(16 + edits.size() * 6 * 16, 0);
    ov::put_u32(bytes.data(), uint32_t(edits.size()));
    uint8_t *slot = bytes.data() + 16;
    for (const Edit &edit : edits) {
        const float values[24] = {
            float(edit.center.x), float(edit.center.y), float(edit.center.z), float(uint32_t(edit.kind)),
            edit.basis[0], edit.basis[1], edit.basis[2], edit.radius,
            edit.half_length, edit.smooth, edit.subtract ? 1.0f : 0.0f, 0.0f,
            edit.basis[3], edit.basis[4], edit.basis[5], 0.0f,
            edit.basis[6], edit.basis[7], edit.basis[8], 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
        };
        for (int i = 0; i < 24; i++) {
            ov::put_f32(slot + i * 4, values[i]);
        }
        slot += 6 * 16;
    }
    return bytes;
}

void DawnSDF::edit_box(const Edit &p_edit, VoxelPos &r_lo, VoxelPos &r_hi) const {
    const int64_t reach = int64_t(std::ceil(p_edit.radius + p_edit.half_length + p_edit.smooth / std::max(voxel_m, 1e-6f))) + 2;
    r_lo = VoxelPos(p_edit.center.x - reach, p_edit.center.y - reach, p_edit.center.z - reach);
    r_hi = VoxelPos(p_edit.center.x + reach, p_edit.center.y + reach, p_edit.center.z + reach);
}

ov::str DawnSDF::get_glsl_code() const {
    // The base, the cascade's height SDF, then the edits. The edits buffer is binding 11 in both the cascade
    // level shader (bindings 0-6) and the legacy mesher (0-10). Distances are LOD0 voxels; the field is metres.
    return base_surface.glsl() + CascadeTerrain::GLSL_HEIGHT_SDF + R"glsl(
layout(set = 0, binding = 11, std430) restrict readonly buffer SdfEdits {
    vec4 edit_data[];
};

vec4 cascade_terrain_unit(vec4 c0, WorldPos p, int lod) {
    return height_sdf(c0, base_at(p, lod), lod);
}

vec4 cascade_metres(vec4 u, int lod) {
    precise vec4 r = u;
    precise float to_m = chunk_param.voxel_size * exact_pow2(lod);
    r.w = r.w * to_m;
    return r;
}

// One primitive at the sample: distance in LOD0 voxels and its outward gradient.
vec2 sdf_edit_primitive(vec4 edit[6], vec3 rel) {
    if (edit[0].w < 0.5) {
        float r = length(rel);
        return vec2(r - edit[1].w, r > 1e-9 ? 1.0 : 0.0);
    }
    vec3 local = vec3(dot(edit[3].xyz, rel), dot(edit[4].xyz, rel), dot(edit[5].xyz, rel));
    float cy = clamp(local.y, -edit[2].x, edit[2].x);
    vec3 q = vec3(local.x, local.y - cy, local.z);
    float r = length(q);
    return vec2(r - edit[1].w, r > 1e-9 ? 1.0 : 0.0);
}

vec3 sdf_edit_gradient(vec4 edit[6], vec3 rel) {
    if (edit[0].w < 0.5) {
        float r = length(rel);
        return r > 1e-9 ? rel / r : vec3(0.0, 1.0, 0.0);
    }
    vec3 local = vec3(dot(edit[3].xyz, rel), dot(edit[4].xyz, rel), dot(edit[5].xyz, rel));
    float cy = clamp(local.y, -edit[2].x, edit[2].x);
    vec3 q = vec3(local.x, local.y - cy, local.z);
    float r = length(q);
    vec3 g = r > 1e-9 ? q / r : vec3(0.0, 1.0, 0.0);
    return vec3(dot(edit[3].xyz, g), dot(edit[4].xyz, g), dot(edit[5].xyz, g));
}

vec4 sdf_apply_edits(vec4 d, WorldPos p, int lod) {
    uint count = uint(edit_data[0].x);
    if (count == 0u) {
        return d;
    }
    vec3 rel = vec3(p.lo) + p.frac - vec3(0.0);
    vec3 normal = d.xyz;
    float distance = d.w / chunk_param.voxel_size; // metres -> LOD0 voxels
    for (uint i = 0u; i < count; i++) {
        uint base = 1u + i * 6u;
        vec4 edit[6];
        for (int k = 0; k < 6; k++) {
            edit[k] = edit_data[base + uint(k)];
        }
        vec3 center = edit[0].xyz;
        vec3 r = rel - center;
        vec2 prim = sdf_edit_primitive(edit, r);
        float k = edit[2].y / chunk_param.voxel_size; // smooth radius in LOD0 voxels
        float before = distance;
        if (edit[2].z > 0.5) {
            float h = max(k - abs(distance - (-prim.x)), 0.0) / max(k, 1e-6);
            distance = max(distance, -prim.x) + (k > 0.0 ? h * h * k * 0.25 : 0.0);
            if (-prim.x >= before || k > 0.0) {
                normal = -sdf_edit_gradient(edit, r);
            }
        } else {
            float h = max(k - abs(distance - prim.x), 0.0) / max(k, 1e-6);
            distance = min(distance, prim.x) - (k > 0.0 ? h * h * k * 0.25 : 0.0);
            if (prim.x <= before || k > 0.0) {
                normal = sdf_edit_gradient(edit, r);
            }
        }
    }
    return vec4(normal, distance * chunk_param.voxel_size);
}

vec4 cascade_terrain_sdf(vec4 c0, WorldPos p, int lod) {
    return sdf_apply_edits(cascade_metres(cascade_terrain_unit(c0, p, lod), lod), p, lod);
}

float cascade_skirt_depth_m(int lod) {
    return chunk_param.skirt_depth * chunk_param.voxel_size * exact_pow2(lod);
}
)glsl";
}

} // namespace ov::dawn
