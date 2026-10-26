#include "terrain.hpp"

#include "chunk_math.hpp"
#include "world_coord.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace ov::dawn {

namespace {

bool boxes_overlap(const VoxelPos &p_a_lo, const VoxelPos &p_a_hi, const VoxelPos &p_b_lo, const VoxelPos &p_b_hi) {
    return p_a_lo.x <= p_b_hi.x && p_b_lo.x <= p_a_hi.x &&
            p_a_lo.y <= p_b_hi.y && p_b_lo.y <= p_a_hi.y &&
            p_a_lo.z <= p_b_hi.z && p_b_lo.z <= p_a_hi.z;
}

} // namespace

void Terrain::initialize(float p_voxel_size, const Float3 &p_spawn_m) {
    (void)p_spawn_m;
    voxel_m = p_voxel_size;
    world_sdf.configure(0xC0FFEEu, 6, 0.15, p_voxel_size, BaseSurface::flat(0));
    world_sdf.chunk_param().chunk_size = 16;
    world_sdf.chunk_param().voxel_size = p_voxel_size;
    world_sdf.chunk_param().skirt_size = 0;
}

void Terrain::clear() {
    chunk_map.clear();
    world_sdf.clear_edits();
}

int Terrain::pending_chunks() const {
    int pending = 0;
    for (const auto &entry : chunk_map) {
        pending += entry.second.dirty ? 1 : 0;
    }
    return pending;
}

void Terrain::update(const Float3 &p_camera_m, int p_budget) {
    const int cells = world_sdf.chunk_param().chunk_size;
    static constexpr float RADII[MAX_LOD + 1] = { 10.0f, 20.0f, 40.0f };

    std::unordered_set<ChunkId, ChunkIdHash> desired;
    const VoxelPos camera_voxel(int64_t(std::llround(p_camera_m.x / voxel_m)), int64_t(std::llround(p_camera_m.y / voxel_m)), int64_t(std::llround(p_camera_m.z / voxel_m)));

    // A quadtree in x/z: roots at the coarsest LOD, subdivided while the camera is close enough. The leaves
    // partition the ground plane, so a coarse chunk never overlaps a fine one.
    const int64_t root_span = int64_t(cells) << MAX_LOD;
    const int64_t root_min_x = WorldCoord::floor_div(camera_voxel.x - root_span / 2, root_span) * root_span;
    const int64_t root_min_z = WorldCoord::floor_div(camera_voxel.z - root_span / 2, root_span) * root_span;

    const auto surface_chunk = [&](int64_t p_min_x, int64_t p_min_z, int p_lod, int64_t p_span) {
        const float cx = (float(p_min_x) + float(p_span) * 0.5f) * voxel_m;
        const float cz = (float(p_min_z) + float(p_span) * 0.5f) * voxel_m;
        // The surface across the leaf: a steep height field can leave one chunk's y band, so every chunk the
        // leaf's surface range touches is kept.
        float min_y = 1e30f;
        float max_y = -1e30f;
        for (int corner = 0; corner < 5; corner++) {
            const float ox = corner == 1 || corner == 3 ? -0.5f : (corner == 2 || corner == 4 ? 0.5f : 0.0f);
            const float oz = corner == 1 || corner == 2 ? -0.5f : (corner == 3 || corner == 4 ? 0.5f : 0.0f);
            const float px = cx + ox * float(p_span) * voxel_m;
            const float pz = cz + oz * float(p_span) * voxel_m;
            Float3 hit{};
            Float3 hit_normal{};
            if (raycast(Float3{ px, 300.0f, pz }, Float3{ 0.0f, -1.0f, 0.0f }, 600.0f, hit, hit_normal)) {
                min_y = std::min(min_y, hit.y);
                max_y = std::max(max_y, hit.y);
            }
        }
        if (min_y > max_y) {
            return;
        }
        const int64_t y_span = int64_t(cells) << p_lod;
        const int64_t y_lo = WorldCoord::floor_div(int64_t(std::floor(min_y / voxel_m)), y_span) * y_span;
        const int64_t y_hi = WorldCoord::floor_div(int64_t(std::floor(max_y / voxel_m)), y_span) * y_span;
        const int64_t center_x = p_min_x + p_span / 2;
        const int64_t center_z = p_min_z + p_span / 2;
        for (int64_t y = y_lo - y_span; y <= y_hi + y_span; y += y_span) {
            const VoxelPos probe(center_x, y + y_span / 2, center_z);
            desired.insert(ChunkMath::chunk_at_voxel(probe, p_lod, world_sdf.chunk_param()));
        }
    };

    struct Node {
        int64_t min_x;
        int64_t min_z;
        int lod;
    };
    std::vector<Node> stack;
    for (int dz = -1; dz <= 1; dz++) {
        for (int dx = -1; dx <= 1; dx++) {
            stack.push_back(Node{ root_min_x + int64_t(dx) * root_span, root_min_z + int64_t(dz) * root_span, MAX_LOD });
        }
    }
    while (!stack.empty()) {
        const Node node = stack.back();
        stack.pop_back();
        const int64_t span = int64_t(cells) << node.lod;
        const float center_x = (float(node.min_x) + float(span) * 0.5f) * voxel_m;
        const float center_z = (float(node.min_z) + float(span) * 0.5f) * voxel_m;
        const float dx = center_x - p_camera_m.x;
        const float dz = center_z - p_camera_m.z;
        const float distance = std::sqrt(dx * dx + dz * dz);
        if (node.lod > 0 && distance < RADII[node.lod]) {
            const int64_t half = span / 2;
            stack.push_back(Node{ node.min_x, node.min_z, node.lod - 1 });
            stack.push_back(Node{ node.min_x + half, node.min_z, node.lod - 1 });
            stack.push_back(Node{ node.min_x, node.min_z + half, node.lod - 1 });
            stack.push_back(Node{ node.min_x + half, node.min_z + half, node.lod - 1 });
        } else {
            surface_chunk(node.min_x, node.min_z, node.lod, span);
        }
    }

    for (auto it = chunk_map.begin(); it != chunk_map.end();) {
        if (desired.find(it->first) == desired.end()) {
            it = chunk_map.erase(it);
        } else {
            ++it;
        }
    }

    for (const ChunkId &id : desired) {
        if (chunk_map.find(id) == chunk_map.end()) {
            Chunk chunk;
            chunk.id = id;
            chunk.lod = ChunkMath::get_lod(id);
            chunk.min = ChunkMath::chunk_min_voxel(id, world_sdf.chunk_param());
            chunk.cells = cells;
            chunk.dirty = true;
            chunk_map.emplace(id, std::move(chunk));
        }
    }

    int budget = p_budget > 0 ? p_budget : int(chunk_map.size());
    int meshed = 0;
    for (auto &entry : chunk_map) {
        if (budget <= 0) {
            break;
        }
        if (entry.second.dirty) {
            mesh_chunk(entry.second);
            budget--;
            meshed++;
        }
    }
    (void)meshed;
}

void Terrain::mesh_chunk(Chunk &r_chunk) {
    r_chunk.mesh = mesh_surface_nets([this, &r_chunk](const VoxelPos &p_voxel) { return world_sdf.distance_m(p_voxel, r_chunk.lod); }, r_chunk.min, r_chunk.cells, voxel_m, r_chunk.lod);
    r_chunk.dirty = false;
    r_chunk.revision++;
}

bool Terrain::raycast(const Float3 &p_origin_m, const Float3 &p_direction, float p_max_m, Float3 &r_hit_m, Float3 &r_normal) const {
    float t = 0.0f;
    while (t < p_max_m) {
        const Float3 p{ p_origin_m.x + p_direction.x * t, p_origin_m.y + p_direction.y * t, p_origin_m.z + p_direction.z * t };
        const VoxelPos voxel(int64_t(std::llround(p.x / voxel_m)), int64_t(std::llround(p.y / voxel_m)), int64_t(std::llround(p.z / voxel_m)));
        const float d = world_sdf.distance_m(voxel);
        if (d < 0.5f) {
            r_hit_m = p;
            r_normal = world_sdf.normal(voxel);
            return true;
        }
        t += std::max(std::fabs(d) * 0.5f, 0.1f);
    }
    return false;
}

void Terrain::mark_edit_dirty(const Edit &p_edit) {
    VoxelPos lo, hi;
    world_sdf.edit_box(p_edit, lo, hi);
    for (auto &entry : chunk_map) {
        Chunk &chunk = entry.second;
        const int64_t step = int64_t(chunk.cells) << chunk.lod;
        const VoxelPos chunk_hi(chunk.min.x + step - 1, chunk.min.y + step - 1, chunk.min.z + step - 1);
        if (boxes_overlap(lo, hi, chunk.min, chunk_hi)) {
            chunk.dirty = true;
        }
    }
}

void Terrain::carve_sphere(const Float3 &p_center_m, float p_radius_m, bool p_subtract, float p_smooth_m) {
    Edit edit;
    edit.kind = Edit::Kind::SPHERE;
    edit.center = VoxelPos(int64_t(std::llround(p_center_m.x / voxel_m)), int64_t(std::llround(p_center_m.y / voxel_m)), int64_t(std::llround(p_center_m.z / voxel_m)));
    edit.radius = p_radius_m / voxel_m;
    edit.smooth = p_smooth_m;
    edit.subtract = p_subtract;
    world_sdf.edit_list().push_back(edit);
    mark_edit_dirty(edit);
}

void Terrain::carve_capsule(const Float3 &p_a_m, const Float3 &p_b_m, float p_radius_m, bool p_subtract, float p_smooth_m) {
    Edit edit;
    edit.kind = Edit::Kind::CAPSULE;
    const Float3 mid{ (p_a_m.x + p_b_m.x) * 0.5f, (p_a_m.y + p_b_m.y) * 0.5f, (p_a_m.z + p_b_m.z) * 0.5f };
    edit.center = VoxelPos(int64_t(std::llround(mid.x / voxel_m)), int64_t(std::llround(mid.y / voxel_m)), int64_t(std::llround(mid.z / voxel_m)));
    edit.radius = p_radius_m / voxel_m;
    const Float3 delta{ p_b_m.x - p_a_m.x, p_b_m.y - p_a_m.y, p_b_m.z - p_a_m.z };
    const float length = std::sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
    edit.half_length = length * 0.5f / voxel_m;
    if (length > 1e-6f) {
        const Float3 axis{ delta.x / length, delta.y / length, delta.z / length };
        const Float3 up = std::fabs(axis.y) > 0.99f ? Float3{ 1.0f, 0.0f, 0.0f } : Float3{ 0.0f, 1.0f, 0.0f };
        Float3 right{ axis.y * up.z - axis.z * up.y, axis.z * up.x - axis.x * up.z, axis.x * up.y - axis.y * up.x };
        const float rl = std::sqrt(right.x * right.x + right.y * right.y + right.z * right.z);
        if (rl > 1e-6f) {
            right = Float3{ right.x / rl, right.y / rl, right.z / rl };
        }
        const Float3 u2{ right.y * axis.z - right.z * axis.y, right.z * axis.x - right.x * axis.z, right.x * axis.y - right.y * axis.x };
        edit.basis[0] = right.x;
        edit.basis[1] = right.y;
        edit.basis[2] = right.z;
        edit.basis[3] = axis.x;
        edit.basis[4] = axis.y;
        edit.basis[5] = axis.z;
        edit.basis[6] = u2.x;
        edit.basis[7] = u2.y;
        edit.basis[8] = u2.z;
    }
    edit.smooth = p_smooth_m;
    edit.subtract = p_subtract;
    world_sdf.edit_list().push_back(edit);
    mark_edit_dirty(edit);
}

void Terrain::clear_edits() {
    world_sdf.clear_edits();
    for (auto &entry : chunk_map) {
        entry.second.dirty = true;
    }
}

} // namespace ov::dawn
