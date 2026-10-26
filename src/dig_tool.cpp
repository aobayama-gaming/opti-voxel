#include "dig_tool.hpp"

#include "tree_archetype.hpp"
#include "world.hpp"

#include <cmath>
#include <cstring>

namespace ov {

namespace {

// A rotation (rows: world -> local) whose local y is p_dir: a capsule's axis.
void axis_rotation(const Float3 &p_dir, float r_rows[9]) {
    Archetype::up_rotation(UpAxis{ p_dir.x, p_dir.y, p_dir.z }, r_rows);
}

} // namespace

bool DigTool::march(const Float3 &p_pos, const Float3 &p_dir, double p_stop_m, double p_min_step_m, VoxelPos &r_hit, double *r_t) const {
    for (double t = 0.0; t < settings.reach_m;) {
        const double p[3] = { p_pos.x + p_dir.x * t, p_pos.y + p_dir.y * t, p_pos.z + p_dir.z * t };
        const double d = world.sdf_at(p);
        if (d <= p_stop_m) {
            r_hit = world.voxel_at(p);
            if (r_t != nullptr) {
                *r_t = t;
            }
            return true;
        }
        t += std::fmax(d * 0.5, p_min_step_m);
    }
    return false;
}

int DigTool::place(int p_type, const VoxelPos &p_anchor, const float *p_rotation, const double p_params_m[2], Archetype::Combine p_op, double p_k_m) {
    const double vs = double(world.settings().chunk_param.voxel_size);
    const float params[8] = { float(p_params_m[0] / vs), float(p_params_m[1] / vs) };
    const int id = world.add_instance(p_type, p_anchor, p_rotation, params, int(p_op), float(p_k_m / vs));
    edits += id >= 0 ? 1 : 0;
    return id;
}

bool DigTool::dig(const Float3 &p_pos, const Float3 &p_dir) {
    VoxelPos hit;
    // Within half a metre: the hole starts at the surface whatever the march's last step.
    if (!march(p_pos, p_dir, 0.5, 0.1, hit)) {
        return false;
    }
    float rotation[9];
    axis_rotation(p_dir, rotation);
    const double params[2] = { settings.dig_length_m, settings.dig_radius_m };
    return place(SDFTable::CAPSULE_TYPE, hit, rotation, params, Archetype::Combine::SMOOTH_SUBTRACT, settings.dig_k_m) >= 0;
}

void DigTool::spray(const Float3 &p_pos, const Float3 &p_dir, double p_dt) {
    spray_wait_s -= p_dt;
    if (stopped || spray_wait_s > 0.0 || !world.is_edit_settled()) {
        return;
    }
    spray_wait_s = settings.spray_period_s;
    VoxelPos hit;
    // To the voxel (sdf_at's lattice): the spheres are a few.
    if (!march(p_pos, p_dir, 0.0, double(world.settings().chunk_param.voxel_size), hit)) {
        return;
    }
    const double params[2] = { settings.spray_radius_m, 0.0 };
    stopped = place(SDFTable::SPHERE_TYPE, hit, nullptr, params, Archetype::Combine::SUBTRACT, 0.0) < 0;
}

void DigTool::tube(const Float3 &p_pos, const Float3 &p_dir) {
    if (stopped) {
        return;
    }
    const double ahead[3] = { p_pos.x + p_dir.x * 0.5, p_pos.y + p_dir.y * 0.5, p_pos.z + p_dir.z * 0.5 };
    const VoxelPos anchor = world.voxel_at(ahead);
    float rotation[9];
    axis_rotation(p_dir, rotation);
    if (tube_id < 0) {
        // The length is measured once, at the press: afterwards the march would find the end of the tube's own hole and
        // lengthen it every frame.
        VoxelPos hit;
        double t = 0.0;
        march(p_pos, p_dir, 0.5, 0.1, hit, &t);
        const double length = std::fmin(std::fmax(t + settings.drag_length_m, settings.drag_length_m), settings.drag_max_m);
        const double params[2] = { length, settings.drag_radius_m };
        tube_id = place(SDFTable::CAPSULE_TYPE, anchor, rotation, params, settings.drag_op, settings.dig_k_m);
        stopped = tube_id < 0;
    } else if (anchor != tube_anchor || std::memcmp(rotation, tube_rotation, sizeof(rotation)) != 0) {
        // Only when the pose changed: every edit makes the re-meshes in flight stale.
        edits += world.move_instance(tube_id, anchor, rotation) ? 1 : 0;
    }
    tube_anchor = anchor;
    std::memcpy(tube_rotation, rotation, sizeof(rotation));
}

void DigTool::release() {
    if (tube_id >= 0) {
        world.remove_instance(tube_id);
        tube_id = -1;
    }
    spray_wait_s = 0.0;
    stopped = false;
}

} // namespace ov
