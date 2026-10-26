#include "dig_tool.hpp"

#include "tree_archetype.hpp"
#include "world.hpp"

#include <cmath>

namespace ov {

bool DigTool::march(const Float3 &p_pos, const Float3 &p_dir, double p_stop_m, double p_min_step_m, VoxelPos &r_hit) const {
    for (double t = 0.0; t < settings.reach_m;) {
        const double p[3] = { p_pos.x + p_dir.x * t, p_pos.y + p_dir.y * t, p_pos.z + p_dir.z * t };
        const double d = world.sdf_at(p);
        if (d <= p_stop_m) {
            r_hit = world.voxel_at(p);
            return true;
        }
        t += std::fmax(d * 0.5, p_min_step_m);
    }
    return false;
}

bool DigTool::place(int p_type, const VoxelPos &p_anchor, const float *p_rotation, const double p_params_m[2], Archetype::Combine p_op, double p_k_m) {
    const double vs = double(world.settings().chunk_param.voxel_size);
    const float params[8] = { float(p_params_m[0] / vs), float(p_params_m[1] / vs) };
    const bool ok = world.add_instance(p_type, p_anchor, p_rotation, params, int(p_op), float(p_k_m / vs)) >= 0;
    edits += ok ? 1 : 0;
    return ok;
}

bool DigTool::dig(const Float3 &p_pos, const Float3 &p_dir) {
    VoxelPos hit;
    // Within half a metre: the hole starts at the surface whatever the march's last step.
    if (!march(p_pos, p_dir, 0.5, 0.1, hit)) {
        return false;
    }
    float rotation[9];
    Archetype::up_rotation(UpAxis{ p_dir.x, p_dir.y, p_dir.z }, rotation); // the capsule's axis along the view
    const double params[2] = { settings.dig_length_m, settings.dig_radius_m };
    return place(SDFTable::CAPSULE_TYPE, hit, rotation, params, Archetype::Combine::SMOOTH_SUBTRACT, settings.dig_k_m);
}

void DigTool::spray(const Float3 &p_pos, const Float3 &p_dir, double p_dt) {
    spray_wait_s -= p_dt;
    if (spray_stopped || spray_wait_s > 0.0 || !world.is_edit_settled()) {
        return;
    }
    spray_wait_s = settings.spray_period_s;
    VoxelPos hit;
    // To the voxel (sdf_at's lattice): the spheres are a few.
    if (!march(p_pos, p_dir, 0.0, double(world.settings().chunk_param.voxel_size), hit)) {
        return;
    }
    const double params[2] = { settings.spray_radius_m, 0.0 };
    spray_stopped = !place(SDFTable::SPHERE_TYPE, hit, nullptr, params, Archetype::Combine::SUBTRACT, 0.0);
}

void DigTool::release() {
    spray_wait_s = 0.0;
    spray_stopped = false;
}

} // namespace ov
