#pragma once

// The player's dig tools on the instance table, for every host (Godot's player.gd through VoxelTerrain, the Dawn viewer):
// - dig: a hole, a SMOOTH_SUBTRACT capsule from where the view meets the terrain running dig_length_m deeper along it;
// - spray (held): a small sphere subtracted where the view meets the surface, one at a time once the last one is meshed
//   (an edit made before the last is meshed makes the re-meshes in flight stale: nothing would show until the release);
// - tube (held): a capsule carried by the camera, from 0.5 m ahead to drag_length_m past the surface the view met at the
//   press, moved whenever its anchor or direction changes and removed at the release.

#include "archetype_record.hpp"
#include "world_coord.hpp"

namespace ov {

class World;

class DigTool {
public:
    struct Settings {
        double dig_radius_m = 1.5;
        double dig_length_m = 6.0;
        double dig_k_m = 0.3; // the hole's and the tube's smooth radius
        double spray_radius_m = 0.1;
        double spray_period_s = 0.03; // at least this long between two spray spheres
        double drag_radius_m = 1.0;
        double drag_length_m = 20.0; // how far past the surface the tube carves
        double drag_max_m = 100.0; // an edit re-meshes every chunk the tube's box (old and new) meets
        Archetype::Combine drag_op = Archetype::Combine::SUBTRACT; // UNION: a solid tube
        double reach_m = 200.0; // how far along the view the tools find the terrain
    };

    explicit DigTool(World &p_world) :
            world(p_world) {}

    Settings settings;

    // A hole along the view (p_dir a unit vector, render space) from where it meets the terrain; false when nothing is
    // within reach or the edit is refused.
    bool dig(const Float3 &p_pos, const Float3 &p_dir);
    // Every frame the spray's button is held, p_dt seconds after the last. A refused edit (the table full) stops it
    // until release().
    void spray(const Float3 &p_pos, const Float3 &p_dir, double p_dt);
    // Every frame the tube's button is held. A refused edit stops it until release().
    void tube(const Float3 &p_pos, const Float3 &p_dir);
    // The button released: the tube removed, the spray ready for the next press.
    void release();
    int edit_count() const { return edits; }

private:
    World &world;
    double spray_wait_s = 0.0;
    bool stopped = false;
    int edits = 0;
    int tube_id = -1;
    VoxelPos tube_anchor;
    float tube_rotation[9] = {};

    // The LOD0 voxel of the first point along the ray where the SDF is at most p_stop_m, stepping half the distance (at
    // least p_min_step_m), and its distance along the ray (r_t, if given); false when nothing is within reach.
    bool march(const Float3 &p_pos, const Float3 &p_dir, double p_stop_m, double p_min_step_m, VoxelPos &r_hit, double *r_t = nullptr) const;
    // One edit at p_anchor, sizes in metres (params[0..1], k): its id, -1 when refused.
    int place(int p_type, const VoxelPos &p_anchor, const float *p_rotation, const double p_params_m[2], Archetype::Combine p_op, double p_k_m);
};

} // namespace ov
