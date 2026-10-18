#pragma once

#include "archetype_ir.hpp"

#include <memory>
#include <vector>

// Instances of archetypes in their own frames, composed with operators, flattened into one Program.
//
// An archetype (Def) is a Program written in its local frame: its INPUT ops read anchor 0, the local origin.
// An instance places it: local = R (p - anchor), the anchor an int64 voxel (the exact sat_diff of INPUT,
// so a far anchor gives the same bits as a near one for the same local point), R a rotation applied op for
// op (mul / add by constants). Gradients come back in world axes through the dual numbers (R^T, for free).
// The noise is not rotated: it follows the instance by an integer lattice shift (offset - anchor), so it is
// the same at every anchor, bit for bit. Children combine with their parent by an operator; the scene
// compiler (compile) inlines every instance's program, so anchors, rotations and parameters become
// constants. A later step replaces those constants with per-chunk table reads.
//
// Bounds: each node gets a world box and a Lipschitz bound L, from its archetype's declared local ones and
// simple rules per operator: outside the box the value is >= the distance to it (voxels), and the
// gradient's length is <= L (per LOD0 voxel). Unit-tested by sampling (test_archetype_instance.cpp).
namespace Archetype {

// Bounds of a perlin octave (ExactNoise): |value| and |gradient| per cell. Conservative; the unit tests
// sample them.
inline constexpr float PERLIN3_MAX = 1.5f;
inline constexpr float PERLIN3_GRAD_MAX = 7.0f;
inline constexpr float PERLIN2_MAX = 2.0f;
inline constexpr float PERLIN2_GRAD_MAX = 8.5f;

struct Def {
    // INPUT ops read anchor 0, which must be (0, 0, 0); PARAMs default to program.params.
    Program program;
    // Outside [lo, hi] (local voxels) the value is >= the distance to it; unbounded: no box.
    bool bounded = true;
    Float3 lo;
    Float3 hi;
    float lipschitz = 1.0f; // INFINITY: unknown
};

// A world box (LOD0 voxels, int64, inclusive) and a Lipschitz bound.
struct Bounds {
    bool bounded = true;
    VoxelPos lo;
    VoxelPos hi;
    float lipschitz = 1.0f;

    // Distance (voxels) from a point to the box, 0 inside.
    double distance(const WorldPos &p_pos) const;
};

enum class Combine : uint8_t {
    UNION, // min(parent, child)
    SMOOTH_UNION, // smooth_min, radius k
    SUBTRACT, // max(parent, -child)
    SMOOTH_SUBTRACT, // smooth_max(parent, -child), radius k
    INTERSECT, // max(parent, child)
    BLEND, // parent (1 - w) + child w, w = clamp(weight, 0, 1) in the child's frame
};

struct Instance {
    std::shared_ptr<const Def> def;
    // The root's: absolute. A child's: relative to its parent's anchor, in world axes (not rotated).
    VoxelPos anchor;
    // local = R (p - anchor), row major, orthonormal.
    float rotation[9] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };
    std::vector<float> params; // overrides the Def's PARAMs
    uint32_t seed = 0; // salts the noise seeds (0: the Def's)

    struct Child;
    std::vector<Child> children;

    void set_rotation_y(double p_radians);
};

struct Instance::Child {
    Combine op = Combine::UNION;
    float k = 0.0f; // smooth radius (voxels)
    std::shared_ptr<const Def> weight; // BLEND
    Instance instance;
};

struct Compiled {
    struct Node {
        Reg reg = -1; // the instance with its children applied
        Bounds bounds;
    };
    Program program; // output: the root node; out_scale / normalize left to the caller
    std::vector<Node> nodes; // children before their parent: the last is the root
    String error; // empty if every Def, rotation and child was valid; else the first error, and no program
};

Compiled compile(const Instance &p_root);

// The box holding both (bounded if both are), the larger Lipschitz bound.
Bounds hull(const Bounds &p_a, const Bounds &p_b);

// Archetypes with their bounds (sizes in voxels).
std::shared_ptr<Def> sphere_def(float p_radius);
std::shared_ptr<Def> box_def(float p_hx, float p_hy, float p_hz);
// Vertical, from y = 0 to p_height.
std::shared_ptr<Def> capsule_def(float p_height, float p_radius);
// A sphere displaced by an fBm (no DIST: the value stays a lower bound of the distance to the box).
std::shared_ptr<Def> rock_def(float p_radius, const ExactNoise::OctaveSet &p_noise);
// A height field y = fbm(x, z), first-order distance (DIST): unbounded, L = 1.
std::shared_ptr<Def> ground_def(const ExactNoise::OctaveSet &p_height);
// BLEND weight: 0.5 + y / p_width along local y (clamped by the operator).
std::shared_ptr<Def> ramp_def(float p_width);

} // namespace Archetype
