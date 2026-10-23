#include "archetype_instance.hpp"

#include <cmath>

namespace Archetype {

double Bounds::distance(const WorldPos &p_pos) const {
    if (!bounded) {
        return 0.0;
    }
    const float frac[3] = { p_pos.frac.x, p_pos.frac.y, p_pos.frac.z };
    double d2 = 0.0;
    for (int a = 0; a < 3; a++) {
        const double below = double(lo[a] - p_pos.voxel[a]) - double(frac[a]);
        const double above = double(p_pos.voxel[a] - hi[a]) + double(frac[a]);
        const double d = std::max(0.0, std::max(below, above));
        d2 += d * d;
    }
    return std::sqrt(d2);
}

void Instance::set_rotation_y(double p_radians) {
    const float c = float(std::cos(p_radians));
    const float s = float(std::sin(p_radians));
    const float r[9] = { c, 0.0f, -s, 0.0f, 1.0f, 0.0f, s, 0.0f, c };
    std::copy(r, r + 9, rotation);
}

namespace {

bool is_identity(const float *p_r) {
    for (int i = 0; i < 9; i++) {
        if (p_r[i] != ((i % 4) == 0 ? 1.0f : 0.0f)) {
            return false;
        }
    }
    return true;
}

bool is_orthonormal(const float *p_r) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double dot = 0.0;
            for (int k = 0; k < 3; k++) {
                dot += double(p_r[i * 3 + k]) * double(p_r[j * 3 + k]);
            }
            if (std::abs(dot - (i == j ? 1.0 : 0.0)) > 1e-5) {
                return false;
            }
        }
    }
    return true;
}

// R_a R_b, double accumulation then a float round: deterministic, CPU only.
void mat_mul(const float *p_a, const float *p_b, float r_out[9]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double s = 0.0;
            for (int k = 0; k < 3; k++) {
                s += double(p_a[i * 3 + k]) * double(p_b[k * 3 + j]);
            }
            r_out[i * 3 + j] = float(s);
        }
    }
}

// A child's world anchor: the parent's anchor plus the child's offset rotated into world axes
// (R_parent^T c), rounded to voxels (deterministic). A child's offset is in its parent's local frame,
// so a composite placed with a rotation turns as a whole.
VoxelPos child_world_anchor(const float *p_parent_r, const VoxelPos &p_c) {
    // An unrotated parent keeps its child's offset bit for bit: the double round trip below would lose
    // voxels above 2^53 (the root's anchor is passed through here).
    if (is_identity(p_parent_r)) {
        return p_c;
    }
    VoxelPos o;
    for (int j = 0; j < 3; j++) {
        double s = 0.0;
        for (int k = 0; k < 3; k++) {
            s += double(p_parent_r[k * 3 + j]) * double(p_c[k]);
        }
        o[j] = int64_t(std::llround(s));
    }
    return o;
}

// The local x, y, z of an instance: R (p - anchor), op for op.
Local frame(Builder &p_b, const VoxelPos &p_anchor, const float *p_r) {
    const Local l = p_b.local(p_b.anchor(p_anchor));
    if (is_identity(p_r)) {
        return l;
    }
    const Reg d[3] = { l.x, l.y, l.z };
    Reg out[3];
    for (int i = 0; i < 3; i++) {
        out[i] = -1;
        for (int j = 0; j < 3; j++) {
            const float m = p_r[i * 3 + j];
            if (m != 0.0f) {
                const Reg term = m == 1.0f ? d[j] : p_b.mul(d[j], m);
                out[i] = out[i] < 0 ? term : p_b.add(out[i], term);
            }
        }
    }
    return Local{ out[0], out[1], out[2] };
}

// p_def's program appended in the frame (anchor, rotation, params, seed) of p_inst. p_rotation is the
// instance's effective world rotation (the composition with its parents, compile_node); a child of a
// rotated parent turns with it.
Reg inline_def(Builder &p_b, const Def &p_def, const VoxelPos &p_anchor, const Instance &p_inst, const float *p_rotation) {
    const Program &p = p_def.program;
    std::vector<Reg> map(p.code.size(), -1);
    Local local;
    bool have_local = false;
    for (size_t i = 0; i < p.code.size(); i++) {
        Instr in = p.code[i];
        switch (in.op) {
            case Op::INPUT:
                if (!have_local) {
                    local = frame(p_b, p_anchor, p_rotation);
                    have_local = true;
                }
                map[i] = in.axis == 0 ? local.x : (in.axis == 1 ? local.y : local.z);
                break;
            case Op::PARAM:
                map[i] = p_b.constant(in.index < int(p_inst.params.size()) ? p_inst.params[in.index] : p.params[in.index]);
                break;
            case Op::PERLIN3:
            case Op::PERLIN2:
                // The lattice follows the anchor: the cell of (p - anchor) + offset.
                in.offset = in.offset - p_anchor;
                in.seed = p_inst.seed == 0 ? in.seed : in.seed ^ ExactNoise::pcg3d(ExactNoise::UVec3{ p_inst.seed, 0x5EEDu, 0u }).x;
                map[i] = p_b.emit(in);
                break;
            default:
                for (Reg *r : { &in.a, &in.b, &in.c, &in.d }) {
                    *r = *r >= 0 ? map[*r] : -1;
                }
                map[i] = p_b.emit(in);
                break;
        }
    }
    return map[p.output];
}

String def_error(const Def &p_def) {
    const std::vector<VoxelPos> &anchors = p_def.program.anchors;
    if (anchors.size() > 1 || (anchors.size() == 1 && anchors[0] != VoxelPos())) {
        return "Archetype: a Def's only anchor must be the local origin.";
    }
    return p_def.program.output < 0 ? "Archetype: a Def without output." : "";
}

Bounds def_world_bounds(const Def &p_def, const VoxelPos &p_anchor, const float *p_r) {
    return world_bounds(p_def.lo, p_def.hi, p_def.lipschitz, p_def.bounded, p_anchor, p_r);
}

double volume(const Bounds &p_b) {
    return p_b.bounded ? double(p_b.hi.x - p_b.lo.x) * double(p_b.hi.y - p_b.lo.y) * double(p_b.hi.z - p_b.lo.z) : INFINITY;
}

// p_parent combined with a child (its value p_child, bounds p_cb): the new value, r_bounds updated.
Reg combine(Builder &p_b, Reg p_parent, Bounds &r_bounds, const Instance::Child &p_child, Reg p_child_reg, const Bounds &p_cb,
        const VoxelPos &p_child_anchor, const float *p_child_rotation) {
    const Bounds a = r_bounds;
    const float l = std::max(a.lipschitz, p_cb.lipschitz);
    switch (p_child.op) {
        case Combine::UNION:
            r_bounds = hull(a, p_cb);
            return p_b.min(p_parent, p_child_reg);
        case Combine::SMOOTH_UNION: {
            // smooth_min >= min - k / 4: the hull grown by k / 4 (a box grown by m is m closer at most).
            r_bounds = hull(a, p_cb);
            const int64_t m = int64_t(std::ceil(p_child.k * 0.25f));
            r_bounds.lo = r_bounds.lo - VoxelPos(m, m, m);
            r_bounds.hi = r_bounds.hi + VoxelPos(m, m, m);
            return p_b.smooth_min(p_parent, p_child_reg, p_child.k);
        }
        case Combine::SUBTRACT:
        case Combine::SMOOTH_SUBTRACT: {
            // >= max(parent, -child) >= parent.
            r_bounds.lipschitz = l;
            const Reg n = p_b.neg(p_child_reg);
            return p_child.op == Combine::SUBTRACT ? p_b.max(p_parent, n) : p_b.smooth_max(p_parent, n, p_child.k);
        }
        case Combine::INTERSECT:
            // >= both: either box holds, keep the smaller.
            r_bounds = volume(p_cb) < volume(a) ? p_cb : a;
            r_bounds.lipschitz = l;
            return p_b.max(p_parent, p_child_reg);
        default: { // BLEND: >= min(parent, child); the weight's gradient times (child - parent) is unbounded.
            r_bounds = hull(a, p_cb);
            r_bounds.lipschitz = INFINITY;
            // The weight in the child's frame (effective rotation, seed), with its own parameters.
            Instance weight_frame;
            std::copy(p_child_rotation, p_child_rotation + 9, weight_frame.rotation);
            weight_frame.seed = p_child.instance.seed;
            const Reg w = p_b.clamp(inline_def(p_b, *p_child.weight, p_child_anchor, weight_frame, p_child_rotation), 0.0f, 1.0f);
            // parent (1 - w) + child w: exactly the parent at w = 0 and the child at w = 1, at any magnitude
            // (parent + (child - parent) w would round the child to the parent's ulp).
            const Reg one = p_b.constant(1.0f);
            const Reg kept = p_b.mul(p_parent, p_b.sub(one, w));
            return p_b.add(kept, p_b.mul(p_child_reg, w));
        }
    }
}

// Empty if p_child can be combined (its instance is checked by compile_node).
String child_error(const Instance::Child &p_child) {
    if ((p_child.op == Combine::SMOOTH_UNION || p_child.op == Combine::SMOOTH_SUBTRACT) && !(p_child.k > 0.0f)) {
        return "Archetype: a smooth operator needs a radius k > 0.";
    }
    if (p_child.op == Combine::BLEND) {
        return p_child.weight ? def_error(*p_child.weight) : String("Archetype: a BLEND child without weight.");
    }
    return String();
}

// -1 and r_out.error set at the first invalid instance or child (nothing more is emitted). The instance
// is placed in its parent's local frame: its world anchor is p_parent_anchor + R_parent^T p_inst.anchor
// and its effective rotation R_inst R_parent, so a composite turns as a whole with its parent.
Reg compile_node(Builder &p_b, const Instance &p_inst, const VoxelPos &p_parent_anchor, const float *p_parent_rotation, Compiled &r_out,
        Bounds &r_bounds) {
    r_out.error = p_inst.def ? def_error(*p_inst.def) : String("Archetype: an instance without a Def.");
    if (r_out.error.is_empty() && (!is_orthonormal(p_inst.rotation) || !is_orthonormal(p_parent_rotation))) {
        r_out.error = "Archetype: an instance rotation is not orthonormal.";
    }
    if (!r_out.error.is_empty()) {
        return -1;
    }
    const VoxelPos anchor = p_parent_anchor + child_world_anchor(p_parent_rotation, p_inst.anchor);
    float rotation[9];
    mat_mul(p_inst.rotation, p_parent_rotation, rotation);
    if (!is_orthonormal(rotation)) {
        r_out.error = "Archetype: a composed rotation is not orthonormal.";
        return -1;
    }
    Reg r = inline_def(p_b, *p_inst.def, anchor, p_inst, rotation);
    r_bounds = def_world_bounds(*p_inst.def, anchor, rotation);
    for (const Instance::Child &child : p_inst.children) {
        r_out.error = child_error(child);
        if (!r_out.error.is_empty()) {
            return -1;
        }
        // The child's effective frame, for its BLEND weight and its own children.
        const VoxelPos child_anchor = anchor + child_world_anchor(rotation, child.instance.anchor);
        float child_rotation[9];
        mat_mul(child.instance.rotation, rotation, child_rotation);
        Bounds cb;
        const Reg c = compile_node(p_b, child.instance, anchor, rotation, r_out, cb);
        if (c < 0) {
            return -1;
        }
        r = combine(p_b, r, r_bounds, child, c, cb, child_anchor, child_rotation);
    }
    r_out.nodes.push_back(Compiled::Node{ r, r_bounds });
    return r;
}

std::shared_ptr<Def> make_def(Builder &p_b, Reg p_out, Float3 p_lo, Float3 p_hi, float p_lipschitz) {
    std::shared_ptr<Def> d = std::make_shared<Def>();
    d->program = p_b.finish(p_out);
    d->lo = p_lo;
    d->hi = p_hi;
    d->lipschitz = p_lipschitz;
    return d;
}

} // namespace

double rotation_margin(Float3 p_lo, Float3 p_hi) {
    double reach = 0.0;
    for (int c = 0; c < 8; c++) {
        const double q[3] = { (c & 1) ? p_hi.x : p_lo.x, (c & 2) ? p_hi.y : p_lo.y, (c & 4) ? p_hi.z : p_lo.z };
        reach = std::max(reach, std::abs(q[0]) + std::abs(q[1]) + std::abs(q[2]));
    }
    return 1.0 + 1e-4 * reach;
}

Bounds world_bounds(Float3 p_lo, Float3 p_hi, float p_lipschitz, bool p_bounded, const VoxelPos &p_anchor, const float *p_r) {
    Bounds b;
    b.lipschitz = p_lipschitz;
    b.bounded = p_bounded;
    if (!b.bounded) {
        return b;
    }
    // The local box's corners back in world axes (R^T q), then rotation_margin.
    double lo[3] = { INFINITY, INFINITY, INFINITY };
    double hi[3] = { -INFINITY, -INFINITY, -INFINITY };
    for (int c = 0; c < 8; c++) {
        const double q[3] = { (c & 1) ? p_hi.x : p_lo.x, (c & 2) ? p_hi.y : p_lo.y, (c & 4) ? p_hi.z : p_lo.z };
        for (int j = 0; j < 3; j++) {
            const double w = double(p_r[j]) * q[0] + double(p_r[3 + j]) * q[1] + double(p_r[6 + j]) * q[2];
            lo[j] = std::min(lo[j], w);
            hi[j] = std::max(hi[j], w);
        }
    }
    const double margin = rotation_margin(p_lo, p_hi);
    for (int j = 0; j < 3; j++) {
        b.lo[j] = p_anchor[j] + int64_t(std::floor(lo[j] - margin));
        b.hi[j] = p_anchor[j] + int64_t(std::ceil(hi[j] + margin));
    }
    return b;
}

Bounds hull(const Bounds &p_a, const Bounds &p_b) {
    Bounds r;
    r.bounded = p_a.bounded && p_b.bounded;
    r.lipschitz = std::max(p_a.lipschitz, p_b.lipschitz);
    for (int j = 0; j < 3; j++) {
        r.lo[j] = std::min(p_a.lo[j], p_b.lo[j]);
        r.hi[j] = std::max(p_a.hi[j], p_b.hi[j]);
    }
    return r;
}

Compiled compile(const Instance &p_root) {
    Compiled out;
    Builder b;
    Bounds bounds;
    const float identity[9] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };
    const Reg r = compile_node(b, p_root, VoxelPos(), identity, out, bounds);
    if (r >= 0) {
        out.program = b.finish(r);
    }
    return out;
}

std::shared_ptr<Def> sphere_def(float p_radius) {
    Builder b;
    const Reg r = b.sphere(b.local(), p_radius);
    return make_def(b, r, Float3{ -p_radius, -p_radius, -p_radius }, Float3{ p_radius, p_radius, p_radius }, 1.0f);
}

std::shared_ptr<Def> box_def(float p_hx, float p_hy, float p_hz) {
    Builder b;
    const Reg r = b.box(b.local(), p_hx, p_hy, p_hz);
    return make_def(b, r, Float3{ -p_hx, -p_hy, -p_hz }, Float3{ p_hx, p_hy, p_hz }, 1.0f);
}

std::shared_ptr<Def> capsule_def(float p_height, float p_radius) {
    Builder b;
    const Reg r = b.capsule(b.local(), p_height, p_radius);
    return make_def(b, r, Float3{ -p_radius, -p_radius, -p_radius }, Float3{ p_radius, p_height + p_radius, p_radius }, 1.0f);
}

std::shared_ptr<Def> rock_def(float p_radius, const ExactNoise::OctaveSet &p_noise) {
    Builder b;
    const Reg s = b.sphere(b.local(), p_radius);
    const Reg r = b.add(s, b.fbm(p_noise));
    // |noise| <= sum amp * max, |grad noise| <= sum amp * grad max / 2^k (per voxel).
    double reach = 0.0;
    double slope = 0.0;
    for (const ExactNoise::Octave &o : p_noise.octaves) {
        reach += o.amp * (p_noise.is_2d ? PERLIN2_MAX : PERLIN3_MAX);
        slope += o.amp * (p_noise.is_2d ? PERLIN2_GRAD_MAX : PERLIN3_GRAD_MAX) * std::ldexp(1.0, -o.k);
    }
    const float e = float(p_radius + reach);
    return make_def(b, r, Float3{ -e, -e, -e }, Float3{ e, e, e }, float(1.0 + slope));
}

std::shared_ptr<Def> ground_def(const ExactNoise::OctaveSet &p_height) {
    Builder b;
    const Local l = b.local();
    const Reg h = b.fbm(p_height);
    std::shared_ptr<Def> d = make_def(b, b.dist(b.sub(l.y, h)), Float3(), Float3(), 1.0f);
    d->bounded = false;
    return d;
}

std::shared_ptr<Def> ramp_def(float p_width) {
    Builder b;
    const Local l = b.local();
    std::shared_ptr<Def> d = make_def(b, b.add(b.mul(l.y, float(1.0 / double(p_width))), 0.5f), Float3(), Float3(), 1.0f / p_width);
    d->bounded = false;
    return d;
}

} // namespace Archetype
