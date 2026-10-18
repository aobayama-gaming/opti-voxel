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
        const double d = MAX(0.0, MAX(below, above));
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

// The local x, y, z of an instance: R (p - anchor), op for op.
Local frame(Builder &p_b, const VoxelPos &p_anchor, const float *p_r) {
    const int a = p_b.anchor(p_anchor);
    const Reg d[3] = { p_b.input(0, a), p_b.input(1, a), p_b.input(2, a) };
    if (is_identity(p_r)) {
        return Local{ d[0], d[1], d[2] };
    }
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

// p_def's program appended in the frame (anchor, rotation, params, seed) of p_inst.
Reg inline_def(Builder &p_b, const Def &p_def, const VoxelPos &p_anchor, const Instance &p_inst) {
    const Program &p = p_def.program;
    std::vector<Reg> map(p.code.size(), -1);
    Local local;
    bool have_local = false;
    for (size_t i = 0; i < p.code.size(); i++) {
        Instr in = p.code[i];
        switch (in.op) {
            case Op::INPUT:
                if (!have_local) {
                    local = frame(p_b, p_anchor, p_inst.rotation);
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

Bounds world_bounds(const Def &p_def, const VoxelPos &p_anchor, const float *p_r) {
    Bounds b;
    b.lipschitz = p_def.lipschitz;
    b.bounded = p_def.bounded;
    if (!b.bounded) {
        return b;
    }
    // The local box's corners back in world axes (R^T q), then one voxel of margin for the float rotation.
    double lo[3] = { INFINITY, INFINITY, INFINITY };
    double hi[3] = { -INFINITY, -INFINITY, -INFINITY };
    for (int c = 0; c < 8; c++) {
        const double q[3] = { (c & 1) ? p_def.hi.x : p_def.lo.x, (c & 2) ? p_def.hi.y : p_def.lo.y, (c & 4) ? p_def.hi.z : p_def.lo.z };
        for (int j = 0; j < 3; j++) {
            const double w = double(p_r[j]) * q[0] + double(p_r[3 + j]) * q[1] + double(p_r[6 + j]) * q[2];
            lo[j] = MIN(lo[j], w);
            hi[j] = MAX(hi[j], w);
        }
    }
    for (int j = 0; j < 3; j++) {
        b.lo[j] = p_anchor[j] + int64_t(std::floor(lo[j])) - 1;
        b.hi[j] = p_anchor[j] + int64_t(std::ceil(hi[j])) + 1;
    }
    return b;
}

Bounds hull(const Bounds &p_a, const Bounds &p_b) {
    Bounds r;
    r.bounded = p_a.bounded && p_b.bounded;
    r.lipschitz = MAX(p_a.lipschitz, p_b.lipschitz);
    for (int j = 0; j < 3; j++) {
        r.lo[j] = MIN(p_a.lo[j], p_b.lo[j]);
        r.hi[j] = MAX(p_a.hi[j], p_b.hi[j]);
    }
    return r;
}

double volume(const Bounds &p_b) {
    return p_b.bounded ? double(p_b.hi.x - p_b.lo.x) * double(p_b.hi.y - p_b.lo.y) * double(p_b.hi.z - p_b.lo.z) : INFINITY;
}

// p_parent combined with a child (its value p_child, bounds p_cb): the new value, r_bounds updated.
Reg combine(Builder &p_b, Reg p_parent, Bounds &r_bounds, const Instance::Child &p_child, Reg p_child_reg, const Bounds &p_cb,
        const VoxelPos &p_child_anchor) {
    const Bounds a = r_bounds;
    const float l = MAX(a.lipschitz, p_cb.lipschitz);
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
            const Reg w = p_b.clamp(inline_def(p_b, *p_child.weight, p_child_anchor, p_child.instance), 0.0f, 1.0f);
            const Reg d = p_b.sub(p_child_reg, p_parent);
            return p_b.add(p_parent, p_b.mul(d, w));
        }
    }
}

Reg compile_node(Builder &p_b, const Instance &p_inst, const VoxelPos &p_parent_anchor, Compiled &r_out, Bounds &r_bounds) {
    const VoxelPos anchor = p_parent_anchor + p_inst.anchor;
    if (r_out.error.is_empty()) {
        r_out.error = def_error(*p_inst.def);
    }
    if (r_out.error.is_empty() && !is_orthonormal(p_inst.rotation)) {
        r_out.error = "Archetype: an instance rotation is not orthonormal.";
    }
    Reg r = inline_def(p_b, *p_inst.def, anchor, p_inst);
    r_bounds = world_bounds(*p_inst.def, anchor, p_inst.rotation);
    for (const Instance::Child &child : p_inst.children) {
        if (child.op == Combine::BLEND && !child.weight) {
            r_out.error = "Archetype: a BLEND child without weight.";
            continue;
        }
        Bounds cb;
        const Reg c = compile_node(p_b, child.instance, anchor, r_out, cb);
        r = combine(p_b, r, r_bounds, child, c, cb, anchor + child.instance.anchor);
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

Compiled compile(const Instance &p_root) {
    Compiled out;
    Builder b;
    Bounds bounds;
    const Reg r = compile_node(b, p_root, VoxelPos(), out, bounds);
    out.program = b.finish(r);
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
