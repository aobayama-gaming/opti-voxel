#include "level_program.hpp"

#include "ov/math.hpp"

#include <algorithm>
#include <utility>

namespace Terrain {

using Cascade::dadd;
using Cascade::dflush;
using Cascade::dmul;
using Cascade::dscale;

namespace {

constexpr Float4 ZERO{ 0.0f, 0.0f, 0.0f, 0.0f };

Float4 neg(const Float4 &p_a) {
    return Float4{ -p_a.x, -p_a.y, -p_a.z, -p_a.w };
}

// SHAPE: k[0] + k[1] a + k[2] sabs(a) + k[3] a^2, each term where its coefficient isn't 0 (a 0 adds nothing: the same
// bits either way).
Float4 shape(const float *p_c, const Float4 &p_a) {
    Float4 r{ 0.0f, 0.0f, 0.0f, p_c[0] };
    if (p_c[1] != 0.0f) {
        r = dadd(r, dscale(p_a, p_c[1]));
    }
    if (p_c[2] != 0.0f || p_c[3] != 0.0f) {
        const Float4 a2 = dmul(p_a, p_a);
        if (p_c[2] != 0.0f) {
            r = dadd(r, dscale(Cascade::smooth_abs(p_a, a2), p_c[2]));
        }
        if (p_c[3] != 0.0f) {
            r = dadd(r, dscale(a2, p_c[3]));
        }
    }
    return r;
}

// NOTE: the note's noise at the point (Cascade::note_noise, its warp from in[0..2]), gradient per lattice spacing.
Float4 note(const LevelProgram &p_prog, const Node &p_node, const Float4 *p_in, const OpPoint &p_at) {
    if (p_at.m < p_node.level) {
        return ZERO; // a run of a coarser LOD: the note is absent (its mean)
    }
    const Float4 n = Cascade::note_noise(p_prog.base, p_node.band, p_at.m, p_at.p, p_node.in.empty() ? nullptr : p_in);
    const float slope = ExactNoise::exact_pow2(-ExactNoise::BAND_SHIFT);
    return dflush(Float4{ n.x * slope, n.y * slope, n.z * slope, n.w });
}

Float4 lin(const Node &p_node, const Float4 *p_in) {
    Float4 acc = ZERO;
    for (size_t i = 0; i < p_node.in.size(); i++) {
        const Float4 t = p_node.k[i] == 1.0f ? p_in[i] : dscale(p_in[i], p_node.k[i]);
        acc = i == 0 ? t : dadd(acc, t);
    }
    return acc;
}

// UP at the point: the base's distance and gradient in voxels of the level, as LOD0 voxels.
Float4 up(const LevelProgram &p_prog, const OpPoint &p_at) {
    const int64_t s = int64_t(1) << p_at.m;
    return Climate::up_value(p_prog.base.at(VoxelPos(p_at.p.x * s, p_at.p.y * s, p_at.p.z * s), p_at.m), p_at.m);
}

Float4 pick(const Node &p_node, const Float4 *p_in, const OpPoint &p_at) {
    return Float4{ 0.0f, 0.0f, 0.0f, float(Climate::pick_of(int(p_node.in.size()), p_in, p_at.p, p_at.m)) };
}

// The op table (level_program.hpp's Op), in Op order.
#define OV_OP(p_body) \
    [](const LevelProgram &p_prog [[maybe_unused]], const Node &p_node [[maybe_unused]], const Float4 *p_in [[maybe_unused]], const OpPoint &p_at [[maybe_unused]]) -> Float4 { p_body }

const OpDef OPS[] = {
    { Op::NOTE, "note", -1, OV_OP(return note(p_prog, p_node, p_in, p_at);), "tp_note(@p, @m, @band, $0, $1, $2)" },
    { Op::CONST, "constant", 0, OV_OP(return (Float4{ 0.0f, 0.0f, 0.0f, p_node.k[0] });), "vec4(0.0, 0.0, 0.0, #0)" },
    { Op::UP, "up", 0, OV_OP(return up(p_prog, p_at);), "cl_scale(base_at(@p, @m), exact_pow2(@m))" },
    { Op::LIN, "lin", -1, OV_OP(return lin(p_node, p_in);), "cl_add($acc, cl_scale($i, #i))" },
    { Op::MUL, "multiply", 2, OV_OP(return dmul(p_in[0], p_in[1]);), "cl_mul($0, $1)" },
    { Op::MIN, "min", 2, OV_OP(return p_in[1].w < p_in[0].w ? p_in[1] : p_in[0];), "($1.w < $0.w ? $1 : $0)" },
    { Op::MAX, "max", 2, OV_OP(return p_in[1].w > p_in[0].w ? p_in[1] : p_in[0];), "($1.w > $0.w ? $1 : $0)" },
    { Op::SMIN, "smin", 2, OV_OP(return Climate::smooth_min(p_in[0], p_in[1], p_node.k[0], p_node.k[1]);), "cl_smin($0, $1, #0, #1)" },
    { Op::SMAX, "smax", 2, OV_OP(return neg(Climate::smooth_min(neg(p_in[0]), neg(p_in[1]), p_node.k[0], p_node.k[1]));), "-cl_smin(-$0, -$1, #0, #1)" },
    { Op::CURVE, "curve", 1, OV_OP(return Climate::eval_curve(p_prog.curves[size_t(p_node.curve)], p_in[0]);), "cl_curve_at(@curve, $0)" },
    { Op::SHAPE, "shape", 1, OV_OP(return shape(p_node.k.data(), p_in[0]);), "tp_shape($0, vec4(#0, #1, #2, #3))" },
    { Op::AXIS, "axis", 1, OV_OP(return Climate::axis_factor(Climate::Axis{ 0, p_node.k[0], p_node.k[1] }, p_in[0]);), "tp_axis($0, #0, #1)" },
    { Op::FILL, "fill", 1, OV_OP(return Climate::fill(p_in[0]);), "cl_fill($0)" },
    { Op::RCP, "rcp", 1, OV_OP(return Climate::blend_rcp(p_in[0]);), "cl_rcp($0)" },
    { Op::NORM, "norm", 2, OV_OP(return Climate::blend_weight(p_in[0], p_in[1]);), "tp_norm($0, $1)" },
    { Op::PICK, "pick", -1, OV_OP(return pick(p_node, p_in, p_at);), "vec4(0.0, 0.0, 0.0, tp_pick($all, #n, @p))" },
};
static_assert(sizeof(OPS) / sizeof(OPS[0]) == size_t(Op::COUNT), "one op table entry per Op");

#undef OV_OP

// The forms' own GLSL functions, op for op with the CPU forms above (after the climate's shared helpers and the cascade's
// noise, base and shaped-band helpers). A NOTE without a warp reads a zero displacement: the same bits (a lattice point
// moved by 0, the chain rule adding 0, -0 flushed to 0).
constexpr const char GLSL_HELPERS[] = R"glsl(
vec4 tp_note(WorldPos p, int m, CascadeBand band, vec4 wx, vec4 wy, vec4 wz) {
    vec4 w[3] = vec4[3](wx, wy, wz);
    WorldPos q = cascade_warped(p, w);
    vec4 g = perlin3(base_stretched(q, int(band.info.w)), int(band.info.z), band.info.y, band.offset_lo.xyz, band.offset_hi.xyz);
    g = base_stretch_gradient(g, int(band.info.w));
    precise float tv = exact_pow2(-m);
    vec3 o;
    for (int b = 0; b < 3; b++) {
        precise float s = 0.0;
        for (int a = 0; a < 3; a++) {
            precise float j0 = w[a][b] * tv;
            float j = abs(j0) < FLUSH ? 0.0 : j0;
            precise float t = j * g[a];
            s = s + (abs(t) < FLUSH ? 0.0 : t);
        }
        o[b] = g[b] + (abs(s) < FLUSH ? 0.0 : s);
    }
    precise vec4 r = vec4(o.x * SLOPE, o.y * SLOPE, o.z * SLOPE, g.w);
    return cl_flush(r);
}

vec4 tp_shape(vec4 a, vec4 c) {
    vec4 r = vec4(0.0, 0.0, 0.0, c.x);
    if (c.y != 0.0) {
        r = cl_add(r, cl_scale(a, c.y));
    }
    if (c.z != 0.0 || c.w != 0.0) {
        vec4 a2 = cl_mul(a, a);
        if (c.z != 0.0) {
            r = cl_add(r, cl_scale(cl_smooth_abs(a, a2), c.z));
        }
        if (c.w != 0.0) {
            r = cl_add(r, cl_scale(a2, c.w));
        }
    }
    return r;
}

vec4 tp_axis(vec4 x, float centre, float inv_span) {
    return cl_axis(x, centre, inv_span, KERNEL_FADE);
}

vec4 tp_norm(vec4 k, vec4 r) {
    vec4 w = cl_mul(k, r);
    w.w = min(w.w, 1.0);
    return w;
}

float tp_pick(vec4 w[MAX_BIOMES], int n, WorldPos p) {
    float best = w[0].w;
    uint count = 1u;
    for (int b = 1; b < n; b++) {
        if (floatBitsToUint(w[b].w) == floatBitsToUint(best)) {
            count++;
        } else if (w[b].w > best) {
            best = w[b].w;
            count = 1u;
        }
    }
    uint k = count == 1u ? 0u : hash_cell(p.lo, p.hi, PICK_SEED) % count;
    for (int b = 0; b < n; b++) {
        if (floatBitsToUint(w[b].w) == floatBitsToUint(best)) {
            if (k == 0u) {
                return float(b);
            }
            k--;
        }
    }
    return 0.0;
}
)glsl";

// Nodes p_in of p_parent subdivided onto p_box (x, then y, then z, a PICK copied from its floor parent), flushed: the
// inputs of the level below, slots 0 .. p_in.size() - 1.
Values subdivide(const LevelProgram &p_prog, const Values &p_parent, const std::vector<int> &p_in, const Cascade::Box &p_box) {
    const size_t groups = p_in.size();
    std::vector<std::vector<float>> cur(groups * Cascade::CHANNELS);
    for (size_t g = 0; g < groups; g++) {
        for (int c = 0; c < Cascade::CHANNELS; c++) {
            cur[g * Cascade::CHANNELS + size_t(c)] = p_parent.ch[size_t(p_parent.slot[size_t(p_in[g])] * Cascade::CHANNELS + c)];
        }
    }
    Cascade::Box box = p_parent.box;
    for (int axis = 0; axis < 3; axis++) {
        Cascade::Box next = box;
        next.lo[axis] = p_box.lo[axis];
        next.hi[axis] = p_box.hi[axis];
        std::vector<std::vector<float>> out(cur.size(), std::vector<float>(size_t(next.count())));
        Cascade::axis_pass(box, next, axis, [&](int64_t p_o, int64_t p_ia, int64_t p_ib, bool p_copy) {
            for (size_t g = 0; g < groups; g++) {
                const size_t base = g * Cascade::CHANNELS;
                const auto get = [&](int p_c, bool p_b) { return cur[base + size_t(p_c)][size_t(p_b ? p_ib : p_ia)]; };
                const auto put = [&](int p_c, float p_v) { out[base + size_t(p_c)][size_t(p_o)] = p_v; };
                if (p_prog.nodes[size_t(p_in[g])].op == Op::PICK) {
                    for (int c = 0; c < Cascade::CHANNELS; c++) {
                        put(c, get(c, false));
                    }
                } else {
                    Cascade::axis_rule(0, axis, p_copy, get, put);
                }
            }
        });
        cur = std::move(out);
        box = next;
    }
    Values v;
    v.m = p_parent.m - 1;
    v.box = p_box;
    v.slot.assign(p_prog.nodes.size(), -1);
    for (size_t g = 0; g < groups; g++) {
        v.slot[size_t(p_in[g])] = int(g);
    }
    for (std::vector<float> &c : cur) {
        for (float &x : c) {
            x = Cascade::flush_value(x);
        }
    }
    v.ch = std::move(cur);
    return v;
}

// Runs p_code at every point of r_values (its inputs there already).
void compute(const LevelProgram &p_prog, const std::vector<int> &p_code, Values &r_values) {
    const size_t count = size_t(r_values.box.count());
    for (const int n : p_code) {
        if (r_values.slot[size_t(n)] < 0) {
            r_values.slot[size_t(n)] = int(r_values.ch.size() / Cascade::CHANNELS);
            r_values.ch.resize(r_values.ch.size() + Cascade::CHANNELS, std::vector<float>(count));
        }
    }
    std::vector<Float4> in;
    Cascade::for_each_point(r_values.box, [&](const Vector3i64 &p_p, int64_t p_i) {
        const OpPoint at{ p_p, r_values.m };
        for (const int n : p_code) {
            const Node &node = p_prog.nodes[size_t(n)];
            in.resize(node.in.size());
            for (size_t i = 0; i < node.in.size(); i++) {
                in[i] = r_values.at(node.in[i], p_i);
            }
            const Float4 v = eval_node(p_prog, node, in.data(), at);
            const size_t s = size_t(r_values.slot[size_t(n)]) * Cascade::CHANNELS;
            r_values.ch[s + Cascade::VALUE][size_t(p_i)] = v.w;
            r_values.ch[s + Cascade::DX][size_t(p_i)] = v.x;
            r_values.ch[s + Cascade::DY][size_t(p_i)] = v.y;
            r_values.ch[s + Cascade::DZ][size_t(p_i)] = v.z;
        }
    });
}

} // namespace

const OpDef &op_def(Op p_op) {
    return OPS[size_t(p_op)];
}

ov::str glsl_helpers() {
    using ExactNoise::glsl_float;
    return ov::replace_all(GLSL_HELPERS, { { "FLUSH", Cascade::glsl_flush() }, { "SLOPE", glsl_float(ExactNoise::exact_pow2(-ExactNoise::BAND_SHIFT)) }, { "KERNEL_FADE", glsl_float(Climate::KERNEL_FADE) }, { "MAX_BIOMES", ov::itos(Climate::MAX_BIOMES) }, { "PICK_SEED", ov::itos(int64_t(Climate::PICK_SEED)) + ov::str("u") } });
}

Float4 eval_node(const LevelProgram &p_prog, const Node &p_node, const Float4 *p_in, const OpPoint &p_at) {
    return op_def(p_node.op).cpu(p_prog, p_node, p_in, p_at);
}

const std::vector<int> &LevelProgram::inputs(int p_m) const {
    static const std::vector<int> NONE;
    return p_m + 1 <= top ? levels[size_t(p_m + 1)].out : NONE;
}

std::vector<int> LevelProgram::last_code(int p_lod) const {
    std::vector<int> code;
    for (int m = ov::min(p_lod, top); m >= 0; m--) {
        code.insert(code.end(), levels[size_t(m)].code.begin(), levels[size_t(m)].code.end());
    }
    return code;
}

Float4 Values::at(int p_node, int64_t p_i) const {
    const size_t s = size_t(slot[size_t(p_node)]) * Cascade::CHANNELS;
    const size_t i = size_t(p_i);
    return Float4{ ch[s + Cascade::DX][i], ch[s + Cascade::DY][i], ch[s + Cascade::DZ][i], ch[s + Cascade::VALUE][i] };
}

void Values::channels(int p_node, int64_t p_i, float r_ch[Cascade::CHANNELS]) const {
    const Float4 v = at(p_node, p_i);
    r_ch[Cascade::VALUE] = v.w;
    r_ch[Cascade::DX] = v.x;
    r_ch[Cascade::DY] = v.y;
    r_ch[Cascade::DZ] = v.z;
}

Values run(const LevelProgram &p_prog, const Cascade::Box &p_box, int p_lod, std::vector<Values> *r_levels) {
    const int top = ov::max(p_prog.top, p_lod);
    const std::vector<Cascade::Box> boxes = Cascade::level_boxes(p_box, p_lod, top);
    Values cur;
    for (int m = top; m >= p_lod; m--) {
        const Cascade::Box &box = boxes[size_t(m - p_lod)];
        Values next;
        if (m == top) {
            next.m = m;
            next.box = box;
            next.slot.assign(p_prog.nodes.size(), -1);
        } else {
            next = subdivide(p_prog, cur, p_prog.inputs(m), box);
        }
        static const std::vector<int> NONE;
        compute(p_prog, m > p_lod ? (m <= p_prog.top ? p_prog.levels[size_t(m)].code : NONE) : p_prog.last_code(p_lod), next);
        if (r_levels != nullptr) {
            r_levels->push_back(next);
        }
        cur = std::move(next);
    }
    return cur;
}

Sample point(const LevelProgram &p_prog, const Vector3i64 &p_p, int p_lod) {
    const Values v = run(p_prog, Cascade::point_box(p_p), p_lod);
    Sample s;
    s.ground = v.has(p_prog.output) ? v.at(p_prog.output, 0) : ZERO;
    s.pick = v.has(p_prog.pick) ? int(v.at(p_prog.pick, 0).w) : 0;
    return s;
}

} // namespace Terrain
