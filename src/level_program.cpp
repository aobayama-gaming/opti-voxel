#include "level_program.hpp"

#include "ov/math.hpp"

#include <algorithm>
#include <utility>

namespace Terrain {

using Cascade::dadd;
using Cascade::dflush;
using Cascade::dmul;
using Cascade::dscale;
using ExactNoise::float_bits;

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
    if (p_at.m > p_node.level) {
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
    { Op::NOTE, "note", -1, OV_OP(return note(p_prog, p_node, p_in, p_at);), "tp_note(@p, @m, @level, @band, $0, $1, $2)" },
    { Op::CONST, "constant", 0, OV_OP(return (Float4{ 0.0f, 0.0f, 0.0f, p_node.k[0] });), "vec4(0.0, 0.0, 0.0, #0)" },
    { Op::UP, "up", 0, OV_OP(return up(p_prog, p_at);), "cl_scale(base_at(@p, @m), exact_pow2(@m))" },
    { Op::LIN, "lin", -1, OV_OP(return lin(p_node, p_in);), "tp_term($i, #i)" },
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
constexpr const char GLSL_WARP_HELPERS[] = R"glsl(
vec4 cl_smooth_abs(vec4 n, vec4 n2) {
    precise float q = n2.w + SABS_E2;
    precise float r = det_rsqrt(q);
    precise float d0 = n.w * r;
    float d = abs(d0) < FLUSH ? 0.0 : d0;
    precise vec4 res = vec4(n.x * d, n.y * d, n.z * d, q * r - SABS_E);
    return cl_flush(res);
}

WorldPos cascade_warped(WorldPos p, vec4 warp[3]) {
    WorldPos q = p;
    ivec3 d;
    for (int a = 0; a < 3; a++) {
        precise float w = clamp(warp[a].w, -WARP_LIMIT, WARP_LIMIT);
        precise float fl = floor(w);
        precise float rest = w - fl;
        if (rest >= 1.0) {
            fl = fl + 1.0;
            rest = 0.0;
        }
        d[a] = int(fl);
        q.frac[a] = rest;
    }
    i64_add(q.lo, q.hi, d);
    return q;
}

)glsl";

constexpr const char GLSL_HELPERS[] = R"glsl(
vec4 tp_note(WorldPos p, int m, int level, CascadeBand band, vec4 wx, vec4 wy, vec4 wz) {
    if (m > level) {
        return vec4(0.0);
    }
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

vec4 tp_term(vec4 a, float k) {
    return k == 1.0 ? a : cl_scale(a, k);
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

// Runs p_code at every point of r_values (its inputs there already): a node folded on the level its constant, on a run's
// last level (p_last) a node no computed node reads skipped.
void compute(const LevelProgram &p_prog, const std::vector<int> &p_code, bool p_last, Values &r_values) {
    const size_t count = size_t(r_values.box.count());
    const int m = r_values.m;
    std::vector<int> code;
    for (const int n : p_code) {
        const Node &node = p_prog.nodes[size_t(n)];
        if (p_last && m > node.live_lod) {
            continue;
        }
        if (r_values.slot[size_t(n)] < 0) {
            r_values.slot[size_t(n)] = int(r_values.ch.size() / Cascade::CHANNELS);
            r_values.ch.resize(r_values.ch.size() + Cascade::CHANNELS, std::vector<float>(count));
        }
        if (m >= node.const_lod) {
            const size_t s = size_t(r_values.slot[size_t(n)]) * Cascade::CHANNELS;
            for (int c = 0; c < Cascade::CHANNELS; c++) {
                std::fill(r_values.ch[s + size_t(c)].begin(), r_values.ch[s + size_t(c)].end(), c == Cascade::VALUE ? node.folded : 0.0f);
            }
        } else {
            code.push_back(n);
        }
    }
    std::vector<Float4> in;
    Cascade::for_each_point(r_values.box, [&](const Vector3i64 &p_p, int64_t p_i) {
        const OpPoint at{ p_p, r_values.m };
        for (const int n : code) {
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
    return ov::replace_all(ov::str(GLSL_WARP_HELPERS) + GLSL_HELPERS,
            { { "SABS_E2", glsl_float(Cascade::SMOOTH_ABS_E * Cascade::SMOOTH_ABS_E) }, { "SABS_E", glsl_float(Cascade::SMOOTH_ABS_E) }, { "WARP_LIMIT", glsl_float(Cascade::WARP_LIMIT) },
                    { "FLUSH", Cascade::glsl_flush() }, { "SLOPE", glsl_float(ExactNoise::exact_pow2(-ExactNoise::BAND_SHIFT)) }, { "KERNEL_FADE", glsl_float(Climate::KERNEL_FADE) },
                    { "MAX_BIOMES", ov::itos(Climate::MAX_BIOMES) }, { "PICK_SEED", ov::itos(int64_t(Climate::PICK_SEED)) + ov::str("u") } });
}

Float4 eval_node(const LevelProgram &p_prog, const Node &p_node, const Float4 *p_in, const OpPoint &p_at) {
    return op_def(p_node.op).cpu(p_prog, p_node, p_in, p_at);
}

void fold_lods(LevelProgram &r_prog) {
    std::vector<Float4> in;
    for (Node &node : r_prog.nodes) {
        node.const_lod = NO_LEVEL;
        node.folded = 0.0f;
        int lod = 0;
        Float4 v = ZERO;
        if (node.op == Op::NOTE) {
            lod = node.level + 1; // absent on a coarser run, whatever its warp
        } else if (node.op == Op::UP || node.op == Op::PICK) {
            continue; // of the point
        } else {
            int zero = NO_LEVEL; // a product's: from where one factor is the constant 0
            in.clear();
            for (const int i : node.in) {
                const Node &a = r_prog.nodes[size_t(i)];
                lod = ov::max(lod, a.const_lod);
                zero = a.const_lod < NO_LEVEL && float_bits(a.folded) == 0u ? ov::min(zero, a.const_lod) : zero;
                in.push_back(Float4{ 0.0f, 0.0f, 0.0f, a.folded });
            }
            if (node.op == Op::MUL && zero < lod) {
                lod = zero; // dmul by +0: every channel a signed 0, flushed to +0
            } else if (lod < NO_LEVEL) {
                v = eval_node(r_prog, node, in.data(), OpPoint{ Vector3i64(), lod }); // no op but NOTE, UP, PICK reads the point
            }
        }
        if (lod < NO_LEVEL && (float_bits(v.x) | float_bits(v.y) | float_bits(v.z)) == 0u) {
            node.const_lod = lod;
            node.folded = v.w;
        }
    }
    // Liveness on the last level, from the roots up: a node computed there (below its const_lod) reads its inputs.
    for (Node &node : r_prog.nodes) {
        node.live_lod = -1;
    }
    r_prog.nodes[size_t(r_prog.output)].live_lod = NO_LEVEL;
    r_prog.nodes[size_t(r_prog.pick)].live_lod = NO_LEVEL;
    for (size_t i = r_prog.nodes.size(); i-- > 0;) {
        const Node &node = r_prog.nodes[i];
        const int reads = node.const_lod < NO_LEVEL ? ov::min(node.live_lod, node.const_lod - 1) : node.live_lod;
        for (const int a : node.in) {
            r_prog.nodes[size_t(a)].live_lod = ov::max(r_prog.nodes[size_t(a)].live_lod, reads);
        }
    }
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

int LevelProgram::last_stop(int p_lod) const {
    const int lv = ov::min(p_lod, top);
    int stop = lv;
    for (int s = lv; s >= 0; s--) {
        for (const int n : levels[size_t(s)].code) {
            const Node &node = nodes[size_t(n)];
            stop = p_lod < node.const_lod && p_lod <= node.live_lod ? s : stop;
        }
    }
    return stop;
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
        compute(p_prog, m > p_lod ? (m <= p_prog.top ? p_prog.levels[size_t(m)].code : NONE) : p_prog.last_code(p_lod), m == p_lod, next);
        if (r_levels != nullptr) {
            r_levels->push_back(next);
        }
        cur = std::move(next);
    }
    return cur;
}

namespace {

// A point run's level boxes hold at most 2 points per axis (level_bound(1, j)), and so do the axis passes' boxes between.
constexpr int POINT_BOX = 8;

// A point run's per-thread storage: each node's values on the level's box (POINT_BOX per node), the cone of the targets
// and its nodes of no level.
struct PointWork {
    std::vector<Float4> cur, next, in;
    std::vector<uint8_t> cone;
    std::vector<int> free;
};

float &channel(Float4 &r_v, int p_c) {
    return p_c == Cascade::VALUE ? r_v.w : (p_c == Cascade::DX ? r_v.x : (p_c == Cascade::DY ? r_v.y : r_v.z));
}

// subdivide()'s three axis passes for one node, from p_from (box p_in) onto p_to (box p_out), then the flush.
void subdivide_point(bool p_copy_all, const Float4 *p_from, const Cascade::Box &p_in, Float4 *r_to, const Cascade::Box &p_out) {
    Float4 a[POINT_BOX], b[POINT_BOX];
    std::copy(p_from, p_from + p_in.count(), a);
    Cascade::Box box = p_in;
    for (int axis = 0; axis < 3; axis++) {
        Cascade::Box next = box;
        next.lo[axis] = p_out.lo[axis];
        next.hi[axis] = p_out.hi[axis];
        Cascade::axis_pass(box, next, axis, [&](int64_t p_o, int64_t p_ia, int64_t p_ib, bool p_copy) {
            if (p_copy_all) {
                b[p_o] = a[p_ia];
                return;
            }
            Cascade::axis_rule(0, axis, p_copy, [&](int p_c, bool p_b) { return channel(a[p_b ? p_ib : p_ia], p_c); }, [&](int p_c, float p_v) { channel(b[p_o], p_c) = p_v; });
        });
        std::copy(b, b + next.count(), a);
        box = next;
    }
    for (int64_t i = 0; i < p_out.count(); i++) {
        r_to[i] = Float4{ Cascade::flush_value(a[i].x), Cascade::flush_value(a[i].y), Cascade::flush_value(a[i].z), Cascade::flush_value(a[i].w) };
    }
}

// compute()'s ops for the nodes of p_code in the cone, at every point of p_box (Λ_p_m).
void compute_point(const LevelProgram &p_prog, const std::vector<int> &p_code, const Cascade::Box &p_box, int p_m, PointWork &r_w) {
    for (const int n : p_code) {
        if (!r_w.cone[size_t(n)]) {
            continue;
        }
        const Node &node = p_prog.nodes[size_t(n)];
        if (p_m >= node.const_lod) {
            std::fill_n(r_w.next.begin() + ptrdiff_t(size_t(n) * POINT_BOX), p_box.count(), Float4{ 0.0f, 0.0f, 0.0f, node.folded });
            continue;
        }
        r_w.in.resize(ov::max(r_w.in.size(), node.in.size()));
        Cascade::for_each_point(p_box, [&](const Vector3i64 &p_p, int64_t p_i) {
            for (size_t i = 0; i < node.in.size(); i++) {
                r_w.in[i] = r_w.next[size_t(node.in[i]) * POINT_BOX + size_t(p_i)];
            }
            r_w.next[size_t(n) * POINT_BOX + size_t(p_i)] = eval_node(p_prog, node, r_w.in.data(), OpPoint{ p_p, p_m });
        });
    }
}

} // namespace

void point_nodes(const LevelProgram &p_prog, const Vector3i64 &p_p, int p_lod, const int *p_targets, int p_count, Float4 *r_values) {
    thread_local PointWork w;
    const size_t n = p_prog.nodes.size();
    w.cone.assign(n, 0);
    w.cur.resize(n * POINT_BOX);
    w.next.resize(n * POINT_BOX);
    for (int t = 0; t < p_count; t++) {
        w.cone[size_t(p_targets[t])] = 1;
    }
    for (size_t i = n; i-- > 0;) {
        if (p_prog.nodes[i].const_lod <= p_lod) {
            continue; // a constant on every level of the run: its inputs unread
        }
        for (const int in : p_prog.nodes[i].in) {
            w.cone[size_t(in)] |= w.cone[i];
        }
    }
    w.free.clear();
    for (size_t i = 0; i < n; i++) {
        if (w.cone[i] && p_prog.nodes[i].level == NO_LEVEL) {
            w.free.push_back(int(i));
        }
    }
    const Cascade::Box lod_box = Cascade::point_box(p_p);
    const int top = ov::max(p_prog.top, p_lod);
    Cascade::Box parent;
    for (int m = top; m >= p_lod; m--) {
        const Cascade::Box box = Cascade::level_box(lod_box, m - p_lod);
        const auto carry = [&](int p_node) {
            const size_t at = size_t(p_node) * POINT_BOX;
            subdivide_point(p_prog.nodes[size_t(p_node)].op == Op::PICK, w.cur.data() + at, parent, w.next.data() + at, box);
        };
        if (m < top) {
            const std::vector<int> &inputs = p_prog.inputs(m);
            for (const int in : inputs) {
                if (w.cone[size_t(in)]) {
                    carry(in);
                }
            }
            // A target computed above that the program doesn't carry here (a field: its biomes' axes read it on its own
            // level), subdivided as its channel would be.
            for (int t = 0; t < p_count; t++) {
                const int level = p_prog.nodes[size_t(p_targets[t])].level;
                if (level != NO_LEVEL && level > m && std::find(inputs.begin(), inputs.end(), p_targets[t]) == inputs.end()) {
                    carry(p_targets[t]);
                }
            }
        }
        if (m > p_lod) {
            if (m <= p_prog.top) {
                compute_point(p_prog, p_prog.levels[size_t(m)].code, box, m, w);
            }
        } else {
            // The cone's nodes of no level first (their inputs have none either): a target no level computes (a single
            // biome's weight) gets its value too.
            compute_point(p_prog, w.free, box, m, w);
            for (int s = ov::min(p_lod, p_prog.top); s >= 0; s--) {
                compute_point(p_prog, p_prog.levels[size_t(s)].code, box, m, w);
            }
        }
        std::swap(w.cur, w.next);
        parent = box;
    }
    for (int t = 0; t < p_count; t++) {
        r_values[t] = w.cur[size_t(p_targets[t]) * POINT_BOX];
    }
}

Sample point(const LevelProgram &p_prog, const Vector3i64 &p_p, int p_lod) {
    const int targets[2] = { p_prog.output, p_prog.pick };
    Float4 v[2];
    point_nodes(p_prog, p_p, p_lod, targets, 2, v);
    return Sample{ v[0], int(v[1].w) };
}

} // namespace Terrain
