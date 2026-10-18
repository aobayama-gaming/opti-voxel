#include "archetype_ir.hpp"

#include <cfloat>
#include <cmath>

namespace Archetype {

using ExactNoise::det_rsqrt;
using ExactNoise::exact_pow2;
using ExactNoise::float_bits;

namespace {

Float4 flat_value(float p_v) {
    return Float4{ 0.0f, 0.0f, 0.0f, p_v };
}

float axis_of(const Float3 &p_v, int p_axis) {
    return p_axis == 0 ? p_v.x : (p_axis == 1 ? p_v.y : p_v.z);
}

bool finite_not_denormal(float p_f) {
    return p_f == 0.0f || (std::isfinite(p_f) && std::fabs(p_f) >= FLT_MIN);
}

// 1 / |grad p_v|, its square floored at 2^GRAD_FLOOR_EXP per LOD0 voxel (p_s = 2^lod): DIST and the epilogue
// (GLSL: inv_grad_length in archetype_glsl.cpp).
float inv_grad_length(const Float4 &p_v, float p_s) {
    const float n2 = p_v.x * p_v.x + p_v.y * p_v.y + p_v.z * p_v.z;
    return det_rsqrt(MAX(n2, exact_pow2(GRAD_FLOOR_EXP) * p_s * p_s));
}

// ADD, SUB, MUL, NEG, ABS: the dual-number rules, flat operands' terms skipped (GLSL: emit_arith).
Float4 eval_arith(const Instr &p_in, const Float4 &p_a, const Float4 &p_b, bool p_fa, bool p_fb) {
    const Float4 &a = p_a;
    const Float4 &b = p_b;
    switch (p_in.op) {
        case Op::ADD:
            if (p_in.flat) {
                return flat_value(a.w + b.w);
            }
            if (p_fa) {
                return Float4{ b.x, b.y, b.z, a.w + b.w };
            }
            if (p_fb) {
                return Float4{ a.x, a.y, a.z, a.w + b.w };
            }
            return Float4{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
        case Op::SUB:
            if (p_in.flat) {
                return flat_value(a.w - b.w);
            }
            if (p_fa) {
                return Float4{ -b.x, -b.y, -b.z, a.w - b.w };
            }
            if (p_fb) {
                return Float4{ a.x, a.y, a.z, a.w - b.w };
            }
            return Float4{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
        case Op::MUL:
            if (p_in.flat) {
                return flat_value(a.w * b.w);
            }
            if (p_fb) {
                return Float4{ a.x * b.w, a.y * b.w, a.z * b.w, a.w * b.w };
            }
            if (p_fa) {
                return Float4{ a.w * b.x, a.w * b.y, a.w * b.z, a.w * b.w };
            }
            return Float4{ a.x * b.w + a.w * b.x, a.y * b.w + a.w * b.y, a.z * b.w + a.w * b.z, a.w * b.w };
        case Op::NEG:
            return p_in.flat ? flat_value(-a.w) : Float4{ -a.x, -a.y, -a.z, -a.w };
        default: // ABS
            if (p_in.flat) {
                return flat_value(std::fabs(a.w));
            }
            return a.w < 0.0f ? Float4{ -a.x, -a.y, -a.z, std::fabs(a.w) } : Float4{ a.x, a.y, a.z, std::fabs(a.w) };
    }
}

// INPUT, DIST, RSQRT and the noises (GLSL: emit_instr).
Float4 eval_special(const Program &p_program, const Instr &p_in, const Float4 *p_r, const WorldPos &p_pos, int p_lod, float p_s,
        const DefRecord *p_record) {
    switch (p_in.op) {
        case Op::INPUT: {
            const VoxelPos &anchor = p_program.anchors[p_in.index];
            const float v = float(WorldCoord::sat_diff(p_pos.voxel[p_in.axis], anchor[p_in.axis], INPUT_LIMIT)) + axis_of(p_pos.frac, p_in.axis);
            return Float4{ p_in.axis == 0 ? p_s : 0.0f, p_in.axis == 1 ? p_s : 0.0f, p_in.axis == 2 ? p_s : 0.0f, v };
        }
        case Op::T_INPUT: {
            // The record's frame: R (p - anchor), the full 9 mul + 6 add (a runtime rotation can't skip
            // zeros as frame() does). q[a] and the dot product are in the generator's order (archetype_glsl).
            const DefRecord &rec = *p_record;
            float q[3];
            for (int a = 0; a < 3; a++) {
                q[a] = float(WorldCoord::sat_diff(p_pos.voxel[a], rec.anchor[a], INPUT_LIMIT)) + axis_of(p_pos.frac, a);
            }
            const float *row = rec.rotation + 3 * p_in.axis;
            const float v = q[0] * row[0] + q[1] * row[1] + q[2] * row[2];
            return Float4{ p_s * row[0], p_s * row[1], p_s * row[2], v };
        }
        case Op::T_PERLIN3:
        case Op::T_PERLIN2: {
            // The lattice follows the record's anchor (offset - anchor, exact int64), the seed is salted.
            const DefRecord &rec = *p_record;
            const Vector3i64 offset = p_in.offset - rec.anchor;
            const uint32_t seed = p_in.seed ^ rec.salt;
            const Float4 n = p_in.op == Op::T_PERLIN3 ? ExactNoise::perlin3(p_pos, p_in.k, seed, offset)
                                                      : ExactNoise::perlin2(p_pos, p_in.k, seed, offset);
            const float f = p_in.value * exact_pow2(p_lod - p_in.k);
            return Float4{ n.x * f, n.y * f, n.z * f, n.w * p_in.value };
        }
        case Op::RSQRT: {
            const Float4 &a = p_r[p_in.a];
            const float y = det_rsqrt(a.w);
            if (p_in.flat) {
                return flat_value(y);
            }
            const float k = -0.5f * y * y * y;
            return Float4{ a.x * k, a.y * k, a.z * k, y };
        }
        case Op::DIST: {
            const Float4 &a = p_r[p_in.a];
            const float q = p_s * inv_grad_length(a, p_s);
            return Float4{ a.x * q, a.y * q, a.z * q, a.w * q };
        }
        default: { // PERLIN3, PERLIN2
            const Float4 n = p_in.op == Op::PERLIN3 ? ExactNoise::perlin3(p_pos, p_in.k, p_in.seed, p_in.offset)
                                                    : ExactNoise::perlin2(p_pos, p_in.k, p_in.seed, p_in.offset);
            const float f = p_in.value * exact_pow2(p_lod - p_in.k);
            return Float4{ n.x * f, n.y * f, n.z * f, n.w * p_in.value };
        }
    }
}

Float4 eval(const Program &p_program, int p_i, const Float4 *p_r, const WorldPos &p_pos, int p_lod, float p_s, const DefRecord *p_record) {
    const Instr &in = p_program.code[p_i];
    switch (in.op) {
        case Op::CONST:
            return flat_value(in.value);
        case Op::PARAM:
            return flat_value(p_program.params[in.index]);
        case Op::T_PARAM:
            return flat_value(p_record->params[in.index]);
        case Op::ADD:
        case Op::SUB:
        case Op::MUL:
            return eval_arith(in, p_r[in.a], p_r[in.b], p_program.code[in.a].flat, p_program.code[in.b].flat);
        case Op::NEG:
        case Op::ABS:
            return eval_arith(in, p_r[in.a], p_r[in.a], p_program.code[in.a].flat, true);
        case Op::MIN:
            return p_r[in.b].w < p_r[in.a].w ? p_r[in.b] : p_r[in.a];
        case Op::MAX:
            return p_r[in.b].w > p_r[in.a].w ? p_r[in.b] : p_r[in.a];
        case Op::SELECT:
            return p_r[in.a].w < p_r[in.b].w ? p_r[in.c] : p_r[in.d];
        case Op::FLOOR:
            return flat_value(std::floor(p_r[in.a].w));
        default:
            return eval_special(p_program, in, p_r, p_pos, p_lod, p_s, p_record);
    }
}

} // namespace

bool is_flat(const Instr &p_in, const std::vector<Instr> &p_code) {
    auto f = [&](Reg p_r) { return p_code[p_r].flat; };
    switch (p_in.op) {
        case Op::CONST:
        case Op::PARAM:
        case Op::T_PARAM:
        case Op::FLOOR:
            return true;
        case Op::INPUT:
        case Op::T_INPUT:
        case Op::DIST:
        case Op::PERLIN3:
        case Op::PERLIN2:
        case Op::T_PERLIN3:
        case Op::T_PERLIN2:
            return false;
        case Op::NEG:
        case Op::ABS:
        case Op::RSQRT:
            return f(p_in.a);
        case Op::SELECT:
            return f(p_in.c) && f(p_in.d);
        default:
            return f(p_in.a) && f(p_in.b);
    }
}

int op_arity(Op p_op) {
    switch (p_op) {
        case Op::CONST:
        case Op::PARAM:
        case Op::INPUT:
        case Op::PERLIN3:
        case Op::PERLIN2:
        case Op::T_INPUT:
        case Op::T_PARAM:
        case Op::T_PERLIN3:
        case Op::T_PERLIN2:
            return 0;
        case Op::NEG:
        case Op::ABS:
        case Op::FLOOR:
        case Op::RSQRT:
        case Op::DIST:
            return 1;
        case Op::SELECT:
            return 4;
        default:
            return 2;
    }
}

int op_cost(Op p_op) {
    switch (p_op) {
        case Op::CONST:
        case Op::PARAM:
        case Op::T_PARAM:
            return 0;
        case Op::MUL:
            return 7;
        case Op::RSQRT:
        case Op::DIST:
            return 14;
        case Op::T_INPUT: // sat_diff + the full 9 mul + 6 add
            return 24;
        case Op::PERLIN3:
        case Op::T_PERLIN3:
            return 260;
        case Op::PERLIN2:
        case Op::T_PERLIN2:
            return 110;
        default:
            return 4;
    }
}

int Program::cost() const {
    int total = 0;
    for (const Instr &in : code) {
        total += op_cost(in.op);
    }
    return total;
}

String Program::validate(int p_lod_count) const {
    if (code.empty() || code.size() > size_t(MAX_REGISTERS)) {
        return vformat("Archetype: %d instructions (1 to %d).", int64_t(code.size()), MAX_REGISTERS);
    }
    if (output < 0 || output >= Reg(code.size())) {
        return vformat("Archetype: output register %d out of range.", output);
    }
    if (!finite_not_denormal(out_scale) || out_scale == 0.0f) {
        return "Archetype: out_scale must be a normal float.";
    }
    for (Reg i = 0; i < Reg(code.size()); i++) {
        const Instr &in = code[i];
        if (uint8_t(in.op) > uint8_t(Op::T_PERLIN2)) {
            return vformat("Archetype: instruction %d: unknown op %d.", i, int(in.op));
        }
        const Reg operands[4] = { in.a, in.b, in.c, in.d };
        for (int j = 0; j < op_arity(in.op); j++) {
            if (operands[j] < 0 || operands[j] >= i) {
                return vformat("Archetype: instruction %d reads register %d (not computed before it).", i, operands[j]);
            }
        }
        if (in.flat != is_flat(in, code)) {
            return vformat("Archetype: instruction %d has a wrong flat flag.", i);
        }
        if (in.op == Op::INPUT && (in.axis > 2 || in.index < 0 || in.index >= int(anchors.size()))) {
            return vformat("Archetype: instruction %d: bad input axis or anchor.", i);
        }
        if (in.op == Op::PARAM && (in.index < 0 || in.index >= int(params.size()))) {
            return vformat("Archetype: instruction %d: parameter %d out of range.", i, in.index);
        }
        if (in.op == Op::T_INPUT && in.axis > 2) {
            return vformat("Archetype: instruction %d: bad table input axis.", i);
        }
        if (in.op == Op::T_PARAM && (in.index < 0 || in.index >= 8)) {
            return vformat("Archetype: instruction %d: table parameter %d out of range (0 to 7).", i, in.index);
        }
        if (in.op == Op::DIST && code[in.a].flat) {
            return vformat("Archetype: instruction %d: DIST of a flat register.", i);
        }
        if (in.op == Op::RSQRT) {
            // det_rsqrt reads the bits: a zero would give different non-zero results for -0 and +0.
            const Instr &g = code[in.a];
            auto positive = [&](Reg p_r) { return code[p_r].op == Op::CONST && code[p_r].value > 0.0f; };
            if (g.op != Op::MAX || !(positive(g.a) || positive(g.b))) {
                return vformat("Archetype: instruction %d: RSQRT of a register that isn't max(x, c > 0).", i);
            }
        }
        if ((in.op == Op::CONST || in.op == Op::PERLIN3 || in.op == Op::PERLIN2 || in.op == Op::T_PERLIN3 || in.op == Op::T_PERLIN2) &&
                !finite_not_denormal(in.value)) {
            return vformat("Archetype: instruction %d: constant %s is denormal or not finite.", i, in.value);
        }
        if (in.op == Op::PERLIN3 || in.op == Op::PERLIN2 || in.op == Op::T_PERLIN3 || in.op == Op::T_PERLIN2) {
            ExactNoise::OctaveSet set;
            set.octaves.push_back(ExactNoise::Octave{ in.k, in.seed, in.offset, in.value });
            if (in.k < 0 || in.k > ExactNoise::MAX_K || !ExactNoise::in_exact_range(in.value)) {
                return vformat("Archetype: instruction %d: noise k=%d amplitude %s out of range.", i, in.k, in.value);
            }
            const String error = ExactNoise::gradient_range_error(set, p_lod_count);
            if (!error.is_empty()) {
                return vformat("Archetype: instruction %d: ", i) + error;
            }
        }
    }
    for (float p : params) {
        if (!finite_not_denormal(p)) {
            return "Archetype: a parameter is denormal or not finite.";
        }
    }
    for (const VoxelPos &a : anchors) {
        for (int axis = 0; axis < 3; axis++) {
            if (a[axis] > (int64_t(1) << 61) || a[axis] < -(int64_t(1) << 61)) {
                return "Archetype: anchor beyond 2^61 voxels.";
            }
        }
    }
    return String();
}

void run_registers(const Program &p_program, const WorldPos &p_pos, int p_lod, std::vector<Float4> &r_regs, const DefRecord *p_record) {
    const float s = exact_pow2(p_lod);
    r_regs.resize(p_program.code.size());
    Float4 *r = r_regs.data();
    for (int i = 0; i < int(p_program.code.size()); i++) {
        r[i] = eval(p_program, i, r, p_pos, p_lod, s, p_record);
    }
}

Float4 finish(const Program &p_program, const Float4 &p_out, int p_lod) {
    const Float4 &o = p_out;
    const float v = o.w * p_program.out_scale;
    if (!p_program.normalize) {
        return Float4{ o.x, o.y, o.z, v };
    }
    const float gr = inv_grad_length(o, exact_pow2(p_lod));
    return Float4{ o.x * gr, o.y * gr, o.z * gr, v };
}

Float4 run(const Program &p_program, const WorldPos &p_pos, int p_lod, const DefRecord *p_record) {
    thread_local std::vector<Float4> regs;
    run_registers(p_program, p_pos, p_lod, regs, p_record);
    return finish(p_program, regs[p_program.output], p_lod);
}

bool exact_domain(const Program &p_program, const std::vector<Float4> &p_regs) {
    const float lo = exact_pow2(-63);
    const float hi = exact_pow2(63);
    for (const Instr &in : p_program.code) {
        if (in.op == Op::RSQRT && !(p_regs[in.a].w > 0.0f)) {
            return false;
        }
    }
    for (const Float4 &r : p_regs) {
        for (const float f : { r.x, r.y, r.z, r.w }) {
            const float m = std::fabs(f);
            if (f != 0.0f && !(m >= lo && m <= hi)) {
                return false;
            }
        }
    }
    return true;
}

uint32_t trace_hash(const std::vector<Float4> &p_regs) {
    uint32_t h = 0x811C9DC5u;
    for (const Float4 &r : p_regs) {
        for (const float f : { r.x, r.y, r.z, r.w }) {
            h = (h ^ zero_canonical(float_bits(f))) * 0x01000193u;
        }
    }
    return h;
}

} // namespace Archetype
