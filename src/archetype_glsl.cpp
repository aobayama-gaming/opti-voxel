#include "archetype_glsl.hpp"

namespace Archetype {

using ExactNoise::exact_pow2;
using ExactNoise::glsl_float;

namespace {

String r(Reg p_reg) {
    return "r" + itos(p_reg);
}

String flat_vec(const String &p_value) {
    return "vec4(0.0, 0.0, 0.0, " + p_value + ")";
}

// det_rsqrt of |grad p_v|^2 floored (inv_grad_length in archetype_ir.cpp): DIST and the epilogue.
String inv_grad_length(const String &p_v) {
    return "det_rsqrt(max(" + p_v + ".x * " + p_v + ".x + " + p_v + ".y * " + p_v + ".y + " + p_v + ".z * " + p_v + ".z, " +
            glsl_float(exact_pow2(GRAD_FLOOR_EXP)) + " * s * s))";
}

// ADD, SUB, MUL, NEG, ABS: GLSL side of eval_arith (archetype_ir.cpp). Vector ops are per component, in
// the interpreter's order.
String arith(const Instr &p_in, const std::vector<Instr> &p_code) {
    String a = r(p_in.a);
    String b = r(p_in.b);
    const bool fa = p_code[p_in.a].flat;
    const bool fb = p_in.b >= 0 && p_code[p_in.b].flat;
    switch (p_in.op) {
        case Op::ADD:
            if (p_in.flat) {
                return flat_vec(a + ".w + " + b + ".w");
            }
            return fa ? "vec4(" + b + ".xyz, " + a + ".w + " + b + ".w)" : (fb ? "vec4(" + a + ".xyz, " + a + ".w + " + b + ".w)" : a + " + " + b);
        case Op::SUB:
            if (p_in.flat) {
                return flat_vec(a + ".w - " + b + ".w");
            }
            return fa ? "vec4(-" + b + ".xyz, " + a + ".w - " + b + ".w)" : (fb ? "vec4(" + a + ".xyz, " + a + ".w - " + b + ".w)" : a + " - " + b);
        case Op::MUL:
            if (p_in.flat) {
                return flat_vec(a + ".w * " + b + ".w");
            }
            if (fb) {
                return a + " * " + b + ".w";
            }
            if (fa) {
                return a + ".w * " + b;
            }
            return "vec4(" + a + ".xyz * " + b + ".w + " + a + ".w * " + b + ".xyz, " + a + ".w * " + b + ".w)";
        case Op::NEG:
            return p_in.flat ? flat_vec("-" + a + ".w") : "-" + a;
        default: // ABS
            return p_in.flat ? flat_vec("abs(" + a + ".w)") : "vec4(" + a + ".w < 0.0 ? -" + a + ".xyz : " + a + ".xyz, abs(" + a + ".w))";
    }
}

// The statement(s) computing register p_i.
String statement(const Program &p_program, int p_i) {
    const Instr &in = p_program.code[p_i];
    String out = "    precise vec4 " + r(p_i) + " = ";
    String a = in.a >= 0 ? r(in.a) : String();
    String id = itos(p_i);
    switch (in.op) {
        case Op::CONST:
            return out + flat_vec(glsl_float(in.value)) + ";\n";
        case Op::PARAM:
            return out + flat_vec(glsl_float(p_program.params[in.index])) + ";\n";
        case Op::T_PARAM:
            return out + flat_vec("rec.params[" + itos(in.index) + "]") + ";\n";
        case Op::T_INPUT: {
            // Mirror of eval_special's T_INPUT: sat_diff on the record's anchor, the full rotation.
            const String row = "rec.rot[" + itos(in.axis) + "]";
            return "    precise ivec3 d" + id + " = sat_diff(p, rec.anchor_lo.xyz, rec.anchor_hi.xyz, " + itos(INPUT_LIMIT) + ");\n" +
                    "    precise float qx" + id + " = float(d" + id + ".x) + p.frac.x;\n" +
                    "    precise float qy" + id + " = float(d" + id + ".y) + p.frac.y;\n" +
                    "    precise float qz" + id + " = float(d" + id + ".z) + p.frac.z;\n" + out + "vec4(s * " + row + ".x, s * " + row +
                    ".y, s * " + row + ".z, qx" + id + " * " + row + ".x + qy" + id + " * " + row + ".y + qz" + id + " * " + row + ".z);\n";
        }
        case Op::INPUT: {
            const VoxelPos &anchor = p_program.anchors[in.index];
            const char *axis = in.axis == 0 ? "x" : (in.axis == 1 ? "y" : "z");
            String g = in.axis == 0 ? "s, 0.0, 0.0" : (in.axis == 1 ? "0.0, s, 0.0" : "0.0, 0.0, s");
            return out + "vec4(" + g + ", float(sat_diff(p, " + ExactNoise::glsl_lo(anchor) + ", " + ExactNoise::glsl_hi(anchor) + ", " +
                    itos(INPUT_LIMIT) + ")." + axis + ") + p.frac." + axis + ");\n";
        }
        case Op::MIN:
            return out + r(in.b) + ".w < " + a + ".w ? " + r(in.b) + " : " + a + ";\n";
        case Op::MAX:
            return out + r(in.b) + ".w > " + a + ".w ? " + r(in.b) + " : " + a + ";\n";
        case Op::SELECT:
            return out + a + ".w < " + r(in.b) + ".w ? " + r(in.c) + " : " + r(in.d) + ";\n";
        case Op::FLOOR:
            return out + flat_vec("floor(" + a + ".w)") + ";\n";
        case Op::RSQRT:
            if (in.flat) {
                return out + flat_vec("det_rsqrt(" + a + ".w)") + ";\n";
            }
            return "    precise float y" + id + " = det_rsqrt(" + a + ".w);\n" + out + "vec4(" + a + ".xyz * (" + glsl_float(-0.5f) + " * y" + id +
                    " * y" + id + " * y" + id + "), y" + id + ");\n";
        case Op::DIST:
            return "    precise float q" + id + " = s * " + inv_grad_length(a) + ";\n" + out + a + " * q" + id + ";\n";
        case Op::PERLIN3:
        case Op::PERLIN2:
            // The band rule (ExactNoise::band_kept): lod is uniform per chunk, so a dropped octave costs nothing.
            return out + "vec4(0.0);\n" + vformat("    if (lod <= %d) {\n", in.k - ExactNoise::BAND_SHIFT) +
                    vformat("        precise vec4 n%d = %s(p, %d, %s, %s, %s);\n", p_i, in.op == Op::PERLIN3 ? "perlin3" : "perlin2", in.k,
                            ExactNoise::glsl_uint(in.seed), ExactNoise::glsl_lo(in.offset), ExactNoise::glsl_hi(in.offset)) +
                    vformat("        %s = vec4(n%d.xyz * (%s * exact_pow2(lod - %d)), n%d.w * %s);\n    }\n", r(p_i), p_i, glsl_float(in.value), in.k, p_i,
                            glsl_float(in.value));
        case Op::T_PERLIN3:
        case Op::T_PERLIN2:
            // Mirror of eval_special's T_PERLIN*: the band rule, then the lattice offset is the Def's minus the
            // record's anchor (exact int64, i64_sub), the seed salted by the record's salt.
            return out + "vec4(0.0);\n" + vformat("    if (lod <= %d) {\n", in.k - ExactNoise::BAND_SHIFT) +
                    vformat("        precise uvec3 o%d = %s;\n", p_i, ExactNoise::glsl_lo(in.offset)) +
                    vformat("        precise ivec3 ohi%d = %s;\n", p_i, ExactNoise::glsl_hi(in.offset)) +
                    vformat("        i64_sub(o%d, ohi%d, rec.anchor_lo.xyz, rec.anchor_hi.xyz);\n", p_i, p_i) +
                    vformat("        precise vec4 n%d = %s(p, %d, %s ^ rec.anchor_lo.w, o%d, ohi%d);\n", p_i,
                            in.op == Op::T_PERLIN3 ? "perlin3" : "perlin2", in.k, ExactNoise::glsl_uint(in.seed), p_i, p_i) +
                    vformat("        %s = vec4(n%d.xyz * (%s * exact_pow2(lod - %d)), n%d.w * %s);\n    }\n", r(p_i), p_i, glsl_float(in.value), in.k, p_i,
                            glsl_float(in.value));
        default:
            return out + arith(in, p_program.code) + ";\n";
    }
}

} // namespace

namespace {

String generate(const Program &p_program, const String &p_name, bool p_trace, bool p_table) {
    String code = "vec4 " + p_name + "(WorldPos p, int lod" + (p_table ? ", DefRecord rec" : "") + (p_trace ? ", out uint trace" : "") + ") {\n";
    code += "    precise float s = exact_pow2(lod);\n";
    for (int i = 0; i < int(p_program.code.size()); i++) {
        code += statement(p_program, i);
    }
    if (p_trace) {
        // trace_hash: FNV-1a of the zero_canonical bits (integer ops: the compiler can't fold them).
        code += "    trace = 0x811C9DC5u;\n    uvec4 u;\n";
        for (int i = 0; i < int(p_program.code.size()); i++) {
            code += "    u = floatBitsToUint(" + r(i) + ");\n";
            for (const char *c : { "x", "y", "z", "w" }) {
                code += vformat("    trace = (trace ^ ((u.%s << 1u) == 0u ? 0u : u.%s)) * 0x01000193u;\n", c, c);
            }
        }
    }
    // Epilogue: finish() in archetype_ir.cpp.
    code += "    precise vec4 o = " + r(p_program.output) + ";\n";
    code += "    precise float v = o.w * " + glsl_float(p_program.out_scale) + ";\n";
    if (p_program.normalize) {
        code += "    precise float gr = " + inv_grad_length("o") + ";\n";
        code += "    return vec4(o.xyz * gr, v);\n}\n";
    } else {
        code += "    return vec4(o.xyz, v);\n}\n";
    }
    return code;
}

} // namespace

String generate_glsl(const Program &p_program, const String &p_name, bool p_trace) {
    return generate(p_program, p_name, p_trace, false);
}

String generate_def_glsl(const Program &p_program, const String &p_name, bool p_trace) {
    return generate(p_program, p_name, p_trace, true);
}

} // namespace Archetype
