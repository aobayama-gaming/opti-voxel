#include "cascade_fold_glsl.hpp"

#include "exact_noise.hpp"

#include <algorithm>

using namespace Cascade;

namespace {

// Cascade::node_axis and node_axis_pass, op for op: products flushed one by one, summed in the CPU's order.
constexpr const char GLSL_FOLD_HELPERS[] = R"glsl(
// Cascade::NodeAxis: whether t > 0, t, 1 - t, the value's and the derivative's weights on (a, slope a, b, slope b).
struct FoldAxis { bool mid; float t; float u; vec4 h; vec4 dh; };

FoldAxis fold_axis(float t) {
    FoldAxis w;
    w.mid = t > 0.0;
    w.t = t;
    precise float u = 1.0 - t;
    precise float t2 = t * t;
    precise float t3 = t2 * t;
    precise float h0 = (t3 * 2.0 - t2 * 3.0) + 1.0;
    precise float h1 = (t3 - t2 * 2.0) + t;
    precise float h2 = t2 * 3.0 - t3 * 2.0;
    precise float h3 = t3 - t2;
    precise float d0 = t2 * 6.0 - t * 6.0;
    precise float d1 = (t2 * 3.0 - t * 4.0) + 1.0;
    precise float d2 = t * 6.0 - t2 * 6.0;
    precise float d3 = t2 * 3.0 - t * 2.0;
    w.u = u;
    w.h = vec4(h0, h1, h2, h3);
    w.dh = vec4(d0, d1, d2, d3);
    return w;
}

float fold_flush(float v) {
    return /*FLUSH_V*/;
}

float fold_sum(vec4 k, float a, float sa, float b, float sb) {
    precise float p0 = k.x * a;
    precise float p1 = k.y * sa;
    precise float p2 = k.z * b;
    precise float p3 = k.w * sb;
    precise float r = ((fold_flush(p0) + fold_flush(p1)) + fold_flush(p2)) + fold_flush(p3);
    return r;
}

float fold_lerp(FoldAxis w, float a, float b) {
    precise float pa = w.u * a;
    precise float pb = w.t * b;
    precise float r = fold_flush(pa) + fold_flush(pb);
    return r;
}

// One axis pass on a group's (value, d/dx, d/dy, d/dz): the low node a, the high one b (Cascade::node_axis_pass).
vec4 fold_pass_x(vec4 a, vec4 b, FoldAxis w) {
    return vec4(fold_sum(w.h, a.x, a.y, b.x, b.y), fold_sum(w.dh, a.x, a.y, b.x, b.y), fold_lerp(w, a.z, b.z), fold_lerp(w, a.w, b.w));
}

vec4 fold_pass_y(vec4 a, vec4 b, FoldAxis w) {
    return vec4(fold_sum(w.h, a.x, a.z, b.x, b.z), fold_lerp(w, a.y, b.y), fold_sum(w.dh, a.x, a.z, b.x, b.z), fold_lerp(w, a.w, b.w));
}

vec4 fold_pass_z(vec4 a, vec4 b, FoldAxis w) {
    return vec4(fold_sum(w.h, a.x, a.w, b.x, b.w), fold_lerp(w, a.y, b.y), fold_lerp(w, a.z, b.z), fold_sum(w.dh, a.x, a.w, b.x, b.w));
}

vec4 fold_load(bool from_a, uint v, uint slot) {
    uint plane_a = uint(scratch_a.length()) / /*STRIDE*/u;
    uint plane_b = uint(scratch_b.length()) / /*STRIDE*/u;
    return from_a ? scratch_a[v * plane_a + slot] : scratch_b[v * plane_b + slot];
}

// The x pass at node slot (and slot + 1 where mid).
vec4 fold_row(bool from_a, uint v, uint slot, FoldAxis wx) {
    vec4 a = fold_load(from_a, v, slot);
    if (!wx.mid) {
        return a;
    }
    return fold_pass_x(a, fold_load(from_a, v, slot + 1u), wx);
}

// Cascade::node_interp of the group in scratch plane v: its cell's low node at slot of a box of nn points per axis.
vec4 fold_group(bool from_a, uint v, uint slot, uint nn, FoldAxis wx, FoldAxis wy, FoldAxis wz, float s) {
    vec4 y0 = fold_row(from_a, v, slot, wx);
    if (wy.mid) {
        y0 = fold_pass_y(y0, fold_row(from_a, v, slot + nn, wx), wy);
    }
    vec4 r = y0;
    if (wz.mid) {
        vec4 y1 = fold_row(from_a, v, slot + nn * nn, wx);
        if (wy.mid) {
            y1 = fold_pass_y(y1, fold_row(from_a, v, slot + nn * nn + nn, wx), wy);
        }
        r = fold_pass_z(y0, y1, wz);
    }
    precise float sx = r.y * s;
    precise float sy = r.z * s;
    precise float sz = r.w * s;
    return vec4(fold_flush(r.x), fold_flush(sx), fold_flush(sy), fold_flush(sz));
}
)glsl";

// The node level's fold function: the cell of the item's point, the bases, then each group's channels.
String fold_function(const Spec &p_spec, const FieldRange &p_range, int p_node) {
    String code = vformat(R"glsl(
// Cascade::node_cell, then node_interp of the groups whose node level is %d (read in the scratch where that level wrote
// them: item c's region, a box of level_bound(%d - lod) points per axis).
void cascade_fold_%d(inout Ch ch, int m, int lod, uvec3 lo_lo, ivec3 lo_hi, uvec3 p_lo, ivec3 p_hi, uint c) {
    int jn = %d - lod;
    int d = %d - m;
    uvec3 n_lo = lo_lo;
    ivec3 n_hi = lo_hi;
    i64_asr(n_lo, n_hi, jn);
    uvec3 q_lo = p_lo;
    ivec3 q_hi = p_hi;
    i64_asr(q_lo, q_hi, d);
    uvec3 q = q_lo - n_lo; // the low node in the node box: a small difference, exact on the low words
    uint nn = cascade_level_bound(jn);
    bool from_a = (jn & 1) != 0;
    uint slot = c * cascade_stride(from_a) + q.x + nn * (q.y + nn * q.z);
    uvec3 o = p_lo & uvec3((1u << uint(d)) - 1u);
    float s = uintBitsToFloat(uint(127 - d) << 23u); // 2^-d
    precise float tx = float(o.x) * s;
    precise float ty = float(o.y) * s;
    precise float tz = float(o.z) * s;
    FoldAxis wx = fold_axis(tx);
    FoldAxis wy = fold_axis(ty);
    FoldAxis wz = fold_axis(tz);
)glsl",
            p_node, p_node, p_node, p_node, p_node);
    for (int g = 1; g < p_spec.group_count(); g++) {
        if (!p_spec.folds(p_range, g) || p_spec.node_level(g) != p_node) {
            continue;
        }
        const int b = p_spec.gpu_base(g);
        code += vformat("    {\n        vec4 r = fold_group(from_a, %du, slot, nn, wx, wy, wz, s);\n", b / 4);
        code += vformat("        ch.c[%d] = r.x;\n        ch.c[%d] = r.y;\n        ch.c[%d] = r.z;\n        ch.c[%d] = r.w;\n    }\n", b, b + 1, b + 2, b + 3);
    }
    return code + "}\n";
}

} // namespace

namespace CascadeFoldGLSL {

std::vector<int> node_levels(const Spec &p_spec, const FieldRange &p_range) {
    std::vector<int> levels;
    for (int g = 1; g < p_spec.group_count(); g++) {
        if (p_spec.folds(p_range, g) && std::find(levels.begin(), levels.end(), p_spec.node_level(g)) == levels.end()) {
            levels.push_back(p_spec.node_level(g));
        }
    }
    std::sort(levels.begin(), levels.end());
    return levels;
}

String carry_flag(const Spec &p_spec, const FieldRange &p_range, int p_group) {
    return p_group > 0 && p_spec.folds(p_range, p_group) ? vformat("g_carry_%d", p_spec.node_level(p_group)) : String();
}

String parent_base(bool p_fold) {
    return p_fold ? "c * cascade_stride((pc.j & 1) == 0)" : "c * np * np * np";
}

String store_base(bool p_fold) {
    return p_fold ? "c * cascade_stride((pc.j & 1) != 0)" : "c * n * n * n";
}

String stride_functions(const String &p_item_size) {
    return vformat(R"glsl(
// Cascade::level_bound of the items (%s points per axis), and the scratch's points per item of buffer A (odd levels) or
// B (even ones): item c's region starts at c times it (CascadeGLSL::scratch_points_a / _b).
uint cascade_level_bound(int j) {
    uint n = %s;
    for (int k = 0; k < j; k++) {
        n = (n + 1u) / 2u + 1u;
    }
    return n;
}

uint cascade_stride(bool a) {
    uint n = cascade_level_bound(a ? 1 : 2);
    return n * n * n;
}
)glsl",
            p_item_size, p_item_size);
}

String flag_declarations(const Spec &p_spec, const FieldRange &p_range) {
    String code;
    for (const int node : node_levels(p_spec, p_range)) {
        code += vformat("bool g_carry_%d = true; // the groups of node level %d are carried at this level (Spec::folded: false where they are folded)\n", node, node);
    }
    return code.is_empty() ? code : code + "\n";
}

String flag_values(const Spec &p_spec, const FieldRange &p_range) {
    String code;
    for (const int node : node_levels(p_spec, p_range)) {
        code += vformat("    g_carry_%d = g_fields && m >= %d;\n", node, node);
    }
    return code;
}

String fold_functions(const Spec &p_spec, const FieldRange &p_range, int p_stride) {
    const std::vector<int> levels = node_levels(p_spec, p_range);
    if (levels.empty()) {
        return String();
    }
    const String v = "v";
    String code = String(GLSL_FOLD_HELPERS).replace("/*STRIDE*/", String::num_int64(p_stride)).replace("/*FLUSH_V*/", glsl_flush_of(v));
    for (const int node : levels) {
        code += fold_function(p_spec, p_range, node);
    }
    return code;
}

String fold_calls(const Spec &p_spec, const FieldRange &p_range) {
    const std::vector<int> levels = node_levels(p_spec, p_range);
    if (levels.empty()) {
        return String();
    }
    String code = "    if (g_fields) {\n        // The folded groups, from their nodes (Cascade::node_interp), before the insertions read them.\n";
    code += "        uvec3 fold_lo = m_lo;\n        ivec3 fold_hi = m_hi;\n        i64_add(fold_lo, fold_hi, ivec3(i));\n";
    for (const int node : levels) {
        code += vformat("        if (m < %d) {\n            cascade_fold_%d(ch, m, lod, lo_lo, lo_hi, fold_lo, fold_hi, c);\n        }\n", node, node);
    }
    return code + "    }\n";
}

} // namespace CascadeFoldGLSL
