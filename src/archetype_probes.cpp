#include "archetype_probes.hpp"

#include "archetype_glsl.hpp"

#include <cmath>
#include <functional>

namespace ArchetypeProbes {

using namespace Archetype;
using ExactNoise::float_bits;

Probe probe(const String &p_name, const Program &p_program, bool p_trace) {
    Probe p;
    const int stride = p_trace ? 5 : 4;
    p.name = "arch_" + p_name;
    p.words = stride * PROBE_LOD_COUNT;
    p.is_float = true;
    String lods;
    for (int i = 0; i < PROBE_LOD_COUNT; i++) {
        lods += (i ? ", " : "") + itos(PROBE_LODS[i]);
    }
    p.glsl = generate_glsl(p_program, p.name, p_trace);
    p.glsl += vformat(R"glsl(void probe_%s(WorldPos p, uint b) {
    const int lods[%d] = int[%d](%s);
    for (uint i = 0u; i < %du; i++) {
        %s
        for (uint j = 0u; j < 4u; j++) {
            uint u = floatBitsToUint(v[j]);
            results[b + %du * i + j] = (u << 1u) == 0u ? 0u : u;
        }
        %s
    }
}
)glsl",
            p.name, PROBE_LOD_COUNT, PROBE_LOD_COUNT, lods, PROBE_LOD_COUNT,
            p_trace ? "uint t;\n        vec4 v = " + p.name + "(p, int(params.lod_bias) + lods[i], t);"
                    : "vec4 v = " + p.name + "(p, int(params.lod_bias) + lods[i]);",
            stride, p_trace ? "results[b + 5u * i + 4u] = t;" : "");
    p.cpu = [p_program, stride](const WorldPos &q, uint32_t *r) {
        thread_local std::vector<Float4> regs;
        bool exact = true;
        for (int i = 0; i < PROBE_LOD_COUNT; i++) {
            run_registers(p_program, q, PROBE_LODS[i], regs);
            const Float4 o = finish(p_program, regs[p_program.output], PROBE_LODS[i]);
            uint32_t *w = r + stride * i;
            w[0] = zero_canonical(float_bits(o.x));
            w[1] = zero_canonical(float_bits(o.y));
            w[2] = zero_canonical(float_bits(o.z));
            w[3] = zero_canonical(float_bits(o.w));
            if (stride == 5) {
                w[4] = trace_hash(regs);
            }
            exact = exact && exact_domain(p_program, regs);
        }
        return exact;
    };
    return p;
}

Instance operator_scene(Combine p_op, const VoxelPos &p_anchor, float p_size) {
    const int k = int(std::lround(std::log2(p_size * 0.25f)));
    Instance root;
    root.def = rock_def(0.4f * p_size, ExactNoise::make_fbm(false, 0xB0C4u, k, 3, 0.05 * p_size, 0.5));
    root.anchor = p_anchor;
    root.set_rotation_y(0.7);
    Instance::Child c;
    c.op = p_op;
    c.k = 0.1f * p_size;
    switch (p_op) {
        case Combine::SMOOTH_UNION:
        case Combine::UNION:
            c.instance.def = capsule_def(0.8f * p_size, 0.08f * p_size);
            c.instance.anchor = VoxelPos(int64_t(0.3f * p_size), 0, 0);
            break;
        case Combine::SUBTRACT:
        case Combine::SMOOTH_SUBTRACT:
            c.instance.def = sphere_def(0.2f * p_size);
            c.instance.anchor = VoxelPos(0, int64_t(0.35f * p_size), 0);
            break;
        case Combine::INTERSECT:
            c.instance.def = box_def(p_size, 0.3f * p_size, p_size);
            c.instance.set_rotation_y(0.3);
            break;
        case Combine::BLEND:
            c.instance.def = rock_def(0.3f * p_size, ExactNoise::make_fbm(false, 0xB1E2u, k, 2, 0.05 * p_size, 0.5));
            c.instance.anchor = VoxelPos(0, -int64_t(0.1f * p_size), int64_t(0.2f * p_size));
            c.instance.seed = 7;
            c.weight = ramp_def(0.5f * p_size);
            break;
    }
    root.children.push_back(c);
    return root;
}

Instance test_scene(const VoxelPos &p_anchor, float p_size) {
    Instance root = operator_scene(Combine::SMOOTH_UNION, p_anchor, p_size);
    for (const Combine op : { Combine::UNION, Combine::SUBTRACT, Combine::SMOOTH_SUBTRACT, Combine::BLEND, Combine::INTERSECT }) {
        Instance::Child c = operator_scene(op, VoxelPos(), p_size).children[0];
        if (op == Combine::UNION) {
            c.instance.anchor = VoxelPos(-int64_t(0.5f * p_size), int64_t(0.2f * p_size), 0);
            c.instance.set_rotation_y(-1.1);
        }
        root.children.push_back(c);
    }
    return root;
}

// p_scene at every anchor, unioned (the first copy is the root).
Instance copies(const std::function<Instance(const VoxelPos &)> &p_scene, const std::vector<VoxelPos> &p_anchors) {
    Instance root = p_scene(p_anchors[0]);
    for (size_t i = 1; i < p_anchors.size(); i++) {
        Instance::Child c;
        c.instance = p_scene(p_anchors[i] - p_anchors[0]);
        root.children.push_back(c);
    }
    return root;
}

namespace {

using ShapeFn = Reg (*)(Builder &, const Local &, float);

struct Shape {
    const char *name;
    ShapeFn fn;
};

const ExactNoise::OctaveSet &rock_noise() {
    static const ExactNoise::OctaveSet set = ExactNoise::make_fbm(false, 0x50C4u, 10, 4, 300.0, 0.5);
    return set;
}

const ExactNoise::OctaveSet &height_noise() {
    static const ExactNoise::OctaveSet set = ExactNoise::make_fbm(true, 0x4E16u, 11, 5, 600.0, 0.5);
    return set;
}

const Shape SHAPES[] = {
    { "length", [](Builder &b, const Local &l, float) { return b.length(l.x, l.y, l.z); } },
    { "sphere", [](Builder &b, const Local &l, float s) { return b.sphere(l, s); } },
    { "box", [](Builder &b, const Local &l, float s) { return b.box(l, 0.6f * s, 0.3f * s, s); } },
    { "capsule", [](Builder &b, const Local &l, float s) { return b.capsule(l, s, 0.25f * s); } },
    { "cylinder", [](Builder &b, const Local &l, float s) { return b.cylinder(l, 0.5f * s, 0.3f * s); } },
    { "plane", [](Builder &b, const Local &l, float s) { return b.plane(l, 0.6f, 0.8f, 0.0f, 0.1f * s); } },
    { "smooth_min",
            [](Builder &b, const Local &l, float s) {
                const Reg a = b.sphere(l, 0.5f * s);
                return b.smooth_min(a, b.capsule(l, s, 0.2f * s), 0.2f * s);
            } },
    { "smooth_max",
            [](Builder &b, const Local &l, float s) {
                const Reg a = b.sphere(l, 0.5f * s);
                return b.smooth_max(a, b.neg(b.box(l, s, 0.1f * s, s)), 0.1f * s);
            } },
    { "rock",
            [](Builder &b, const Local &l, float s) {
                const Reg a = b.sphere(l, s);
                return b.dist(b.add(a, b.fbm(rock_noise())));
            } },
    { "height",
            [](Builder &b, const Local &l, float) {
                const Reg h = b.fbm(height_noise());
                return b.dist(b.sub(l.y, h));
            } },
};

} // namespace

std::vector<Probe> make(const std::vector<VoxelPos> &p_anchors, const std::vector<VoxelPos> &p_frame_anchors, float p_size, int p_fuzz) {
    std::vector<Probe> probes;
    auto scene_probe = [&](const String &p_name, const Instance &p_root, bool p_trace) {
        Compiled c = compile(p_root);
        ERR_FAIL_COND_V_MSG(!c.error.is_empty(), Probe(), c.error);
        c.program.normalize = true;
        c.program.out_scale = 0.5f;
        return probe(p_name, c.program, p_trace);
    };
    // One copy per origin for the scenes (they are ~200 instructions each).
    std::vector<VoxelPos> scene_anchors;
    for (const VoxelPos &a : p_frame_anchors) {
        scene_anchors.push_back(a + VoxelPos(2300, -900, -1700));
    }
    const char *op_names[] = { "union", "smooth_union", "subtract", "smooth_subtract", "intersect", "blend" };
    for (int op = 0; op < 6; op++) {
        probes.push_back(scene_probe(op_names[op], copies([&](const VoxelPos &a) { return operator_scene(Combine(op), a, p_size); }, scene_anchors), true));
    }
    auto scene = [&](const VoxelPos &a) { return test_scene(a, p_size); };
    probes.push_back(scene_probe("scene", copies(scene, scene_anchors), true));
    // The same scene at the same offset from every origin: compared across origins (origin_relative).
    Probe frames = scene_probe("frames", copies(scene, p_frame_anchors), false);
    frames.origin_relative = true;
    probes.push_back(frames);
    for (const Shape &shape : SHAPES) {
        Builder b;
        Reg u = -1;
        for (const VoxelPos &a : p_anchors) {
            const Reg d = shape.fn(b, b.local(b.anchor(a)), p_size);
            u = u < 0 ? d : b.min(u, d);
        }
        Program prog = b.finish(u);
        prog.normalize = true;
        prog.out_scale = 0.5f;
        probes.push_back(probe(shape.name, prog, true));
    }
    for (int i = 0; i < p_fuzz; i++) {
        probes.push_back(probe(vformat("fuzz_%d", i), random_program(uint32_t(i), p_anchors, 8 + i % 33), true));
    }
    return probes;
}

} // namespace ArchetypeProbes
