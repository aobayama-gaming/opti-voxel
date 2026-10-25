#include "archetype_probes.hpp"

#include "archetype_glsl.hpp"

#include <cmath>
#include <functional>

namespace ArchetypeProbes {

using namespace Archetype;
using ExactNoise::float_bits;

Probe probe(const ov::str &p_name, const Program &p_program, bool p_trace) {
    Probe p;
    const int stride = p_trace ? 5 : 4;
    p.name = "arch_" + p_name;
    p.words = stride * PROBE_LOD_COUNT;
    p.is_float = true;
    ov::str lods;
    for (int i = 0; i < PROBE_LOD_COUNT; i++) {
        lods += (i ? ", " : "") + ov::itos(PROBE_LODS[i]);
    }
    p.glsl = generate_glsl(p_program, p.name, p_trace);
    p.glsl += ov::format(R"glsl(void probe_%s(WorldPos p, uint b) {
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

namespace {

using ShapeFn = Reg (*)(Builder &, const Local &, float);

struct Shape {
    const char *name;
    ShapeFn fn;
    // The same at the same offset from any origin: false for the fBm shapes (an absolute noise lattice) and the plane
    // (unbounded: a copy 2^31 voxels away reads saturated inputs, a huge negative value, and wins the min).
    bool origin_relative = true;
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
    { "plane", [](Builder &b, const Local &l, float s) { return b.plane(l, 0.6f, 0.8f, 0.0f, 0.1f * s); }, false },
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
            },
            false },
    { "height",
            [](Builder &b, const Local &l, float) {
                const Reg h = b.fbm(height_noise());
                return b.dist(b.sub(l.y, h));
            },
            false },
};

} // namespace

std::vector<Probe> make(const std::vector<VoxelPos> &p_anchors, const std::vector<VoxelPos> &p_frame_anchors, float p_size, int p_fuzz) {
    std::vector<Probe> probes;
    // A union of p_shapes, one copy per anchor of p_at.
    auto shapes_probe = [&](const ov::str &p_name, const std::vector<const Shape *> &p_shapes, const std::vector<VoxelPos> &p_at, bool p_trace) {
        Builder b;
        Reg u = -1;
        for (const VoxelPos &a : p_at) {
            const Local l = b.local(b.anchor(a));
            for (const Shape *shape : p_shapes) {
                const Reg d = shape->fn(b, l, p_size);
                u = u < 0 ? d : b.min(u, d);
            }
        }
        Program prog = b.finish(u);
        prog.normalize = true;
        prog.out_scale = 0.5f;
        return probe(p_name, prog, p_trace);
    };
    std::vector<const Shape *> relative;
    for (const Shape &shape : SHAPES) {
        probes.push_back(shapes_probe(shape.name, { &shape }, p_anchors, true));
        if (shape.origin_relative) {
            relative.push_back(&shape);
        }
    }
    // Every noise-free shape at the same offset from every origin: compared across origins (origin_relative).
    Probe frames = shapes_probe("frames", relative, p_frame_anchors, false);
    frames.origin_relative = true;
    probes.push_back(frames);
    for (int i = 0; i < p_fuzz; i++) {
        probes.push_back(probe(ov::format("fuzz_%d", i), random_program(uint32_t(i), p_anchors, 8 + i % 33), true));
    }
    return probes;
}

} // namespace ArchetypeProbes
