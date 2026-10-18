#include "archetype_probes.hpp"

#include "archetype_glsl.hpp"

namespace ArchetypeProbes {

using namespace Archetype;
using ExactNoise::float_bits;

Probe probe(const String &p_name, const Program &p_program) {
    Probe p;
    p.name = "arch_" + p_name;
    p.words = 5 * PROBE_LOD_COUNT;
    p.is_float = true;
    String lods;
    for (int i = 0; i < PROBE_LOD_COUNT; i++) {
        lods += (i ? ", " : "") + itos(PROBE_LODS[i]);
    }
    p.glsl = generate_glsl(p_program, p.name, true);
    p.glsl += vformat(R"glsl(void probe_%s(WorldPos p, uint b) {
    const int lods[%d] = int[%d](%s);
    for (uint i = 0u; i < %du; i++) {
        uint t;
        vec4 v = %s(p, int(params.lod_bias) + lods[i], t);
        for (uint j = 0u; j < 4u; j++) {
            uint u = floatBitsToUint(v[j]);
            results[b + 5u * i + j] = (u << 1u) == 0u ? 0u : u;
        }
        results[b + 5u * i + 4u] = t;
    }
}
)glsl",
            p.name, PROBE_LOD_COUNT, PROBE_LOD_COUNT, lods, PROBE_LOD_COUNT, p.name);
    p.cpu_exact = [p_program](const WorldPos &q, uint32_t *r) {
        std::vector<Float4> regs;
        bool exact = true;
        for (int i = 0; i < PROBE_LOD_COUNT; i++) {
            run_registers(p_program, q, PROBE_LODS[i], regs);
            const Float4 o = finish(p_program, regs[p_program.output], PROBE_LODS[i]);
            r[5 * i + 0] = zero_canonical(float_bits(o.x));
            r[5 * i + 1] = zero_canonical(float_bits(o.y));
            r[5 * i + 2] = zero_canonical(float_bits(o.z));
            r[5 * i + 3] = zero_canonical(float_bits(o.w));
            r[5 * i + 4] = trace_hash(regs);
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

std::vector<Probe> make(const std::vector<VoxelPos> &p_anchors, float p_size, int p_fuzz) {
    std::vector<Probe> probes;
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
        probes.push_back(probe(shape.name, prog));
    }
    for (int i = 0; i < p_fuzz; i++) {
        probes.push_back(probe(vformat("fuzz_%d", i), random_program(uint32_t(i), p_anchors, 8 + i % 33)));
    }
    return probes;
}

} // namespace ArchetypeProbes
