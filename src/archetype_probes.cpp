#include "archetype_probes.hpp"

#include "archetype_glsl.hpp"

#include <cmath>
#include <functional>

namespace ArchetypeProbes {

using namespace Archetype;
using ExactNoise::float_bits;

namespace {

// A record at p_anchor: identity rotation, no parameters, salt 0.
DefRecord record_at(const VoxelPos &p_anchor) {
    DefRecord rec;
    rec.anchor = p_anchor;
    return rec;
}

// The GLSL DefRecord constructor of p_rec (GLSL_DEF_RECORD_STRUCT's fields; the head and box are unused here).
ov::str glsl_record(const DefRecord &p_rec) {
    ov::str rot;
    for (int row = 0; row < 3; row++) {
        rot += ov::format("%svec4(%s, %s, %s, 0.0)", row ? ", " : "", ExactNoise::glsl_float(p_rec.rotation[3 * row]),
                ExactNoise::glsl_float(p_rec.rotation[3 * row + 1]), ExactNoise::glsl_float(p_rec.rotation[3 * row + 2]));
    }
    ov::str params;
    for (int i = 0; i < 8; i++) {
        params += (i ? ", " : "") + ExactNoise::glsl_float(p_rec.params[i]);
    }
    return ov::format("DefRecord(uvec4(0u), uvec4(%s, %s), ivec4(%s, 0), vec4[3](%s), float[8](%s), ivec4(0), ivec4(0))", ExactNoise::glsl_lo(p_rec.anchor),
            ExactNoise::glsl_uint(p_rec.salt), ExactNoise::glsl_hi(p_rec.anchor), rot, params);
}

// FNV-1a step of the per-record trace hashes (the GLSL side in probe()).
uint32_t fold_trace(uint32_t p_h, uint32_t p_t) {
    return (p_h ^ p_t) * 0x01000193u;
}

} // namespace

Probe probe(const ov::str &p_name, const Program &p_program, const std::vector<VoxelPos> &p_anchors, bool p_trace) {
    std::vector<DefRecord> records;
    for (const VoxelPos &a : p_anchors) {
        records.push_back(record_at(a));
    }
    Probe p;
    const int stride = p_trace ? 5 : 4;
    const int count = int(records.size());
    p.name = "arch_" + p_name;
    p.words = stride * PROBE_LOD_COUNT;
    p.is_float = true;
    ov::str lods;
    for (int i = 0; i < PROBE_LOD_COUNT; i++) {
        lods += (i ? ", " : "") + ov::itos(PROBE_LODS[i]);
    }
    ov::str recs;
    for (int c = 0; c < count; c++) {
        recs += (c ? ",\n        " : "") + glsl_record(records[c]);
    }
    p.glsl = generate_def_glsl(p_program, p.name, p_trace);
    // The union of the copies (the MIN op's rule: a later copy wins only below), the traces folded in order.
    p.glsl += ov::format(R"glsl(void probe_%s(WorldPos p, uint b) {
    const int lods[%d] = int[%d](%s);
    DefRecord recs[%d] = DefRecord[%d](%s);
    for (uint i = 0u; i < %du; i++) {
        int lod = int(params.lod_bias) + lods[i];
        vec4 v = vec4(0.0);
        uint t = 0x811C9DC5u;
        for (int c = 0; c < %d; c++) {
            %s
            if (c == 0 || w.w < v.w) {
                v = w;
            }
            %s
        }
        for (uint j = 0u; j < 4u; j++) {
            uint u = floatBitsToUint(v[j]);
            results[b + %du * i + j] = (u << 1u) == 0u ? 0u : u;
        }
        %s
    }
}
)glsl",
            p.name, PROBE_LOD_COUNT, PROBE_LOD_COUNT, lods, count, count, recs, PROBE_LOD_COUNT, count,
            p_trace ? "uint tc;\n            vec4 w = " + p.name + "(p, lod, recs[c], tc);" : "vec4 w = " + p.name + "(p, lod, recs[c]);",
            p_trace ? "t = (t ^ tc) * 0x01000193u;" : "", stride, p_trace ? "results[b + 5u * i + 4u] = t;" : "");
    p.cpu = [p_program, records, stride](const WorldPos &q, uint32_t *r) {
        thread_local std::vector<Float4> regs;
        bool exact = true;
        for (int i = 0; i < PROBE_LOD_COUNT; i++) {
            Float4 o;
            uint32_t t = 0x811C9DC5u;
            for (size_t c = 0; c < records.size(); c++) {
                run_registers(p_program, q, PROBE_LODS[i], regs, &records[c]);
                const Float4 w = finish(p_program, regs[p_program.output], PROBE_LODS[i]);
                if (c == 0 || w.w < o.w) {
                    o = w;
                }
                t = fold_trace(t, trace_hash(regs));
                exact = exact && exact_domain(p_program, regs);
            }
            uint32_t *out = r + stride * i;
            out[0] = zero_canonical(float_bits(o.x));
            out[1] = zero_canonical(float_bits(o.y));
            out[2] = zero_canonical(float_bits(o.z));
            out[3] = zero_canonical(float_bits(o.w));
            if (stride == 5) {
                out[4] = t;
            }
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
    // The same at the same offset from any origin: false for the plane (unbounded: a copy 2^31 voxels away reads
    // saturated inputs, a huge negative value, and wins the min) and the noise shapes (their lattice follows the
    // record since the IR lost its absolute frame, but no frame test has compared them across origins yet: a run
    // that does would let them join the frames probe).
    bool origin_relative = true;
};

const ExactNoise::OctaveSet &rock_noise() {
    static const ExactNoise::OctaveSet set = ExactNoise::make_fbm(false, 0x50C4u, 10, 4, 300.0, 0.5);
    return set;
}

const ExactNoise::OctaveSet &height_noise() {
    static const ExactNoise::OctaveSet set = ExactNoise::make_fbm(false, 0x4E16u, 11, 5, 600.0, 0.5);
    return set;
}

// The set's octaves summed from 0, in the record's lattice.
Reg noise_sum(Builder &b, const ExactNoise::OctaveSet &p_set) {
    Reg sum = b.constant(0.0f);
    for (const ExactNoise::Octave &o : p_set.octaves) {
        const Reg n = b.t_perlin3(o.k, o.seed, o.offset, o.amp);
        sum = b.add(sum, n);
    }
    return sum;
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
                return b.dist(b.add(a, noise_sum(b, rock_noise())));
            },
            false },
    { "height",
            [](Builder &b, const Local &l, float) {
                const Reg h = noise_sum(b, height_noise());
                return b.dist(b.sub(l.y, h));
            },
            false },
};

} // namespace

std::vector<Probe> make(const std::vector<VoxelPos> &p_anchors, const std::vector<VoxelPos> &p_frame_anchors, float p_size, int p_fuzz) {
    std::vector<Probe> probes;
    // A union of p_shapes in one record's frame, one record per anchor of p_at.
    auto shapes_probe = [&](const ov::str &p_name, const std::vector<const Shape *> &p_shapes, const std::vector<VoxelPos> &p_at, bool p_trace) {
        Builder b;
        const Local l = b.t_local();
        Reg u = -1;
        for (const Shape *shape : p_shapes) {
            const Reg d = shape->fn(b, l, p_size);
            u = u < 0 ? d : b.min(u, d);
        }
        Program prog = b.finish(u);
        prog.normalize = true;
        prog.out_scale = 0.5f;
        return probe(p_name, prog, p_at, p_trace);
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
        probes.push_back(probe(ov::format("fuzz_%d", i), random_program(uint32_t(i), 8 + i % 33), p_anchors, true));
    }
    return probes;
}

} // namespace ArchetypeProbes
