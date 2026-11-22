#include "test_common.hpp"

#include "godot_conv.hpp"

#include <godot_cpp/variant/string.hpp>

#include "archetype_glsl.hpp"
#include "archetype_ir.hpp"

#include <cmath>
#include <functional>
#include <string>

using namespace Archetype;
using ExactNoise::exact_pow2;
using ExactNoise::float_bits;

namespace {

// A point p_offset LOD0 voxels (fraction included) from p_anchor.
WorldPos near(const VoxelPos &p_anchor, double p_x, double p_y, double p_z) {
    WorldPos p;
    const double v[3] = { p_x, p_y, p_z };
    float *frac[3] = { &p.frac.x, &p.frac.y, &p.frac.z };
    for (int a = 0; a < 3; a++) {
        const double fl = std::floor(v[a]);
        p.voxel[a] = p_anchor[a] + int64_t(fl);
        *frac[a] = float(v[a] - fl);
    }
    return p;
}

// Deterministic points within +-p_range voxels of p_anchor, fractions multiples of 2^-18.
std::vector<WorldPos> random_points(const VoxelPos &p_anchor, int p_count, double p_range, uint32_t p_seed) {
    std::vector<WorldPos> out;
    for (int i = 0; i < p_count; i++) {
        const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ uint32_t(i), p_seed, 0xA4C7u });
        auto coord = [&](uint32_t p_u) { return (double(p_u >> 14) * std::ldexp(1.0, -18) - 0.5) * 2.0 * p_range; };
        out.push_back(near(p_anchor, coord(h.x), coord(h.y), coord(h.z)));
    }
    return out;
}

const VoxelPos FAR_ANCHOR((int64_t(1) << 60) - 12345, -(int64_t(1) << 32) + 7, int64_t(3) << 40);

// A record at p_anchor: identity rotation, salt 0 (the T_* ops' frame).
DefRecord record_at(const VoxelPos &p_anchor) {
    DefRecord rec;
    rec.anchor = p_anchor;
    return rec;
}
const DefRecord ORIGIN = record_at(VoxelPos());

// The set's octaves summed from 0, in the record's lattice.
Reg noise_sum(Builder &b, const ExactNoise::OctaveSet &p_set) {
    Reg sum = b.constant(0.0f);
    for (const ExactNoise::Octave &o : p_set.octaves) {
        const Reg n = b.t_perlin3(o.k, o.seed, o.offset, o.amp);
        sum = b.add(sum, n);
    }
    return sum;
}

// Value at p (LOD 0, the record ORIGIN) for the finite differences: the point moved by p_d voxels along p_axis.
float value_at(const Program &p_program, const WorldPos &p_pos, int p_axis, double p_d) {
    WorldPos q = p_pos;
    float *frac[3] = { &q.frac.x, &q.frac.y, &q.frac.z };
    const double v = double(*frac[p_axis]) + p_d;
    const double fl = std::floor(v);
    q.voxel[p_axis] += int64_t(fl);
    *frac[p_axis] = float(v - fl);
    std::vector<Float4> regs;
    run_registers(p_program, q, 0, regs, &ORIGIN);
    return regs[p_program.output].w;
}

// Central differences (h = 2^-6 voxel) vs the forward-mode gradient at LOD 0, away from kinks (where the
// one-sided differences disagree). Returns the share of points checked; r_worst the largest error,
// relative to max(1, |grad|).
double check_gradient(const Program &p_program, const std::vector<WorldPos> &p_points, double &r_worst) {
    const double h = std::ldexp(1.0, -6);
    int checked = 0;
    r_worst = 0.0;
    std::vector<Float4> regs;
    for (const WorldPos &p : p_points) {
        run_registers(p_program, p, 0, regs, &ORIGIN);
        const Float4 g = regs[p_program.output];
        const float v0 = g.w;
        // The differences' float noise: a few ulps of the largest intermediate value, over h.
        double vmax = 0.0;
        for (const Float4 &r : regs) {
            vmax = std::max(vmax, std::fabs(double(r.w)));
        }
        const double noise = 8.0 * vmax * std::ldexp(1.0, -23) / h;
        bool smooth = true;
        double err = 0.0;
        const float ga[3] = { g.x, g.y, g.z };
        for (int a = 0; a < 3 && smooth; a++) {
            const double fwd = (double(value_at(p_program, p, a, h)) - v0) / h;
            const double bwd = (v0 - double(value_at(p_program, p, a, -h))) / h;
            const double scale = std::max(1.0, std::fabs(double(ga[a]))) + noise;
            smooth = std::fabs(fwd - bwd) < 0.02 * scale;
            err = std::max(err, std::fabs(0.5 * (fwd + bwd) - double(ga[a])) / scale);
        }
        if (smooth) {
            checked++;
            r_worst = std::max(r_worst, err);
        }
    }
    return double(checked) / double(p_points.size());
}

} // namespace

TEST_CASE("Archetype: every op matches a hand-written float32 reference") {
    const ExactNoise::Octave n3{ 5, 0x1234u, Vector3i64(17, -(int64_t(1) << 40), 3), 2.5f };
    DefRecord rec = record_at(FAR_ANCHOR);
    rec.salt = 0x5A17u;
    Builder b;
    const Local l = b.t_local();
    const Reg c = b.constant(-1.75f);
    const Reg eps = b.constant(exact_pow2(-8));
    const Reg ops[] = { b.add(l.x, l.y), b.sub(l.x, l.z), b.mul(l.x, l.y), b.mul(l.x, c), b.neg(l.z), b.abs(l.x), b.min(l.x, l.y),
        b.max(l.y, l.z), b.select(l.x, l.y, l.z, c), b.floor(l.y), b.rsqrt(b.max(b.abs(l.z), eps)), b.dist(b.mul(l.x, l.y)),
        b.t_perlin3(n3.k, n3.seed, n3.offset, n3.amp), b.add(c, l.y), b.sub(c, l.y) };
    const Program prog = b.finish(ops[0]);
    CHECK(prog.validate(25).empty());

    int compared = 0;
    for (const int lod : { 0, 3, 14 }) {
        const float s = exact_pow2(lod);
        for (const WorldPos &p : random_points(FAR_ANCHOR, 64, 3000.0, 7)) {
            const float x = float(p.voxel.x - FAR_ANCHOR.x) + p.frac.x;
            const float y = float(p.voxel.y - FAR_ANCHOR.y) + p.frac.y;
            const float z = float(p.voxel.z - FAR_ANCHOR.z) + p.frac.z;
            const float cv = -1.75f;
            const float zs = z < 0.0f ? -s : s;
            // RSQRT of max(|z|, 2^-8).
            const Float4 gz = exact_pow2(-8) > std::fabs(z) ? Float4{ 0.0f, 0.0f, 0.0f, exact_pow2(-8) }
                                                            : Float4{ z < 0.0f ? -0.0f : 0.0f, z < 0.0f ? -0.0f : 0.0f, zs, std::fabs(z) };
            const float ry = ExactNoise::det_rsqrt(gz.w);
            const float ky = -0.5f * ry * ry * ry;
            const Float4 xy{ s * y + x * 0.0f, 0.0f * y + x * s, 0.0f * y + x * 0.0f, x * y }; // a.g * b.w + a.w * b.g
            const float q = s * ExactNoise::det_rsqrt(std::max(xy.x * xy.x + xy.y * xy.y + xy.z * xy.z, exact_pow2(-40) * s * s));
            // The record's lattice: the octave's offset minus the anchor, its seed salted.
            const Float4 p3 = ExactNoise::perlin3(p, n3.k, n3.seed ^ rec.salt, n3.offset - FAR_ANCHOR);
            const float f3 = n3.amp * exact_pow2(lod - n3.k);
            // T_INPUT with the identity rotation: x * 1 + y * 0 + z * 0, gradient s times the row.
            const Float4 want[] = {
                { s, s, 0.0f, x + y },
                { s, 0.0f, -s, x - z },
                xy,
                { s * cv, 0.0f * cv, 0.0f * cv, x * cv },
                { -0.0f, -0.0f, -s, -z },
                { x < 0.0f ? -s : s, x < 0.0f ? -0.0f : 0.0f, x < 0.0f ? -0.0f : 0.0f, std::fabs(x) },
                y < x ? Float4{ 0.0f, s, 0.0f, y } : Float4{ s, 0.0f, 0.0f, x },
                z > y ? Float4{ 0.0f, 0.0f, s, z } : Float4{ 0.0f, s, 0.0f, y },
                x < y ? Float4{ 0.0f, 0.0f, s, z } : Float4{ 0.0f, 0.0f, 0.0f, cv },
                { 0.0f, 0.0f, 0.0f, std::floor(y) },
                { gz.x * ky, gz.y * ky, gz.z * ky, ry },
                { xy.x * q, xy.y * q, xy.z * q, xy.w * q },
                // The band rule: an octave finer than 8 cells of the LOD is +0.
                ExactNoise::band_kept(n3.k, lod) ? Float4{ p3.x * f3, p3.y * f3, p3.z * f3, p3.w * n3.amp } : Float4{},
                { 0.0f, s, 0.0f, cv + y },
                { -0.0f, -s, -0.0f, cv - y },
            };
            std::vector<Float4> regs;
            run_registers(prog, p, lod, regs, &rec);
            for (size_t i = 0; i < sizeof(ops) / sizeof(ops[0]); i++) {
                CAPTURE(i);
                CAPTURE(lod);
                const Float4 &r = regs[ops[i]];
                CHECK(same_bits(r, want[i]));
                compared++;
            }
        }
    }
    MESSAGE("ops compared bit for bit: ", compared);
}

TEST_CASE("Archetype: the shapes are exact on axis points") {
    Builder b;
    const Local l = b.t_local();
    const Reg sphere = b.sphere(l, 10.0f);
    const Reg box = b.box(l, 4.0f, 5.0f, 6.0f);
    const Reg capsule = b.capsule(l, 8.0f, 2.0f);
    const Reg cylinder = b.cylinder(l, 3.0f, 2.0f);
    const Reg plane = b.plane(l, 0.0f, 1.0f, 0.0f, 2.0f);
    const Program prog = b.finish(sphere);
    CHECK(prog.validate(25).empty());
    struct Case {
        Reg reg;
        double x, y, z, want;
    };
    const Case cases[] = { { sphere, 13, 0, 0, 3 }, { sphere, 0, -4, 0, -6 }, { sphere, 0, 0, 10.5, 0.5 }, { box, 7, 0, 0, 3 },
        { box, 1, 0, 0, -3 }, { box, 0, 0, -6.25, 0.25 }, { box, 7, 9, 0, 5 }, { capsule, 5, 4, 0, 3 }, { capsule, 0, 11, 0, 1 },
        { capsule, 0, -3, 0, 1 }, { cylinder, 5, 0, 0, 3 }, { cylinder, 0, 4, 0, 1 }, { cylinder, 0, -2.5, 1, -0.5 },
        { plane, 9, 5, -3, 3 } };
    std::vector<Float4> regs;
    for (const Case &c : cases) {
        CAPTURE(c.reg);
        CAPTURE(c.x);
        CAPTURE(c.y);
        CAPTURE(c.z);
        run_registers(prog, near(VoxelPos(), c.x, c.y, c.z), 0, regs, &ORIGIN);
        // det_rsqrt and the products: a few ulps of the length.
        CHECK(std::fabs(double(regs[c.reg].w) - c.want) <= 1e-6 * (std::fabs(c.want) + 20.0));
    }
}

TEST_CASE("Archetype: gradients match finite differences") {
    ExactNoise::OctaveSet set = ExactNoise::make_fbm(false, 0xF0Bu, 6, 3, 4.0, 0.5);
    Builder b;
    const Local l = b.t_local();
    const Reg sphere = b.sphere(l, 20.0f);
    const Reg rock = b.add(sphere, noise_sum(b, set));
    const Reg shapes[] = { sphere, b.box(l, 9.0f, 14.0f, 6.0f), b.capsule(l, 12.0f, 5.0f), b.cylinder(l, 8.0f, 11.0f),
        b.plane(l, 0.6f, 0.8f, 0.0f, 1.0f), b.smooth_min(sphere, b.capsule(l, 30.0f, 4.0f), 6.0f),
        b.smooth_max(rock, b.neg(b.sphere(l, 12.0f)), 3.0f), rock, b.dist(rock) };
    const std::vector<WorldPos> points = random_points(VoxelPos(5, -3, 9), 400, 40.0, 11);
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        CAPTURE(i);
        const Program prog = b.finish(shapes[i]);
        CHECK(prog.validate(25).empty());
        double worst = 0.0;
        const double share = check_gradient(prog, points, worst);
        MESSAGE("shape ", i, ": ", share * 100.0, "% of the points smooth, worst relative error ", worst);
        CHECK(share > 0.9);
        // DIST's gradient is the normalised one (first order): not the derivative of its value.
        if (i != sizeof(shapes) / sizeof(shapes[0]) - 1) {
            CHECK(worst < 1e-2);
        }
    }
    int fuzz_checked = 0;
    double fuzz_worst = 0.0;
    for (uint32_t seed = 0; seed < 50; seed++) {
        const Program prog = random_program(seed, 40);
        double worst = 0.0;
        bool has_dist = false;
        for (const Instr &in : prog.code) {
            has_dist = has_dist || in.op == Op::DIST;
        }
        const double share = check_gradient(prog, random_points(VoxelPos(), 50, 60.0, seed), worst);
        if (!has_dist && share > 0.5) {
            fuzz_checked++;
            fuzz_worst = std::max(fuzz_worst, worst);
        }
    }
    MESSAGE("fuzzed programs without DIST checked by finite differences: ", fuzz_checked, ", worst relative error ", fuzz_worst);
    CHECK(fuzz_checked >= 10);
    CHECK(fuzz_worst < 2e-2);
}

// Whether the band rule keeps the same noise ops at LODs p_a and p_b.
static bool same_bands(const Program &p_prog, int p_a, int p_b) {
    for (const Instr &in : p_prog.code) {
        if (in.op == Op::T_PERLIN3 && ExactNoise::band_kept(in.k, p_a) != ExactNoise::band_kept(in.k, p_b)) {
            return false;
        }
    }
    return true;
}

TEST_CASE("Archetype: the band rule drops a noise op to +0 exactly below 8 cells of the LOD") {
    // k - 4 .. k - 2 around each op's cutoff lod = k - 3, at far points: kept = the octave's bits, dropped = +0.
    for (const int k : { 3, 6, 12, 15 }) {
        CAPTURE(k);
        Builder b;
        const Reg n3 = b.t_perlin3(k, 0xBA5Du, Vector3i64(int64_t(1) << 41, -9, 77), 1.5f);
        const Program prog = b.finish(n3);
        CHECK(prog.validate(25).empty());
        for (int lod = std::max(k - 5, 0); lod <= k - 1; lod++) {
            CAPTURE(lod);
            CHECK(ExactNoise::band_kept(k, lod) == (lod <= k - 3));
            std::vector<Float4> regs;
            for (const WorldPos &p : random_points(FAR_ANCHOR, 64, double(int64_t(1) << std::min(k + 4, 40)), uint32_t(k * 31 + lod))) {
                run_registers(prog, p, lod, regs, &ORIGIN);
                if (ExactNoise::band_kept(k, lod)) {
                    const Float4 a = ExactNoise::perlin3(p, k, 0xBA5Du, Vector3i64(int64_t(1) << 41, -9, 77));
                    const float f = 1.5f * exact_pow2(lod - k);
                    CHECK(same_bits(regs[n3], Float4{ a.x * f, a.y * f, a.z * f, a.w * 1.5f }));
                } else {
                    CHECK(same_bits(regs[n3], Float4{}));
                }
            }
        }
    }
}

TEST_CASE("Archetype: values are the same bits at every LOD, gradients scale by exactly 2^lod") {
    int64_t points = 0;
    int64_t in_domain = 0;
    int64_t compared = 0;
    for (uint32_t seed = 0; seed < 100; seed++) {
        Program prog = random_program(seed, 40);
        prog.normalize = true;
        CHECK(prog.validate(25).empty());
        std::vector<Float4> r0, rl;
        for (const WorldPos &p : random_points(VoxelPos(), 20, 2e4, seed)) {
            points++;
            run_registers(prog, p, 0, r0, &ORIGIN);
            const Float4 n0 = finish(prog, r0[prog.output], 0);
            bool ok = exact_domain(prog, r0);
            for (const int lod : { 1, 5, 14, 24 }) {
                if (!same_bands(prog, 0, lod)) {
                    continue; // the band rule drops an octave between LOD 0 and this one: another function
                }
                run_registers(prog, p, lod, rl, &ORIGIN);
                ok = ok && exact_domain(prog, rl);
                if (!ok) {
                    break;
                }
                const float s = exact_pow2(lod);
                const Float4 &a = r0[prog.output];
                const Float4 &c = rl[prog.output];
                CHECK(same_bits(Float4{ a.x * s, a.y * s, a.z * s, a.w }, c));
                CHECK(same_bits(finish(prog, c, lod), n0));
                compared++;
            }
            in_domain += ok ? 1 : 0;
        }
    }
    MESSAGE("points in the exact domain at LODs 0, 1, 5, 14, 24: ", in_domain, " of ", points);
    CHECK(in_domain * 10 >= points * 8);
    MESSAGE("(point, LOD) pairs with the same bands as LOD 0, compared: ", compared);
    CHECK(compared > 1000);
}

TEST_CASE("Archetype: validate rejects malformed programs") {
    Builder b;
    const Local l = b.t_local();
    const Reg s = b.sphere(l, 3.0f);
    const Program good = b.finish(s);
    CHECK(good.validate(25).empty());
    const std::function<void(Program &)> breakers[] = {
        [](Program &p) { p.code[2].a = 5; }, // reads a later register (a T_INPUT has no operand: use an ADD)
        [](Program &p) { p.output = int(p.code.size()); },
        [](Program &p) { p.code[3].flat = !p.code[3].flat; },
        [](Program &p) {
            for (Instr &in : p.code) {
                if (in.op == Op::CONST) {
                    in.value = 1e-40f;
                    break;
                }
            }
        },
        [](Program &p) { p.code[0].axis = 3; },
        [](Program &p) { p.out_scale = 0.0f; },
        [](Program &p) {
            Instr in;
            in.op = Op::T_PERLIN3;
            in.k = 30;
            in.value = 1.0f; // gradient factor 2^-30 at LOD 0
            p.code.push_back(in);
        },
        [](Program &p) {
            Instr in;
            in.op = Op::DIST;
            in.a = int(p.code.size()) - 2; // the sphere's radius constant
            p.code.push_back(in);
        },
        [](Program &p) {
            Instr in;
            in.op = Op::RSQRT;
            in.a = int(p.code.size()) - 1; // the sphere: 0 on its surface, unguarded
            p.code.push_back(in);
        },
        [](Program &p) { p.code[3].op = Op(200); },
    };
    int i = 0;
    for (const auto &breaker : breakers) {
        CAPTURE(i++);
        Program bad = good;
        CHECK(bad.code[2].op == Op::T_INPUT);
        bad.code[2].op = Op::ADD;
        bad.code[2].a = 0;
        bad.code[2].b = 1;
        bad.code[2].flat = false;
        CHECK(bad.validate(25).empty());
        breaker(bad);
        CHECK_FALSE(bad.validate(25).empty());
    }
}

TEST_CASE("Archetype: smooth_min / smooth_max are exactly min / max beyond their radius, at any magnitude") {
    // A shape near its anchor against a far one (a saturated T_INPUT reads 2^30 voxels): the near value's bits.
    Builder b;
    const Local l = b.t_local();
    const Reg near_shape = b.sphere(l, 6.0f);
    const Reg far_shape = b.add(l.x, 1073741824.0f);
    const Reg far_neg = b.neg(far_shape);
    const Reg u = b.smooth_min(near_shape, far_shape, 8.0f);
    const Reg u2 = b.smooth_min(far_shape, near_shape, 8.0f);
    const Reg m = b.smooth_max(near_shape, far_neg, 8.0f);
    const Program prog = b.finish(u);
    CHECK(prog.validate(25).empty());
    std::vector<Float4> regs;
    for (const int lod : { 0, 14 }) {
        for (const WorldPos &p : random_points(VoxelPos(), 500, 40.0, 21)) {
            run_registers(prog, p, lod, regs, &ORIGIN);
            CHECK(same_bits(regs[u], regs[near_shape]));
            CHECK(same_bits(regs[u2], regs[near_shape]));
            CHECK(same_bits(regs[m], regs[near_shape]));
        }
    }
}

TEST_CASE("Archetype: fuzzed programs are valid and use every op") {
    int seen[int(Op::LAST) + 1] = {};
    for (uint32_t seed = 0; seed < 200; seed++) {
        const Program prog = random_program(seed, 40);
        CAPTURE(seed);
        CHECK(prog.validate(25).empty());
        CHECK(prog.code.size() <= 43);
        for (const Instr &in : prog.code) {
            seen[int(in.op)]++;
        }
    }
    for (int op = 0; op <= int(Op::LAST); op++) {
        CAPTURE(op);
        CHECK((op == int(Op::PARAM) || op == int(Op::T_PARAM) || seen[op] > 0));
    }
}

// The generated GLSL is proven against the interpreter by FrameSpike (the arch_* probes): a change of its
// text needs a tools/frame_test.sh run, then a new pin. FNV-1a of the text of fixed programs.
TEST_CASE("Archetype: the generated GLSL is pinned and uses exact operations only") {
    String all;
    for (uint32_t seed = 0; seed < 20; seed++) {
        const Program prog = random_program(seed, 40);
        const String code = godot_str(generate_def_glsl(prog, ov::format("f%d", seed), (seed & 1u) != 0));
        // One statement per instruction.
        CHECK(code.count("    precise vec4 r") == int(prog.code.size()));
        all += code;
    }
    Builder b;
    const Local l = b.t_local();
    const Reg s = b.sphere(l, 30.0f);
    const Reg c = b.smooth_min(s, b.capsule(l, 40.0f, 6.0f), 8.0f);
    const Reg x = b.smooth_max(c, b.neg(b.box(l, 10.0f, 4.0f, 9.0f)), 2.0f);
    const Reg y = b.min(x, b.cylinder(l, 5.0f, 7.0f));
    Program prog = b.finish(b.dist(b.add(y, noise_sum(b, ExactNoise::make_fbm(false, 5u, 8, 3, 10.0, 0.5)))));
    prog.normalize = true;
    prog.out_scale = 0.5f;
    all += godot_str(generate_def_glsl(prog, "shapes"));
    const String without_det = all.replace("det_rsqrt(", "");
    for (const char *banned : { "/", "mix(", "sqrt(", "normalize(", "length(", "dot(", "fma(", "#" }) {
        CAPTURE(std::string(banned));
        CHECK_FALSE(without_det.contains(banned));
    }
    // Float constants only as raw bits: the one decimal literal is 0.0.
    const String rest = all.replace("0.0", "");
    auto digit = [](char32_t p_c) { return p_c >= '0' && p_c <= '9'; };
    bool decimal = false;
    for (int i = 1; i + 1 < rest.length(); i++) {
        decimal = decimal || (rest[i] == '.' && digit(rest[i - 1]) && digit(rest[i + 1]));
    }
    CHECK_FALSE(decimal);
    const godot::CharString text = all.utf8();
    uint32_t h = 0x811C9DC5u;
    for (int i = 0; i < text.length(); i++) {
        h = (h ^ uint8_t(text[i])) * 0x01000193u;
    }
    MESSAGE("generated GLSL: ", text.length(), " bytes, FNV-1a ", h);
    CHECK(text.length() == 246279);
    CHECK(h == 2562581086u);
}
