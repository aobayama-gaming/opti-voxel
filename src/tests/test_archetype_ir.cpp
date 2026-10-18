#include "test_common.hpp"

#include "archetype_ir.hpp"

#include <cmath>
#include <functional>

using namespace Archetype;
using ExactNoise::exact_pow2;
using ExactNoise::float_bits;

namespace {

bool same_bits(const Float4 &p_a, const Float4 &p_b) {
    return float_bits(p_a.x) == float_bits(p_b.x) && float_bits(p_a.y) == float_bits(p_b.y) && float_bits(p_a.z) == float_bits(p_b.z) &&
            float_bits(p_a.w) == float_bits(p_b.w);
}

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

// Value at p (LOD 0) for the finite differences: the point moved by p_d voxels along p_axis.
float value_at(const Program &p_program, const WorldPos &p_pos, int p_axis, double p_d) {
    WorldPos q = p_pos;
    float *frac[3] = { &q.frac.x, &q.frac.y, &q.frac.z };
    const double v = double(*frac[p_axis]) + p_d;
    const double fl = std::floor(v);
    q.voxel[p_axis] += int64_t(fl);
    *frac[p_axis] = float(v - fl);
    std::vector<Float4> regs;
    run_registers(p_program, q, 0, regs);
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
        run_registers(p_program, p, 0, regs);
        const Float4 g = regs[p_program.output];
        const float v0 = g.w;
        // The differences' float noise: a few ulps of the largest intermediate value, over h.
        double vmax = 0.0;
        for (const Float4 &r : regs) {
            vmax = MAX(vmax, std::fabs(double(r.w)));
        }
        const double noise = 8.0 * vmax * std::ldexp(1.0, -23) / h;
        bool smooth = true;
        double err = 0.0;
        const float ga[3] = { g.x, g.y, g.z };
        for (int a = 0; a < 3 && smooth; a++) {
            const double fwd = (double(value_at(p_program, p, a, h)) - v0) / h;
            const double bwd = (v0 - double(value_at(p_program, p, a, -h))) / h;
            const double scale = MAX(1.0, std::fabs(double(ga[a]))) + noise;
            smooth = std::fabs(fwd - bwd) < 0.02 * scale;
            err = MAX(err, std::fabs(0.5 * (fwd + bwd) - double(ga[a])) / scale);
        }
        if (smooth) {
            checked++;
            r_worst = MAX(r_worst, err);
        }
    }
    return double(checked) / double(p_points.size());
}

} // namespace

TEST_CASE("Archetype: every op matches a hand-written float32 reference") {
    const ExactNoise::Octave n3{ 5, 0x1234u, Vector3i64(17, -(int64_t(1) << 40), 3), 2.5f };
    const ExactNoise::Octave n2{ 7, 0x99u, Vector3i64(-5, 0, int64_t(1) << 33), 0.75f };
    Builder b;
    b.anchor(FAR_ANCHOR);
    const Local l = b.local();
    const Reg c = b.constant(-1.75f);
    const Reg ops[] = { b.add(l.x, l.y), b.sub(l.x, l.z), b.mul(l.x, l.y), b.mul(l.x, c), b.neg(l.z), b.abs(l.x), b.min(l.x, l.y),
        b.max(l.y, l.z), b.select(l.x, l.y, l.z, c), b.floor(l.y), b.rsqrt(b.abs(l.z)), b.dist(b.mul(l.x, l.y)),
        b.perlin3(n3.k, n3.seed, n3.offset, n3.amp), b.perlin2(n2.k, n2.seed, n2.offset, n2.amp), b.add(c, l.y), b.sub(c, l.y) };
    const Program prog = b.finish(ops[0]);
    CHECK(prog.validate(25).is_empty());

    int compared = 0;
    for (const int lod : { 0, 3, 14 }) {
        const float s = exact_pow2(lod);
        for (const WorldPos &p : random_points(FAR_ANCHOR, 64, 3000.0, 7)) {
            const float x = float(p.voxel.x - FAR_ANCHOR.x) + p.frac.x;
            const float y = float(p.voxel.y - FAR_ANCHOR.y) + p.frac.y;
            const float z = float(p.voxel.z - FAR_ANCHOR.z) + p.frac.z;
            const float cv = -1.75f;
            const float ry = ExactNoise::det_rsqrt(std::fabs(z));
            const float ky = -0.5f * ry * ry * ry;
            const float zs = z < 0.0f ? -s : s;
            const Float4 xy{ s * y + x * 0.0f, 0.0f * y + x * s, 0.0f * y + x * 0.0f, x * y }; // a.g * b.w + a.w * b.g
            const float q = s * ExactNoise::det_rsqrt(MAX(xy.x * xy.x + xy.y * xy.y + xy.z * xy.z, exact_pow2(-40) * s * s));
            const Float4 p3 = ExactNoise::perlin3(p, n3.k, n3.seed, n3.offset);
            const Float4 p2 = ExactNoise::perlin2(p, n2.k, n2.seed, n2.offset);
            const float f3 = n3.amp * exact_pow2(lod - n3.k);
            const float f2 = n2.amp * exact_pow2(lod - n2.k);
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
                { (z < 0.0f ? -0.0f : 0.0f) * ky, (z < 0.0f ? -0.0f : 0.0f) * ky, zs * ky, ry },
                { xy.x * q, xy.y * q, xy.z * q, xy.w * q },
                { p3.x * f3, p3.y * f3, p3.z * f3, p3.w * n3.amp },
                { p2.x * f2, p2.y * f2, p2.z * f2, p2.w * n2.amp },
                { 0.0f, s, 0.0f, cv + y },
                { -0.0f, -s, -0.0f, cv - y },
            };
            std::vector<Float4> regs;
            run_registers(prog, p, lod, regs);
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
    const Local l = b.local();
    const Reg sphere = b.sphere(l, 10.0f);
    const Reg box = b.box(l, 4.0f, 5.0f, 6.0f);
    const Reg capsule = b.capsule(l, 8.0f, 2.0f);
    const Reg cylinder = b.cylinder(l, 3.0f, 2.0f);
    const Reg plane = b.plane(l, 0.0f, 1.0f, 0.0f, 2.0f);
    const Program prog = b.finish(sphere);
    CHECK(prog.validate(25).is_empty());
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
        run_registers(prog, near(VoxelPos(), c.x, c.y, c.z), 0, regs);
        // det_rsqrt and the products: a few ulps of the length.
        CHECK(std::fabs(double(regs[c.reg].w) - c.want) <= 1e-6 * (std::fabs(c.want) + 20.0));
    }
}

TEST_CASE("Archetype: gradients match finite differences") {
    ExactNoise::OctaveSet set = ExactNoise::make_fbm(false, 0xF0Bu, 6, 3, 4.0, 0.5);
    Builder b;
    const Local l = b.local();
    const Reg sphere = b.sphere(l, 20.0f);
    const Reg rock = b.add(sphere, b.fbm(set));
    const Reg shapes[] = { sphere, b.box(l, 9.0f, 14.0f, 6.0f), b.capsule(l, 12.0f, 5.0f), b.cylinder(l, 8.0f, 11.0f),
        b.plane(l, 0.6f, 0.8f, 0.0f, 1.0f), b.smooth_min(sphere, b.capsule(l, 30.0f, 4.0f), 6.0f),
        b.smooth_max(rock, b.neg(b.sphere(l, 12.0f)), 3.0f), rock, b.dist(rock) };
    const std::vector<WorldPos> points = random_points(VoxelPos(5, -3, 9), 400, 40.0, 11);
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        CAPTURE(i);
        const Program prog = b.finish(shapes[i]);
        CHECK(prog.validate(25).is_empty());
        double worst = 0.0;
        const double share = check_gradient(prog, points, worst);
        MESSAGE("shape ", i, ": ", share * 100.0, "% of the points smooth, worst relative error ", worst);
        CHECK(share > 0.9);
        // DIST's gradient is the normalised one (first order): not the derivative of its value.
        if (i != sizeof(shapes) / sizeof(shapes[0]) - 1) {
            CHECK(worst < 1e-2);
        }
    }
    std::vector<VoxelPos> anchors = { VoxelPos(), VoxelPos(40, -20, 7) };
    int fuzz_checked = 0;
    double fuzz_worst = 0.0;
    for (uint32_t seed = 0; seed < 50; seed++) {
        const Program prog = random_program(seed, anchors, 40);
        double worst = 0.0;
        bool has_dist = false;
        for (const Instr &in : prog.code) {
            has_dist = has_dist || in.op == Op::DIST;
        }
        const double share = check_gradient(prog, random_points(VoxelPos(), 50, 60.0, seed), worst);
        if (!has_dist && share > 0.5) {
            fuzz_checked++;
            fuzz_worst = MAX(fuzz_worst, worst);
        }
    }
    MESSAGE("fuzzed programs without DIST checked by finite differences: ", fuzz_checked, ", worst relative error ", fuzz_worst);
    CHECK(fuzz_checked >= 10);
    CHECK(fuzz_worst < 2e-2);
}

TEST_CASE("Archetype: the IR fbm gives ExactNoise::fbm's bits") {
    for (const bool is_2d : { false, true }) {
        const ExactNoise::OctaveSet set = ExactNoise::make_fbm(is_2d, 0x7E7A11u, 9, 5, 75.0, 0.5);
        Builder b;
        const Program prog = b.finish(b.fbm(set));
        CHECK(prog.validate(25).is_empty());
        for (const int lod : { 0, 5, 14 }) {
            for (const WorldPos &p : random_points(FAR_ANCHOR, 200, 1e5, 3)) {
                const Float4 want = ExactNoise::fbm(set, p, lod);
                const Float4 got = run(prog, p, lod);
                CHECK(float_bits(got.w) == float_bits(want.w));
                // The same up to the sign of a zero (the IR's first octave isn't added to +0).
                CHECK((got.x == want.x && got.y == want.y && got.z == want.z));
            }
        }
    }
}

TEST_CASE("Archetype: values are the same bits at every LOD, gradients scale by exactly 2^lod") {
    const std::vector<VoxelPos> anchors = { VoxelPos(), VoxelPos(3000, -77, 5) };
    int64_t points = 0;
    int64_t in_domain = 0;
    for (uint32_t seed = 0; seed < 100; seed++) {
        Program prog = random_program(seed, anchors, 40);
        prog.normalize = true;
        CHECK(prog.validate(25).is_empty());
        std::vector<Float4> r0, rl;
        for (const WorldPos &p : random_points(VoxelPos(), 20, 2e4, seed)) {
            points++;
            run_registers(prog, p, 0, r0);
            const Float4 n0 = finish(prog, r0[prog.output], 0);
            bool ok = exact_domain(r0);
            for (const int lod : { 1, 5, 14, 24 }) {
                run_registers(prog, p, lod, rl);
                ok = ok && exact_domain(rl);
                if (!ok) {
                    break;
                }
                const float s = exact_pow2(lod);
                const Float4 &a = r0[prog.output];
                const Float4 &c = rl[prog.output];
                CHECK(same_bits(Float4{ a.x * s, a.y * s, a.z * s, a.w }, c));
                CHECK(same_bits(finish(prog, c, lod), n0));
            }
            in_domain += ok ? 1 : 0;
        }
    }
    MESSAGE("points in the exact domain at LODs 0, 1, 5, 14, 24: ", in_domain, " of ", points);
    CHECK(in_domain * 10 >= points * 8);
}

TEST_CASE("Archetype: validate rejects malformed programs") {
    Builder b;
    const Local l = b.local();
    const Reg s = b.sphere(l, 3.0f);
    const Program good = b.finish(s);
    CHECK(good.validate(25).is_empty());
    const std::function<void(Program &)> breakers[] = {
        [](Program &p) { p.code[2].a = 5; }, // reads a later register (an INPUT has no operand: use an ADD)
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
        [](Program &p) { p.code[0].index = 3; },
        [](Program &p) { p.out_scale = 0.0f; },
        [](Program &p) {
            Instr in;
            in.op = Op::PERLIN3;
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
    };
    int i = 0;
    for (const auto &breaker : breakers) {
        CAPTURE(i++);
        Program bad = good;
        CHECK(bad.code[2].op == Op::INPUT);
        bad.code[2].op = Op::ADD;
        bad.code[2].a = 0;
        bad.code[2].b = 1;
        bad.code[2].flat = false;
        CHECK(bad.validate(25).is_empty());
        breaker(bad);
        CHECK_FALSE(bad.validate(25).is_empty());
    }
}

TEST_CASE("Archetype: fuzzed programs are valid and use every op") {
    const std::vector<VoxelPos> anchors = { VoxelPos(), VoxelPos(-5, 1 << 20, 3) };
    int seen[int(Op::PERLIN2) + 1] = {};
    for (uint32_t seed = 0; seed < 200; seed++) {
        const Program prog = random_program(seed, anchors, 40);
        CAPTURE(seed);
        CHECK(prog.validate(25).is_empty());
        CHECK(prog.code.size() <= 43);
        for (const Instr &in : prog.code) {
            seen[int(in.op)]++;
        }
    }
    for (int op = 0; op <= int(Op::PERLIN2); op++) {
        CAPTURE(op);
        CHECK((op == int(Op::PARAM) || seen[op] > 0));
    }
}
