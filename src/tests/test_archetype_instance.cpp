#include "test_common.hpp"

#include "archetype_instance.hpp"
#include "archetype_probes.hpp"

#include <cmath>
#include <string>

using namespace Archetype;
using ExactNoise::exact_pow2;
using ExactNoise::float_bits;

namespace {

bool same_bits(const Float4 &p_a, const Float4 &p_b) {
    return float_bits(p_a.x) == float_bits(p_b.x) && float_bits(p_a.y) == float_bits(p_b.y) && float_bits(p_a.z) == float_bits(p_b.z) &&
            float_bits(p_a.w) == float_bits(p_b.w);
}

// p_anchor + an offset in [-p_range, p_range)^3 voxels, fraction in steps of 2^-18.
WorldPos random_point(const VoxelPos &p_anchor, double p_range, uint32_t p_i, uint32_t p_seed) {
    const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ p_i, p_seed, 0x1A57u });
    WorldPos p;
    const uint32_t u[3] = { h.x, h.y, h.z };
    float *frac[3] = { &p.frac.x, &p.frac.y, &p.frac.z };
    for (int a = 0; a < 3; a++) {
        const double v = (double(u[a] >> 8) * std::ldexp(1.0, -24) - 0.5) * 2.0 * p_range;
        const double fl = std::floor(v);
        p.voxel[a] = p_anchor[a] + int64_t(fl);
        *frac[a] = float(std::floor((v - fl) * 262144.0) / 262144.0);
    }
    return p;
}

Program scene_program(const Instance &p_root) {
    Compiled c = compile(p_root);
    CHECK(c.error.is_empty());
    c.program.normalize = true;
    c.program.out_scale = 0.5f;
    CHECK(c.program.validate(25).is_empty());
    return c.program;
}

} // namespace

TEST_CASE("Archetype: an instance gives the same bits for the same local point at any anchor") {
    const int64_t P32 = int64_t(1) << 32;
    // 0, across the 2^32 low-word wrap, ~1e15 m at 0.5 m voxels, +-2^60.
    const VoxelPos anchors[] = { VoxelPos(), VoxelPos(P32 - 3, -P32 + 2, P32 - 1), VoxelPos(2000000000000000, -7, -2000000000000003),
        VoxelPos((int64_t(1) << 60) - 1001, 5, -(int64_t(1) << 60) + 999), VoxelPos(-(int64_t(1) << 60) + 3, -(int64_t(1) << 60), 77) };
    std::vector<Program> programs;
    for (const VoxelPos &a : anchors) {
        programs.push_back(scene_program(ArchetypeProbes::test_scene(a, 64.0f)));
    }
    int64_t compared = 0;
    for (uint32_t i = 0; i < 2000; i++) {
        const WorldPos local = random_point(VoxelPos(), 80.0, i, 1);
        std::vector<Float4> ref_regs;
        for (const int lod : { 0, 5, 24 }) {
            run_registers(programs[0], local, lod, ref_regs);
            const Float4 ref = finish(programs[0], ref_regs[programs[0].output], lod);
            const uint32_t ref_trace = trace_hash(ref_regs);
            for (size_t k = 1; k < programs.size(); k++) {
                WorldPos p = local;
                p.voxel += anchors[k];
                std::vector<Float4> regs;
                run_registers(programs[k], p, lod, regs);
                CAPTURE(k);
                CHECK(same_bits(finish(programs[k], regs[programs[k].output], lod), ref));
                CHECK(trace_hash(regs) == ref_trace);
                compared++;
            }
        }
    }
    MESSAGE("local points x LODs x far anchors compared bit for bit (output and every register): ", compared);
}

TEST_CASE("Archetype: perlin octaves stay within the bounds the instance bounds use") {
    float vmax[2] = {};
    float gmax[2] = {};
    for (uint32_t i = 0; i < 200000; i++) {
        const WorldPos p = random_point(VoxelPos(int64_t(i) << 20, 0, -int64_t(i) * 977), 4096.0, i, 5);
        for (int d = 0; d < 2; d++) {
            const Float4 n = d == 0 ? ExactNoise::perlin3(p, 4, i * 31u, Vector3i64()) : ExactNoise::perlin2(p, 4, i * 31u, Vector3i64());
            vmax[d] = MAX(vmax[d], std::fabs(n.w));
            gmax[d] = MAX(gmax[d], std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z));
        }
    }
    MESSAGE("perlin3 max |value| ", vmax[0], " |grad| per cell ", gmax[0], "; perlin2 ", vmax[1], " ", gmax[1]);
    CHECK(vmax[0] <= PERLIN3_MAX);
    CHECK(gmax[0] <= PERLIN3_GRAD_MAX);
    CHECK(vmax[1] <= PERLIN2_MAX);
    CHECK(gmax[1] <= PERLIN2_GRAD_MAX);
}

TEST_CASE("Archetype: node bounds hold on 100,000 random points per scene") {
    struct Scene {
        std::string name;
        Instance root;
    };
    std::vector<Scene> scenes;
    scenes.push_back({ "test_scene", ArchetypeProbes::test_scene(VoxelPos(100, -50, 30), 64.0f) });
    for (int op = 0; op < 6; op++) {
        scenes.push_back({ "operator " + std::to_string(op), ArchetypeProbes::operator_scene(Combine(op), VoxelPos(-(int64_t(1) << 40), 9, 0), 48.0f) });
    }
    for (const Scene &scene : scenes) {
        Compiled c = compile(scene.root);
        CHECK(c.error.is_empty());
        const Bounds &root = c.nodes.back().bounds;
        CHECK(root.bounded);
        // Points over the root box grown by half its size on each side.
        VoxelPos centre;
        double half = 0.0;
        for (int a = 0; a < 3; a++) {
            centre[a] = root.lo[a] + (root.hi[a] - root.lo[a]) / 2;
            half = MAX(half, double(root.hi[a] - root.lo[a]));
        }
        int64_t outside = 0;
        int64_t gradients = 0;
        int64_t violations = 0;
        double worst_margin = INFINITY; // min over points outside a box of value - distance
        double worst_slope = 0.0; // max |grad| / L
        std::vector<Float4> regs;
        for (uint32_t i = 0; i < 100000; i++) {
            const WorldPos p = random_point(centre, half, i, 9);
            run_registers(c.program, p, 0, regs);
            for (const Compiled::Node &node : c.nodes) {
                const Float4 &r = regs[node.reg];
                const double d = node.bounds.distance(p);
                if (node.bounds.bounded && d > 0.0) {
                    outside++;
                    worst_margin = MIN(worst_margin, double(r.w) - d);
                    violations += double(r.w) < d - 1e-4 * (d + 1.0) ? 1 : 0;
                }
                if (std::isfinite(node.bounds.lipschitz)) {
                    gradients++;
                    const double g = std::sqrt(double(r.x) * r.x + double(r.y) * r.y + double(r.z) * r.z);
                    worst_slope = MAX(worst_slope, g / node.bounds.lipschitz);
                    violations += g > node.bounds.lipschitz * (1.0 + 1e-5) ? 1 : 0;
                }
            }
        }
        CAPTURE(scene.name);
        MESSAGE(scene.name, ": ", c.nodes.size(), " nodes, ", outside, " (point, node) pairs outside the box (worst value - distance ",
                worst_margin, "), ", gradients, " gradients (worst |grad| / L ", worst_slope, ")");
        CHECK(outside > 0);
        CHECK(violations == 0);
    }
}

TEST_CASE("Archetype: operator identities") {
    const std::shared_ptr<Def> rock = rock_def(20.0f, ExactNoise::make_fbm(false, 3u, 4, 2, 2.0, 0.5));
    const std::shared_ptr<Def> capsule = capsule_def(30.0f, 5.0f);
    // Union is commutative: the same value (bits); the same gradient off ties.
    Instance ab;
    ab.def = rock;
    ab.set_rotation_y(0.4);
    Instance::Child cb;
    cb.instance.def = capsule;
    cb.instance.anchor = VoxelPos(15, -10, 3);
    ab.children.push_back(cb);
    Instance ba;
    ba.def = capsule;
    ba.anchor = VoxelPos(15, -10, 3);
    Instance::Child ca;
    ca.instance.def = rock;
    ca.instance.anchor = VoxelPos(-15, 10, -3);
    ca.instance.set_rotation_y(0.4);
    ba.children.push_back(ca);
    const Program pab = scene_program(ab);
    const Program pba = scene_program(ba);
    // Subtracting a far, disjoint shape changes nothing near the rock.
    Instance alone;
    alone.def = rock;
    Instance cut = alone;
    Instance::Child far;
    far.op = Combine::SUBTRACT;
    far.instance.def = capsule;
    far.instance.anchor = VoxelPos(1000, 0, 0);
    cut.children.push_back(far);
    const Program palone = scene_program(alone);
    const Program pcut = scene_program(cut);
    int ties = 0;
    for (uint32_t i = 0; i < 20000; i++) {
        const WorldPos p = random_point(VoxelPos(), 60.0, i, 21);
        const Float4 x = run(pab, p, 0);
        const Float4 y = run(pba, p, 0);
        CHECK(float_bits(x.w) == float_bits(y.w));
        if (x.x != y.x || x.y != y.y || x.z != y.z) {
            ties++;
        }
        CHECK(same_bits(run(palone, p, 3), run(pcut, p, 3)));
    }
    CHECK(ties == 0);
}

TEST_CASE("Archetype: compile reports invalid instances") {
    Instance bad;
    bad.def = sphere_def(3.0f);
    bad.rotation[0] = 2.0f;
    CHECK_FALSE(compile(bad).error.is_empty());
    std::shared_ptr<Def> moved = sphere_def(3.0f);
    moved->program.anchors[0] = VoxelPos(1, 0, 0);
    Instance m;
    m.def = moved;
    CHECK_FALSE(compile(m).error.is_empty());
    Instance blend;
    blend.def = sphere_def(3.0f);
    Instance::Child c;
    c.op = Combine::BLEND;
    c.instance.def = sphere_def(2.0f);
    blend.children.push_back(c);
    CHECK_FALSE(compile(blend).error.is_empty());
}
