#include "test_common.hpp"

#include <atomic>
#include <thread>

#include "archetype_instance.hpp"
#include "archetype_probes.hpp"
#include "sdf_archetype.hpp"

#include <cmath>
#include <string>

using namespace Archetype;
using ExactNoise::exact_pow2;
using ExactNoise::float_bits;

namespace {

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
    scenes.push_back({ "demo terrain", SDFArchetype::scene(0.5) });
    scenes.push_back({ "shapes gallery", SDFArchetype::shapes(0.5) });
    for (const Scene &scene : scenes) {
        Compiled c = compile(scene.root);
        CHECK(c.error.is_empty());
        // Points over the hull of the bounded nodes' boxes, grown by half its size on each side.
        Bounds root;
        root.bounded = false;
        for (const Compiled::Node &node : c.nodes) {
            if (node.bounds.bounded) {
                root = root.bounded ? hull(root, node.bounds) : node.bounds;
            }
        }
        CHECK(root.bounded);
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
    // Both trees are the same scene (the two shapes swapped, identity rotations, so a child's offset in
    // its parent's frame is its world offset): a noisy rock at the world origin and a capsule at C.
    const VoxelPos C(15, -10, 3);
    Instance ab;
    ab.def = rock;
    Instance::Child cb;
    cb.instance.def = capsule;
    cb.instance.anchor = C;
    ab.children.push_back(cb);
    Instance ba;
    ba.def = capsule;
    ba.anchor = C;
    Instance::Child ca;
    ca.instance.def = rock;
    ca.instance.anchor = VoxelPos(-C.x, -C.y, -C.z);
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
    bool first_mismatch = true;
    for (uint32_t i = 0; i < 20000; i++) {
        const WorldPos p = random_point(VoxelPos(), 60.0, i, 21);
        const Float4 x = run(pab, p, 0);
        const Float4 y = run(pba, p, 0);
        if (float_bits(x.w) != float_bits(y.w) && ties >= 0 && first_mismatch) {
            first_mismatch = false;
            MESSAGE("commutativity mismatch at ", WorldCoord::to_string(p.voxel), " frac (", p.frac.x, ",", p.frac.y, ",", p.frac.z, "): ab w=", x.w, " ba w=", y.w);
        }
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
    // A weight Def must sit at the local origin too; with a valid one the blend compiles.
    blend.children[0].weight = moved;
    CHECK_FALSE(compile(blend).error.is_empty());
    blend.children[0].weight = ramp_def(4.0f);
    CHECK(compile(blend).error.is_empty());
    // A child without Def, a rotation with a zero row, a smooth operator without radius: an error, no program
    // (compiling on would dereference null or emit operands -1).
    Instance parent;
    parent.def = sphere_def(3.0f);
    parent.children.push_back(Instance::Child());
    Compiled none = compile(parent);
    CHECK_FALSE(none.error.is_empty());
    CHECK(none.program.code.empty());
    Instance zero_row;
    zero_row.def = sphere_def(3.0f);
    zero_row.rotation[0] = 0.0f;
    CHECK_FALSE(compile(zero_row).error.is_empty());
    Instance smooth;
    smooth.def = sphere_def(3.0f);
    Instance::Child sc;
    sc.op = Combine::SMOOTH_UNION;
    sc.instance.def = sphere_def(2.0f);
    smooth.children.push_back(sc);
    CHECK_FALSE(compile(smooth).error.is_empty());
    smooth.children[0].k = 1.0f;
    CHECK(compile(smooth).error.is_empty());
}

TEST_CASE("SDFArchetype scatter: evaluate() and its gradient stay finite") {
    ChunkParam param;
    param.voxel_size = 0.5f;
    SDFArchetype sdf;
    sdf.set_scatter(1000, 200.0);
    sdf.configure(param, 15);
    CHECK(sdf.get_snapshot()->records.size() == 1008);
    std::vector<uint32_t> list;
    int64_t nonfinite = 0;
    int64_t checked = 0;
    for (uint32_t i = 0; i < 8000; i++) {
        const WorldPos p = random_point(VoxelPos(-100, 600, -300), 420.0, i, 7);
        for (const int lod : { 0, 2, 4 }) {
            sdf.get_snapshot()->point_candidates(p.voxel, lod, list);
            const Float4 r = finish_table(fold(sdf.get_ground(), sdf.get_library(), sdf.get_snapshot()->records, list.data(),
                                                  uint32_t(list.size()), p, lod),
                    lod, 0.5f, true);
            if (!std::isfinite(r.x) || !std::isfinite(r.y) || !std::isfinite(r.z) || !std::isfinite(r.w)) {
                nonfinite++;
            }
            checked++;
        }
    }
    MESSAGE("scatter: ", checked, " evaluations, ", nonfinite, " non-finite");
    CHECK(nonfinite == 0);
}

TEST_CASE("SDFArchetype table: valid at 0.5 m and 5 cm voxels, evaluate() is the fold, far away the bare ground") {
    for (const float voxel : { 0.5f, 0.05f }) {
        CAPTURE(voxel);
        ChunkParam param;
        param.voxel_size = voxel;
        SDFArchetype sdf;
        sdf.configure(param, 25);
        const Program &ground = sdf.get_ground();
        CHECK(ground.validate(25).is_empty());
        for (int i = 0; i < sdf.get_library().size(); i++) {
            CHECK(sdf.get_library()[i].program.validate(25).is_empty());
        }
        CHECK(sdf.get_snapshot());
        CHECK(sdf.get_snapshot()->records.size() == 8); // 4 rocks + 3 trunks + 1 cave
        // The library and ground evaluate: evaluate() is the fold over the point's candidates, box-tested.
        int below = 0;
        std::vector<uint32_t> list;
        for (uint32_t i = 0; i < 2000; i++) {
            const WorldPos p = random_point(VoxelPos(std::llround(-50.0 / voxel), std::llround(300.0 / voxel), std::llround(-140.0 / voxel)), 60.0 / voxel,
                    i, 33);
            for (const int lod : { 0, 4 }) {
                sdf.get_snapshot()->point_candidates(p.voxel, lod, list);
                const Float4 r = finish_table(fold(ground, sdf.get_library(), sdf.get_snapshot()->records, list.data(), uint32_t(list.size()),
                                                      p, lod),
                        lod, float(voxel), true);
                CHECK(float_bits(sdf.evaluate(p, lod)) == float_bits(r.w));
                below += (lod == 0 && r.w < 0.0f) ? 1 : 0;
            }
        }
        CHECK(below > 200);
        CHECK(below < 1800);
        // Far away (a far origin's view) the instances' boxes are nowhere near: the bare ground, every bit.
        const VoxelPos far(2000000000000000, std::llround(300.0 / voxel), -2000000000000000);
        for (uint32_t i = 0; i < 500; i++) {
            const WorldPos p = random_point(far, 200.0 / voxel, i, 34);
            for (const int lod : { 0, 4 }) {
                sdf.get_snapshot()->point_candidates(p.voxel, lod, list);
                // No instance's box contains a far point: the fold is the bare ground.
                const float g = finish_table(run(ground, p, lod), lod, float(voxel), true).w;
                CHECK(float_bits(sdf.evaluate(p, lod)) == float_bits(g));
            }
        }
    }
}

// Edits replace SDFArchetype's snapshot on the main thread while the discoverer thread evaluates and runs
// cull_exempt: a plain shared_ptr assignment there freed the snapshot under the reader (SIGSEGV).
TEST_CASE("SDFArchetype: evaluating on another thread while the main thread moves an instance") {
    ChunkParam param;
    SDFArchetype sdf;
    sdf.configure(param, 15);
    const VoxelPos anchor = sdf.get_snapshot()->records[0].anchor;
    std::atomic<bool> stop{ false };
    std::atomic<int64_t> evaluations{ 0 };
    std::thread reader([&]() {
        uint32_t i = 0;
        while (!stop.load()) {
            WorldPos p;
            p.voxel = anchor + VoxelPos(int64_t(i % 17) - 8, int64_t(i % 13) - 6, int64_t(i % 11) - 5);
            (void)sdf.evaluate(p, int(i % 3));
            (void)sdf.cull_exempt(ChunkId(1, 1, 1), 0);
            i++;
            evaluations.fetch_add(1);
        }
    });
    for (int step = 0; step < 5000; step++) {
        CHECK(sdf.move_instance(0, anchor + VoxelPos(step % 7, 0, step % 5), nullptr));
    }
    stop.store(true);
    reader.join();
    CHECK(evaluations.load() > 0);
}

TEST_CASE("SDFArchetype: the shapes gallery is valid, its ground flat, its cubes holed through, its die pipped") {
    ChunkParam param;
    const double vs = param.voxel_size;
    SDFArchetype sdf(SDFArchetype::Scene::SHAPES);
    sdf.configure(param, 25);
    CHECK(sdf.get_ground().validate(25).is_empty());
    const Instance scene = SDFArchetype::shapes(vs);
    CHECK(compile(scene).error.is_empty());
    // The SDF (metres) at p_anchor + (p_x, p_y, p_z) metres.
    auto sdf_at = [&](const VoxelPos &p_anchor, double p_x, double p_y, double p_z) {
        const double d[3] = { p_x / vs, p_y / vs, p_z / vs };
        WorldPos p;
        float *frac[3] = { &p.frac.x, &p.frac.y, &p.frac.z };
        for (int a = 0; a < 3; a++) {
            p.voxel[a] = p_anchor[a] + int64_t(std::floor(d[a]));
            *frac[a] = float(d[a] - std::floor(d[a]));
        }
        return sdf.evaluate(p, 0);
    };
    // Far from the shapes: 10 m above the flat ground.
    CHECK(std::abs(sdf_at(scene.anchor, 5000.0, 10.0, 5000.0) - 10.0f) < 1e-3f);
    // Probes on vertical lines and rings around them: the yaw of an instance doesn't move them.
    auto ring = [&](const VoxelPos &p_centre, double p_radius, double p_y) {
        float worst = -INFINITY;
        for (int a = 0; a < 16; a++) {
            const double phi = a * Math_PI / 8.0;
            worst = MAX(worst, sdf_at(p_centre, p_radius * std::cos(phi), p_y, p_radius * std::sin(phi)));
        }
        return worst;
    };
    // The ladder: air at the centre and up through the top face, solid in the wall around the top hole.
    for (int i = 0; i < 8; i++) {
        CAPTURE(i);
        const VoxelPos c = scene.anchor + scene.children[i].instance.anchor;
        const double h = 8.0 * double(1 << i);
        CHECK(sdf_at(c, 0.0, 0.0, 0.0) > 0.0f);
        CHECK(sdf_at(c, 0.0, 0.95 * h, 0.0) > 0.0f);
        CHECK(ring(c, 0.88 * h, 0.92 * h) < 0.0f);
    }
    // The die (half size 6 m): solid, but the top's one pip.
    const VoxelPos die = scene.anchor + scene.children[8].instance.anchor;
    CHECK(sdf_at(die, 0.0, 0.0, 0.0) < 0.0f);
    CHECK(sdf_at(die, 0.0, 4.0, 0.0) < 0.0f);
    CHECK(sdf_at(die, 0.0, 5.5, 0.0) > 0.0f);
    // The staircase: step k's top at k + 1 m.
    const Instance &stairs = scene.children[9].instance;
    for (int k = 0; k < 8; k++) {
        CAPTURE(k);
        // A child's anchor is in its parent's frame: R^T of the offset, rounded to voxels.
        VoxelPos off;
        if (k > 0) {
            const VoxelPos &o = stairs.children[k - 1].instance.anchor;
            const float *r = stairs.rotation;
            for (int a = 0; a < 3; a++) {
                off[a] = std::llround(double(r[a]) * double(o.x) + double(r[3 + a]) * double(o.y) + double(r[6 + a]) * double(o.z));
            }
        }
        const VoxelPos c = scene.anchor + stairs.anchor + off;
        CHECK(sdf_at(c, 0.0, k + 0.5, 0.0) < 0.0f);
        CHECK(sdf_at(c, 0.0, k + 1.5, 0.0) > 0.0f);
    }
    // The tunnelled cube (half size 6 m, tunnels 2.5 m): air down the vertical tunnel, solid around it.
    const VoxelPos tunnels = scene.anchor + scene.children[10].instance.anchor;
    CHECK(sdf_at(tunnels, 0.0, 0.0, 0.0) > 0.0f);
    CHECK(sdf_at(tunnels, 0.0, 5.0, 0.0) > 0.0f);
    CHECK(ring(tunnels, 4.5, 5.0) < 0.0f);
}

// B6: the band rule makes two LODs see different surfaces (an octave or an instance dropped or eroded between
// them). A skirt pushes its chunk's rim towards air by skirt_depth voxels of its LOD, so at a seam between a LOD L
// and a LOD L+1 chunk the field may differ by at most the skirt depth of the chunk whose surface is the outer one,
// or a crack shows. On the demo ground with a dense scatter, at points near the LOD-L surface of columns of
// Λ_{L+1}, with the skirt depth the terrain asks for (min_skirt_depth).
TEST_CASE("SDFArchetype band: the field step between LOD L and L+1 stays within the skirt depth") {
    ChunkParam param;
    param.voxel_size = 0.5f;
    SDFArchetype sdf;
    sdf.set_scatter(4000, 200.0);
    const float skirt = MAX(param.skirt_depth, sdf.min_skirt_depth(param, 15));
    param.skirt_depth = skirt;
    sdf.configure(param, 15);
    MESSAGE("skirt depth asked for: ", skirt, " voxels of the chunk's LOD");
    CHECK(skirt >= ERODE_CELLS);
    const double datum = SDFBase::DATUM_M / 0.5;
    auto at = [&](const VoxelPos &p_column, double p_y, int p_lod) {
        WorldPos p;
        p.voxel = VoxelPos(p_column.x, int64_t(std::floor(p_y)), p_column.z);
        p.frac = Float3{ 0.0f, float(p_y - std::floor(p_y)), 0.0f };
        return double(sdf.evaluate(p, p_lod));
    };
    for (int lod = 0; lod <= 6; lod++) {
        CAPTURE(lod);
        const double cell = std::ldexp(1.0, lod);
        double worst = 0.0;
        int64_t points = 0;
        const int64_t unit = int64_t(2) << lod; // Λ_{L+1}
        for (uint32_t i = 0; i < 300; i++) {
            const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ i, uint32_t(lod), 0x5EA4u });
            const VoxelPos column(WorldCoord::floor_div(-100 + int64_t(h.x % 400u) - 200, unit) * unit, 0,
                    WorldCoord::floor_div(-300 + int64_t(h.y % 400u) - 200, unit) * unit);
            // The LOD-L top surface: scan down in cells.
            double y = datum + 300.0;
            while (at(column, y - cell, lod) > 0.0 && y > datum - 300.0) {
                y -= cell;
            }
            for (double d = -2.0 * cell; d <= 2.0 * cell; d += 0.25 * cell) {
                const double fine = at(column, y + d, lod);
                if (std::fabs(fine) > double(param.voxel_size) * cell) {
                    continue; // only near the surface, where the skirt has to hide the step
                }
                const double step = at(column, y + d, lod + 1) - fine;
                // step > 0: the coarse field is more air there, its surface the inner one: the fine rim must cover.
                const double depth_m = double(skirt) * double(param.voxel_size) * std::ldexp(1.0, step > 0.0 ? lod : lod + 1);
                worst = MAX(worst, std::fabs(step) / depth_m);
                points++;
            }
        }
        MESSAGE("LOD ", lod, " | ", lod + 1, ": ", points, " points near the surface, worst step ", worst, " of the outer chunk's skirt depth");
        CHECK(points > 300);
        CHECK(worst <= 1.0);
    }
}

// An edit only re-meshes the LODs the changed instance is listed at: a small rock added next to the spawn
// asks for its LOD 0 chunk, not for the coarse chunks over it (their lists never held it).
TEST_CASE("SDFArchetype band: an edit re-meshes only the LODs its instance is listed at") {
    ChunkParam param;
    param.voxel_size = 0.5f;
    SDFArchetype sdf;
    sdf.configure(param, 15);
    const float params[8] = { 4.0f, 0.4f, 0, 0, 0, 0, 0, 0 }; // a 2 m rock (sphere type 0: radius 4 voxels)
    const VoxelPos at(-104, 600, -190);
    const int id = sdf.add_instance(0, at, nullptr, params, int(Combine::UNION), 0.0f);
    CHECK(id >= 0);
    if (id < 0) {
        return;
    }
    const TableRecord &rec = sdf.get_snapshot()->records[id];
    MESSAGE("a radius-4 sphere is listed at LODs [", int(rec.min_lod), ", ", int(rec.max_lod), ")");
    CHECK(rec.max_lod < 15);
    for (int lod = 0; lod < 15; lod++) {
        const int64_t span = int64_t(param.chunk_voxel_dimension()) << lod;
        const int64_t unit = int64_t(1) << lod;
        const ChunkId c((WorldCoord::floor_div(at.x, span) * 2 + 1) * unit, (WorldCoord::floor_div(at.y, span) * 2 + 1) * unit,
                (WorldCoord::floor_div(at.z, span) * 2 + 1) * unit);
        CAPTURE(lod);
        CHECK(sdf.edit_affects_chunk(c) == (lod < rec.max_lod));
    }
}

// Constant cost per chunk: at a fixed density of rocks, the longest list at each LOD doesn't grow with the
// number of rocks (10x more rocks on 10x the area), and from LOD 6 on the rocks are eroded away and dropped
// (at LOD 4 a chunk outgrows the small scene's square, so only the bigger one fills it):
// the lists only hold the demo's big objects.
TEST_CASE("SDFArchetype band: list lengths are flat in the number of instances at a fixed density") {
    int64_t longest[2][13] = {};
    for (int run = 0; run < 2; run++) {
        const int n = run == 0 ? 2000 : 20000;
        const double area = 200.0 * std::sqrt(double(n) / 2000.0);
        ChunkParam param;
        param.voxel_size = 0.5f;
        SDFArchetype sdf;
        sdf.set_scatter(n, area);
        sdf.configure(param, 15);
        const std::shared_ptr<const InstanceSnapshot> snap = sdf.get_snapshot();
        std::vector<uint32_t> list;
        for (int lod = 0; lod <= 12; lod++) {
            for (uint32_t i = 0; i < 48; i++) {
                const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ i, uint32_t(lod), 0x7111u });
                const double px = -50.0 + (double(h.x >> 8) / 16777216.0 - 0.5) * area;
                const double pz = -150.0 + (double(h.y >> 8) / 16777216.0 - 0.5) * area;
                const int64_t span = int64_t(param.chunk_voxel_dimension()) << lod;
                const int64_t unit = int64_t(1) << lod;
                const ChunkId c((WorldCoord::floor_div(int64_t(px / 0.5), span) * 2 + 1) * unit, (WorldCoord::floor_div(600, span) * 2 + 1) * unit,
                        (WorldCoord::floor_div(int64_t(pz / 0.5), span) * 2 + 1) * unit);
                snap->chunk_list(c, list);
                longest[run][lod] = MAX(longest[run][lod], int64_t(list.size()));
            }
        }
    }
    for (int lod = 0; lod <= 12; lod++) {
        CAPTURE(lod);
        MESSAGE("LOD ", lod, ": longest list ", longest[0][lod], " (2,000 rocks) vs ", longest[1][lod], " (20,000 on 10x the area)");
        if (lod <= 3) {
            // Up to LOD 3 a chunk (112 m) is smaller than the 200 m square: the same density, the same lists.
            CHECK(double(longest[1][lod]) <= 1.5 * double(longest[0][lod]) + 8.0);
        } else if (lod >= 6) {
            CHECK(longest[1][lod] <= 8); // the demo objects at most: every scattered rock is dropped by LOD 6
        }
    }
}

// Edits go through the overlay (a small index over the edited records, the base kept): after random adds,
// moves and removes, every chunk list and point query equals the one of a whole new index over the same
// records; past REBASE edited records the base is rebuilt. The GPU gets the pages an edit changed.
TEST_CASE("SDFArchetype table: the edit overlay lists exactly what a whole new index lists") {
    ChunkParam param;
    param.voxel_size = 0.5f;
    SDFArchetype sdf;
    sdf.set_scatter(3000, 150.0);
    sdf.configure(param, 15);
    const std::shared_ptr<const InstanceSnapshot> first = sdf.get_snapshot();
    int64_t lists = 0, entries = 0, edits = 0;
    std::vector<uint32_t> touched;
    for (uint32_t step = 0; step < 2600; step++) {
        const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ step, 0xED17u, 0x0FA7u });
        const uint32_t n = sdf.get_snapshot()->records.size();
        const VoxelPos at(-100 + int64_t(h.y % 300u) - 150, 600 + int64_t(h.z % 40u) - 20, -300 + int64_t(h.x % 300u) - 150);
        int id = -1;
        if (h.x % 3u == 0u) {
            const float params[8] = { 3.0f + float(h.y % 6u), 0, 0, 0, 0, 0, 0, 0 };
            id = sdf.add_instance(0, at, nullptr, params, int(h.z % 4u), 1.0f);
        } else if (h.x % 3u == 1u) {
            id = int(h.y % n);
            id = sdf.move_instance(id, at, nullptr) ? id : -1;
        } else {
            id = int(h.z % n);
            id = sdf.remove_instance(id) ? id : -1;
        }
        if (id < 0) {
            continue;
        }
        edits++;
        touched.push_back(uint32_t(id));
        if (step % 200 != 199) {
            continue;
        }
        const std::shared_ptr<const InstanceSnapshot> snap = sdf.get_snapshot();
        const std::shared_ptr<const InstanceSnapshot> whole =
                build_snapshot(snap->library, snap->ground, snap->out_scale, snap->normalize, snap->records, snap->param, snap->max_lod, snap->version);
        std::vector<uint32_t> a, b;
        for (uint32_t i = 0; i < 64; i++) {
            const VoxelPos &v = snap->records[touched[(i * 7919u) % touched.size()]].anchor;
            for (int lod = 0; lod <= 6; lod++) {
                const int64_t span = int64_t(param.chunk_voxel_dimension()) << lod;
                const int64_t unit = int64_t(1) << lod;
                const ChunkId c((WorldCoord::floor_div(v.x, span) * 2 + 1) * unit, (WorldCoord::floor_div(v.y, span) * 2 + 1) * unit,
                        (WorldCoord::floor_div(v.z, span) * 2 + 1) * unit);
                snap->chunk_list(c, a);
                whole->chunk_list(c, b);
                CHECK(a == b);
                snap->point_candidates(v + VoxelPos(1, 2, 3), lod, a);
                whole->point_candidates(v + VoxelPos(1, 2, 3), lod, b);
                CHECK(a == b);
                CHECK(snap->near_instance(c, lod) == true); // an instance is there (or was: conservative)
                lists++;
                entries += int64_t(b.size());
            }
        }
        MESSAGE("after ", edits, " edits: overlay ", snap->overlay_ids.size(), " records, base ", std::string(snap->base == first->base ? "kept" : "rebuilt"));
    }
    MESSAGE("overlay vs whole index: ", lists, " chunk lists and point queries, ", entries, " entries");
    CHECK(sdf.get_snapshot()->base != first->base); // more than REBASE records edited: rebuilt at least once
    CHECK(entries > 1000);
    // An edit uploads the page it changed, not the table.
    uint32_t v = 0, first_record = 0;
    const uint32_t before = sdf.get_snapshot()->version;
    const float params[8] = { 4.0f, 0, 0, 0, 0, 0, 0, 0 };
    CHECK(sdf.add_instance(0, VoxelPos(-100, 600, -300), nullptr, params, 0, 0.0f) >= 0);
    const PackedByteArray bytes = sdf.get_gpu_table()->record_bytes(before, v, first_record);
    CHECK(v == before + 1);
    CHECK(bytes.size() > 0);
    CHECK(bytes.size() <= int64_t(RecordStore::PAGE) * RECORD_BYTES);
    CHECK(first_record % RecordStore::PAGE == 0);
}
