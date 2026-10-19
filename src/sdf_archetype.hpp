#pragma once

#include "sdf_table.hpp"

#include <cmath>
#include <memory>

// Demo terrain of the archetype table (terrain_mode 3, --terrain=archetypes): an fBm ground (a compiled Program)
// under the instance table (SDFTable: the demo objects, the scatter and the forest in front of the main_scene
// spawn). A function of the absolute position: far origins see the ground only.
// Ground::SHAPES (terrain_mode 4, --terrain=shapes) has no table instances: shapes() compiled into the ground program
// (hard CSG for dual contouring, nested composites the table does not express).
class SDFArchetype : public SDFTable {
public:
    enum class Ground { DEMO,
        SHAPES };

    explicit SDFArchetype(Ground p_ground = Ground::DEMO) :
            which(p_ground) {}

    static constexpr uint32_t HEIGHT_SEED = 0xA2C4E7u;
    // The ground's mountains (metres).
    static constexpr double HEIGHT_WAVELENGTH_M = 1.0 / 0.0035;
    static constexpr double HEIGHT_AMPLITUDE_M = 50.0;
    static constexpr int HEIGHT_OCTAVES = 5;
    // Scales the octaves to the spread of the simplex noise the mountains were tuned with (std 0.36 for one
    // octave): perlin2 has std 0.48.
    static constexpr double NORM_2D = 0.75;

    // A wavelength of p_wavelength_m metres as 2^k voxels of p_voxel metres.
    static int k_of(double p_wavelength_m, double p_voxel) { return int(std::lround(std::log2(p_wavelength_m / p_voxel))); }

    // The mountains' fBm at voxel size p_voxel (heights in LOD0 voxels).
    static ExactNoise::OctaveSet height_fbm(double p_voxel, uint32_t p_seed = HEIGHT_SEED) {
        return ExactNoise::make_fbm(true, p_seed, k_of(HEIGHT_WAVELENGTH_M, p_voxel), HEIGHT_OCTAVES, HEIGHT_AMPLITUDE_M / p_voxel * NORM_2D, 0.5);
    }

    // The band rule's steps between LOD L and L+1: an instance erodes (SDFTable); a ground octave k, dropped from
    // LOD k - 2 on, moves the ground by up to amp * PERLIN2_MAX, i.e. that many cells of LOD k - 3. Both at once next
    // to a rock on the ground (its smooth blend follows both): the sum.
    float min_skirt_depth(const ChunkParam &p_param, int p_lod_count) const override {
        double ground = 0.0;
        if (which == Ground::DEMO) {
            for (const ExactNoise::Octave &o : height_fbm(p_param.voxel_size).octaves) {
                if (o.k - ExactNoise::BAND_SHIFT + 1 < p_lod_count) {
                    ground = MAX(ground, double(o.amp) * Archetype::PERLIN2_MAX * std::ldexp(1.0, ExactNoise::BAND_SHIFT - o.k));
                }
            }
        }
        return float(ground) + SDFTable::min_skirt_depth(p_param, p_lod_count);
    }

    void configure(const ChunkParam &p_param, int p_lod_count) override {
        SDFBase::configure(p_param, p_lod_count);
        const double vs = p_param.voxel_size;
        height = height_fbm(vs);
        datum = std::llround(DATUM_M / vs);
        // The ground register: the ground Def placed at the datum (a compiled program, constants only), or the
        // shapes gallery, which is the ground program itself (the table starts empty; edits still work).
        Archetype::Instance ground;
        ground.def = Archetype::ground_def(height);
        ground.anchor = VoxelPos(0, datum, 0);
        if (which == Ground::SHAPES) {
            ground = shapes(vs);
        }
        const Archetype::Compiled compiled = Archetype::compile(ground);
        if (!compiled.error.is_empty()) {
            ERR_PRINT(compiled.error);
        }
        const String error = compiled.program.validate(p_lod_count);
        if (!error.is_empty()) {
            ERR_PRINT(error);
        }
        build_table(compiled.program, vs, p_lod_count, which == Ground::SHAPES ? Scene::EMPTY : Scene::DEMO);
    }

    float evaluate(const WorldPos &p_pos, int p_lod) const override {
        thread_local std::vector<uint32_t> list;
        const std::shared_ptr<const Archetype::InstanceSnapshot> snap = current();
        snap->point_candidates(p_pos.voxel, p_lod, list);
        const Archetype::Float4 r = Archetype::fold(snap->ground, *snap->library, snap->records, list.data(), uint32_t(list.size()), p_pos, p_lod);
        return Archetype::finish_table(r, p_lod, snap->out_scale, snap->normalize).w;
    }

    void evaluate_batch(const VoxelPos *p_points, int p_count, int p_lod, float *r_out) const override {
        evaluate_batch_parallel(p_points, p_count, p_lod, r_out);
    }

    String get_glsl_code() const override {
        const std::shared_ptr<const Archetype::InstanceSnapshot> snap = current();
        return Archetype::generate_table_sdf(snap->ground, *snap->library, snap->out_scale, snap->normalize);
    }

    // The mountains' surface (the scenes stand on the fBm ground, also under the shapes gallery's flat one).
    int64_t ground_y(int64_t p_x, int64_t p_z) const override {
        WorldPos p;
        p.voxel = VoxelPos(p_x, 0, p_z);
        return datum + int64_t(std::floor(ExactNoise::fbm(height, p, 0).w));
    }

    const Archetype::Program &get_ground() const { return current()->ground; }

    // The compiled (PR 1) reference of the same scene (bounds test, arch_demo probe).
    static Archetype::Instance scene(double p_voxel) {
        using namespace Archetype;
        auto v = [p_voxel](double p_metres) { return float(p_metres / p_voxel); };
        const ExactNoise::OctaveSet height = height_fbm(p_voxel);
        const int64_t datum = std::llround(DATUM_M / p_voxel);
        Instance root;
        root.def = ground_def(height);
        root.anchor = VoxelPos(0, datum, 0);
        auto ground = [&](double p_x, double p_z) {
            WorldPos p;
            p.voxel = VoxelPos(std::llround(p_x / p_voxel), 0, std::llround(p_z / p_voxel));
            return int64_t(std::floor(ExactNoise::fbm(height, p, 0).w));
        };
        auto add = [&](Combine p_op, double p_k_m, const std::shared_ptr<Def> &p_def, double p_x, double p_z, double p_dy_m, double p_yaw) {
            Instance::Child c;
            c.op = p_op;
            c.k = v(p_k_m);
            c.instance.def = p_def;
            c.instance.anchor = VoxelPos(std::llround(p_x / p_voxel), ground(p_x, p_z) + std::llround(p_dy_m / p_voxel), std::llround(p_z / p_voxel));
            c.instance.set_rotation_y(p_yaw);
            c.instance.seed = uint32_t(root.children.size() + 1);
            root.children.push_back(c);
        };
        const struct {
            double x, z, r, yaw;
        } rocks[] = { { -40.0, -150.0, 4.0, 0.3 }, { -62.0, -132.0, 2.5, 1.2 }, { -25.0, -128.0, 3.0, 2.0 }, { -75.0, -165.0, 5.0, 0.7 } };
        for (const auto &r : rocks) {
            const ExactNoise::OctaveSet bumps = ExactNoise::make_fbm(false, ROCK_SEED, k_of(2.0 * r.r, p_voxel), 2, 0.1 * r.r / p_voxel, 0.5);
            add(Combine::SMOOTH_UNION, 1.5, rock_def(v(r.r), bumps), r.x, r.z, -0.4 * r.r, r.yaw);
        }
        for (const double x : { -50.0, -33.0, -58.0 }) {
            add(Combine::SMOOTH_UNION, 1.0, capsule_def(v(12.0), v(1.0)), x, -160.0 + 0.3 * x, -1.0, 0.0);
        }
        add(Combine::SMOOTH_SUBTRACT, 1.0, sphere_def(v(7.0)), -10.0, -170.0, -3.0, 0.0);
        return root;
    }

    // The shapes gallery at voxel size p_voxel (metres): hard unions, subtractions and intersections (sharp
    // creases, no smoothing) on a flat ground at GROUND_M, around the main_scene spawn (SPAWN_X, SPAWN_Z).
    // In front (-z): a ladder of eight cubes with a sphere hole through every face, each twice the size and
    // twice the distance of the previous one, so the same size on screen about one LOD coarser each (LOD 0
    // to ~7 at lod_pixel_error 4), turned to show the camera the same two faces. Behind (+z): a die (a cube
    // intersected with a sphere, its pips subtracted), a staircase and a cube tunnelled through along its
    // three axes, turned off the voxel axes.
    static constexpr double SPAWN_X = -52.0;
    static constexpr double SPAWN_Z = -95.0;
    static constexpr double GROUND_M = DATUM_M - 50.0;
    static Archetype::Instance shapes(double p_voxel) {
        using namespace Archetype;
        const Metres m{ p_voxel };
        Instance root;
        root.def = ground_def(ExactNoise::OctaveSet()); // no octave: y = 0
        // 50 m below the datum: halfway between the page shader's 100 m contour lines. On one (300 m), a flat
        // ground has fwidth ~ 0 and float noise picks the contour shade per triangle: it flickers.
        root.anchor = VoxelPos(0, m.at(GROUND_M), 0);
        // Centres at 0.9 of the half size: a little sunk, so the voxel rounding never lifts one off the ground.
        for (int i = 0; i < 8; i++) {
            const double d = 80.0 * double(1 << i);
            const double h = 0.1 * d;
            const double theta = (-45.5 + 13.0 * i) * Math_PI / 180.0;
            Instance::Child cube = m.child(Combine::UNION, box_def(m.v(h), m.v(h), m.v(h)), SPAWN_X + d * std::sin(theta), 0.9 * h, SPAWN_Z - d * std::cos(theta), 0.5 - theta);
            cube.instance.children.push_back(m.child(Combine::SUBTRACT, sphere_def(m.v(1.2 * h)), 0.0, 0.0, 0.0, 0.0));
            root.children.push_back(cube);
        }
        const double h = 6.0;
        root.children.push_back(die(m, h, SPAWN_X - 25.0, SPAWN_Z + 40.0));
        // The staircase: eight 1 m steps, 2 m deep, 8 m wide, rising away from the spawn; columns centred on the ground.
        Instance::Child stairs = m.child(Combine::UNION, box_def(m.v(4.0), m.v(1.0), m.v(1.0)), SPAWN_X, 0.0, SPAWN_Z + 33.0, 0.3);
        for (int k = 1; k < 8; k++) {
            stairs.instance.children.push_back(m.child(Combine::UNION, box_def(m.v(4.0), m.v(k + 1.0), m.v(1.0)), 0.0, 0.0, 2.0 * k, 0.0));
        }
        root.children.push_back(stairs);
        // The tunnelled cube: 5 m square tunnels along x, y and z.
        Instance::Child tunnels = m.child(Combine::UNION, box_def(m.v(h), m.v(h), m.v(h)), SPAWN_X + 25.0, 0.9 * h, SPAWN_Z + 40.0, 0.6);
        for (int a = 0; a < 3; a++) {
            const float e[3] = { m.v(a == 0 ? h + 1.0 : 2.5), m.v(a == 1 ? h + 1.0 : 2.5), m.v(a == 2 ? h + 1.0 : 2.5) };
            tunnels.instance.children.push_back(m.child(Combine::SUBTRACT, box_def(e[0], e[1], e[2]), 0.0, 0.0, 0.0, 0.0));
        }
        root.children.push_back(tunnels);
        return root;
    }

private:
    Ground which;
    ExactNoise::OctaveSet height; // the mountains (configure)
    int64_t datum = 0; // LOD0 voxels

    // shapes() in metres, rounded to voxels.
    struct Metres {
        double voxel;
        float v(double p_metres) const { return float(p_metres / voxel); }
        int64_t at(double p_metres) const { return std::llround(p_metres / voxel); }
        // p_def at (p_x, p_y, p_z) in its parent's frame (a child's anchor and rotation are relative to its
        // parent's, archetype_instance.hpp), turned by p_yaw.
        Archetype::Instance::Child child(Archetype::Combine p_op, const std::shared_ptr<Archetype::Def> &p_def, double p_x, double p_y, double p_z, double p_yaw) const {
            Archetype::Instance::Child c;
            c.op = p_op;
            c.instance.def = p_def;
            c.instance.anchor = VoxelPos(at(p_x), at(p_y), at(p_z));
            c.instance.set_rotation_y(p_yaw);
            return c;
        }
    };

    // The die of shapes(), half size p_h, on the ground at (p_x, p_z): pips on a 3 x 3 grid of p_h / 2 steps, faces
    // +y (1), +z (2), +x (3), -x (4), -z (5); 6 is underground.
    static Archetype::Instance::Child die(const Metres &p_m, double p_h, double p_x, double p_z) {
        using namespace Archetype;
        const double yaw = 0.4;
        Instance::Child die = p_m.child(Combine::UNION, box_def(p_m.v(p_h), p_m.v(p_h), p_m.v(p_h)), p_x, 0.95 * p_h, p_z, yaw);
        die.instance.children.push_back(p_m.child(Combine::INTERSECT, sphere_def(p_m.v(1.55 * p_h)), 0.0, 0.0, 0.0, 0.0));
        const char *faces[5] = { "000010000", "100000001", "100010001", "101000101", "101010101" };
        const int axis[5] = { 1, 2, 0, 0, 2 };
        const double sign[5] = { 1.0, 1.0, 1.0, -1.0, -1.0 };
        const std::shared_ptr<Def> pip = sphere_def(p_m.v(0.2 * p_h));
        for (int f = 0; f < 5; f++) {
            for (int g = 0; g < 9; g++) {
                if (faces[f][g] == '1') {
                    double p[3];
                    p[axis[f]] = sign[f] * p_h;
                    p[(axis[f] + 1) % 3] = double(g % 3 - 1) * 0.5 * p_h;
                    p[(axis[f] + 2) % 3] = double(g / 3 - 1) * 0.5 * p_h;
                    die.instance.children.push_back(p_m.child(Combine::SUBTRACT, pip, p[0], p[1], p[2], 0.0));
                }
            }
        }
        return die;
    }
};
