#pragma once

#include "archetype_glsl.hpp"
#include "archetype_instance.hpp"
#include "sdf_base.hpp"

#include <cmath>

// Demo terrain of the archetype IR (terrain_mode 3, --terrain=archetypes): one scene of instances compiled
// into one Program (archetype_instance.hpp), its GLSL generated as terrain_sdf (archetype_glsl.hpp) and its
// CPU evaluation the interpreter, bit for bit (FrameSpike probe arch_demo). An fBm ground (first-order
// distance of a 2D height field, height_fbm) with a few noisy rocks and tree trunks merged
// in by smooth unions and one cave mouth smoothly subtracted from a slope, all placed by absolute anchors
// in front of the main_scene spawn. A function of the absolute position: far origins see the ground only.
// Sizes are metres rounded to voxels in configure(), so another voxel size is another world.
class SDFArchetype : public SDFBase {
public:
    static constexpr uint32_t HEIGHT_SEED = 0xA2C4E7u;
    static constexpr uint32_t ROCK_SEED = 0x20C4u;
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

    void configure(const ChunkParam &p_param, int p_lod_count) override {
        SDFBase::configure(p_param, p_lod_count);
        Archetype::Compiled c = Archetype::compile(scene(p_param.voxel_size));
        program = c.program;
        program.out_scale = p_param.voxel_size;
        program.normalize = true;
        const String error = c.error.is_empty() ? program.validate(p_lod_count) : c.error;
        if (!error.is_empty()) {
            ERR_PRINT(error);
        }
    }

    float evaluate(const WorldPos &p_pos, int p_lod) const override { return Archetype::run(program, p_pos, p_lod).w; }

    // The discoverer's corner batches over the WorkerThreadPool (run() keeps its registers per thread).
    void evaluate_batch(const VoxelPos *p_points, int p_count, int p_lod, float *r_out) const override {
        evaluate_batch_parallel(p_points, p_count, p_lod, r_out);
    }

    String get_glsl_code() const override { return Archetype::generate_glsl(program, "terrain_sdf"); }

    const Archetype::Program &get_program() const { return program; }

    // The scene at voxel size p_voxel (metres).
    static Archetype::Instance scene(double p_voxel) {
        using namespace Archetype;
        auto v = [p_voxel](double p_metres) { return float(p_metres / p_voxel); };
        const ExactNoise::OctaveSet height = height_fbm(p_voxel);
        const int64_t datum = std::llround(DATUM_M / p_voxel);
        Instance root;
        root.def = ground_def(height);
        root.anchor = VoxelPos(0, datum, 0);
        // The ground's height (voxels above the datum) at a point (metres, x and z).
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
            // Relative to the root (the datum).
            c.instance.anchor = VoxelPos(std::llround(p_x / p_voxel), ground(p_x, p_z) + std::llround(p_dy_m / p_voxel), std::llround(p_z / p_voxel));
            c.instance.set_rotation_y(p_yaw);
            c.instance.seed = uint32_t(root.children.size() + 1);
            root.children.push_back(c);
        };
        // Rocks: 2.5 to 5 m spheres, partly buried, bumps of a tenth of the radius over twice the radius.
        const struct {
            double x, z, r, yaw;
        } rocks[] = { { -40.0, -150.0, 4.0, 0.3 }, { -62.0, -132.0, 2.5, 1.2 }, { -25.0, -128.0, 3.0, 2.0 }, { -75.0, -165.0, 5.0, 0.7 } };
        for (const auto &r : rocks) {
            const ExactNoise::OctaveSet bumps = ExactNoise::make_fbm(false, ROCK_SEED, k_of(2.0 * r.r, p_voxel), 2, 0.1 * r.r / p_voxel, 0.5);
            add(Combine::SMOOTH_UNION, 1.5, rock_def(v(r.r), bumps), r.x, r.z, -0.4 * r.r, r.yaw);
        }
        // Tree trunks, 12 m, rooted 1 m below the ground.
        for (const double x : { -50.0, -33.0, -58.0 }) {
            add(Combine::SMOOTH_UNION, 1.0, capsule_def(v(12.0), v(1.0)), x, -160.0 + 0.3 * x, -1.0, 0.0);
        }
        // A cave mouth: a 7 m sphere carved into the slope, its centre 3 m below the ground.
        add(Combine::SMOOTH_SUBTRACT, 1.0, sphere_def(v(7.0)), -10.0, -170.0, -3.0, 0.0);
        return root;
    }

private:
    Archetype::Program program;
};
