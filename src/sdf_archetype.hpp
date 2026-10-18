#pragma once

#include "archetype_gpu_table.hpp"
#include "archetype_table.hpp"
#include "instance_table.hpp"
#include "sdf_base.hpp"
#include "sdf_dummy.hpp"

#include <cmath>

// Demo terrain of the archetype table (terrain_mode 3, --terrain=archetypes): an fBm ground (a compiled
// Program) plus a table of instances (rocks, tree trunks and a cave mouth) in front of the main_scene
// spawn. Each chunk folds only the instances near it (instance_table.hpp); the library
// (archetype_table.hpp) is compiled once and reads the instances' records from a GPU buffer. A function
// of the absolute position: far origins see the ground only.
class SDFArchetype : public SDFBase {
public:
    static constexpr uint32_t HEIGHT_SEED = 0xA2C4E7u;
    static constexpr uint32_t ROCK_SEED = 0x20C4u;

    void configure(const ChunkParam &p_param, int p_lod_count) override {
        SDFBase::configure(p_param, p_lod_count);
        create_scene(p_param.voxel_size, p_lod_count);
        const String ground_error = snapshot->ground.validate(p_lod_count);
        if (!ground_error.is_empty()) {
            ERR_PRINT(ground_error);
        }
        for (int i = 0; i < library.size(); i++) {
            const String error = library[i].program.validate(p_lod_count);
            if (!error.is_empty()) {
                ERR_PRINT(vformat("Archetype library type %d: ", i) + error);
            }
        }
    }

    float evaluate(const WorldPos &p_pos, int p_lod) const override {
        thread_local std::vector<uint32_t> list;
        snapshot->point_candidates(p_pos.voxel, list);
        const Archetype::Float4 r = Archetype::fold(snapshot->ground, *snapshot->library, snapshot->records.data(), list.data(), uint32_t(list.size()), p_pos, p_lod);
        return Archetype::finish_table(r, p_lod, snapshot->out_scale, snapshot->normalize).w;
    }

    void evaluate_batch(const VoxelPos *p_points, int p_count, int p_lod, float *r_out) const override {
        evaluate_batch_parallel(p_points, p_count, p_lod, r_out);
    }

    String get_glsl_code() const override {
        return Archetype::generate_table_sdf(snapshot->ground, *snapshot->library, snapshot->out_scale, snapshot->normalize);
    }

    SDFGpuTable *get_gpu_table() override { return &gpu; }

    bool cull_exempt(const ChunkId &p_node, int p_lod) const override {
        // The node's world box grown by the cull band (2 chunk sizes at its LOD), against every instance's
        // influence box. O(records) with a short circuit: fine for the demo's small tables.
        if (!snapshot) {
            return false;
        }
        const VoxelPos min = ChunkMath::chunk_min_voxel(p_node, param);
        const int64_t span = int64_t(param.chunk_voxel_dimension()) << p_lod;
        const int64_t grow = 2 * span;
        const VoxelPos lo = min - VoxelPos(grow, grow, grow);
        const VoxelPos hi = min + VoxelPos(span + grow, span + grow, span + grow);
        for (const Archetype::TableRecord &rec : snapshot->records) {
            bool overlap = true;
            for (int a = 0; a < 3; a++) {
                const int64_t rlo = rec.anchor[a] + rec.box_lo[a];
                const int64_t rhi = rec.anchor[a] + rec.box_hi[a];
                if (hi[a] < rlo || lo[a] > rhi) {
                    overlap = false;
                    break;
                }
            }
            if (overlap) {
                return true;
            }
        }
        return false;
    }

    const std::shared_ptr<const Archetype::InstanceSnapshot> &get_snapshot() const { return snapshot; }
    const Archetype::DefLibrary &get_library() const { return *snapshot->library; }
    const Archetype::Program &get_ground() const { return snapshot->ground; }

    // The compiled (PR 1) reference of the same scene (bounds test, arch_demo probe).
    static Archetype::Instance scene(double p_voxel) {
        using namespace Archetype;
        auto v = [p_voxel](double p_metres) { return float(p_metres / p_voxel); };
        const ExactNoise::OctaveSet height = SDFDummy::height_fbm(p_voxel, HEIGHT_SEED);
        const int64_t datum = std::llround(SDFDummy::DATUM_M / p_voxel);
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
            const ExactNoise::OctaveSet bumps = ExactNoise::make_fbm(false, ROCK_SEED, SDFDummy::k_of(2.0 * r.r, p_voxel), 2, 0.1 * r.r / p_voxel, 0.5);
            add(Combine::SMOOTH_UNION, 1.5, rock_def(v(r.r), bumps), r.x, r.z, -0.4 * r.r, r.yaw);
        }
        for (const double x : { -50.0, -33.0, -58.0 }) {
            add(Combine::SMOOTH_UNION, 1.0, capsule_def(v(12.0), v(1.0)), x, -160.0 + 0.3 * x, -1.0, 0.0);
        }
        add(Combine::SMOOTH_SUBTRACT, 1.0, sphere_def(v(7.0)), -10.0, -170.0, -3.0, 0.0);
        return root;
    }

private:
    Archetype::DefLibrary library;
    std::shared_ptr<const Archetype::InstanceSnapshot> snapshot;
    Archetype::GpuTable gpu;

    void create_scene(double p_voxel, int p_lod_count) {
        using namespace Archetype;
        auto v = [p_voxel](double p_metres) { return float(p_metres / p_voxel); };
        const ExactNoise::OctaveSet height = SDFDummy::height_fbm(p_voxel, HEIGHT_SEED);
        const int64_t datum = std::llround(SDFDummy::DATUM_M / p_voxel);
        auto ground = [&](double p_x, double p_z) {
            WorldPos p;
            p.voxel = VoxelPos(std::llround(p_x / p_voxel), 0, std::llround(p_z / p_voxel));
            return int64_t(std::floor(ExactNoise::fbm(height, p, 0).w));
        };
        library = DefLibrary();
        const int sphere = library.add(make_sphere_type());
        const int capsule = library.add(make_capsule_type());
        const int rock = library.add(make_rock_type(int(std::lround(std::log2(8.0 / p_voxel))), ROCK_SEED));

        std::vector<TableRecord> records;
        uint32_t seed = 1;
        auto add = [&](Combine p_op, double p_k_m, int p_type, const float *p_params, double p_x, double p_z, double p_dy_m, double p_yaw) {
            TableRecord rec;
            rec.def = uint8_t(p_type);
            rec.op = p_op;
            rec.k = v(p_k_m);
            rec.inv_k = rec.k != 0.0f ? float(1.0 / double(rec.k)) : 0.0f;
            for (int i = 0; i < 8; i++) {
                rec.params[i] = p_params[i];
            }
            rec.anchor = VoxelPos(std::llround(p_x / p_voxel), datum + ground(p_x, p_z) + std::llround(p_dy_m / p_voxel), std::llround(p_z / p_voxel));
            const float c = float(std::cos(p_yaw)), s = float(std::sin(p_yaw));
            const float rot[9] = { c, 0, -s, 0, 1, 0, s, 0, c };
            std::copy(rot, rot + 9, rec.rotation);
            const uint32_t salt_seed = seed++;
            rec.salt = salt_seed == 0 ? 0u : ExactNoise::pcg3d(ExactNoise::UVec3{ salt_seed, 0x5EEDu, 0u }).x;
            const Bounds wb = record_world_bounds(library[rec.def], rec);
            for (int a = 0; a < 3; a++) {
                rec.box_lo[a] = int32_t(wb.lo[a] - rec.anchor[a]);
                rec.box_hi[a] = int32_t(wb.hi[a] - rec.anchor[a]);
            }
            rec.lipschitz = wb.lipschitz;
            rec.margin_c = 2.0f * wb.lipschitz + param.skirt_depth;
            records.push_back(rec);
        };
        const struct {
            double x, z, r, yaw;
        } rocks[] = { { -40.0, -150.0, 4.0, 0.3 }, { -62.0, -132.0, 2.5, 1.2 }, { -25.0, -128.0, 3.0, 2.0 }, { -75.0, -165.0, 5.0, 0.7 } };
        for (const auto &r : rocks) {
            const float params[8] = { v(r.r), 0.1f * v(r.r), 0, 0, 0, 0, 0, 0 };
            add(Combine::SMOOTH_UNION, 1.5, rock, params, r.x, r.z, -0.4 * r.r, r.yaw);
        }
        for (const double x : { -50.0, -33.0, -58.0 }) {
            const float params[8] = { v(12.0), v(1.0), 0, 0, 0, 0, 0, 0 };
            add(Combine::SMOOTH_UNION, 1.0, capsule, params, x, -160.0 + 0.3 * x, -1.0, 0.0);
        }
        {
            const float params[8] = { v(7.0), 0, 0, 0, 0, 0, 0, 0 };
            add(Combine::SMOOTH_SUBTRACT, 1.0, sphere, params, -10.0, -170.0, -3.0, 0.0);
        }
        // The ground register: the ground Def placed at the datum (a compiled program, constants only).
        Archetype::Instance ground_inst;
        ground_inst.def = ground_def(height);
        ground_inst.anchor = VoxelPos(0, datum, 0);
        const Program ground_program = compile(ground_inst).program;
        snapshot = Archetype::build_snapshot(std::make_shared<const DefLibrary>(library), ground_program, float(p_voxel), true, std::move(records), param,
                p_lod_count - 1, 1);
        gpu.set_snapshot(snapshot);
    }
};
