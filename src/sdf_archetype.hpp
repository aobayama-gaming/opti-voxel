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

    // A deterministic scatter of p_count rocks and pillars in a square of side p_area (metres) around
    // the demo objects (--instances / --instance_area). Set before configure().
    void set_scatter(int p_count, double p_area) {
        scatter_count = p_count;
        scatter_area = p_area;
    }

    void configure(const ChunkParam &p_param, int p_lod_count) override {
        SDFBase::configure(p_param, p_lod_count);
        create_scene(p_param.voxel_size, p_lod_count);
        const String ground_error = snapshot->ground.validate(p_lod_count);
        if (!ground_error.is_empty()) {
            ERR_PRINT(ground_error);
        }
        for (int i = 0; i < library->size(); i++) {
            const String error = (*library)[i].program.validate(p_lod_count);
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

    // Runtime edits: rebuild the snapshot (copy on write) and bump the version; the GPU re-uploads the
    // records when it changes. last_edit_box() / edit_affects_chunk() tell the terrain what to re-mesh.
    int add_instance(int p_type, const VoxelPos &p_anchor, const float *p_rotation, const float *p_params, int p_op, float p_k) override {
        if (!library || p_type < 0 || p_type >= library->size()) {
            return -1;
        }
        Archetype::TableRecord rec;
        rec.def = uint8_t(p_type);
        rec.op = Archetype::Combine(p_op);
        rec.k = p_k;
        rec.inv_k = p_k > 0.0f ? float(1.0 / double(p_k)) : 0.0f;
        rec.anchor = p_anchor;
        if (p_rotation) {
            std::copy(p_rotation, p_rotation + 9, rec.rotation);
        }
        if (p_params) {
            std::copy(p_params, p_params + 8, rec.params);
        }
        finish_record(rec);
        const VoxelBox region = record_box(rec);
        records.push_back(rec);
        edit_k = rec.k;
        edit_margin_c = rec.margin_c;
        rebuild(region);
        return int(records.size() - 1);
    }

    bool remove_instance(int p_id) override {
        if (p_id < 0 || p_id >= int(records.size()) || records[p_id].removed) {
            return false;
        }
        const VoxelBox region = record_box(records[p_id]);
        records[p_id].removed = true;
        edit_k = records[p_id].k;
        edit_margin_c = records[p_id].margin_c;
        rebuild(region);
        return true;
    }

    bool move_instance(int p_id, const VoxelPos &p_anchor, const float *p_rotation) override {
        if (p_id < 0 || p_id >= int(records.size()) || records[p_id].removed) {
            return false;
        }
        Archetype::TableRecord &rec = records[p_id];
        const VoxelBox old_box = record_box(rec);
        rec.anchor = p_anchor;
        if (p_rotation) {
            std::copy(p_rotation, p_rotation + 9, rec.rotation);
        }
        finish_record(rec);
        const VoxelBox new_box = record_box(rec);
        VoxelBox region;
        region.valid = true;
        for (int a = 0; a < 3; a++) {
            region.lo[a] = MIN(old_box.lo[a], new_box.lo[a]);
            region.hi[a] = MAX(old_box.hi[a], new_box.hi[a]);
        }
        edit_k = rec.k;
        edit_margin_c = rec.margin_c;
        rebuild(region);
        return true;
    }

    // Whether a chunk's samples can be changed by the last edit (its box grown by M(lod) meets the
    // edit's region): the chunks to re-mesh.
    bool edit_affects_chunk(const ChunkId &p_chunk) const override {
        if (!edit_box.valid) {
            return false;
        }
        const int lod = ChunkMath::get_lod(p_chunk);
        const int64_t span = int64_t(param.chunk_voxel_dimension()) << lod;
        const VoxelPos min = ChunkMath::chunk_min_voxel(p_chunk, param);
        const float m = edit_k + edit_margin_c * ExactNoise::exact_pow2(lod);
        const int64_t grow = int64_t(std::ceil(m)) + 2;
        VoxelBox box;
        box.valid = true;
        for (int a = 0; a < 3; a++) {
            box.lo[a] = min[a] - grow;
            box.hi[a] = min[a] + span + grow;
        }
        return box.intersects(edit_box);
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
    std::shared_ptr<const Archetype::DefLibrary> library;
    Archetype::Program ground_program;
    float out_scale = 1.0f;
    int lod_count = 24;
    uint32_t version = 0;
    std::vector<Archetype::TableRecord> records; // editable: the snapshot is rebuilt from these
    std::shared_ptr<const Archetype::InstanceSnapshot> snapshot;
    Archetype::GpuTable gpu;
    float edit_k = 0.0f; // the changed records' largest margin inputs (edit_affects_chunk)
    float edit_margin_c = 0.0f;
    int scatter_count = 0; // --instances
    double scatter_area = 200.0; // --instance_area (metres)

    void rebuild(const VoxelBox &p_region) {
        version++;
        snapshot = Archetype::build_snapshot(library, ground_program, out_scale, true, records, param, lod_count - 1, version);
        gpu.set_snapshot(snapshot);
        edit_box = p_region;
    }

    void finish_record(Archetype::TableRecord &r_rec) const {
        const Archetype::Bounds wb = Archetype::record_world_bounds((*library)[r_rec.def], r_rec);
        for (int a = 0; a < 3; a++) {
            r_rec.box_lo[a] = int32_t(wb.lo[a] - r_rec.anchor[a]);
            r_rec.box_hi[a] = int32_t(wb.hi[a] - r_rec.anchor[a]);
        }
        r_rec.lipschitz = wb.lipschitz;
        r_rec.margin_c = 2.0f * wb.lipschitz + param.skirt_depth;
    }

    VoxelBox record_box(const Archetype::TableRecord &p_rec) const {
        VoxelBox b;
        b.valid = true;
        for (int a = 0; a < 3; a++) {
            b.lo[a] = p_rec.anchor[a] + p_rec.box_lo[a];
            b.hi[a] = p_rec.anchor[a] + p_rec.box_hi[a];
        }
        return b;
    }

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
        DefLibrary lib;
        const int sphere = lib.add(make_sphere_type());
        const int capsule = lib.add(make_capsule_type());
        const int rock = lib.add(make_rock_type(int(std::lround(std::log2(8.0 / p_voxel))), ROCK_SEED));
        library = std::make_shared<const DefLibrary>(lib);

        records.clear();
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
            finish_record(rec);
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
        // The scatter: deterministic rocks and pillars on the ground, around the demo objects.
        const double centre_x = -50.0, centre_z = -150.0;
        for (int i = 0; i < scatter_count; i++) {
            const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ uint32_t(i), 0x5CA7u, 0x7E12u });
            const double px = centre_x + (double(h.x >> 8) / 16777216.0 - 0.5) * scatter_area;
            const double pz = centre_z + (double(h.y >> 8) / 16777216.0 - 0.5) * scatter_area;
            const float r = 1.0f + float(h.z % 300u) * 0.01f;
            if (i % 2 == 0) {
                const float params[8] = { v(double(r)), 0.1f * v(double(r)), 0, 0, 0, 0, 0, 0 };
                add(Combine::SMOOTH_UNION, 1.0, rock, params, px, pz, -0.4 * double(r), double(h.x % 628u) * 0.01);
            } else {
                const float params[8] = { v(2.0 + double(r)), v(0.5), 0, 0, 0, 0, 0, 0 };
                add(Combine::SMOOTH_UNION, 0.8, capsule, params, px, pz, 0.0, 0.0);
            }
        }
        // The ground register: the ground Def placed at the datum (a compiled program, constants only).
        Archetype::Instance ground_inst;
        ground_inst.def = ground_def(height);
        ground_inst.anchor = VoxelPos(0, datum, 0);
        ground_program = compile(ground_inst).program;
        out_scale = float(p_voxel);
        lod_count = p_lod_count;
        edit_k = 0.0f;
        edit_margin_c = 0.0f;
        // The GPU buffers hold the scatter plus room for runtime edits (the mesher reads this at init).
        gpu.set_capacity(MAX<uint32_t>(16384u, uint32_t(records.size()) + 4096u));
        rebuild(VoxelBox());
    }
};
