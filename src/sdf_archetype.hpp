#pragma once

#include "archetype_gpu_table.hpp"
#include "archetype_table.hpp"
#include "instance_table.hpp"
#include "sdf_base.hpp"

#include <atomic>
#include <cfloat>
#include <cmath>
#include <memory>

// Demo terrain of the archetype table (terrain_mode 3, --terrain=archetypes): an fBm ground (a compiled
// Program) plus a table of instances (rocks, tree trunks and a cave mouth) in front of the main_scene
// spawn. Each chunk folds only the instances near it (instance_table.hpp); the library
// (archetype_table.hpp) is compiled once and reads the instances' records from a GPU buffer. A function
// of the absolute position: far origins see the ground only.
// Scene::SHAPES (terrain_mode 4, --terrain=shapes) has no table instances: shapes() compiled into the
// ground program (hard CSG for dual contouring, nested composites the table does not express).
class SDFArchetype : public SDFBase {
public:
    enum class Scene { DEMO,
        SHAPES };

    explicit SDFArchetype(Scene p_scene = Scene::DEMO) :
            which(p_scene) {}

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

    // A deterministic scatter of p_count rocks and pillars in a square of side p_area (metres) around
    // the demo objects (--instances / --instance_area). Set before configure().
    void set_scatter(int p_count, double p_area) {
        scatter_count = p_count;
        scatter_area = p_area;
    }

    void configure(const ChunkParam &p_param, int p_lod_count) override {
        SDFBase::configure(p_param, p_lod_count);
        create_scene(p_param.voxel_size, p_lod_count);
        const String ground_error = current()->ground.validate(p_lod_count);
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
        const std::shared_ptr<const Archetype::InstanceSnapshot> snap = current();
        snap->point_candidates(p_pos.voxel, p_lod, list);
        const Archetype::Float4 r = Archetype::fold(snap->ground, *snap->library, snap->records.data(), list.data(), uint32_t(list.size()), p_pos, p_lod);
        return Archetype::finish_table(r, p_lod, snap->out_scale, snap->normalize).w;
    }

    void evaluate_batch(const VoxelPos *p_points, int p_count, int p_lod, float *r_out) const override {
        evaluate_batch_parallel(p_points, p_count, p_lod, r_out);
    }

    String get_glsl_code() const override {
        const std::shared_ptr<const Archetype::InstanceSnapshot> snap = current();
        return Archetype::generate_table_sdf(snap->ground, *snap->library, snap->out_scale, snap->normalize);
    }

    SDFGpuTable *get_gpu_table() override { return &gpu; }

    bool cull_exempt(const ChunkId &p_node, int p_lod) const override {
        // The node's world box grown by the cull band (2 chunk sizes at its LOD), against every instance's
        // influence box. O(records) with a short circuit: fine for the demo's small tables.
        const std::shared_ptr<const Archetype::InstanceSnapshot> snap = current();
        if (!snap) {
            return false;
        }
        const VoxelPos min = ChunkMath::chunk_min_voxel(p_node, param);
        const int64_t span = int64_t(param.chunk_voxel_dimension()) << p_lod;
        const int64_t grow = 2 * span;
        const VoxelPos lo = min - VoxelPos(grow, grow, grow);
        const VoxelPos hi = min + VoxelPos(span + grow, span + grow, span + grow);
        for (const Archetype::TableRecord &rec : snap->records) {
            if (rec.removed) {
                continue;
            }
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
        const String error = edit_error(p_rotation, p_params, p_op, p_k);
        ERR_FAIL_COND_V_MSG(!error.is_empty(), -1, "add_instance: " + error);
        ERR_FAIL_COND_V_MSG(records.size() >= gpu.instance_capacity(), -1,
                vformat("add_instance: the table is full (%d instances, removed ones included).", gpu.instance_capacity()));
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
        const String error = edit_error(p_rotation, nullptr, 0, 0.0f);
        ERR_FAIL_COND_V_MSG(!error.is_empty(), false, "move_instance: " + error);
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
        // The chunk's samples (span + skirt overhang) against the edit region grown by M(lod).
        Archetype::TableRecord grown;
        grown.k = edit_k;
        grown.margin_c = edit_margin_c;
        grown.anchor = edit_box.lo;
        for (int a = 0; a < 3; a++) {
            grown.box_hi[a] = int32_t(MIN<int64_t>(edit_box.hi[a] - edit_box.lo[a], INT32_MAX));
        }
        VoxelPos lo, hi;
        Archetype::chunk_sample_box(p_chunk, param, lo, hi);
        return Archetype::influence_meets(grown, ChunkMath::get_lod(p_chunk), lo, hi);
    }

    // The current snapshot. Edits replace it on the main thread while the discoverer thread evaluates and
    // the render thread uploads: every reader takes one atomic copy and works on it.
    std::shared_ptr<const Archetype::InstanceSnapshot> current() const { return std::atomic_load(&snapshot); }
    std::shared_ptr<const Archetype::InstanceSnapshot> get_snapshot() const { return current(); }
    const Archetype::DefLibrary &get_library() const { return *library; }
    const Archetype::Program &get_ground() const { return ground_program; }

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
    Scene which;

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

    void rebuild(const VoxelBox &p_region) {
        version++;
        const std::shared_ptr<const Archetype::InstanceSnapshot> snap =
                Archetype::build_snapshot(library, ground_program, out_scale, true, records, param, lod_count - 1, version);
        std::atomic_store(&snapshot, snap);
        gpu.set_snapshot(snap);
        edit_box = p_region;
    }

    // Why an edit's inputs are refused (empty if they are fine): the GPU and the CPU only agree on finite,
    // normal floats, the box and the fold assume an orthonormal rotation, the table folds the four top-level
    // operators, and a smooth radius is >= 0.
    static String edit_error(const float *p_rotation, const float *p_params, int p_op, float p_k) {
        auto bad = [](float p_v) { return !std::isfinite(p_v) || (p_v != 0.0f && std::fabs(p_v) < FLT_MIN); };
        if (p_op < int(Archetype::Combine::UNION) || p_op > int(Archetype::Combine::SMOOTH_SUBTRACT)) {
            return vformat("op %d is not 0 union, 1 smooth union, 2 subtract or 3 smooth subtract.", p_op);
        }
        if (bad(p_k) || p_k < 0.0f) {
            return "k must be finite and >= 0.";
        }
        for (int i = 0; p_params && i < 8; i++) {
            if (bad(p_params[i])) {
                return vformat("param %d is not a finite normal float.", i);
            }
        }
        if (p_rotation) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    double d = 0.0;
                    for (int c = 0; c < 3; c++) {
                        d += double(p_rotation[3 * i + c]) * double(p_rotation[3 * j + c]);
                    }
                    if (bad(p_rotation[3 * i + j]) || std::abs(d - (i == j ? 1.0 : 0.0)) > 1e-5) {
                        return "the rotation is not orthonormal (rows within 1e-5).";
                    }
                }
            }
        }
        return String();
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
        const ExactNoise::OctaveSet height = height_fbm(p_voxel);
        const int64_t datum = std::llround(DATUM_M / p_voxel);
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
        if (which == Scene::SHAPES) {
            // The shapes gallery is the ground program itself; the table starts empty (edits still work).
            records.clear();
            ground_inst = shapes(p_voxel);
        }
        const Archetype::Compiled ground_compiled = compile(ground_inst);
        if (!ground_compiled.error.is_empty()) {
            ERR_PRINT(ground_compiled.error);
        }
        ground_program = ground_compiled.program;
        out_scale = float(p_voxel);
        lod_count = p_lod_count;
        edit_k = 0.0f;
        edit_margin_c = 0.0f;
        // The GPU buffers hold the scatter plus room for runtime edits (the mesher reads this at init).
        gpu.set_capacity(MAX<uint32_t>(16384u, uint32_t(records.size()) + 4096u));
        rebuild(VoxelBox());
    }
};
