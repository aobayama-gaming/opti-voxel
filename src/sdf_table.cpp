#include "sdf_table.hpp"

#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/classes/worker_thread_pool.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>

using namespace Archetype;

int SDFTable::add_instance(int p_type, const VoxelPos &p_anchor, const float *p_rotation, const float *p_params, int p_op, float p_k) {
    if (!library || p_type < 0 || p_type >= library->size()) {
        return -1;
    }
    const String error = edit_error(p_rotation, p_params, p_op, p_k);
    ERR_FAIL_COND_V_MSG(!error.is_empty(), -1, "add_instance: " + error);
    ERR_FAIL_COND_V_MSG(records.size() >= gpu.instance_capacity(), -1,
            vformat("add_instance: the table is full (%d instances, removed ones included).", gpu.instance_capacity()));
    TableRecord rec;
    rec.def = uint8_t(p_type);
    rec.op = Combine(p_op);
    rec.set_k(p_k);
    rec.anchor = p_anchor;
    if (p_rotation) {
        std::copy(p_rotation, p_rotation + 9, rec.rotation);
    }
    if (p_params) {
        std::copy(p_params, p_params + 8, rec.params);
    }
    const String fit = record_fit_error((*library)[rec.def], rec, max_instance_extent());
    ERR_FAIL_COND_V_MSG(!fit.is_empty(), -1, "add_instance: " + fit);
    finish_record(rec);
    records.push_back(rec, version + 1);
    finish_edit(rec, rec, record_box(rec), records.size() - 1);
    return int(records.size() - 1);
}

bool SDFTable::remove_instance(int p_id) {
    if (p_id < 0 || p_id >= int(records.size()) || records[p_id].removed) {
        return false;
    }
    TableRecord rec = records[p_id];
    const VoxelBox region = record_box(rec);
    rec.removed = true;
    records.set(p_id, rec, version + 1);
    finish_edit(rec, rec, region, uint32_t(p_id));
    return true;
}

bool SDFTable::move_instance(int p_id, const VoxelPos &p_anchor, const float *p_rotation) {
    if (p_id < 0 || p_id >= int(records.size()) || records[p_id].removed) {
        return false;
    }
    const String error = edit_error(p_rotation, nullptr, 0, 0.0f);
    ERR_FAIL_COND_V_MSG(!error.is_empty(), false, "move_instance: " + error);
    TableRecord rec = records[p_id];
    const TableRecord before = rec;
    const VoxelBox old_box = record_box(rec);
    rec.anchor = p_anchor;
    if (p_rotation) {
        std::copy(p_rotation, p_rotation + 9, rec.rotation);
    }
    const String fit = record_fit_error((*library)[rec.def], rec, max_instance_extent());
    ERR_FAIL_COND_V_MSG(!fit.is_empty(), false, "move_instance: " + fit);
    finish_record(rec);
    records.set(p_id, rec, version + 1);
    const VoxelBox new_box = record_box(rec);
    VoxelBox region;
    region.valid = true;
    for (int a = 0; a < 3; a++) {
        region.lo[a] = MIN(old_box.lo[a], new_box.lo[a]);
        region.hi[a] = MAX(old_box.hi[a], new_box.hi[a]);
    }
    finish_edit(before, rec, region, uint32_t(p_id));
    return true;
}

bool SDFTable::edit_affects_chunk(const ChunkId &p_chunk) const {
    if (!edit_box.valid) {
        return false;
    }
    // The band rule: the changed instances are listed below edit_lod_max only, so a small dig doesn't re-mesh the
    // coarse chunks around it.
    const int lod = ChunkMath::get_lod(p_chunk);
    if (lod >= edit_lod_max) {
        return false;
    }
    // The chunk's samples (span + skirt overhang) against the edit region grown by M(lod).
    TableRecord grown;
    grown.k = edit_k;
    grown.margin_c = edit_margin_c;
    grown.anchor = edit_box.lo;
    for (int a = 0; a < 3; a++) {
        grown.box_hi[a] = int32_t(MIN<int64_t>(edit_box.hi[a] - edit_box.lo[a], INT32_MAX));
    }
    VoxelPos lo, hi;
    chunk_sample_box(p_chunk, param, lo, hi);
    return influence_meets(grown, lod, lo, hi);
}

String SDFTable::edit_error(const float *p_rotation, const float *p_params, int p_op, float p_k) {
    auto bad = [](float p_v) { return !std::isfinite(p_v) || (p_v != 0.0f && std::fabs(p_v) < FLT_MIN); };
    if (p_op < int(Combine::UNION) || p_op > int(Combine::SMOOTH_SUBTRACT)) {
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

void SDFTable::publish(const std::shared_ptr<const InstanceSnapshot> &p_snapshot, const VoxelBox &p_region) {
    std::atomic_store(&snapshot, p_snapshot);
    gpu.set_snapshot(p_snapshot);
    edit_box = p_region;
}

int64_t SDFTable::max_instance_extent() const {
    const int64_t top_span = int64_t(param.chunk_voxel_dimension()) << (lod_count - 1);
    return MIN(4 * top_span, int64_t(1) << 29);
}

void SDFTable::finish_record(TableRecord &r_rec) const {
    Archetype::finish_record((*library)[r_rec.def], r_rec, param.skirt_depth, instance_band, erode_rule);
}

void SDFTable::finish_edit(const TableRecord &p_before, const TableRecord &p_after, const VoxelBox &p_region, uint32_t p_id) {
    edit_k = p_after.k;
    edit_margin_c = p_after.margin_c;
    // The LODs the edit can change: below the larger of the record's max_lod before and after it.
    edit_lod_max = MAX(p_before.max_lod, p_after.max_lod);
    version++;
    publish(edit_snapshot(*current(), records, { p_id }, version), p_region);
}

VoxelBox SDFTable::record_box(const TableRecord &p_rec) {
    VoxelBox b;
    b.valid = true;
    for (int a = 0; a < 3; a++) {
        b.lo[a] = p_rec.anchor[a] + p_rec.box_lo[a];
        b.hi[a] = p_rec.anchor[a] + p_rec.box_hi[a];
    }
    return b;
}

void SDFTable::add_forest(const TreeTypes &p_types, double p_voxel, double p_centre_x, double p_centre_z, Dictionary &r_stats) {
    std::vector<TreeSite> sites;
    forest_sites(forest_count, forest_area, p_centre_x, p_centre_z, p_voxel, [this](int64_t p_x, int64_t p_z) { return ground_y(p_x, p_z); }, sites);
    std::vector<TableRecord> parts;
    int64_t most = 0;
    for (const TreeSite &site : sites) {
        if (in_lake(site.base.x, site.base.z)) {
            continue;
        }
        parts.clear();
        grow_tree(tree_species()[site.species], site.base, site.seed, p_voxel, p_types, parts);
        const uint32_t first = records.size();
        if (admit(parts, "tree")) {
            trees.push_back(TreeSpan{ first, uint32_t(parts.size()), site.species, site.seed });
            most = MAX(most, int64_t(parts.size()));
        }
    }
    Dictionary forest;
    forest["count"] = int64_t(trees.size());
    int64_t total = 0;
    for (const TreeSpan &t : trees) {
        total += t.count;
    }
    forest["records"] = total;
    forest["records_per_tree_avg"] = trees.empty() ? 0.0 : double(total) / double(trees.size());
    forest["records_per_tree_max"] = most;
    r_stats["trees"] = forest;
}

void SDFTable::add_biomes(const Types &p_types, Dictionary &r_stats) {
    if (!(biome_radius > 0.0)) {
        return;
    }
    const uint64_t start_us = Time::get_singleton()->get_ticks_usec();
    const double vs = param.voxel_size;
    const int64_t cx = std::llround(SCENE_X / vs), cz = std::llround(SCENE_Z / vs);
    const int64_t r = std::llround(biome_radius / vs);
    std::vector<Biome::Site> all, sites;
    for (int f = 0; f < Biome::FEATURE_COUNT; f++) {
        Biome::sites(Biome::Feature(f), cx - r, cz - r, cx + r, cz + r, vs, all);
    }
    for (const Biome::Site &site : all) {
        const double dx = double(site.x - cx), dz = double(site.z - cz);
        if (dx * dx + dz * dz <= double(r) * double(r)) {
            sites.push_back(site);
        }
    }
    // Each site samples the ground at a few dozen columns (ground_y, const): on the worker pool, kept in site order.
    struct Job {
        const SDFTable *table;
        const Biome::Types *types;
        const Biome::Site *sites;
        Biome::Placed *placed;
        uint8_t *ok;
        double voxel;
        static void run(void *p_job, uint32_t p_i) {
            const Job &job = *static_cast<const Job *>(p_job);
            const SDFTable *t = job.table;
            job.ok[p_i] = Biome::place(job.sites[p_i], [t](int64_t p_x, int64_t p_z) { return t->ground_y(p_x, p_z); }, *t->library, *job.types, job.voxel, job.placed[p_i]);
        }
    };
    std::vector<Biome::Placed> placed(sites.size());
    std::vector<uint8_t> ok(sites.size(), 0);
    Job job{ this, &p_types.biome, sites.data(), placed.data(), ok.data(), vs };
    if (!sites.empty()) {
        WorkerThreadPool *pool = WorkerThreadPool::get_singleton();
        pool->wait_for_group_task_completion(pool->add_native_group_task(&Job::run, &job, int(sites.size()), -1, true, "SDFTable::add_biomes"));
    }
    Array listed;
    int64_t counts[Biome::FEATURE_COUNT] = {};
    int64_t too_steep = 0;
    for (size_t i = 0; i < sites.size(); i++) {
        if (!ok[i]) {
            too_steep++;
            continue;
        }
        if (!admit(placed[i].records, "biome feature")) {
            continue;
        }
        const Biome::Placed &p = placed[i];
        if (p.lava) {
            lava_pools.push_back(p.pool);
        }
        counts[p.at.feature]++;
        if (listed.size() >= SITES_LISTED_MAX) {
            continue;
        }
        Dictionary d;
        d["kind"] = Biome::kind(p.at.feature).name;
        d["x_m"] = double(p.at.x) * vs;
        d["z_m"] = double(p.at.z) * vs;
        d["y_m"] = double(p.top_y) * vs; // the lava surface, the island's rim plane
        d["radius_m"] = p.at.radius * vs;
        listed.push_back(d);
    }
    Dictionary biomes;
    for (int f = 0; f < Biome::FEATURE_COUNT; f++) {
        biomes[Biome::kind(Biome::Feature(f)).name] = counts[f];
    }
    biomes["lakes_too_steep"] = too_steep;
    biomes["build_ms"] = double(Time::get_singleton()->get_ticks_usec() - start_us) * 1e-3;
    biomes["sites"] = listed;
    r_stats["biomes"] = biomes;
}

bool SDFTable::in_lake(int64_t p_x, int64_t p_z) const {
    for (const Biome::LavaPool &p : lava_pools) {
        const double dx = double(p_x - p.top.x), dz = double(p_z - p.top.z), keep = Biome::LAKE_KEEP_OUT * p.radius;
        if (dx * dx + dz * dz < keep * keep) {
            return true;
        }
    }
    return false;
}

bool SDFTable::admit(std::vector<TableRecord> &r_records, const char *p_what) {
    for (const TableRecord &rec : r_records) {
        const String fit = record_fit_error((*library)[rec.def], rec, max_instance_extent());
        ERR_FAIL_COND_V_MSG(!fit.is_empty(), false, vformat("SDFTable: %s left out: ", p_what) + fit);
    }
    for (TableRecord &rec : r_records) {
        finish_record(rec);
        records.push_back(rec, version + 1);
    }
    return true;
}

float SDFTable::voxels(double p_metres) const {
    return float(p_metres / param.voxel_size);
}

void SDFTable::place(Combine p_op, double p_k_m, int p_type, const float *p_params, double p_x, double p_z, double p_dy_m, double p_yaw) {
    const double vs = param.voxel_size;
    TableRecord rec;
    rec.def = uint8_t(p_type);
    rec.op = p_op;
    rec.set_k(voxels(p_k_m));
    std::copy(p_params, p_params + 8, rec.params);
    const int64_t x = std::llround(p_x / vs), z = std::llround(p_z / vs);
    rec.anchor = VoxelPos(x, ground_y(x, z) + std::llround(p_dy_m / vs), z);
    const float c = float(std::cos(p_yaw)), s = float(std::sin(p_yaw));
    const float rot[9] = { c, 0, -s, 0, 1, 0, s, 0, c };
    std::copy(rot, rot + 9, rec.rotation);
    rec.salt = ExactNoise::pcg3d(ExactNoise::UVec3{ records.size() + 1, 0x5EEDu, 0u }).x;
    // The same limits as a runtime edit (a gallery cube past 4 top-LOD chunk spans with few LODs).
    std::vector<TableRecord> one{ rec };
    admit(one, "scene record");
}

void SDFTable::add_shapes(const Types &p_types) {
    auto v = [this](double p_metres) { return voxels(p_metres); };
    // The cube ladder: centres at 0.9 of the half size, a little sunk, so the voxel rounding never lifts one off the
    // ground.
    for (int i = 0; i < 8; i++) {
        const double d = 80.0 * double(1 << i);
        const double h = 0.1 * d;
        const double theta = (-45.5 + 13.0 * i) * Math_PI / 180.0;
        const float params[8] = { v(h), v(h), v(h), 0, v(1.2 * h), 0, 0, 0 };
        place(Combine::UNION, 0.0, p_types.holed_cube, params, SHAPES_X + d * std::sin(theta), SHAPES_Z - d * std::cos(theta), 0.9 * h, 0.5 - theta);
    }
    const double h = 6.0;
    const float die[8] = { v(h), 0, 0, 0, 0, 0, 0, 0 };
    place(Combine::UNION, 0.0, p_types.die, die, SHAPES_X - 25.0, SHAPES_Z + 40.0, 0.95 * h, 0.4);
    // Eight 1 m steps, 2 m deep, 8 m wide, rising away from the spawn.
    const float stairs[8] = { v(4.0), v(1.0), v(1.0), 0, 0, 0, 0, 0 };
    place(Combine::UNION, 0.0, p_types.stairs, stairs, SHAPES_X, SHAPES_Z + 33.0, 0.0, 0.3);
    // 5 m square tunnels along x, y and z.
    const float tunnels[8] = { v(h), v(2.5), 0, 0, 0, 0, 0, 0 };
    place(Combine::UNION, 0.0, p_types.tunnel_cube, tunnels, SHAPES_X + 25.0, SHAPES_Z + 40.0, 0.9 * h, 0.6);
}

void SDFTable::add_scatter(const Types &p_types) {
    auto v = [this](double p_metres) { return voxels(p_metres); };
    for (int i = 0; i < scatter_count; i++) {
        const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ uint32_t(i), 0x5CA7u, 0x7E12u });
        const double px = SCENE_X + (ExactNoise::unit_of(h.x) - 0.5) * scatter_area;
        const double pz = SCENE_Z + (ExactNoise::unit_of(h.y) - 0.5) * scatter_area;
        if (in_lake(std::llround(px / param.voxel_size), std::llround(pz / param.voxel_size))) {
            continue;
        }
        const float r = 1.0f + float(h.z % 300u) * 0.01f;
        if (i % 2 == 0) {
            const float params[8] = { v(double(r)), 0.1f * v(double(r)), 0, 0, 0, 0, 0, 0 };
            place(Combine::SMOOTH_UNION, 1.0, p_types.rock, params, px, pz, -0.4 * double(r), double(h.x % 628u) * 0.01);
        } else {
            const float params[8] = { v(2.0 + double(r)), v(0.5), 0, 0, 0, 0, 0, 0 };
            place(Combine::SMOOTH_UNION, 0.8, p_types.capsule, params, px, pz, 0.0, 0.0);
        }
    }
}

void SDFTable::build_table(int p_lod_count) {
    const double vs = param.voxel_size;
    // Appended in this order: the GDScript side names the indices (CAPSULE_TYPE). The gallery's types only with it.
    DefLibrary lib;
    Types types;
    lib.add(make_sphere_type());
    types.capsule = lib.add(make_capsule_type());
    types.rock = lib.add(make_rock_type(ExactNoise::lattice_k(8.0, vs), ROCK_SEED));
    const int branch = lib.add(make_branch_type());
    const int crown = lib.add(make_rock_type(crown_k(vs), CROWN_SEED));
    const int cone = lib.add(make_cone_type(crown_k(vs), CROWN_SEED));
    if (scene == Scene::SHAPES) {
        types.holed_cube = lib.add(make_swiss_cube_type(1));
        types.die = lib.add(make_die_type());
        types.stairs = lib.add(make_stairs_type());
        types.tunnel_cube = lib.add(make_tunnel_cube_type());
    }
    if (biome_radius > 0.0) {
        types.biome = Biome::add_types(lib, vs);
    }
    library = std::make_shared<const DefLibrary>(lib);
    for (int i = 0; i < library->size(); i++) {
        const String error = (*library)[i].program.validate(p_lod_count);
        if (!error.is_empty()) {
            ERR_PRINT(vformat("Archetype library type %d: ", i) + error);
        }
    }
    lod_count = p_lod_count;

    records.clear();
    trees.clear();
    lava_pools.clear();
    if (scene == Scene::SHAPES) {
        add_shapes(types);
    }
    // The biome features first: the scatter and the forest keep out of the lakes (their basins cut the ground away).
    Dictionary scene_stats;
    add_biomes(types, scene_stats);
    add_scatter(types);
    add_forest(crown_types(branch, crown, lib[crown], cone), vs, SCENE_X, SCENE_Z, scene_stats);
    gpu.set_scene_stats(scene_stats);
    edit_k = 0.0f;
    edit_margin_c = 0.0f;
    edit_lod_max = 0;
    // The GPU buffers hold the scene plus room for runtime edits (the mesher reads this at init).
    gpu.set_capacity(MAX<uint32_t>(16384u, uint32_t(records.size()) + 4096u));
    version++;
    publish(build_snapshot(library, float(vs), true, records, param, lod_count - 1, version), VoxelBox());
}
