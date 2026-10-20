#include "sdf_table.hpp"

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
    rec.k = p_k;
    rec.inv_k = p_k > 0.0f ? float(1.0 / double(p_k)) : 0.0f;
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

void SDFTable::add_forest(const TreeTypes &p_types, double p_voxel, double p_centre_x, double p_centre_z) {
    std::vector<TreeSite> sites;
    forest_sites(forest_count, forest_area, p_centre_x, p_centre_z, p_voxel, [this](int64_t p_x, int64_t p_z) { return ground_y(p_x, p_z); }, sites);
    std::vector<TableRecord> parts;
    int64_t most = 0;
    for (const TreeSite &site : sites) {
        parts.clear();
        grow_tree(tree_species()[site.species], site.base, site.seed, p_voxel, p_types, parts);
        trees.push_back(TreeSpan{ records.size(), uint32_t(parts.size()), site.species, site.seed });
        for (TableRecord &rec : parts) {
            finish_record(rec);
            records.push_back(rec, version + 1);
        }
        most = MAX(most, int64_t(parts.size()));
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
    Dictionary scene;
    scene["trees"] = forest;
    gpu.set_scene_stats(scene);
}

void SDFTable::build_table(const Program &p_ground, double p_voxel, int p_lod_count, Scene p_scene) {
    auto v = [p_voxel](double p_metres) { return float(p_metres / p_voxel); };
    auto ground = [&](double p_x, double p_z) { return ground_y(std::llround(p_x / p_voxel), std::llround(p_z / p_voxel)); };
    // Appended in this order: the GDScript side names the indices (CAPSULE_TYPE).
    DefLibrary lib;
    const int sphere = lib.add(make_sphere_type());
    const int capsule = lib.add(make_capsule_type());
    const int rock = lib.add(make_rock_type(int(std::lround(std::log2(8.0 / p_voxel))), ROCK_SEED));
    const int branch = lib.add(make_branch_type());
    const int crown = lib.add(make_rock_type(crown_k(p_voxel), CROWN_SEED));
    const int cone = lib.add(make_cone_type(crown_k(p_voxel), CROWN_SEED));
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
        rec.anchor = VoxelPos(std::llround(p_x / p_voxel), ground(p_x, p_z) + std::llround(p_dy_m / p_voxel), std::llround(p_z / p_voxel));
        const float c = float(std::cos(p_yaw)), s = float(std::sin(p_yaw));
        const float rot[9] = { c, 0, -s, 0, 1, 0, s, 0, c };
        std::copy(rot, rot + 9, rec.rotation);
        const uint32_t salt_seed = seed++;
        rec.salt = salt_seed == 0 ? 0u : ExactNoise::pcg3d(ExactNoise::UVec3{ salt_seed, 0x5EEDu, 0u }).x;
        finish_record(rec);
        records.push_back(rec, version + 1);
    };
    if (p_scene == Scene::DEMO) {
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
        const float params[8] = { v(7.0), 0, 0, 0, 0, 0, 0, 0 };
        add(Combine::SMOOTH_SUBTRACT, 1.0, sphere, params, -10.0, -170.0, -3.0, 0.0);
    }
    if (p_scene != Scene::EMPTY) {
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
        add_forest(crown_types(branch, crown, lib[crown], cone), p_voxel, centre_x, centre_z);
    }
    edit_k = 0.0f;
    edit_margin_c = 0.0f;
    edit_lod_max = 0;
    // The GPU buffers hold the scene plus room for runtime edits (the mesher reads this at init).
    gpu.set_capacity(MAX<uint32_t>(16384u, uint32_t(records.size()) + 4096u));
    version++;
    publish(build_snapshot(library, p_ground, float(p_voxel), true, records, param, lod_count - 1, version), VoxelBox());
}
