#include "sdf_table.hpp"

#include "godot_conv.hpp"
#include "moment_layers.hpp"
#include "parallel_for.hpp"

#include "ov/executor.hpp"
#include <godot_cpp/classes/time.hpp>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <tuple>

using namespace Archetype;

int SDFTable::add_instance(int p_type, const VoxelPos &p_anchor, const float *p_rotation, const float *p_params, int p_op, float p_k) {
    if (!library || p_type < 0 || p_type >= library->size()) {
        return -1;
    }
    const String error = edit_error(p_rotation, p_params, p_op, p_k);
    ERR_FAIL_COND_V_MSG(!error.is_empty(), -1, "add_instance: " + error);
    ERR_FAIL_COND_V_MSG(records.size() >= gpu.table_capacity(), -1,
            vformat("add_instance: the table is full (%d instances, removed ones included).", gpu.table_capacity()));
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
    ERR_FAIL_COND_V_MSG(records[p_id].op == Combine::ACCUM, false, "remove_instance: a moment blob is the table's, not an instance.");
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
    ERR_FAIL_COND_V_MSG(records[p_id].op == Combine::ACCUM, false, "move_instance: a moment blob is the table's, not an instance.");
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
    finish_edit(before, rec, old_box.merged(record_box(rec)), uint32_t(p_id));
    return true;
}

bool SDFTable::edit_affects_chunk(const ChunkId &p_chunk) const {
    if (!edit_box.valid) {
        return false;
    }
    const int lod = ChunkMath::get_lod(p_chunk);
    VoxelPos lo, hi;
    chunk_sample_box(p_chunk, param, lo, hi);
    for (const EditRegion &r : edit_regions) {
        // The LODs it is listed at only (the band rule: a small dig doesn't re-mesh the coarse chunks around it; a blob:
        // its layer's alone), its region grown by M(lod) against the chunk's samples (span + skirt overhang).
        if (lod < r.lod_min || lod >= r.lod_max) {
            continue;
        }
        TableRecord grown;
        grown.k = r.k;
        grown.margin_c = r.margin_c;
        grown.anchor = r.box.lo;
        for (int a = 0; a < 3; a++) {
            grown.box_hi[a] = int32_t(MIN<int64_t>(r.box.hi[a] - r.box.lo[a], INT32_MAX));
        }
        if (influence_meets(grown, lod, lo, hi)) {
            return true;
        }
    }
    return false;
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
    swap_snapshot(p_snapshot);
    edit_box = p_region;
}

void SDFTable::swap_snapshot(const std::shared_ptr<const InstanceSnapshot> &p_snapshot) {
    std::atomic_store(&snapshot, p_snapshot);
    gpu.set_snapshot(p_snapshot);
}

int64_t SDFTable::max_instance_extent() const {
    const int64_t top_span = int64_t(param.chunk_voxel_dimension()) << (lod_count - 1);
    return MIN(4 * top_span, int64_t(1) << 29);
}

void SDFTable::finish_record(TableRecord &r_rec) const {
    finish_table_record(*library, record_finish(), r_rec);
}

void SDFTable::finish_edit(const TableRecord &p_before, const TableRecord &p_after, const VoxelBox &p_region, uint32_t p_id) {
    // The LODs the edit can change: below the larger of the record's max_lod before and after it; then the moment
    // blobs it changes, each at its layer's LOD.
    edit_regions.assign(1, EditRegion{ p_region, MAX(p_before.k, p_after.k), MAX(p_before.margin_c, p_after.margin_c), 0, MAX(p_before.max_lod, p_after.max_lod) });
    std::vector<uint32_t> ids{ p_id };
    update_moment_blobs(p_id, ids);
    VoxelBox all = p_region;
    for (const EditRegion &r : edit_regions) {
        all = all.merged(r.box);
    }
    version++;
    publish(rebuild.edit(*current(), records, ids, version), all);
}

bool SDFTable::cull_exempt(const ChunkId &p_node, int p_lod, float p_past_band_m) const {
    const std::shared_ptr<const InstanceSnapshot> snap = current();
    if (snap && snap->near_instance(p_node, p_lod)) {
        return true;
    }
    // The corners are the bare ground's distance: the node holds a point within its band of the ground, so within the
    // band plus the reach of a source's records when it can hold one (the corner test's own slope bound).
    const int64_t span = int64_t(param.chunk_voxel_dimension()) << p_lod;
    const VoxelPos lo = ChunkMath::chunk_min_voxel(p_node, param);
    const VoxelBox node{ lo, lo + VoxelPos(span, span, span), true };
    for (const std::shared_ptr<const ScatterSource> &s : streamed) {
        const int64_t reach = s->ground_reach(p_lod);
        if (reach > 0 && double(p_past_band_m) <= double(reach) * param.voxel_size && node.clipped(s->setup().scatter.zone.grown(reach)).valid) {
            return true;
        }
    }
    return false;
}

SDFBase::Admit SDFTable::admit(const ChunkId &p_chunk, uint64_t p_owner, uint64_t p_tag, uint64_t p_rank, uint64_t &r_words) {
    r_words = 0;
    if (!stream) {
        return Admit::READY;
    }
    switch (stream->admit(p_chunk, p_owner, p_tag, p_rank, &r_words)) {
        case InstanceStream::READY:
            return Admit::READY;
        case InstanceStream::PARKED:
            return Admit::PARKED;
        case InstanceStream::LATER:
            break;
    }
    return Admit::LATER;
}

void SDFTable::take_ready(uint64_t p_owner, std::vector<std::pair<ChunkId, uint64_t>> &r_out) {
    if (stream) {
        stream->take_ready(p_owner, r_out);
    } else {
        r_out.clear();
    }
}

void SDFTable::poll() {
    poll_rebuild();
    if (field_gpu_starting && field_gpu_starting->state() != FieldGPU::STARTING) {
        if (field_gpu_starting->state() == FieldGPU::READY) {
            attach_field_gpu(field_gpu_starting);
        }
        field_gpu_starting.reset(); // attached, or failed (its thread warned)
    }
    const std::shared_ptr<FieldGPU> attached = std::atomic_load(&field_gpu);
    if (attached && attached->state() == FieldGPU::FAILED) {
        attach_field_gpu(nullptr); // a batch failed: the CPU's field, a walk per task again
    }
    if (stream) {
        stream->poll();
    }
}

RecordView SDFTable::point_list(const InstanceSnapshot &p_snap, const VoxelPos &p_voxel, int p_lod, std::vector<uint32_t> &r_list, bool p_view) const {
    p_snap.point_candidates(p_voxel, p_lod, r_list);
    if (!stream) {
        return RecordView(p_snap.records);
    }
    thread_local std::vector<TableRecord> flat;
    flat.clear();
    compose_list(
            r_list, gpu.get_scene_end(), [&](uint32_t p_id) { flat.push_back(p_snap.records[p_id]); },
            [&] {
                if (p_view) {
                    stream->view_records(p_voxel, p_lod, flat);
                } else {
                    stream->point_records(p_voxel, p_lod, flat);
                }
            });
    r_list.resize(flat.size());
    std::iota(r_list.begin(), r_list.end(), 0u);
    return RecordView(flat.data());
}

void SDFTable::stop_streaming() {
    gpu.set_capacity(gpu.table_capacity());
    stream.reset(); // the last reference: waits for its drain tasks
    streamed.clear();
}

void SDFTable::start_field_gpu(const ChunkParam &p_param) {
    stop_field_gpu();
    if (stream && get_cascade()) {
        field_gpu_starting = std::make_shared<FieldGPU>();
        field_gpu_starting->start(this, p_param, FIELD_GPU_SCRATCH_MB);
    }
}

void SDFTable::stop_field_gpu() {
    field_gpu_starting.reset(); // joins its thread, which may still be compiling (~0.5 s at most)
    attach_field_gpu(nullptr); // a walk still reading it falls back to the CPU's field
}

void SDFTable::attach_field_gpu(std::shared_ptr<FieldGPU> p_gpu) {
    std::atomic_store(&field_gpu, p_gpu);
    if (stream) {
        stream->set_walk_batch(p_gpu ? walk_batch_gpu : 1);
    }
}

void SDFTable::ground_field_batch(const VoxelPos *p_voxels, const int *p_lods, int p_count, ExactNoise::Float4 *r_out) const {
    auto cpu = [this, p_voxels, p_lods](int p_count_cpu, ExactNoise::Float4 *r_values) {
        parallel_for(uint32_t(p_count_cpu), "ground_field_batch", [&](uint32_t p_i) { r_values[p_i] = ground_field(p_voxels[p_i], p_lods[p_i]); });
    };
    const std::shared_ptr<FieldGPU> gpu = std::atomic_load(&field_gpu);
    if (!gpu || !gpu->ground_field(p_voxels, p_lods, p_count, r_out)) {
        cpu(p_count, r_out);
        return;
    }
    if (walk_verify) {
        std::vector<ExactNoise::Float4> want(static_cast<size_t>(p_count));
        cpu(p_count, want.data());
        int64_t differing = 0;
        for (int i = 0; i < p_count; i++) {
            if (!ExactNoise::same_bits(r_out[i], want[i])) {
                if (differing++ == 0) {
                    ERR_PRINT(vformat("FieldGPU: voxel %s at LOD %d: GPU (%s %s %s %s), CPU (%s %s %s %s).", godot_str(p_voxels[i]), p_lods[i], r_out[i].x, r_out[i].y,
                            r_out[i].z, r_out[i].w, want[i].x, want[i].y, want[i].z, want[i].w));
                }
                r_out[i] = want[i]; // the walk goes on with the oracle's bits
            }
        }
        verified_points += p_count;
        verify_differences += differing;
    }
}

Dictionary SDFTable::walk_gpu_stats() const {
    const std::shared_ptr<FieldGPU> attached = std::atomic_load(&field_gpu);
    const std::shared_ptr<FieldGPU> gpu = attached ? attached : field_gpu_starting;
    if (!gpu) {
        return Dictionary();
    }
    Dictionary d = gpu->get_stats();
    d["verified_points"] = verified_points.load();
    d["verify_differences"] = verify_differences.load();
    return d;
}

void SDFTable::poll_rebuild() {
    if (rebuild.ready()) {
        finish_rebuild();
    }
}

void SDFTable::finish_rebuild() {
    if (rebuild.running()) {
        const std::shared_ptr<const InstanceSnapshot> now = current();
        swap_snapshot(rebuild.take(*now, now->records, now->version));
    }
}

bool SDFTable::BlobKey::operator<(const BlobKey &p_o) const {
    return std::tie(lod, x, y, z, material) < std::tie(p_o.lod, p_o.x, p_o.y, p_o.z, p_o.material);
}

void SDFTable::moment_blobs_reference(std::vector<MomentBlob> &r_out) const {
    r_out.clear();
    if (moment_blob_type >= 0) {
        moment_layers.rebuild(r_out);
    }
}

void SDFTable::update_moment_blobs(uint32_t p_id, std::vector<uint32_t> &r_ids) {
    if (moment_blob_type < 0) {
        return;
    }
    std::vector<MomentBlobChange> changes;
    moment_layers.update(p_id, record_item(*library, records[p_id]), changes);
    for (const MomentBlobChange &c : changes) {
        const BlobKey key{ c.blob.lod, c.blob.cell[0], c.blob.cell[1], c.blob.cell[2], c.blob.material };
        const auto it = blob_ids.find(key);
        if (!c.kept) {
            if (it != blob_ids.end() && !records[it->second].removed) { // gone: removed, its id kept for the key
                const TableRecord old = records[it->second];
                TableRecord removed = old;
                removed.removed = true;
                records.set(it->second, removed, version + 1);
                blob_changed(it->second, old, removed, key.lod, r_ids);
            }
            continue;
        }
        TableRecord rec = blob_record(*library, moment_blob_type, c.blob, param.skirt_depth);
        if (!blob_fits(rec)) {
            continue; // refused like at the build
        }
        if (it != blob_ids.end()) { // replaced when different (or back, in its old record)
            const TableRecord old = records[it->second];
            if (old.removed || rec.anchor != old.anchor || std::memcmp(rec.rotation, old.rotation, sizeof(rec.rotation)) != 0 ||
                    std::memcmp(rec.params, old.params, sizeof(rec.params)) != 0 || rec.material != old.material) {
                records.set(it->second, rec, version + 1);
                blob_changed(it->second, old, rec, key.lod, r_ids);
            }
            continue;
        }
        ERR_CONTINUE_MSG(records.size() >= gpu.table_capacity(), "SDFTable: a moment blob left out: the table is full.");
        add_blob_record(key, rec);
        TableRecord none = rec;
        none.removed = true;
        blob_changed(records.size() - 1, none, rec, key.lod, r_ids);
    }
}

bool SDFTable::blob_fits(const TableRecord &p_rec) const {
    return record_fit_error((*library)[p_rec.def], p_rec, max_instance_extent()).is_empty() && records.size() <= uint32_t(SDFGpuTable::LIST_ID_MASK);
}

void SDFTable::add_blob_record(const BlobKey &p_key, const TableRecord &p_rec) {
    blob_ids[p_key] = records.size();
    records.push_back(p_rec, version + 1);
    has_moment_blobs = true;
}

void SDFTable::blob_changed(uint32_t p_id, const TableRecord &p_old, const TableRecord &p_new, int p_lod, std::vector<uint32_t> &r_ids) {
    VoxelBox box = record_box(p_old.removed ? p_new : p_old);
    if (!p_new.removed) {
        box = box.merged(record_box(p_new));
    }
    edit_regions.push_back(EditRegion{ box, 0.0f, MAX(p_old.margin_c, p_new.margin_c), p_lod, p_lod + 1 });
    r_ids.push_back(p_id);
}

RecordFinish SDFTable::record_finish() const {
    return RecordFinish{ param.skirt_depth, instance_band, erode_rule };
}

CellScatter SDFTable::scene_source(uint32_t p_salt, int p_count, double p_area_m) const {
    const double vs = param.voxel_size;
    CellScatter s;
    s.salt = p_salt;
    s.side = forest_cell_side(p_count, p_area_m, vs);
    if (s.side > 0) {
        // The scene's centre is still found down a column (ground_y).
        const int64_t x = std::llround(SCENE_X / vs), z = std::llround(SCENE_Z / vs);
        const VoxelPos centre(x, ground_y(x, z), z);
        const int64_t half = std::llround(0.5 * p_area_m / vs);
        const VoxelPos reach(half, 4 * half, half);
        s.zone = VoxelBox{ centre - reach, centre + reach, true };
    }
    s.ground = [this](const VoxelPos &p_v, int p_lod) { return ground_field(p_v, p_lod); };
    s.ground_batch = [this](const VoxelPos *p_voxels, const int *p_lods, int p_count, ExactNoise::Float4 *r_out) {
        ground_field_batch(p_voxels, p_lods, p_count, r_out);
    };
    s.up = [this](const VoxelPos &p_v) { return up_at(p_v); };
    if (!lava_pools.empty()) {
        s.accept = [this](const VoxelPos &p_v) { return !in_lake(p_v.x, p_v.z); };
    }
    return s;
}

CellScatter SDFTable::forest_source(const TreeTypes &p_types) const {
    CellScatter s = scene_source(TREE_SITE_SALT, forest_count, forest_area);
    const double vs = param.voxel_size;
    set_forest_scatter(s, vs, *library, record_finish(), p_types, max_instance_extent());
    return s;
}

CellScatter SDFTable::scatter_source(const Types &p_types) const {
    CellScatter s = scene_source(SCATTER_SITE_SALT, scatter_count, scatter_area);
    const float vs = param.voxel_size;
    const int rock = p_types.rock, capsule = p_types.capsule;
    s.generate = [rock, capsule, vs](const Site &p_site, const UpAxis &p_up, int, std::vector<TableRecord> &r_out) { // one record: the same at every LOD
        const ExactNoise::UVec3 h = ExactNoise::hash3(p_site.seed, 0x5CA7u, 0x7E12u);
        const float r = (1.0f + float(h.x % 300u) * 0.01f) / vs; // voxels
        const bool is_rock = (p_site.variant & 1u) == 0u;
        TableRecord rec;
        rec.def = uint8_t(is_rock ? rock : capsule);
        rec.op = Combine::SMOOTH_UNION;
        rec.set_k((is_rock ? 1.0f : 0.8f) / vs);
        rec.salt = h.z;
        float frame[9];
        up_rotation(p_up, frame);
        // A rock turns about the up by a hashed yaw (rows: yaw times the up frame), sunk 0.4 of its radius along it.
        const double yaw = is_rock ? double(h.y % 628u) * 0.01 : 0.0;
        const float c = float(std::cos(yaw)), sn = float(std::sin(yaw));
        const float turn[9] = { c, 0, -sn, 0, 1, 0, sn, 0, c };
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                rec.rotation[3 * i + j] = turn[3 * i] * frame[j] + turn[3 * i + 1] * frame[3 + j] + turn[3 * i + 2] * frame[6 + j];
            }
        }
        const double dy = is_rock ? -0.4 * double(r) : 0.0;
        rec.anchor = p_site.base + VoxelPos(std::llround(dy * p_up[0]), std::llround(dy * p_up[1]), std::llround(dy * p_up[2]));
        if (is_rock) {
            rec.params[0] = r;
            rec.params[1] = 0.1f * r;
        } else {
            rec.params[0] = 2.0f / vs + r;
            rec.params[1] = 0.5f / vs;
        }
        r_out.push_back(rec);
    };
    return s;
}

void SDFTable::add_source(const CellScatter &p_source, const char *p_name, Dictionary &r_stats, std::vector<TreeSpan> *r_spans) {
    const uint64_t start_us = Time::get_singleton()->get_ticks_usec();
    std::vector<CellSite> sites;
    if (p_source.valid()) {
        p_source.sites(p_source.zone, sites);
    }
    // Each site's records are independent (up_at and the library are const): on the worker pool, a batch at a time (the
    // records wait for their turn), then admitted in site order (the record ids don't change).
    constexpr size_t BATCH = 4096;
    std::vector<std::vector<TableRecord>> grown(MIN(BATCH, sites.size()));
    int64_t placed = 0, total = 0, most = 0;
    // Per material: the sites' summed volume and equal-covariance semi-axes (span_moments), for the means.
    std::map<uint32_t, std::pair<double, std::array<double, 3>>> per_material;
    for (size_t i = 0; i < sites.size(); i++) {
        if (i % BATCH == 0) {
            parallel_for(uint32_t(MIN(BATCH, sites.size() - i)), "SDFTable::add_source", [&](uint32_t p_j) {
                grown[p_j].clear();
                p_source.grow(sites[i + p_j].site, -1, grown[p_j]);
            });
        }
        const Site &site = sites[i].site;
        std::vector<TableRecord> &parts = grown[i % BATCH];
        const uint32_t first = records.size();
        if (parts.empty() || !admit_records(parts, p_name)) {
            continue;
        }
        if (r_spans) {
            r_spans->push_back(TreeSpan{ first, uint32_t(parts.size()), tree_species_of(site.variant), site.seed });
        }
        placed++;
        total += int64_t(parts.size());
        most = MAX(most, int64_t(parts.size()));
        const MaterialMoments mm = span_moments(*library, parts.data(), uint32_t(parts.size()), site.base);
        for (const std::pair<uint32_t, Moments> &m : mm.parts) {
            double a[3];
            m.second.semi_axes(a);
            std::pair<double, std::array<double, 3>> &sum = per_material[m.first];
            sum.first += m.second.m0;
            for (int j = 0; j < 3; j++) {
                sum.second[j] += a[j];
            }
        }
    }
    // A site's mean weight per material (m^3, overlapping records counted twice) and spread (semi-axes, metres).
    const double vs = param.voxel_size;
    const double per_site = placed == 0 ? 0.0 : vs / double(placed);
    Dictionary materials_stats;
    for (const std::pair<const uint32_t, std::pair<double, std::array<double, 3>>> &m : per_material) {
        Dictionary d;
        d["volume_m3_avg"] = m.second.first * per_site * vs * vs;
        d["semi_axes_m_avg"] = vformat("%.2f %.2f %.2f", m.second.second[0] * per_site, m.second.second[1] * per_site, m.second.second[2] * per_site);
        materials_stats[materials.name(m.first)] = d;
    }
    Dictionary d;
    d["count"] = placed;
    d["records"] = total;
    d["records_per_site_avg"] = placed == 0 ? 0.0 : double(total) / double(placed);
    d["records_per_site_max"] = most;
    d["moments"] = materials_stats;
    d["build_ms"] = double(Time::get_singleton()->get_ticks_usec() - start_us) * 1e-3;
    r_stats[p_name] = d;
}

void SDFTable::add_stream(uint32_t p_slot_base, Dictionary &r_stats) {
    const uint64_t start_us = Time::get_singleton()->get_ticks_usec();
    for (const Source &source : sources) {
        if (!source.scatter.valid()) {
            continue;
        }
        const uint64_t t0 = Time::get_singleton()->get_ticks_usec();
        streamed.push_back(std::make_shared<const ScatterSource>(source_setup(source, uint32_t(streamed.size()))));
        const SourceProfile &prof = streamed.back()->profile();
        Dictionary d;
        d["streamed"] = true;
        d["records_per_site_avg"] = prof.records_per_site;
        d["reach_m"] = prof.reach * param.voxel_size;
        d["lod_end"] = prof.lod_end;
        d["profile_classes"] = int64_t(prof.lost.size());
        d["profile_shared"] = streamed.back()->profile_shared();
        d["profile_ms"] = double(Time::get_singleton()->get_ticks_usec() - t0) * 1e-3;
        r_stats[source.name] = d;
    }
    if (streamed.empty()) {
        return;
    }
    InstanceStream::Settings settings;
    settings.slot_base = p_slot_base;
    settings.slot_capacity = MIN<uint32_t>(InstanceStream::slot_budget(streamed), SDFGpuTable::LIST_ID_MASK + 1 - p_slot_base);
    stream = std::make_shared<InstanceStream>(streamed, param, settings, StreamExecutor::pool());
    if (std::atomic_load(&field_gpu)) {
        stream->set_walk_batch(walk_batch_gpu); // a reconfigure keeps the attached field
    }
    Dictionary d;
    d["slots"] = int64_t(settings.slot_capacity);
    d["setup_ms"] = double(Time::get_singleton()->get_ticks_usec() - start_us) * 1e-3;
    r_stats["stream_setup"] = d;
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
        ov::executor().parallel_for(uint32_t(sites.size()), "SDFTable::add_biomes", &Job::run, &job);
    }
    Array listed;
    int64_t counts[Biome::FEATURE_COUNT] = {};
    int64_t too_steep = 0;
    for (size_t i = 0; i < sites.size(); i++) {
        if (!ok[i]) {
            too_steep++;
            continue;
        }
        if (!admit_records(placed[i].records, "biome feature")) {
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

void SDFTable::add_moment_blobs(int p_blob_type, Dictionary &r_stats) {
    if (!moment_blobs || !instance_band) {
        return;
    }
    const uint64_t start_us = Time::get_singleton()->get_ticks_usec();
    moment_blob_type = p_blob_type;
    MomentLayerParams params = moment_params();
    params.lod_min = lod_count;
    params.lod_max = lod_count - 1;
    std::vector<MomentItem> items(records.size());
    int64_t weighed = 0;
    for (uint32_t i = 0; i < records.size(); i++) {
        items[i] = record_item(*library, records[i]);
        if (!items[i].moments.empty()) {
            params.lod_min = MIN(params.lod_min, items[i].first_lost_lod());
            weighed++;
        }
    }
    moment_layers.build(items, params);
    std::vector<MomentBlob> blobs;
    moment_layers.kept(blobs);
    Array per_lod;
    per_lod.resize(lod_count);
    per_lod.fill(int64_t(0));
    blob_ids.clear();
    int64_t refused = 0;
    for (const MomentBlob &b : blobs) {
        const TableRecord rec = blob_record(*library, moment_blob_type, b, param.skirt_depth);
        if (!blob_fits(rec)) {
            refused++;
            continue;
        }
        add_blob_record(BlobKey{ b.lod, b.cell[0], b.cell[1], b.cell[2], b.material }, rec);
        per_lod[b.lod] = int64_t(per_lod[b.lod]) + 1;
    }
    Dictionary d;
    d["items"] = weighed;
    d["blobs"] = int64_t(blobs.size());
    d["blobs_per_lod"] = per_lod;
    d["build_ms"] = double(Time::get_singleton()->get_ticks_usec() - start_us) * 1e-3;
    r_stats["moments"] = d;
    if (refused > 0) {
        ERR_PRINT(vformat("SDFTable: %d moment blobs left out (too wide, or the table is full).", refused));
    }
}

MomentLayerParams SDFTable::moment_params() const {
    MomentLayerParams params;
    // Fainter, even with its neighbours, than what draws a material: it would only lift the surface by an eighth of a cell.
    params.min_occupancy = MOMENT_MATERIAL_ISO;
    return params;
}

ScatterSource::Setup SDFTable::source_setup(const Source &p_source, uint32_t p_id) const {
    ScatterSource::Setup s;
    s.id = p_id;
    s.library = library;
    s.scatter = p_source.scatter;
    s.blob_type = moment_blobs && instance_band ? blob_type : -1;
    s.finish = record_finish();
    s.layers = moment_params();
    s.max_extent = max_instance_extent();
    s.chunk_voxels = param.chunk_voxel_dimension();
    s.lod_count = lod_count;
    s.part_cell_sides = stream_part_sides;
    return s;
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

bool SDFTable::admit_records(std::vector<TableRecord> &r_records, const char *p_what) {
    ERR_FAIL_COND_V_MSG(records.size() + r_records.size() > uint32_t(SDFGpuTable::LIST_ID_MASK) + 1, false,
            vformat("SDFTable: %s left out: the table holds at most %d instances (SDFGpuTable::LIST_ID_BITS).", p_what, SDFGpuTable::LIST_ID_MASK + 1));
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
    admit_records(one, "scene record");
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

void SDFTable::build_table(int p_lod_count) {
    const double vs = param.voxel_size;
    // Appended in this order: the GDScript side names the indices (CAPSULE_TYPE). The gallery's types only with it.
    DefLibrary lib;
    Types types;
    lib.add(make_sphere_type());
    types.capsule = lib.add(make_capsule_type());
    types.rock = lib.add(make_rock_type(ExactNoise::lattice_k(8.0, vs), ROCK_SEED));
    const TreeTypes tree_types = add_tree_types(lib, vs);
    if (scene == Scene::SHAPES) {
        types.holed_cube = lib.add(make_swiss_cube_type(1));
        types.die = lib.add(make_die_type());
        types.stairs = lib.add(make_stairs_type());
        types.tunnel_cube = lib.add(make_tunnel_cube_type());
    }
    if (biome_radius > 0.0) {
        types.biome = Biome::add_types(lib, vs);
    }
    const int blob = lib.add(make_moment_blob_type());
    blob_type = blob;
    library = std::make_shared<const DefLibrary>(lib);
    for (int i = 0; i < library->size(); i++) {
        const String error = godot_str((*library)[i].program.validate(p_lod_count));
        if (!error.is_empty()) {
            ERR_PRINT(vformat("Archetype library type %d: ", i) + error);
        }
    }
    lod_count = p_lod_count;

    stop_streaming(); // its tasks read the old sources
    rebuild.cancel(); // a base of the old records
    records.clear();
    has_moment_blobs = false;
    moment_blob_type = -1;
    blob_ids.clear();
    moment_layers = MomentLayers();
    trees.clear();
    sources.clear();
    lava_pools.clear();
    if (scene == Scene::SHAPES) {
        add_shapes(types);
    }
    // The biome features first: the scatter and the forest keep out of the lakes (their basins cut the ground away).
    Dictionary scene_stats;
    add_biomes(types, scene_stats);
    sources = { Source{ "scatter", scatter_source(types) }, Source{ "trees", forest_source(tree_types) } };
    if (!stream_sources) {
        add_source(sources[0].scatter, sources[0].name, scene_stats);
        add_source(sources[1].scatter, sources[1].name, scene_stats, &trees);
    }
    add_moment_blobs(blob, scene_stats);
    edit_regions.clear();
    // The GPU buffers hold the scene plus room for runtime edits, then the streamed slots (the mesher reads this at init).
    const uint32_t capacity = MIN<uint32_t>(MAX<uint32_t>(16384u, uint32_t(records.size()) + 4096u), SDFGpuTable::LIST_ID_MASK + 1);
    if (stream_sources) {
        add_stream(capacity, scene_stats);
    }
    gpu.set_capacity(capacity, stream);
    gpu.set_scene(scene_stats, records.size());
    version++;
    publish(build_snapshot(library, float(vs), true, records, param, lod_count - 1, version), VoxelBox());
}
