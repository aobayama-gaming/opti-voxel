#include "mesh_validator.hpp"

#include "GPU_mesher.hpp"
#include "debug_log.hpp"
#include "page_format.hpp"
#include "quad_page_pool.hpp"

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/rendering_server.hpp>

#include <cmath>

namespace {

struct Tri {
    Vector3 p[3];
    Vector3 n[3];
};

// Smallest |SDF| (CPU evaluation, in cells) over the grid corners of the cell holding `p_grid` (a
// point of the chunk's vertex grid). A sign decision taken on a near-zero corner can legitimately
// differ between two GPU evaluations.
double min_abs_sdf_cells(SDFBase *p_sdf, const ChunkId &p_chunk, const Vector3 &p_grid, const ChunkParam &p_param, float p_cell) {
    const Vector3i base((int)std::floor(p_grid.x), (int)std::floor(p_grid.y), (int)std::floor(p_grid.z));
    double best = 1e30;
    for (int i = 0; i < 8; i++) {
        const Vector3 corner(base.x + (i & 1), base.y + ((i >> 1) & 1), base.z + ((i >> 2) & 1));
        best = MIN(best, std::abs((double)p_sdf->evaluate(ChunkMath::grid_to_render(p_chunk, corner, p_param))) / p_cell);
    }
    return best;
}

// Triangles in the chunk's vertex grid, written in render space (metres relative to the origin).
void write_obj(const String &p_path, const std::vector<Tri> &p_tris, const ChunkId &p_chunk, const ChunkParam &p_param) {
    Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::WRITE);
    ERR_FAIL_COND(f.is_null());
    for (const Tri &t : p_tris) {
        for (int k = 0; k < 3; k++) {
            const Vector3 p = ChunkMath::grid_to_render(p_chunk, t.p[k], p_param);
            f->store_line(vformat("v %f %f %f", p.x, p.y, p.z));
        }
    }
    for (size_t i = 0; i < p_tris.size(); i++) {
        f->store_line(vformat("f %d %d %d", (int64_t)i * 3 + 1, (int64_t)i * 3 + 2, (int64_t)i * 3 + 3));
    }
}

} // namespace

void MeshValidator::start(const std::vector<Target> &p_targets, SDFBase *p_sdf, ChunkParam p_chunk_param, int p_lod_numbers,
        QuadPagePool *p_pool, const String &p_dump_dir) {
    targets = p_targets;
    sdf = p_sdf;
    chunk_param = p_chunk_param;
    pool = p_pool;
    dump_dir = p_dump_dir;
    mutex.instantiate();
    start_usec = now_usec();

    // Legacy mesher on the same chunks, drawn into a scenario nobody renders.
    RenderingServer *rs = RenderingServer::get_singleton();
    legacy_scenario = rs->scenario_create();
    legacy_state.set_max_lod(p_lod_numbers);
    legacy = memnew(GPUMesher);
    legacy->initialize(sdf, chunk_param, MAX((int)targets.size(), 1), p_lod_numbers, legacy_scenario);
    legacy->set_trim_indices(true);
    LODHashSet chunks;
    chunks.max_lod = p_lod_numbers;
    for (const Target &t : targets) {
        chunks.insert(t.chunk_id);
    }
    legacy->send_chunk(chunks);

    // Direct side: read back every page holding a target.
    HashMap<int, RID> pages;
    for (const Target &t : targets) {
        pages[t.page] = t.page_multimesh;
    }
    pages_requested = pages.size();
    for (const KeyValue<int, RID> &kv : pages) {
        pages_to_request.push_back({ kv.key, kv.value });
    }
    VLOG(1, "MeshValidator: validating ", (int64_t)targets.size(), " chunks in ", pages_requested, " pages");
}

void MeshValidator::_rt_read_page(int p_page, RID p_multimesh) {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    rd->texture_get_data_async(pool->rt_get_page_rd(p_page, p_multimesh).texture, 0,
            callable_mp(this, &MeshValidator::_rt_on_page).bind(p_page));
}

void MeshValidator::_rt_on_page(const PackedByteArray &p_data, int p_page) {
    mutex->lock();
    page_data[p_page] = p_data;
    mutex->unlock();
}

bool MeshValidator::step() {
    if (done) {
        return true;
    }
    legacy->process_queue(); // no-op once the batch is dispatched
    legacy->get_results(legacy_state);

    for (int i = 0; i < PAGES_PER_FRAME && !pages_to_request.empty(); i++) {
        const std::pair<int, RID> page = pages_to_request.back();
        pages_to_request.pop_back();
        RenderingServer::get_singleton()->call_on_render_thread(
                callable_mp(this, &MeshValidator::_rt_read_page).bind(page.first, page.second));
    }

    mutex->lock();
    const bool pages_ready = (int)page_data.size() == pages_requested;
    mutex->unlock();
    const bool legacy_ready = legacy_state.loaded_chunk.size() == targets.size();
    if (!pages_ready || !legacy_ready) {
        if (now_usec() - start_usec > 30 * 1000000ULL) {
            report["error"] = vformat("timeout: pages %d/%d legacy %d/%d", (int64_t)page_data.size(), pages_requested,
                    (int64_t)legacy_state.loaded_chunk.size(), (int64_t)targets.size());
            report["ok"] = false;
            done = true;
        }
        return done;
    }
    _compare();

    for (KeyValue<ChunkId, ChunkOutput> &kv : legacy_state.loaded_chunk) {
        kv.value.free();
    }
    legacy_state.loaded_chunk.clear();
    legacy->stop();
    memdelete(legacy);
    legacy = nullptr;
    RenderingServer::get_singleton()->free_rid(legacy_scenario);
    done = true;
    return true;
}

void MeshValidator::_compare() {
    RenderingServer *rs = RenderingServer::get_singleton();
    int64_t legacy_tris_total = 0;
    int64_t direct_tris_total = 0;
    int64_t matched = 0;
    int64_t winding_flipped = 0;
    int64_t legacy_unmatched = 0;
    int64_t direct_unmatched = 0;
    double max_pos_err_cells = 0.0;
    double max_normal_err_deg = 0.0;
    double sum_normal_err_deg = 0.0;
    int64_t normal_samples = 0;
    int64_t normals_over_5deg = 0;
    int chunks_exact = 0;
    Array per_chunk;
    Array anomalies; // flipped / unmatched / bad-normal triangles with their nearest |sdf|
    PackedFloat64Array baseline_sdf; // same measure on ordinary matched triangles
    int64_t unexplained = 0;
    int64_t bad_headers = 0; // groups of a chunk whose header names another chunk
    auto record = [&](const char *p_kind, const ChunkId &p_chunk, const Tri &p_tri, float p_cell, double p_extra) {
        const Vector3 c = (p_tri.p[0] + p_tri.p[1] + p_tri.p[2]) / 3.0f;
        const double near_zero = min_abs_sdf_cells(sdf, p_chunk, c, chunk_param, p_cell);
        unexplained += near_zero > 1e-3 ? 1 : 0;
        if (anomalies.size() >= 40) {
            return; // keep the report small; every anomaly is still classified above
        }
        Dictionary a;
        a["kind"] = p_kind;
        a["chunk"] = WorldCoord::to_string(p_chunk);
        a["centroid"] = ChunkMath::grid_to_render(p_chunk, c, chunk_param);
        a["min_abs_sdf_cells"] = near_zero;
        a["extra"] = p_extra;
        anomalies.push_back(a);
    };
    bool dumped = false;

    // Both sides are compared in the chunk's vertex grid (cell units): positions never go through
    // absolute coordinates, so the comparison does not depend on where the world origin is.
    for (const Target &t : targets) {
        const float lod_size = float(ChunkMath::get_lod_size(t.chunk_id));
        const float cell = chunk_param.voxel_size * lod_size;
        const float tol = 0.02f + 1e-4f / cell;
        const Vector3i chunk_lo = WorldCoord::lo32_i(t.chunk_id);

        // --- direct triangles from the page texture: 4 corners per quad, triangles A,B,D / A,D,C
        std::vector<Tri> direct;
        {
            mutex->lock();
            const PackedByteArray bytes = page_data[t.page];
            mutex->unlock();
            const uint16_t *page = reinterpret_cast<const uint16_t *>(bytes.ptr());
            const uint32_t rows = bytes.size() / PageFormat::ROW_BYTES;
            const float scale = PageFormat::pos_scale(chunk_param.chunk_size);
            for (uint32_t q = t.offset; q < t.offset + t.count && q / PageFormat::QUAD_GROUP < rows; q++) {
                if (q % PageFormat::QUAD_GROUP == 0 && PageFormat::header_chunk(PageFormat::row(page, q / PageFormat::QUAD_GROUP)) != chunk_lo) {
                    bad_headers++;
                }
                Vector3 p[4];
                Vector3 n[4];
                for (int k = 0; k < 4; k++) {
                    const uint16_t *c = PageFormat::corner(page, q, k);
                    p[k] = PageFormat::decode_local(c, scale);
                    n[k] = PageFormat::is_zero(c) ? Vector3() : PageFormat::decode_normal(c);
                }
                direct.push_back({ { p[0], p[1], p[3] }, { n[0], n[1], n[3] } });
                direct.push_back({ { p[0], p[3], p[2] }, { n[0], n[3], n[2] } });
            }
        }

        // --- legacy triangles: the mesh vertices already are vertex-grid coordinates
        std::vector<Tri> legacy_tris;
        if (legacy_state.loaded_chunk.has(t.chunk_id)) {
            const ChunkOutput &out = legacy_state.loaded_chunk[t.chunk_id];
            if (out.mesh_rid.is_valid()) {
                const Array arrays = rs->mesh_surface_get_arrays(out.mesh_rid, 0);
                const PackedVector3Array v = arrays[Mesh::ARRAY_VERTEX];
                const PackedVector3Array vn = arrays[Mesh::ARRAY_NORMAL];
                const PackedInt32Array idx = arrays[Mesh::ARRAY_INDEX];
                for (int i = 0; i + 2 < idx.size(); i += 3) {
                    Tri tri;
                    for (int k = 0; k < 3; k++) {
                        tri.p[k] = v[idx[i + k]];
                        tri.n[k] = vn.size() == v.size() ? vn[idx[i + k]].normalized() : Vector3();
                    }
                    legacy_tris.push_back(tri);
                }
            }
        }

        // --- match: spatial hash of direct triangles by centroid
        const float bucket = tol * 8.0f;
        auto key = [&](const Vector3 &c) {
            return Vector3i((int)std::floor(c.x / bucket), (int)std::floor(c.y / bucket), (int)std::floor(c.z / bucket));
        };
        HashMap<Vector3i, std::vector<int>> grid;
        for (int i = 0; i < (int)direct.size(); i++) {
            const Tri &d = direct[i];
            grid[key((d.p[0] + d.p[1] + d.p[2]) / 3.0f)].push_back(i);
        }
        std::vector<bool> direct_used(direct.size(), false);
        int64_t chunk_matched = 0;
        int64_t chunk_flipped = 0;
        for (const Tri &l : legacy_tris) {
            const Vector3i k0 = key((l.p[0] + l.p[1] + l.p[2]) / 3.0f);
            int best = -1;
            bool best_flipped = false;
            int best_rot = 0;
            double best_err = 1e30;
            for (int dx = -1; dx <= 1 && best < 0; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dz = -1; dz <= 1; dz++) {
                        const std::vector<int> *cell_list = grid.getptr(k0 + Vector3i(dx, dy, dz));
                        if (!cell_list) {
                            continue;
                        }
                        for (int di : *cell_list) {
                            if (direct_used[di]) {
                                continue;
                            }
                            const Tri &d = direct[di];
                            for (int flip = 0; flip < 2; flip++) {
                                for (int rot = 0; rot < 3; rot++) {
                                    double err = 0.0;
                                    for (int k = 0; k < 3; k++) {
                                        const int dk = flip ? (rot + 3 - k) % 3 : (rot + k) % 3;
                                        err = MAX(err, (double)l.p[k].distance_to(d.p[dk]));
                                    }
                                    // Same winding wins over flipped: a sliver (two corners closer than
                                    // the tolerance) matches both windings and has no meaningful one,
                                    // so it only counts as flipped when no same-winding match exists.
                                    const bool better = best < 0 || (!flip && best_flipped) || (bool(flip) == best_flipped && err < best_err);
                                    if (err <= tol && better) {
                                        best = di;
                                        best_err = err;
                                        best_flipped = flip;
                                        best_rot = rot;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (best < 0) {
                legacy_unmatched++;
                record("legacy_unmatched", t.chunk_id, l, cell, 0.0);
                continue;
            }
            direct_used[best] = true;
            chunk_matched++;
            chunk_flipped += best_flipped ? 1 : 0;
            if (best_flipped) {
                record("winding_flipped", t.chunk_id, l, cell, 0.0);
            } else if (chunk_matched % 4000 == 1 && baseline_sdf.size() < 200) {
                const Vector3 c = (l.p[0] + l.p[1] + l.p[2]) / 3.0f;
                baseline_sdf.push_back(min_abs_sdf_cells(sdf, t.chunk_id, c, chunk_param, cell));
            }
            max_pos_err_cells = MAX(max_pos_err_cells, best_err);
            const Tri &d = direct[best];
            for (int k = 0; k < 3; k++) {
                const int dk = best_flipped ? (best_rot + 3 - k) % 3 : (best_rot + k) % 3;
                if (l.n[k] == Vector3() || d.n[dk] == Vector3()) {
                    continue;
                }
                const double ang = Math::rad_to_deg(std::acos(CLAMP(l.n[k].dot(d.n[dk]), -1.0f, 1.0f)));
                max_normal_err_deg = MAX(max_normal_err_deg, ang);
                sum_normal_err_deg += ang;
                normal_samples++;
                normals_over_5deg += ang > 5.0 ? 1 : 0;
                if (ang > 20.0) {
                    record("normal_over_20deg", t.chunk_id, l, cell, ang);
                }
            }
        }
        int64_t chunk_direct_unmatched = 0;
        for (bool used : direct_used) {
            chunk_direct_unmatched += used ? 0 : 1;
        }
        const int64_t chunk_legacy_unmatched = (int64_t)legacy_tris.size() - chunk_matched;

        legacy_tris_total += legacy_tris.size();
        direct_tris_total += direct.size();
        matched += chunk_matched;
        winding_flipped += chunk_flipped;
        direct_unmatched += chunk_direct_unmatched;
        const bool exact = chunk_legacy_unmatched == 0 && chunk_direct_unmatched == 0 && chunk_flipped == 0;
        chunks_exact += exact ? 1 : 0;

        if (!exact) {
            Dictionary c;
            c["chunk"] = WorldCoord::to_string(t.chunk_id);
            c["lod"] = ChunkMath::get_lod(t.chunk_id);
            c["legacy_tris"] = (int64_t)legacy_tris.size();
            c["direct_tris"] = (int64_t)direct.size();
            c["legacy_unmatched"] = chunk_legacy_unmatched;
            c["direct_unmatched"] = chunk_direct_unmatched;
            c["winding_flipped"] = chunk_flipped;
            per_chunk.push_back(c);
            if (!dumped && !dump_dir.is_empty()) {
                const String base = dump_dir.path_join(vformat("chunk_%d_%d_%d", t.chunk_id.x, t.chunk_id.y, t.chunk_id.z));
                write_obj(base + String("_legacy.obj"), legacy_tris, t.chunk_id, chunk_param);
                write_obj(base + String("_direct.obj"), direct, t.chunk_id, chunk_param);
                c["dumped"] = base;
                dumped = true;
            }
        }
    }

    report["chunks"] = (int64_t)targets.size();
    report["chunks_exact"] = chunks_exact;
    report["legacy_tris"] = legacy_tris_total;
    report["direct_tris"] = direct_tris_total;
    report["matched_tris"] = matched;
    report["legacy_unmatched_tris"] = legacy_unmatched;
    report["direct_unmatched_tris"] = direct_unmatched;
    report["winding_flipped_tris"] = winding_flipped;
    report["max_position_error_cells"] = max_pos_err_cells;
    report["max_normal_error_deg"] = max_normal_err_deg;
    report["mean_normal_error_deg"] = normal_samples ? sum_normal_err_deg / normal_samples : 0.0;
    report["normals_over_5deg"] = normals_over_5deg;
    report["mismatching_chunks"] = per_chunk;
    report["anomalies"] = anomalies;
    {
        PackedFloat64Array sorted = baseline_sdf;
        sorted.sort();
        Dictionary b;
        b["n"] = sorted.size();
        b["min"] = sorted.is_empty() ? 0.0 : sorted[0];
        b["p10"] = sorted.is_empty() ? 0.0 : sorted[sorted.size() / 10];
        b["median"] = sorted.is_empty() ? 0.0 : sorted[sorted.size() / 2];
        report["baseline_min_abs_sdf_cells"] = b;
    }
    report["match_ratio"] = legacy_tris_total ? (double)matched / legacy_tris_total : 1.0;
    // Legacy and direct evaluate the same SDF with differently compiled shaders on different devices,
    // so a sign decision on a corner where |sdf| ~ 0 may legitimately differ (winding flip, edge
    // crossing present on one side only, gradient branch of max()). Such differences are
    // "explained"; anything next to a clearly non-zero corner is a real bug.
    report["unexplained_anomalies"] = unexplained;
    report["bad_group_headers"] = bad_headers;
    report["ok"] = legacy_tris_total > 0 && legacy_tris_total == direct_tris_total && unexplained == 0 && bad_headers == 0 &&
            (double)matched / legacy_tris_total > 0.9999 && (normal_samples == 0 || sum_normal_err_deg / normal_samples < 0.5);
}
