#include "mesh_validator.hpp"

#include "GPU_mesher.hpp"
#include "debug_log.hpp"
#include "material_library.hpp"
#include "page_format.hpp"
#include "quad_page_pool.hpp"

#include "ov/files.hpp"
#include "ov/math.hpp"
#include "ov/on_error.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>

namespace {

struct Tri {
    Float3 p[3];
    Float3 n[3];
    uint32_t m[3] = {}; // global materials
};

// godot's Vector3 float arithmetic, operation for operation (the comparison's tolerances were tuned on it).
Float3 centroid(const Float3 p_p[3]) {
    return Float3{ (p_p[0].x + p_p[1].x + p_p[2].x) / 3.0f, (p_p[0].y + p_p[1].y + p_p[2].y) / 3.0f, (p_p[0].z + p_p[1].z + p_p[2].z) / 3.0f };
}

float distance(const Float3 &p_a, const Float3 &p_b) {
    const float dx = p_b.x - p_a.x;
    const float dy = p_b.y - p_a.y;
    const float dz = p_b.z - p_a.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Zero for a zero vector.
Float3 normalized(const Float3 &p_v) {
    return dot(p_v, p_v) == 0.0f ? Float3() : p_v.normalized();
}

// The GPU meshes the samples with their skirt, not the bare SDF: past the chunk bounds it adds up to
// p_depth_cells (SDFBase::skirt_depth_m in cells). Same formula here, so the corners are judged on the field that
// was meshed.
double skirt_offset_cells(const Float3 &p_corner, const ChunkParam &p_param, double p_depth_cells) {
    if (p_param.skirt_size <= 0 || p_param.skirt_depth <= 0.0f) {
        return 0.0;
    }
    const double lo = double(p_param.skirt_size);
    const double hi = lo + double(p_param.chunk_voxel_dimension());
    auto inside = [lo, hi](double c) { return ov::min(c - lo, hi - c); };
    const double d = ov::min(inside(p_corner.x), ov::min(inside(p_corner.y), inside(p_corner.z)));
    return ov::clamp(-d, 0.0, 1.0) * p_depth_cells;
}

// Smallest |SDF| (CPU evaluation, in cells) over the grid corners of the cell holding `p_grid` (a
// point of the chunk's vertex grid). A sign decision taken on a near-zero corner can legitimately
// differ between two GPU evaluations.
double min_abs_sdf_cells(SDFBase *p_sdf, const ChunkId &p_chunk, const Float3 &p_grid, const ChunkParam &p_param, float p_cell) {
    const ov::ivec3 base((int)std::floor(p_grid.x), (int)std::floor(p_grid.y), (int)std::floor(p_grid.z));
    double best = 1e30;
    const int lod = ChunkMath::get_lod(p_chunk);
    const double depth_cells = double(p_sdf->skirt_depth_m(lod)) / p_cell;
    for (int i = 0; i < 8; i++) {
        const Float3 corner{ float(base.x + (i & 1)), float(base.y + ((i >> 1) & 1)), float(base.z + ((i >> 2) & 1)) };
        const double sdf_cells = (double)p_sdf->evaluate(ChunkMath::sample_pos(p_chunk, corner, p_param), lod) / p_cell;
        best = ov::min(best, std::abs(sdf_cells + skirt_offset_cells(corner, p_param, depth_cells)));
    }
    return best;
}

// Triangles in the chunk's vertex grid, written in render space (metres relative to the origin).
void write_obj(const ov::str &p_path, const std::vector<Tri> &p_tris, const ChunkId &p_chunk, const ChunkParam &p_param) {
    ov::str text;
    for (const Tri &t : p_tris) {
        for (int k = 0; k < 3; k++) {
            const Float3 p = ChunkMath::grid_to_render(p_chunk, t.p[k], p_param);
            text += ov::format("v %f %f %f\n", p.x, p.y, p.z);
        }
    }
    for (size_t i = 0; i < p_tris.size(); i++) {
        text += ov::format("f %d %d %d\n", int64_t(i) * 3 + 1, int64_t(i) * 3 + 2, int64_t(i) * 3 + 3);
    }
    OV_ERR_FAIL_COND_MSG(!ov::write_file(p_path, text), "MeshValidator: cannot write " + p_path + ".");
}

} // namespace

std::vector<MeshValidator::Target> MeshValidator::pick_targets(const std::vector<Target> &p_candidates, int p_count) {
    std::vector<Target> sets[2];
    for (const Target &t : p_candidates) {
        sets[t.palette.offset != PaletteAlloc::GROUND_BLOCK ? 1 : 0].push_back(t); // records' materials past the base
    }
    const int with_materials = ov::min(p_count / 2, (int)sets[1].size());
    const int counts[2] = { ov::min(p_count - with_materials, (int)sets[0].size()), with_materials };
    std::vector<Target> targets;
    for (int s = 0; s < 2; s++) {
        for (int i = 0; i < counts[s]; i++) {
            targets.push_back(sets[s][(size_t)i * sets[s].size() / counts[s]]);
        }
    }
    return targets;
}

void MeshValidator::start(const std::vector<Target> &p_targets, SDFBase *p_sdf, ChunkParam p_chunk_param, int p_lod_numbers, QuadPagePool *p_pool,
        MeshHost *p_legacy_host, const ov::str &p_dump_dir, bool p_dual_contouring) {
    targets = p_targets;
    sdf = p_sdf;
    chunk_param = p_chunk_param;
    pool = p_pool;
    legacy_host = p_legacy_host;
    dump_dir = p_dump_dir;
    start_usec = now_usec();

    // Legacy mesher on the same chunks, its meshes in a host that doesn't draw them.
    legacy_state.set_max_lod(p_lod_numbers);
    legacy = new GPUMesher();
    // Same vertex placement as the direct side, or every triangle mismatches (two copies of one
    // function: see gpu_glsl.hpp).
    legacy->set_dual_contouring(p_dual_contouring);
    legacy->initialize(sdf, chunk_param, std::max((int)targets.size(), 1), p_lod_numbers, legacy_host);
    legacy->set_trim_indices(true);
    LODHashSet chunks;
    chunks.max_lod = p_lod_numbers;
    for (const Target &t : targets) {
        chunks.insert(t.chunk_id);
    }
    legacy->send_chunk(chunks, ScanView()); // no view: LOD order

    // Direct side: read back every page holding a target.
    std::set<int> pages;
    for (const Target &t : targets) {
        pages.insert(t.page);
    }
    pages_requested = int(pages.size());
    for (const int page : pages) {
        pages_to_request.push_back(page);
    }
    VLOG(1, "MeshValidator: validating ", (int64_t)targets.size(), " chunks in ", pages_requested, " pages");
}

void MeshValidator::_rt_read_page(int p_page) {
    rt_readbacks_in_flight++;
    pool->get_device()->read_texture_async(pool->rt_get_page_rd(p_page).texture, [this, p_page](const ov::bytes &p_data) { _rt_on_page(p_data, p_page); });
}

void MeshValidator::_rt_on_page(const ov::bytes &p_data, int p_page) {
    mutex.lock();
    page_data[p_page] = p_data;
    mutex.unlock();
    rt_readbacks_in_flight--;
    if (rt_destroying && rt_readbacks_in_flight == 0) {
        delete this;
    }
}

bool MeshValidator::step() {
    if (done) {
        return true;
    }
    legacy->process_queue(); // no-op once the batch is dispatched
    legacy->get_results(legacy_state);

    for (int i = 0; i < PAGES_PER_FRAME && !pages_to_request.empty(); i++) {
        const int page = pages_to_request.back();
        pages_to_request.pop_back();
        pool->get_device()->post([this, page]() { _rt_read_page(page); });
    }

    mutex.lock();
    const bool pages_ready = (int)page_data.size() == pages_requested;
    mutex.unlock();
    const bool legacy_ready = legacy_state.loaded_chunk.size() == targets.size();
    if (!pages_ready || !legacy_ready) {
        if (now_usec() - start_usec > 30 * 1000000ULL) {
            report["error"] = ov::format("timeout: pages %d/%d legacy %d/%d", (int64_t)page_data.size(), pages_requested,
                    (int64_t)legacy_state.loaded_chunk.size(), (int64_t)targets.size());
            report["ok"] = false;
            _release_legacy();
            done = true;
        }
        return done;
    }
    _compare();
    _release_legacy();
    done = true;
    return true;
}

void MeshValidator::_release_legacy() {
    if (!legacy) {
        return;
    }
    for (auto &kv : legacy_state.loaded_chunk) {
        kv.second.free();
    }
    legacy_state.loaded_chunk.clear();
    legacy->stop();
    delete legacy;
    legacy = nullptr;
}

void MeshValidator::destroy() {
    _release_legacy();
    // FIFO: page reads posted by step() run before _rt_destroy and are counted.
    pool->get_device()->post([this]() { _rt_destroy(); });
}

void MeshValidator::_rt_destroy() {
    rt_destroying = true;
    if (rt_readbacks_in_flight == 0) {
        delete this;
    }
}

MeshValidator::~MeshValidator() {
    _release_legacy();
}

void MeshValidator::_compare() {
    int64_t legacy_tris_total = 0;
    int64_t direct_tris_total = 0;
    int64_t matched = 0;
    int64_t winding_flipped = 0;
    int64_t legacy_unmatched = 0;
    int64_t direct_unmatched = 0;
    int64_t ambiguous_slivers = 0; // stranded by the greedy matching
    int64_t ambiguous_flips = 0; // matched to a flipped twin only
    double max_pos_err_cells = 0.0;
    double max_normal_err_deg = 0.0;
    double sum_normal_err_deg = 0.0;
    int64_t normal_samples = 0;
    int64_t normals_over_5deg = 0;
    int chunks_exact = 0;
    ov::Value per_chunk = ov::Value::array();
    ov::Value anomalies = ov::Value::array(); // flipped / unmatched / bad-normal triangles with their nearest |sdf|
    std::vector<double> baseline_sdf; // same measure on ordinary matched triangles
    int64_t unexplained = 0;
    int64_t sliver_normals_skipped = 0; // matched sliver triangles: normal partners are ambiguous
    int64_t bad_headers = 0; // groups of a chunk whose header names another chunk or palette block
    int64_t material_samples = 0, material_records = 0, material_mismatch = 0; // per paired vertex
    auto record = [&](const char *p_kind, const ChunkId &p_chunk, const Tri &p_tri, float p_cell, double p_extra) {
        const Float3 c = centroid(p_tri.p);
        const double near_zero = min_abs_sdf_cells(sdf, p_chunk, c, chunk_param, p_cell);
        unexplained += near_zero > 1e-3 ? 1 : 0;
        if (anomalies.size() >= 40) {
            return; // keep the report small; every anomaly is still classified above
        }
        ov::Value a = ov::Value::object();
        a["kind"] = p_kind;
        a["chunk"] = WorldCoord::to_string(p_chunk);
        const Float3 at = ChunkMath::grid_to_render(p_chunk, c, chunk_param);
        a["centroid"] = ov::format("(%s, %s, %s)", at.x, at.y, at.z);
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
        const ov::ivec3 chunk_lo = WorldCoord::lo32_i(t.chunk_id);

        // --- direct triangles from the page texture: 4 corners per quad, triangles A,B,D / A,D,C
        std::vector<Tri> direct;
        {
            mutex.lock();
            const ov::bytes bytes = page_data[t.page];
            mutex.unlock();
            const uint16_t *page = reinterpret_cast<const uint16_t *>(bytes.data());
            const uint32_t rows = bytes.size() / PageFormat::ROW_BYTES;
            const float scale = PageFormat::pos_scale(chunk_param.chunk_size);
            const std::vector<uint32_t> &palette = *t.palette.palette;
            for (uint32_t q = t.offset; q < t.offset + t.count && q / PageFormat::QUAD_GROUP < rows; q++) {
                if (q % PageFormat::QUAD_GROUP == 0) {
                    const uint16_t *row = PageFormat::row(page, q / PageFormat::QUAD_GROUP);
                    if (PageFormat::header_chunk(row) != chunk_lo || PageFormat::header_palette(row) != t.palette.offset) {
                        bad_headers++;
                    }
                }
                Float3 p[4];
                Float3 n[4];
                uint32_t m[4];
                for (int k = 0; k < 4; k++) {
                    const uint16_t *c = PageFormat::corner(page, q, k);
                    p[k] = PageFormat::decode_local(c, scale);
                    n[k] = PageFormat::is_zero(c) ? Float3() : PageFormat::decode_normal(c);
                    const uint32_t index = PageFormat::decode_palette(c);
                    m[k] = index < palette.size() ? palette[index] : UINT32_MAX; // out of the palette: never matches
                }
                direct.push_back({ { p[0], p[1], p[3] }, { n[0], n[1], n[3] }, { m[0], m[1], m[3] } });
                direct.push_back({ { p[0], p[3], p[2] }, { n[0], n[3], n[2] }, { m[0], m[3], m[2] } });
            }
        }

        // --- legacy triangles: the mesh vertices already are vertex-grid coordinates
        std::vector<Tri> legacy_tris;
        const ChunkOutput *legacy_out = legacy_state.loaded_chunk.getptr(t.chunk_id);
        ChunkMesh mesh;
        if (legacy_out != nullptr && legacy_host->read_mesh(legacy->mesh_of(*legacy_out), mesh)) {
            const std::vector<Float3> &v = mesh.points;
            const std::vector<Float3> &vn = mesh.normals;
            const std::vector<int32_t> &idx = mesh.indices;
            const std::vector<float> &vm = mesh.materials; // GPUMesher: the global material
            for (size_t i = 0; i + 2 < idx.size(); i += 3) {
                Tri tri;
                for (int k = 0; k < 3; k++) {
                    tri.p[k] = v[idx[i + k]];
                    tri.n[k] = vn.size() == v.size() ? normalized(vn[idx[i + k]]) : Float3();
                    tri.m[k] = vm.size() == v.size() ? uint32_t(vm[idx[i + k]]) : UINT32_MAX - 1;
                }
                legacy_tris.push_back(tri);
            }
        }

        // --- match: spatial hash of direct triangles by centroid
        const float bucket = tol * 8.0f;
        using Key = std::array<int, 3>;
        auto key = [&](const Float3 &c) { return Key{ (int)std::floor(c.x / bucket), (int)std::floor(c.y / bucket), (int)std::floor(c.z / bucket) }; };
        std::map<Key, std::vector<int>> grid;
        for (int i = 0; i < (int)direct.size(); i++) {
            grid[key(centroid(direct[i].p))].push_back(i);
        }
        // A quad smaller than the tolerance matches any triangle of its neighbourhood, so the greedy
        // one-to-one pass below can strand a triangle whose partner was taken by a neighbour, or leave
        // it only its flipped twin. Such a sliver (every edge within 2 * tol) is accepted if some
        // triangle of the other side, used or not, lies within the tolerance (in the same winding, for
        // a flipped match).
        auto sliver = [&](const Tri &p_t) {
            return distance(p_t.p[0], p_t.p[1]) <= 2.0f * tol && distance(p_t.p[1], p_t.p[2]) <= 2.0f * tol && distance(p_t.p[2], p_t.p[0]) <= 2.0f * tol;
        };
        auto near_any = [&](const Tri &p_t, const std::vector<Tri> &p_others, bool p_same_winding) {
            const Float3 c = centroid(p_t.p);
            for (const Tri &o : p_others) {
                if (distance(centroid(o.p), c) > 3.0f * tol) {
                    continue;
                }
                for (int flip = 0; flip < (p_same_winding ? 1 : 2); flip++) {
                    for (int rot = 0; rot < 3; rot++) {
                        double err = 0.0;
                        for (int k = 0; k < 3; k++) {
                            const int ok = flip ? (rot + 3 - k) % 3 : (rot + k) % 3;
                            err = ov::max(err, (double)distance(p_t.p[k], o.p[ok]));
                        }
                        if (err <= tol) {
                            return true;
                        }
                    }
                }
            }
            return false;
        };
        int64_t chunk_ambiguous = 0; // stranded slivers, not in chunk_matched
        int64_t chunk_ambiguous_flips = 0; // matched, flipped twin only
        std::vector<bool> direct_used(direct.size(), false);
        int64_t chunk_matched = 0;
        int64_t chunk_flipped = 0;
        std::vector<int> legacy_left; // unmatched after the one-to-one pass
        for (int li = 0; li < (int)legacy_tris.size(); li++) {
            const Tri &l = legacy_tris[li];
            const Key k0 = key(centroid(l.p));
            int best = -1;
            bool best_flipped = false;
            int best_rot = 0;
            double best_err = 1e30;
            for (int dx = -1; dx <= 1 && best < 0; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dz = -1; dz <= 1; dz++) {
                        const auto cell_list = grid.find(Key{ k0[0] + dx, k0[1] + dy, k0[2] + dz });
                        if (cell_list == grid.end()) {
                            continue;
                        }
                        for (int di : cell_list->second) {
                            if (direct_used[di]) {
                                continue;
                            }
                            const Tri &d = direct[di];
                            for (int flip = 0; flip < 2; flip++) {
                                for (int rot = 0; rot < 3; rot++) {
                                    double err = 0.0;
                                    for (int k = 0; k < 3; k++) {
                                        const int dk = flip ? (rot + 3 - k) % 3 : (rot + k) % 3;
                                        err = ov::max(err, (double)distance(l.p[k], d.p[dk]));
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
                legacy_left.push_back(li);
                continue;
            }
            direct_used[best] = true;
            chunk_matched++;
            // A sliver whose same-winding partner was taken by a neighbour is not a flip.
            const bool ambiguous_flip = best_flipped && sliver(l) && near_any(l, direct, true);
            chunk_ambiguous_flips += ambiguous_flip ? 1 : 0;
            const bool flipped = best_flipped && !ambiguous_flip;
            chunk_flipped += flipped ? 1 : 0;
            if (flipped) {
                record("winding_flipped", t.chunk_id, l, cell, 0.0);
            } else if (chunk_matched % 4000 == 1 && baseline_sdf.size() < 200) {
                baseline_sdf.push_back(min_abs_sdf_cells(sdf, t.chunk_id, centroid(l.p), chunk_param, cell));
            }
            max_pos_err_cells = ov::max(max_pos_err_cells, best_err);
            const Tri &d = direct[best];
            // A sliver's partner is any triangle of its cluster within the tolerance (e.g. the 6 quads
            // closing around a lone corner whose |sdf| is a few 1e-3 cells, all within ~0.01 cells of
            // it): its vertices can't be paired, so its normals are not compared.
            if (sliver(l)) {
                sliver_normals_skipped++;
                continue;
            }
            for (int k = 0; k < 3; k++) {
                const int dk = best_flipped ? (best_rot + 3 - k) % 3 : (best_rot + k) % 3;
                // The same vertex (cell) on both sides: the same samples, the same vertex_material rule.
                material_samples++;
                material_records += l.m[k] != Material::GROUND ? 1 : 0;
                if (l.m[k] != d.m[dk]) {
                    material_mismatch++;
                    if (material_mismatch <= 8) {
                        record("material_mismatch", t.chunk_id, l, cell, double(d.m[dk]));
                    }
                }
                if (l.n[k] == Float3() || d.n[dk] == Float3()) {
                    continue;
                }
                const double ang = ov::rad_to_deg(std::acos(ov::clamp(dot(l.n[k], d.n[dk]), -1.0f, 1.0f)));
                max_normal_err_deg = ov::max(max_normal_err_deg, ang);
                sum_normal_err_deg += ang;
                normal_samples++;
                normals_over_5deg += ang > 5.0 ? 1 : 0;
                if (ang > 20.0) {
                    record("normal_over_20deg", t.chunk_id, l, cell, ang);
                }
            }
        }
        int64_t chunk_legacy_unmatched = 0;
        for (int li : legacy_left) {
            const Tri &l = legacy_tris[li];
            if (sliver(l) && near_any(l, direct, false)) {
                chunk_ambiguous++;
                continue;
            }
            chunk_legacy_unmatched++;
            record("legacy_unmatched", t.chunk_id, l, cell, 0.0);
        }
        int64_t chunk_direct_unmatched = 0;
        for (int di = 0; di < (int)direct.size(); di++) {
            if (!direct_used[di] && !(sliver(direct[di]) && near_any(direct[di], legacy_tris, false))) {
                chunk_direct_unmatched++;
            }
        }
        legacy_unmatched += chunk_legacy_unmatched;
        ambiguous_slivers += chunk_ambiguous;
        ambiguous_flips += chunk_ambiguous_flips;

        legacy_tris_total += legacy_tris.size();
        direct_tris_total += direct.size();
        matched += chunk_matched;
        winding_flipped += chunk_flipped;
        direct_unmatched += chunk_direct_unmatched;
        const bool exact = chunk_legacy_unmatched == 0 && chunk_direct_unmatched == 0 && chunk_flipped == 0;
        chunks_exact += exact ? 1 : 0;

        if (!exact) {
            ov::Value c = ov::Value::object();
            c["chunk"] = WorldCoord::to_string(t.chunk_id);
            c["lod"] = ChunkMath::get_lod(t.chunk_id);
            c["legacy_tris"] = (int64_t)legacy_tris.size();
            c["direct_tris"] = (int64_t)direct.size();
            c["legacy_unmatched"] = chunk_legacy_unmatched;
            c["direct_unmatched"] = chunk_direct_unmatched;
            c["winding_flipped"] = chunk_flipped;
            if (!dumped && !dump_dir.empty()) {
                const ov::str base = ov::path_join(dump_dir, ov::format("chunk_%d_%d_%d", t.chunk_id.x, t.chunk_id.y, t.chunk_id.z));
                write_obj(base + "_legacy.obj", legacy_tris, t.chunk_id, chunk_param);
                write_obj(base + "_direct.obj", direct, t.chunk_id, chunk_param);
                c["dumped"] = base;
                dumped = true;
            }
            per_chunk.push_back(c);
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
    report["ambiguous_slivers"] = ambiguous_slivers;
    report["ambiguous_sliver_flips"] = ambiguous_flips;
    report["max_position_error_cells"] = max_pos_err_cells;
    report["max_normal_error_deg"] = max_normal_err_deg;
    report["mean_normal_error_deg"] = normal_samples ? sum_normal_err_deg / normal_samples : 0.0;
    report["normals_over_5deg"] = normals_over_5deg;
    report["sliver_normals_skipped"] = sliver_normals_skipped;
    report["mismatching_chunks"] = per_chunk;
    report["anomalies"] = anomalies;
    {
        std::vector<double> sorted = baseline_sdf;
        std::sort(sorted.begin(), sorted.end());
        ov::Value b = ov::Value::object();
        b["n"] = int64_t(sorted.size());
        b["min"] = sorted.empty() ? 0.0 : sorted[0];
        b["p10"] = sorted.empty() ? 0.0 : sorted[sorted.size() / 10];
        b["median"] = sorted.empty() ? 0.0 : sorted[sorted.size() / 2];
        report["baseline_min_abs_sdf_cells"] = b;
    }
    report["match_ratio"] = legacy_tris_total ? (double)(matched + ambiguous_slivers) / legacy_tris_total : 1.0;
    // Legacy and direct evaluate the same SDF with differently compiled shaders on different devices.
    // The terrain SDF is bit-exact (exact_noise.hpp, `precise`), so its signs agree; a debug shape that
    // is not (test rig) may still disagree on the sign of a corner where |sdf| ~ 0 (winding flip, edge
    // crossing present on one side only). Such differences are "explained"; anything next to a
    // clearly non-zero corner is a real bug.
    report["unexplained_anomalies"] = unexplained;
    report["bad_group_headers"] = bad_headers;
    report["material_samples"] = material_samples;
    report["material_records"] = material_records; // paired vertices whose material is not the ground
    report["material_mismatch"] = material_mismatch;
    report["ok"] = legacy_tris_total > 0 && legacy_tris_total == direct_tris_total && unexplained == 0 && bad_headers == 0 && material_mismatch == 0 &&
            (double)(matched + ambiguous_slivers) / legacy_tris_total > 0.9999 && (normal_samples == 0 || sum_normal_err_deg / normal_samples < 0.5);
}
