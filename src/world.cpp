#include "world.hpp"

#include "trace.hpp"

#include "ov/executor.hpp"
#include "ov/json.hpp"
#include "ov/math.hpp"
#include "ov/on_error.hpp"

#include <cmath>
#include <map>

namespace ov {

namespace {

// A frame() section: its time goes to its stat and, while tracing, to a "terrain" span of the same name.
class Section {
public:
    Section(TimingStat &r_stat, const char *p_name) :
            stat(r_stat), span("terrain", p_name), t0(now_usec()) {}
    ~Section() { stat.add(now_usec() - t0); }
    Section(const Section &) = delete;
    Section &operator=(const Section &) = delete;

private:
    TimingStat &stat;
    const Trace::Scope span;
    const uint64_t t0;
};

// The planet's radius (Settings::planet_radius_m), none under the shapes gallery (its flat preset).
double planet_radius_m(const World::Settings &p_config) {
    return p_config.terrain_mode == 4 ? 0.0 : p_config.planet_radius_m;
}

// A planet's +y pole less the flat datum (LOD0 voxels): where the flat world's voxel 0 goes, so the poses made for its
// datum stand the same over the pole (start's default origin, scene_shift).
VoxelPos pole_lift(const BaseSurface &p_planet, double p_voxel_size) {
    return p_planet.tangent_point(0, 0) - VoxelPos(0, std::llround(BaseSurface::DATUM_M / p_voxel_size), 0);
}

// The render position (metres) of LOD0 voxel p_voxel, the render origin at p_origin, in double.
void render_m(const VoxelPos &p_voxel, const VoxelPos &p_origin, double p_voxel_size, double r_pos[3]) {
    r_pos[0] = double(p_voxel.x - p_origin.x) * p_voxel_size;
    r_pos[1] = double(p_voxel.y - p_origin.y) * p_voxel_size;
    r_pos[2] = double(p_voxel.z - p_origin.z) * p_voxel_size;
}

} // namespace

World::World() :
        sdf(std::make_unique<SDFCascade>()) {}

World::~World() {
    stop();
}

bool World::start(const Projection &p_projection, const Hosts &p_hosts, const Float3 *p_camera_pos, const Float3 *p_camera_forward,
        double p_camera_far_m) {
    ChunkParam &chunk_param = config.chunk_param;
    const int lod_numbers = config.lod_numbers;
    OV_ERR_FAIL_COND_V_MSG(started, false, "World: already started.");
    // One chunk-id unit is half a LOD0 chunk, a whole number of voxels.
    OV_ERR_FAIL_COND_V_MSG(chunk_param.chunk_voxel_dimension() % 2 != 0, false, "World: chunk_size - 2 * skirt_size must be even.");
    OV_ERR_FAIL_COND_V_MSG(lod_numbers < 1 || lod_numbers > ChunkMath::MAX_LOD_NUMBERS, false, "World: lod_numbers out of range.");
    // Both meshers need a device (GPU Direct the main one, Legacy a local one, which needs it too). Without one
    // (headless, the Compatibility renderer) the page pool would dereference null on the device's thread: the terrain
    // stays empty instead.
    OV_ERR_FAIL_COND_V_MSG(p_hosts.device == nullptr, false, "World: no GPU device (headless or the Compatibility renderer): the terrain is disabled.");
    OV_ERR_FAIL_COND_V_MSG(config.pipeline_mode == PIPELINE_GPU_DIRECT ? p_hosts.pages == nullptr : p_hosts.meshes == nullptr, false,
            "World: no host to draw the mesher's output.");
    hosts = p_hosts;

    started = true;
    start_usec = now_usec();

    chunk_param.origin_chunk = start_origin();
    if (chunk_param.origin_voxel() != config.world_origin_request) {
        VLOG(1, "World: world origin at ", WorldCoord::to_string(chunk_param.origin_voxel()), " voxels");
    }
    world_origin_chunk = chunk_param.origin_chunk;

    // One terrain, the cascade: mode 2 is its continents preset, 4 its flat preset under the shapes gallery.
    if (config.terrain_mode == 2) {
        sdf->preset = SDFCascade::Preset::CONTINENTS;
    } else if (config.terrain_mode == 4) {
        sdf->preset = SDFCascade::Preset::FLAT;
        sdf->set_scene(SDFTable::Scene::SHAPES);
    }
    sdf->set_scatter(config.archetype_instances, config.archetype_instance_area);
    sdf->set_forest(config.archetype_trees, config.archetype_tree_area);
    sdf->set_stream_sources(config.archetype_stream);
    sdf->set_stream_part_sides(config.archetype_stream_part_sides);
    sdf->set_biome_radius(config.biome_radius);
    sdf->planet_radius_m = planet_radius_m(config);
    if (!config.climate_path.empty()) {
        // Parsed and checked by configure() (an ERR there: no climate).
        const str err = Climate::load(config.climate_path, sdf->climate_data);
        if (!err.empty()) {
            OV_ERR_PRINT("World: " + err + " No climate.");
        }
    }
    sdf->set_instance_band(config.archetype_band && config.terrain_mode != 4); // the gallery: a crease test at every LOD, never eroded
    sdf->set_moment_blobs(config.archetype_moments);
    sdf->set_instance_erode(config.archetype_erode);
    sdf->set_instance_resolve(config.archetype_resolve);
    sdf->set_walk_batch_gpu(size_t(config.walk_batch));
    const float min_skirt = sdf->min_skirt_depth(chunk_param, lod_numbers);
    if (chunk_param.skirt_depth < min_skirt) {
        VLOG(1, "World: skirt_depth raised from ", chunk_param.skirt_depth, " to ", min_skirt, " voxels (the terrain's band steps)");
        chunk_param.skirt_depth = min_skirt;
    }
    sdf->configure(chunk_param, lod_numbers);
    sdf->set_walk_verify(config.walk_verify);
    if (config.walk_gpu) {
        sdf->start_field_gpu(chunk_param); // attached by sdf->poll() once its device is ready
    }

    state.set_max_lod(lod_numbers);
    state.verify = config.destroy_verify;

    chunk_discoverer = new ChunkDiscoverer();
    chunk_discoverer->set_config(chunk_param, sdf.get());
    chunk_discoverer->set_cull_mode(ChunkDiscoverer::CullMode(config.discovery_cull_mode));
    chunk_discoverer->set_cache(config.discovery_cache, config.discovery_verify);
    chunk_discoverer->set_gpu(config.discoverer_mode != 0);
    chunk_discoverer->set_compare(config.discovery_compare);
    // The pixel error is measured against the camera's vertical FOV and the 3D render's height. The view cone holds the
    // frustum: its half-angle is the half-FOV along the screen diagonal.
    const float fov_rad = p_projection.fov_rad > 0.0f ? p_projection.fov_rad : ov::deg_to_rad(75.0f);
    chunk_discoverer->setup_lod(lod_numbers, config.lod_pixel_error, p_projection.viewport_height, fov_rad, config.lod_pixel_error_back,
            ChunkMath::view_cone_half_angle(fov_rad, p_projection.aspect, ov::deg_to_rad(config.lod_view_margin_deg)),
            config.lod_pixel_errors);
    chunk_discoverer->set_view_memory(config.lod_view_memory_m);
    // The far plane is a depth along the view axis: the farthest point drawn is at the frustum's corners.
    far_reach = 1.0 / std::cos(double(ChunkMath::view_cone_half_angle(fov_rad, p_projection.aspect, 0.0f)));
    chunk_discoverer->start();

    if (config.pipeline_mode == PIPELINE_GPU_DIRECT) {
        direct_mesher = new GPUDirectMesher();
        direct_mesher->set_dual_contouring(config.dual_contouring);
        direct_mesher->initialize(sdf.get(), chunk_param, lod_numbers, hosts.device, hosts.pages, config.direct);
        direct_mesher->set_debug_view(config.debug_view);
    } else {
        gpu_mesher = new GPUMesher();
        gpu_mesher->set_dual_contouring(config.dual_contouring);
        gpu_mesher->initialize(sdf.get(), chunk_param, uint32_t(config.legacy_batch_size), lod_numbers, hosts.meshes);
        gpu_mesher->set_trim_indices(config.legacy_trim_indices);
    }
    last_dispatch_times = DispatchTimes(); // the new mesher's start from 0
    // Only for a host that draws them (binds their textures).
    if (config.atmosphere && hosts.atmosphere_textures) {
        atmosphere = new Atmosphere();
        atmosphere->start(hosts.device, sdf->get_base(), double(chunk_param.voxel_size), sdf->get_relief_m(), config.atmosphere_scale,
                hosts.atmosphere_textures);
        if (config.clouds && hosts.cloud_textures) {
            clouds = new Clouds();
            clouds->start(hosts.device, atmosphere, sdf->get_base(), double(chunk_param.voxel_size), sdf->get_relief_m(), config.atmosphere_scale,
                    hosts.cloud_textures);
            clouds->set_weather(cloud_cover, cloud_wind);
        }
    }

    if (p_camera_pos != nullptr && p_camera_forward != nullptr) {
        chunk_discoverer->set_camera(ChunkMath::chunk_at_render(*p_camera_pos, config.chunk_param), *p_camera_forward, p_camera_far_m * far_reach);
        last_camera = *p_camera_pos;
        has_camera = true;
    }
    VLOG(1, "World: started, pipeline_mode=", int(config.pipeline_mode));
    return true;
}

int World::add_instance(int p_type, const VoxelPos &p_anchor, const float *p_rotation, const float *p_params, int p_op, float p_k) {
    const int id = sdf->add_instance(p_type, p_anchor, p_rotation, p_params, p_op, p_k);
    if (id >= 0) {
        _apply_instance_edit();
    }
    return id;
}

bool World::remove_instance(int p_id) {
    const bool ok = sdf->remove_instance(p_id);
    if (ok) {
        _apply_instance_edit();
    }
    return ok;
}

bool World::move_instance(int p_id, const VoxelPos &p_anchor, const float *p_rotation) {
    const bool ok = sdf->move_instance(p_id, p_anchor, p_rotation);
    if (ok) {
        _apply_instance_edit();
    }
    return ok;
}

void World::_apply_instance_edit() {
    const uint64_t t0 = now_usec();
    const VoxelBox &box = sdf->last_edit_box();
    if (!box.valid) {
        return;
    }
    // The loaded chunks whose samples the edit can change: their box grown by the locality margin at
    // their LOD meets the edit's region.
    std::vector<ChunkId> remesh;
    for (const auto &kv : state.loaded_chunk) {
        if (sdf->edit_affects_chunk(kv.first)) {
            remesh.push_back(kv.first);
        }
    }
    if (direct_mesher) {
        direct_mesher->remesh_chunks(remesh);
    } else if (gpu_mesher) {
        gpu_mesher->remesh_chunks(remesh);
    }
    if (chunk_discoverer) {
        chunk_discoverer->invalidate(box);
    }
    destroy_dirty = true;
    edit_stats.edits++;
    edit_stats.remesh_last = int64_t(remesh.size());
    edit_stats.remesh_total += int64_t(remesh.size());
    edit_stats.apply.add(now_usec() - t0);
}

void World::_record_dispatch_times(const DispatchTimes &p_now) {
    process_stats.gate_admit.add(p_now.gate_admit - last_dispatch_times.gate_admit);
    process_stats.build_lists.add(p_now.build_lists - last_dispatch_times.build_lists);
    process_stats.record_bytes.add(p_now.record_bytes - last_dispatch_times.record_bytes);
    last_dispatch_times = p_now;
}

void World::frame(const Float3 *p_camera_pos, const Float3 *p_camera_forward, double p_camera_far_m) {
    if (!started) {
        return;
    }

    Section total(process_stats.process_total, "process");
    state.frame = exec_time;
    ov::executor().pump(); // an inline executor's queued tasks and woken workers (a threaded one runs them itself)
    {
        Section section(process_stats.stream_poll, "stream_poll");
        sdf->poll(); // an edit's rebuilt index base, the streamed keys computed since the last frame (before the meshers ask)
    }

    if (p_camera_pos != nullptr && p_camera_forward != nullptr) {
        chunk_discoverer->set_camera(ChunkMath::chunk_at_render(*p_camera_pos, config.chunk_param), *p_camera_forward, p_camera_far_m * far_reach);
        last_camera = *p_camera_pos;
        has_camera = true;
    }
    if (atmosphere && p_camera_pos != nullptr) {
        const BaseSurface::Render base = base_render();
        atmosphere->frame(*p_camera_pos, sun, base);
        if (clouds) {
            clouds->frame(*p_camera_pos, sun, base, render_origin_voxel());
        }
    }

    {
        Section section(process_stats.scan_ingest, "scan_ingest");
        std::unique_ptr<LODHashSet> new_scanned_chunk;
        std::unique_ptr<ScanDelta> scan_delta;
        if (chunk_discoverer->get_scanned_chunks(new_scanned_chunk, scan_delta, scan_view)) {
            scan_received = true;
            destroy_dirty = true;
            {
                Section apply(process_stats.scan_apply, "scan_apply");
                state.add_scan_chunk(std::move(new_scanned_chunk), *scan_delta);
            }
            if (direct_mesher) {
                direct_mesher->send_chunk(state.dirty_chunk, scan_view);
            } else {
                gpu_mesher->send_chunk(state.dirty_chunk, scan_view);
            }
        }
    }

    // Only a new scan or new results change what can be destroyed.
    if (destroy_dirty) {
        destroy_dirty = false;
        {
            Section section(process_stats.destroy_from_scan, "destroy_from_scan");
            state.update_destroy();
        }
        if (config.destroy_verify) {
            int64_t mismatches = 0;
            {
                Section section(process_stats.destroy_verify, "destroy_verify");
                mismatches = state.verify_against_full();
            }
            if (mismatches > 0) {
                OV_ERR_PRINT(ov::format("World: incremental destroy check differs from the full recompute: %d differences (frame %d).", mismatches,
                        exec_time));
            }
        }
    }

    {
        Section section(process_stats.mesher_dispatch, "mesher_dispatch");
        if (direct_mesher) {
            direct_mesher->process_queue();
        } else {
            gpu_mesher->process_queue();
        }
    }
    _record_dispatch_times(direct_mesher ? direct_mesher->get_dispatch_times() : gpu_mesher->get_dispatch_times());

    {
        Section section(process_stats.mesher_collect, "mesher_collect");
        const bool results = direct_mesher ? direct_mesher->get_results(state) : gpu_mesher->get_results(state);
        destroy_dirty = destroy_dirty || results;
    }

    {
        Section section(process_stats.destroy_chunks, "destroy_chunks");
        state.destroy_marked();
    }

    // After destruction (to_destroy_chunk holds copies of the outputs it moves), and never while the
    // validator or auditor hold page ranges.
    {
        Section section(process_stats.compact, "compact");
        if (direct_mesher && !validator && !auditor) {
            direct_mesher->compact(state);
        }
    }

    // The stalled chunks hide the sub-cubes their replacements cover. After compaction: the rows are current.
    {
        Section section(process_stats.coverage, "coverage");
        state.push_coverage();
    }

    if (auditor && auditor->step()) {
        audit = auditor->get_report();
        auditor->destroy();
        auditor = nullptr;
        VLOG(1, "World: page audit: ", to_json(audit));
    }

    if (discovery_audit.running) {
        _poll_discovery_audit();
    }

    if (validator && validator->step()) {
        validation = validator->get_report();
        _free_validator();
        VLOG(1, "World: validation report: ", to_json(validation));
    }

    exec_time++;

    if (frames_until_loaded < 0 && is_load_complete(has_camera ? &last_camera : nullptr)) {
        frames_until_loaded = exec_time;
        usec_until_loaded = int64_t(now_usec() - start_usec);
        VLOG(1, "World: load complete after ", exec_time, " frames, ", double(usec_until_loaded) / 1000.0, " ms");
    }
}

void World::_free_validator() {
    validator->destroy(); // its legacy mesher stops now (its meshes freed), the object goes on the device's thread
    validator = nullptr;
    validator_host.reset();
}

bool World::debug_validate(int p_count, const str &p_dump_dir) {
    OV_ERR_FAIL_COND_V_MSG(!direct_mesher, false, "debug_validate needs pipeline_mode = GPU Direct.");
    OV_ERR_FAIL_COND_V_MSG(validator, false, "A validation is already running.");
    OV_ERR_FAIL_COND_V_MSG(!hosts.hidden_meshes, false, "debug_validate needs a host for the legacy meshes.");
    std::vector<MeshValidator::Target> candidates;
    for (const auto &kv : state.loaded_chunk) {
        if (kv.second.page >= 0) {
            MeshValidator::Target t;
            t.chunk_id = kv.first;
            t.page = kv.second.page;
            t.offset = kv.second.quad_offset;
            t.count = kv.second.quad_count;
            t.palette = kv.second.palette;
            candidates.push_back(t);
        }
    }
    OV_ERR_FAIL_COND_V_MSG(candidates.empty(), false, "No non-empty loaded chunk to validate.");
    // (hash map order is arbitrary but stable enough for a spread of LODs)
    const std::vector<MeshValidator::Target> targets = MeshValidator::pick_targets(candidates, p_count);
    validation = ov::Value::object();
    validator_host = hosts.hidden_meshes();
    validator = new MeshValidator();
    validator->start(targets, sdf.get(), config.chunk_param, config.lod_numbers, direct_mesher->get_pool(), validator_host.get(), p_dump_dir,
            config.dual_contouring);
    return true;
}

bool World::debug_audit_pages() {
    OV_ERR_FAIL_COND_V_MSG(!direct_mesher, false, "debug_audit_pages needs pipeline_mode = GPU Direct.");
    OV_ERR_FAIL_COND_V_MSG(auditor, false, "An audit is already running.");
    std::map<int, size_t> index_of_page;
    std::vector<PageAuditor::PageInfo> pages;
    for (const auto &kv : state.loaded_chunk) {
        if (kv.second.page < 0) {
            continue;
        }
        if (index_of_page.count(kv.second.page) == 0) {
            index_of_page[kv.second.page] = pages.size();
            PageAuditor::PageInfo info;
            info.page = kv.second.page;
            pages.push_back(info);
        }
        pages[index_of_page[kv.second.page]].live.push_back({ kv.second.quad_offset, kv.second.quad_count, kv.first, kv.second.palette });
    }
    audit = ov::Value::object();
    auditor = new PageAuditor();
    auditor->start(direct_mesher->get_pool(), pages, config.chunk_param);
    return true;
}

bool World::debug_audit_discovery() {
    OV_ERR_FAIL_COND_V_MSG(!direct_mesher, false, "debug_audit_discovery needs pipeline_mode = GPU Direct.");
    OV_ERR_FAIL_COND_V_MSG(discovery_audit.running, false, "A discovery audit is already running.");
    OV_ERR_FAIL_COND_V_MSG(!scan_received, false, "No scan received yet.");

    const uint64_t t0 = now_usec();
    const LODHashSet &scanned = *state.scanned_chunk;
    LODHashSet reference;
    ChunkDiscoverer::ScanCounters counters;
    chunk_discoverer->scan(scan_view, ChunkDiscoverer::CULL_NONE, reference, counters);

    int64_t scanned_not_in_ref = 0;
    for (const ChunkId &id : scanned) {
        if (!reference.has(id)) {
            scanned_not_in_ref++;
        }
    }
    std::vector<ChunkId> culled;
    for (const ChunkId &id : reference) {
        if (!scanned.has(id)) {
            culled.push_back(id);
        }
    }

    // LOD difference across chunk faces (the skirt hides a jump of 1).
    std::vector<int64_t> faces;
    face_lod_jumps(scanned, chunk_discoverer->get_max_lod(), faces);
    int max_jump = 0;
    ov::Value jumps = ov::Value::object(); // jump -> faces
    for (int j = 0; j < int(faces.size()); j++) {
        if (faces[j] > 0) {
            jumps[itos(j)] = faces[j];
            max_jump = j;
        }
    }

    discovery_audit = DiscoveryAudit();
    discovery_audit.running = true;
    discovery_audit.start_usec = t0;
    discovery_audit.expected = int64_t(culled.size());
    ov::Value &base = discovery_audit.base;
    base["camera_chunk"] = WorldCoord::to_string(scan_view.camera_chunk);
    base["cull_mode"] = config.discovery_cull_mode;
    base["reference"] = int64_t(reference.size());
    base["reference_nodes_visited"] = counters.nodes_visited;
    base["scanned"] = int64_t(scanned.size());
    base["culled"] = int64_t(culled.size());
    base["scanned_not_in_reference"] = scanned_not_in_ref;
    base["max_lod_jump"] = max_jump;
    base["lod_jump_faces"] = jumps;
    base["reference_ms"] = double(now_usec() - t0) / 1000.0;
    discovery_audit_result = ov::Value::object();

    direct_mesher->audit_count_chunks(culled);
    VLOG(1, "World: discovery audit started: reference ", int64_t(reference.size()), " chunks, scanned ", int64_t(scanned.size()), ", checking ",
            int64_t(culled.size()), " culled chunks");
    return true;
}

void World::_poll_discovery_audit() {
    std::vector<std::pair<ChunkId, uint32_t>> counts;
    direct_mesher->take_audit_counts(counts);
    for (const std::pair<ChunkId, uint32_t> &c : counts) {
        discovery_audit.checked++;
        if (c.second == 0) {
            continue;
        }
        discovery_audit.misses++;
        discovery_audit.miss_quads += c.second;
        const int lod = ChunkMath::get_lod(c.first);
        ov::Value &by_lod = discovery_audit.misses_by_lod[itos(lod)];
        by_lod = by_lod.as_int() + 1;
        if (discovery_audit.miss_examples.size() < 20) {
            ov::Value e = ov::Value::object();
            e["id"] = WorldCoord::to_string(c.first);
            e["lod"] = lod;
            e["quads"] = int64_t(c.second);
            e["distance_chunks"] = scan_view.camera_chunk.distance_to(c.first);
            discovery_audit.miss_examples.push_back(e);
        }
    }
    if (!direct_mesher->is_audit_idle()) {
        return;
    }
    ov::Value r = discovery_audit.base;
    r["checked"] = discovery_audit.checked;
    r["misses"] = discovery_audit.misses;
    r["miss_quads"] = discovery_audit.miss_quads;
    r["misses_by_lod"] = discovery_audit.misses_by_lod;
    r["miss_examples"] = discovery_audit.miss_examples;
    r["ms"] = double(now_usec() - discovery_audit.start_usec) / 1000.0;
    r["ok"] = discovery_audit.misses == 0 && discovery_audit.checked == discovery_audit.expected && r["scanned_not_in_reference"].as_int() == 0;
    discovery_audit_result = r;
    discovery_audit.running = false;
    VLOG(1, "World: discovery audit: ", to_json(r));
}

bool World::is_load_complete(const Float3 *p_camera_pos) {
    if (!started || !scan_received || !state.dirty_chunk.is_empty()) {
        return false;
    }
    // The received scan must be the camera's current one: a camera moved (bench --above_ground) or turned
    // after start is otherwise "loaded" with the first scan while the next one is still running.
    // The view of the last set_camera (frame) carries the remembered views; a camera moved since is
    // caught by its chunk, a turn since by the next frame.
    if (p_camera_pos != nullptr && (ChunkDiscoverer::view_changed(scan_view, chunk_discoverer->get_view()) || ChunkMath::chunk_at_render(*p_camera_pos, config.chunk_param) != scan_view.camera_chunk)) {
        return false;
    }
    if (direct_mesher) {
        return direct_mesher->is_idle();
    }
    return gpu_mesher->is_idle();
}

bool World::is_edit_settled() {
    if (!started) {
        return false;
    }
    if (direct_mesher) {
        return direct_mesher->is_edit_settled();
    }
    return gpu_mesher->is_idle();
}

double World::load_radius() const {
    if (chunk_discoverer == nullptr) {
        return 0.0;
    }
    // The camera is within one step of its top-LOD ancestor, the grid spans `count` steps on each
    // side of it and each top chunk reaches one step (its half extent) past its centre.
    int64_t step;
    int64_t count;
    chunk_discoverer->get_scan_grid(step, count);
    const double ids = double((count + 2) * step) * std::sqrt(3.0);
    return ids * config.chunk_param.id_unit_m();
}

double World::sdf_at(const double p_render_pos[3]) const {
    WorldPos p;
    p.voxel = voxel_at(p_render_pos); // the cascade only exists on lattice points: the LOD0 voxel corner (within one voxel)
    return sdf->evaluate_view(p, 0); // called every frame: streamed instances come once resident
}

VoxelPos World::voxel_at(const double p_render_pos[3]) const {
    const double vs = double(config.chunk_param.voxel_size);
    const double offset[3] = { p_render_pos[0] / vs, p_render_pos[1] / vs, p_render_pos[2] / vs };
    return WorldCoord::offset_pos(config.chunk_param.origin_voxel(), offset).voxel;
}

void World::tangent_position(double p_u_m, double p_v_m, double r_render_pos[3]) const {
    const double vs = double(config.chunk_param.voxel_size);
    render_m(sdf->get_base().tangent_point(std::llround(p_u_m / vs), std::llround(p_v_m / vs)), config.chunk_param.origin_voxel(), vs, r_render_pos);
}

ChunkId World::start_origin() const {
    const ChunkParam &chunk_param = config.chunk_param;
    const int64_t root_units = int64_t(1) << config.lod_numbers;
    const int64_t root_voxels = root_units * chunk_param.half_chunk_voxels();
    VoxelPos request = config.world_origin_request;
    if (const BaseSurface planet = settings_planet(); planet.kind() == BaseSurface::Kind::SPHERE && request == VoxelPos(0, 0, 0)) {
        request = pole_lift(planet, double(chunk_param.voxel_size));
    }
    return ChunkId(WorldCoord::floor_div(request.x, root_voxels) * root_units, WorldCoord::floor_div(request.y, root_voxels) * root_units,
            WorldCoord::floor_div(request.z, root_voxels) * root_units);
}

BaseSurface World::settings_planet() const {
    return planet_radius_m(config) > 0.0 ? SDFCascade::planet_base(planet_radius_m(config), double(config.chunk_param.voxel_size)) : BaseSurface();
}

VoxelPos World::render_origin_voxel() const {
    ChunkParam param = config.chunk_param;
    if (!started) {
        param.origin_chunk = start_origin();
    }
    return param.origin_voxel();
}

void World::scene_shift(double r_shift_m[3]) const {
    const BaseSurface planet = settings_planet();
    const VoxelPos origin = render_origin_voxel();
    render_m(planet.kind() == BaseSurface::Kind::SPHERE ? pole_lift(planet, double(config.chunk_param.voxel_size)) : origin, origin,
            double(config.chunk_param.voxel_size), r_shift_m);
}

UpAxis World::up_at(const double p_render_pos[3]) const {
    return sdf->up_at(voxel_at(p_render_pos));
}

BaseSurface::Render World::base_render() const {
    return sdf->get_base().render(config.chunk_param.origin_voxel(), double(config.chunk_param.voxel_size));
}

void World::set_cloud_weather(float p_cover, const Float3 &p_wind_mps) {
    cloud_cover = p_cover;
    cloud_wind = p_wind_mps;
    if (clouds) {
        clouds->set_weather(p_cover, p_wind_mps);
    }
}

bool World::recenter(const Float3 &p_camera_pos, Float3 &r_shift) {
    if (!started || !(config.rebase_distance_m > 0.0f)) {
        return false;
    }
    const float d = ov::max(std::abs(p_camera_pos.x), ov::max(std::abs(p_camera_pos.y), std::abs(p_camera_pos.z)));
    if (!(d > config.rebase_distance_m)) {
        return false;
    }
    return rebase(ChunkMath::recentered_origin(p_camera_pos, config.chunk_param), r_shift);
}

bool World::rebase(const ChunkId &p_origin, Float3 &r_shift) {
    OV_ERR_FAIL_COND_V_MSG(!started, false, "World: rebase before start.");
    OV_ERR_FAIL_COND_V_MSG(!ChunkMath::is_render_origin(p_origin), false, "World: a render origin is an even chunk id.");
    // The validator, the auditor and the discovery audit hold render positions and page ranges from their start.
    if (validator || auditor || discovery_audit.running) {
        return false;
    }
    ChunkParam &chunk_param = config.chunk_param;
    r_shift = ChunkMath::origin_shift_m(chunk_param.origin_chunk, p_origin, chunk_param);
    chunk_param.origin_chunk = p_origin;
    if (direct_mesher) {
        direct_mesher->set_render_origin(p_origin);
    } else {
        gpu_mesher->set_render_origin(p_origin);
    }
    last_camera = last_camera - r_shift;
    if (clouds) {
        clouds->shift(r_shift);
    }
    rebases++;
    VLOG(1, "World: render origin moved to chunk ", WorldCoord::to_string(p_origin), ", render positions by -",
            ov::format("%.1f, %.1f, %.1f", r_shift.x, r_shift.y, r_shift.z), " m");
    return true;
}

void World::set_debug_view(int p_view) {
    config.debug_view = p_view;
    if (direct_mesher) {
        direct_mesher->set_debug_view(p_view);
    }
}

ov::Value World::process_times_ms() const {
    ov::Value d = ov::Value::object();
    for (const auto &[name, stat] : process_stats.sections()) {
        d[name] = double(stat->total) / 1000.0;
    }
    return d;
}

ov::Value World::debug_stats() {
    ov::Value d = ov::Value::object();
    if (!started) {
        return d;
    }
    const ChunkParam &chunk_param = config.chunk_param;
    d["pipeline_mode"] = int(config.pipeline_mode);
    d["terrain_mode"] = config.terrain_mode;
    ov::Value origin = ov::Value::object();
    origin["requested_voxel"] = WorldCoord::to_string(config.world_origin_request);
    origin["voxel"] = WorldCoord::to_string(world_origin_chunk * chunk_param.half_chunk_voxels());
    origin["chunk"] = WorldCoord::to_string(world_origin_chunk);
    origin["render_voxel"] = WorldCoord::to_string(chunk_param.origin_voxel());
    origin["render_chunk"] = WorldCoord::to_string(chunk_param.origin_chunk);
    origin["rebases"] = rebases;
    d["world_origin"] = origin;
    ov::Value table = sdf->get_gpu_table()->get_stats();
    const ov::Value walk = sdf->walk_gpu_stats();
    if (!walk.is_empty()) {
        table["walk_gpu"] = walk;
    }
    d["table"] = table;
    if (const Climate::Table *climate = sdf->get_climate()) {
        ov::Value c = ov::Value::object();
        c["path"] = config.climate_path;
        c["fields"] = int64_t(climate->fields.size());
        c["biomes"] = int64_t(climate->biomes.size());
        c["modulated"] = sdf->get_cascade()->modulation != nullptr;
        d["climate"] = c;
    }
    d["frames"] = int64_t(exec_time);
    d["load_complete"] = is_load_complete(has_camera ? &last_camera : nullptr);
    d["frames_until_loaded"] = frames_until_loaded;
    d["ms_until_loaded"] = usec_until_loaded < 0 ? -1.0 : double(usec_until_loaded) / 1000.0;

    int64_t live_triangles = 0;
    int64_t drawn_triangles = 0;
    int64_t non_empty_chunks = 0;
    for (const auto &kv : state.loaded_chunk) {
        live_triangles += kv.second.triangle_count;
        drawn_triangles += kv.second.drawn_triangles;
        non_empty_chunks += kv.second.triangle_count > 0 ? 1 : 0;
    }

    ov::Value chunks = ov::Value::object();
    chunks["scanned"] = int64_t(state.scanned_chunk->size());
    chunks["loaded"] = int64_t(state.loaded_chunk.size());
    chunks["loaded_non_empty"] = non_empty_chunks;
    chunks["pending"] = int64_t(state.dirty_chunk.size());
    // Settle check: once the load is complete both must be 0 (a scanned chunk not loaded is a hole).
    int64_t loaded_not_scanned = 0;
    int64_t scanned_not_loaded = 0;
    for (const auto &kv : state.loaded_chunk) {
        loaded_not_scanned += state.scanned_chunk->has(kv.first) ? 0 : 1;
    }
    for (const ChunkId &id : *state.scanned_chunk) {
        scanned_not_loaded += state.loaded_chunk.has(id) ? 0 : 1;
    }
    chunks["loaded_not_scanned"] = loaded_not_scanned;
    chunks["scanned_not_loaded"] = scanned_not_loaded;
    chunks["live_triangles"] = live_triangles;
    chunks["drawn_triangles"] = drawn_triangles;
    d["chunks"] = chunks;

    ov::Value process = ov::Value::object();
    for (const auto &[name, stat] : process_stats.sections()) {
        process[name] = stat->to_value();
    }
    d["process"] = process;

    const VoxelState::Stats &vs = state.stats;
    ov::Value destroy = ov::Value::object();
    destroy["leaving"] = int64_t(state.leaving.size());
    destroy["leaving_max"] = vs.leaving_max;
    destroy["stalled"] = int64_t(state.stalled.size());
    destroy["rescans"] = vs.rescans;
    destroy["rescan_reanchored"] = vs.rescan_reanchored;
    destroy["rescan_ms_total"] = double(vs.rescan_usec) / 1000.0;
    destroy["verify"] = config.destroy_verify;
    destroy["verify_checks"] = vs.verify_checks;
    destroy["verify_mismatches"] = vs.verify_mismatches;
    destroy["delta_mismatches"] = vs.delta_mismatches;
    destroy["octree_violations"] = vs.octree_violations;
    destroy["coverage_hidden_subcubes"] = state.hidden_subcubes(); // now
    destroy["coverage_written"] = vs.coverage_written;
    destroy["coverage_written_max"] = vs.coverage_written_max;
    // Frames a stalled parent stays drawn over its children (first stalled sub-cube to the last cleared), cumulative.
    destroy["handoff_frames"] = to_value(vs.handoff_frames);
    d["destroy"] = destroy;

    d["chunk_rids_alive"] = DebugLog::chunk_rids_created.load() - DebugLog::chunk_rids_freed.load();
    d["discoverer"] = chunk_discoverer->get_stats();
    d["mesher"] = direct_mesher ? direct_mesher->get_stats() : gpu_mesher->get_stats();

    ov::Value edits = ov::Value::object();
    edits["edits"] = edit_stats.edits;
    edits["remesh_total"] = edit_stats.remesh_total;
    edits["remesh_last"] = edit_stats.remesh_last;
    edits["apply"] = edit_stats.apply.to_value();
    d["edits"] = edits;
    return d;
}

void World::stop() {
    if (!started) {
        return;
    }
    started = false;

    // Debug tools first: the validator owns a legacy mesher, and both hold pages of the direct one.
    // A run may be in flight: they delete themselves once their readbacks have arrived.
    if (validator) {
        _free_validator();
    }
    if (auditor) {
        auditor->destroy();
        auditor = nullptr;
    }

    chunk_discoverer->stop();
    delete chunk_discoverer;
    chunk_discoverer = nullptr;
    sdf->stop_field_gpu(); // its thread frees its device; a walk still computing falls back to the CPU's field

    // Release loaded chunks first: GPU Direct chunks hand their range back to the mesher.
    for (auto &kv : state.loaded_chunk) {
        kv.second.free();
    }
    state.loaded_chunk.clear();

    if (clouds) {
        clouds->destroy(); // deletes itself on the device's thread, before the atmosphere's textures go
        clouds = nullptr;
    }
    if (atmosphere) {
        atmosphere->destroy(); // deletes itself on the device's thread
        atmosphere = nullptr;
    }
    if (direct_mesher) {
        direct_mesher->destroy(); // deletes itself on the device's thread
        direct_mesher = nullptr;
    }
    if (gpu_mesher) {
        gpu_mesher->stop();
        delete gpu_mesher;
        gpu_mesher = nullptr;
    }

    VLOG(1, "World: teardown, chunk instances created=", DebugLog::chunk_rids_created.load(), " freed=", DebugLog::chunk_rids_freed.load());
    state.clear();
}

} // namespace ov
