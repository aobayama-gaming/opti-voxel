#pragma once

#include "archetype_gpu_table.hpp"
#include "archetype_table.hpp"
#include "biome_features.hpp"
#include "biome_pick.hpp"
#include "cell_scatter.hpp"
#include "field_gpu.hpp"
#include "instance_table.hpp"
#include "moment_layers.hpp"
#include "ov/math.hpp"
#include "ov/value.hpp"
#include "scatter_source.hpp"
#include "sdf_base.hpp"
#include "species.hpp"
#include "tree_archetype.hpp"

#include <algorithm>
#include <atomic>
#include <map>
#include <memory>

// The instance half of the terrain: every point folds the instances near it onto the ground's register
// (instance_table.hpp, archetype_table.hpp fold_onto; the ground is SDFCascade's). This class owns everything about
// the instances: the library, the records and their snapshot (read from other threads), the GPU table, the runtime
// edits (replayed from a world's edits file at configure), and the sources the instance lists place on the ground
// wherever it is, which only need the subclass's base and ground_under.
class SDFTable : public SDFBase {
public:
    static constexpr uint32_t ROCK_SEED = 0x20C4u;
    // The scatter's cell sites (cell_sites.hpp; the forest's is Archetype::TREE_SITE_SALT).
    static constexpr uint32_t SCATTER_SITE_SALT = 0x5CA77u;
    // A runtime edit as an edits file keeps it (WorldFile::parse_edits): a library type by name (TableDef::name: the
    // DefLibrary's indices are the build's, not the file's), the operator, the anchor (LOD0 voxels), the rotation's rows
    // and the parameters (voxels), the smooth radius (voxels).
    struct Edit {
        ov::str type;
        Archetype::Combine op = Archetype::Combine::UNION;
        VoxelPos anchor;
        float rotation[9] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };
        float params[8] = {};
        float k = 0.0f;
        // The same bits (the floats compared as bits).
        bool operator==(const Edit &p_o) const;
        bool operator!=(const Edit &p_o) const { return !(*this == p_o); }
    };
    // The edits configure replays after the scene (a world file's "edits", a restart's live ones), each the instance
    // add_instance makes of it: their ids come past the scene's, so they fold after the streamed instances too
    // (compose_list: a subtraction carves the streamed trees). An entry naming no type of the library, or one
    // add_instance refuses, refuses the world (configure's world error). Set before configure().
    void set_edits(const std::vector<Edit> &p_edits) { start_edits = p_edits; }
    // Whether an edit configure replays is hard CSG (k 0: the shapes gallery's), whose creases differ between LODs by up
    // to a cell when kept whole (set_instance_band(false)): SDFCascade's skirt.
    bool has_hard_edits() const {
        return std::any_of(start_edits.begin(), start_edits.end(), [](const Edit &p_e) { return p_e.k == 0.0f; });
    }
    // The live edits in id order: the replayed ones and add_instance's, as moved since, the removed ones left out (what
    // an edits file saves: WorldFile::save_edits). Records the scene placed are never edits.
    std::vector<Edit> live_edits() const;

    // A box given in metres: a square of side side_m metres around (x_m, z_m) of the base's tangent plane (in front of
    // the main_scene spawn by default), over every height the ground can reach in it (region_box: 4 half sides up and
    // down from the ground under its centre); side 0: none.
    struct Region {
        double side_m = 0.0;
        double x_m = -50.0;
        double z_m = -150.0;
    };
    // The box a whole build walks (set_stream_sources(false): the stream's oracle, the tests and checks that edit the
    // records): the sources have no end, so a whole build without one is refused (configure's world error). Unused
    // while streaming. Set before configure().
    void set_whole_region(const Region &p_region) { whole_region = p_region; }
    // Its box (region_box; once configured: the ground under its centre), invalid for none.
    VoxelBox whole_box() const { return region_box(whole_region); }
    // Where the biome features stand (a world file's "feature_region"): a world listing features without one is refused.
    // TODO(feature stream): the features are placed whole over this box because Biome::sites walks every cell of it and
    // the trees' keep-out scans every lake (in_lake). A streamed feature source keyed by feature-cell tiles, Biome::place
    // memoised per cell (a pure function of it) and the keep-out read from the memo of the cells around a site would put
    // them everywhere, like the trees, and close this region. Set before configure().
    void set_feature_region(const Region &p_region) { feature_region = p_region; }
    // What grows where no biome lists its own "instances" (a world file's "instances"; the world without biomes): each
    // entry a species, its cell and its share (Climate::InstanceShare). Every entry of a kind (trees, rocks) and a cell,
    // over these and the biomes' lists, is one source (cell_scatter.hpp) over the whole ground, each of its sites growing
    // what its biome lists there (BiomePick, pick_share). Set before configure().
    void set_instances(const std::vector<Climate::InstanceShare> &p_instances) { instances = p_instances; }
    // Every species of p_kind in p_species at share 1, about p_count sites on a square of side p_side_m metres (the cell
    // forest_cell_side gives): a scene given as a count (the tests, the checks).
    static std::vector<Climate::InstanceShare> counted(const std::vector<Archetype::Species> &p_species, Archetype::Species::Kind p_kind, int p_count,
            double p_side_m);
    // The species the instances name (a world file's "species", none without one): the trees the
    // tree sources grow, the rocks the rock sources place. Set before configure().
    void set_species(const std::vector<Archetype::Species> &p_species) { species = p_species; }
    // Whether the scatter and the forest stream (instance_stream.hpp: their keys computed as the chunks ask for them, their
    // records in GPU slots past the table's), else placed whole in the table at configure (the reference; the tests that
    // edit their records). Set before configure().
    void set_stream_sources(bool p_enabled) { stream_sources = p_enabled; }
    // The streamed sources' part layers: those whose cell is under this many source cells read the records (1 by default:
    // proxies from the source cell on; ScatterSource::PART_CELL_SIDES: the whole build's blobs). Set before configure().
    // The forest's trees, in id order (placed whole only).
    const std::vector<Archetype::TreeSpan> &get_trees() const { return trees; }
    // The procedural sources of the scene (cell_scatter.hpp), by name: the rocks' ("scatter"), then the trees' ("trees"),
    // a kind's cells past its finest named "<kind>_<cell in voxels>".
    struct Source {
        ov::str name;
        Archetype::Species::Kind kind = Archetype::Species::TREE;
        Archetype::CellScatter scatter;
    };
    const std::vector<Source> &get_sources() const { return sources; }
    // A source as a field of keys (scatter_source.hpp), with the table's library, finish, moment layers and limits.
    Archetype::ScatterSource::Setup source_setup(const Source &p_source, uint32_t p_id) const;
    // How the table finishes its records (the band rule unless set_instance_band(false), the erosion, the skirt).
    Archetype::RecordFinish record_finish() const;
    // The moment blobs a whole rebuild of the current records would make, with the joins the table was built with: what
    // the edits' incremental updates must match (tests).
    void moment_blobs_reference(std::vector<Archetype::MomentBlob> &r_out) const;
    // The features (biome_features.hpp: lava lakes, floating islands) where no biome lists its own (a world file's
    // "features"; the world without biomes): each entry a kind, its cell, the chance a cell holds one and its radius range
    // (Climate::FeatureShare). They stand in the feature region, a cell's candidate where its biome's entry of its kind
    // and cell keeps it (BiomePick). Set before configure().
    void set_features(const std::vector<Climate::FeatureShare> &p_features) { features = p_features; }
    // The lowest altitude a tree or a rock may stand at (metres above the base: a world's sea level plus its shore's
    // above_m; -INFINITY: anywhere). Set before configure().
    void set_site_floor_m(double p_altitude_m) { site_floor_m = p_altitude_m; }
    // Whether a tree or a rock may stand on LOD0 voxel p_base (the sources' accept): not in or near a lava lake (in_lake),
    // not under the site floor. On the CPU after either walk (CellScatter::accept_sites: the GPU walk only reads the
    // ground), so walk_gpu 0 and 1 keep the same sites.
    bool site_allowed(const VoxelPos &p_base) const;
    // The materials the records name (TableRecord::material): the page shader's material_tex.
    const Material::Library &get_materials() const override { return materials; }

    // The band rule for the instances (Archetype::set_band); false keeps them at every LOD (a world's "instance_band").
    // Set before configure().
    void set_instance_band(bool p_enabled) { instance_band = p_enabled; }
    bool get_instance_band() const { return instance_band; }
    // The band rule's erosion rate in cells per LOD (Archetype::set_band). Set before configure().
    void set_instance_erode(float p_cells) { erode_rule.rate = p_cells; }
    // The band rule's resolution: a part is kept while its half thickness holds this many cells (Archetype::set_band).
    void set_instance_resolve(float p_cells) { erode_rule.resolve = p_cells; }
    // The far field of the dropped instances (moment_layers.hpp; with the band rule only, default on): the blobs of their
    // moments, one per LOD layer, cell and material, placed after the scene's records. Set before configure().
    void set_moment_blobs(bool p_enabled) { moment_blobs = p_enabled; }

    // An instance erodes by at most this many cells between LOD L and L+1 (the band rule; 0 without it).
    float instance_erode_cells() const { return instance_band ? erode_rule.rate : 0.0f; }
    // The skirts must hide the band rule's step, and the moment blobs' when the world places instances that can have
    // some (places_instances: asked before configure, the blobs aren't built yet): each layer's are listed at its LOD alone, so the
    // far-field surfaces of a LOD L and a LOD L+1 chunk differ by up to MOMENT_SKIRT_CELLS cells of L (0.93 of the
    // skirt on a 4,000-rock scatter at LOD 3 | 4, the worst measured: test_cascade_table).
    static constexpr float MOMENT_SKIRT_CELLS = 2.5f;
    // The locality margins (skirt_depth) take it whole; the skirt walls only where a blob touches the sample
    // (SDFCascade's moment skirt), so the rest of the terrain keeps the band rule's.
    float min_skirt_depth(const ChunkParam &, int) const override {
        const bool blobs = instance_band && moment_blobs && places_instances();
        return instance_erode_cells() + (blobs ? MOMENT_SKIRT_CELLS : 0.0f);
    }
    // The moment skirt in cells: MOMENT_SKIRT_CELLS when the table has moment layers (edits may add blobs where the build
    // placed none; the skirt only reaches the samples a blob touches), else 0.
    float moment_skirt_cells() const { return moment_blob_type >= 0 ? MOMENT_SKIRT_CELLS : 0.0f; }

    SDFGpuTable *get_gpu_table() override { return &gpu; }
    using SDFBase::get_gpu_table; // the const form

    // The node's world box grown by the cull band (2 chunk sizes at its LOD) against every instance's box, band or
    // not: a culled coarse node hides its subtree, where the instance is (instance_table.hpp). A streamed source's records
    // stand on the ground anywhere: a node is kept while its corners are within the records' ground_reach of it.
    bool cull_exempt(const ChunkId &p_node, int p_lod, float p_past_band_m) const override;

    // The streamed sources' gating (InstanceStream::admit; every chunk READY without them).
    Admit admit(const ChunkId &p_chunk, uint64_t p_owner, uint64_t p_tag, uint64_t p_rank, uint64_t &r_words) override;
    void set_tag(uint64_t p_owner, uint64_t p_tag, const std::function<int64_t(const ChunkId &)> &p_rank,
            const std::function<void(const ChunkId &)> &p_kept) override {
        if (stream) {
            stream->set_tag(p_owner, p_tag, p_rank, p_kept);
        }
    }
    void take_ready(uint64_t p_owner, std::vector<std::pair<ChunkId, uint64_t>> &r_out) override;
    size_t parked_count(uint64_t p_owner) const override { return stream ? stream->parked_count(p_owner) : 0; }
    // poll_rebuild(), then the stream's poll.
    void poll() override;
    // The streamed sources (null when none streams).
    const Archetype::InstanceStream *get_stream() const { return stream.get(); }

    // The lockstep site walks' field on the GPU (VoxelTerrain.walk_gpu, FieldGPU). start_field_gpu after configure() (the
    // point passes need the configured terrain): its thread makes its device; poll() attaches it once ready (the stream
    // then walks WALK_BATCH_GPU keys per drain task, the sources' ground_batch reads it) and detaches it if it fails.
    // stop_field_gpu before the terrain goes. Nothing without a stream or a cascade.
    static constexpr size_t WALK_BATCH_GPU = 64;
    static constexpr uint32_t FIELD_GPU_SCRATCH_MB = 64;
    void start_field_gpu(const ChunkParam &p_param);
    void stop_field_gpu();
    // Attached now (null detaches): the walk batch follows. poll() calls it; tests attach directly.
    void attach_field_gpu(std::shared_ptr<FieldGPU> p_gpu);
    // Every batch the GPU answers read again on the CPU, ERR on a difference (debug: --walk_verify).
    void set_walk_verify(bool p_verify) { walk_verify = p_verify; }
    // ground_field at many points, bit for bit (CellScatter::ground_batch). Thread-safe.
    void ground_field_batch(const VoxelPos *p_voxels, const int *p_lods, int p_count, ExactNoise::Float4 *r_out) const;
    // The GPU field's stats (FieldGPU::get_stats) and the verify counts; empty when none was started.
    ov::Value walk_gpu_stats() const;
    // The records the CPU folds at p_voxel for a LOD-p_lod sample, in fold order (compose_list): r_list indexes the
    // returned view, valid until the next call on this thread. Any thread.
    // p_view: the streamed keys not landed yet are left out instead of computed (InstanceStream::view_records; main thread).
    Archetype::RecordView point_list(const Archetype::InstanceSnapshot &p_snap, const VoxelPos &p_voxel, int p_lod, std::vector<uint32_t> &r_list,
            bool p_view = false) const;

    // Runtime edits: rebuild the snapshot (copy on write) and bump the version; the GPU re-uploads the records when
    // it changes. last_edit_box() / edit_affects_chunk() tell the terrain what to re-mesh. p_type: a library index
    // (get_library().find(name)). An added instance is a live edit (live_edits) until removed.
    int add_instance(int p_type, const VoxelPos &p_anchor, const float *p_rotation, const float *p_params, int p_op, float p_k) override;
    bool remove_instance(int p_id) override;
    bool move_instance(int p_id, const VoxelPos &p_anchor, const float *p_rotation) override;
    // Whether a chunk's samples can be changed by the last edit (its box grown by M(lod) meets the edit's region):
    // the chunks to re-mesh.
    bool edit_affects_chunk(const ChunkId &p_chunk) const override;
    // Publishes the index base an edit started building on the pool, once it is done (main thread, every frame). The
    // lists don't change, nor the version: nothing to upload or re-mesh.
    void poll_rebuild();
    // The same, waiting for the base if it is still being built (tests).
    void finish_rebuild();

    // The current snapshot. Edits replace it on the main thread while the discoverer thread evaluates and the render
    // thread uploads: every reader takes one atomic copy and works on it.
    std::shared_ptr<const Archetype::InstanceSnapshot> current() const { return std::atomic_load(&snapshot); }
    std::shared_ptr<const Archetype::InstanceSnapshot> get_snapshot() const { return current(); }
    const Archetype::DefLibrary &get_library() const { return *library; }

    // The ground's base surface (the flat datum or a planet): its up and its tangent plane place the scenes.
    virtual const BaseSurface &get_base() const = 0;
    // The ground's top solid voxel on the line through p_start along the base's up there, instances left out: where the
    // scenes and the biome features stand (thread-safe).
    virtual VoxelPos ground_under(const VoxelPos &p_start) const = 0;
    // The ground under the point (p_u, p_v) LOD0 voxels of the base's tangent plane (BaseSurface::tangent_point): where
    // the boxes given in metres (Region) stand.
    VoxelPos ground_at(int64_t p_u, int64_t p_v) const { return ground_under(get_base().tangent_point(p_u, p_v)); }
    // Why an edit's inputs are refused (empty if they are fine): the GPU and the CPU only agree on finite, normal
    // floats, the box and the fold assume an orthonormal rotation, the table folds the four top-level operators, and
    // a smooth radius is >= 0.
    static ov::str edit_error(const float *p_rotation, const float *p_params, int p_op, float p_k);

protected:
    // The library, the scene's records, the replayed edits (set_edits) and the first snapshot, its values in LOD0
    // voxels times the voxel size (metres). Call from configure(), after SDFBase's and once ground_under works. Empty,
    // or why the world is refused (an edit refused: nothing published).
    ov::str build_table(int p_lod_count);
    // Waits for the stream's tasks and drops it: they call the subclass's ground_field, so its destructor calls this first.
    void stop_streaming();

    // The climate the scenes' sites read their biome from (BiomePick), and its program: none here, the subclass's
    // (SDFCascade: its world's climate). Valid from configure() on.
    virtual const Climate::Table *placing_climate() const { return nullptr; }
    virtual const Terrain::LevelProgram *placing_program() const { return nullptr; }
    // Empty, or the first instance entry (the world's or a biome's, of placing_climate) naming a species the world's
    // species don't have: a world error (SDFCascade::configure), never a list grown without it.
    ov::str species_error() const;
    // Whether the world lists instances or features (its own lists; the subclass adds its biomes'), asked before configure
    // (min_skirt_depth: the moment blobs' skirt).
    virtual bool places_instances() const { return !instances.empty() || !features.empty(); }

    // The base palette (SDFGpuTable::base_palette): GROUND, then p_ground's materials, registered by name (their
    // parameters replaced on a reconfigure). Call from configure(); empty: {GROUND}.
    void set_ground_materials(const std::vector<std::pair<ov::str, Material::Params>> &p_ground) {
        std::vector<uint32_t> base{ Material::GROUND };
        for (const std::pair<ov::str, Material::Params> &m : p_ground) {
            base.push_back(materials.set(m.first, m.second));
        }
        gpu.set_base_palette(base);
    }

private:
    // The library indices the scenes place.
    struct Types {
        int capsule = -1;
        int rock = -1;
        Biome::Types biome;
    };

    std::shared_ptr<const Archetype::DefLibrary> library;
    int lod_count = 24;
    uint32_t version = 0;
    Archetype::RecordStore records; // editable (paged, copy on write): each snapshot shares its pages
    std::shared_ptr<const Archetype::InstanceSnapshot> snapshot;
    Archetype::BaseRebuild rebuild; // the next index base, while the edits past REBASE go on
    Archetype::GpuTable gpu; // the meshers read and upload through get_gpu_table()
    // What the last edit changed (edit_affects_chunk): one region per changed record, its margin inputs and the LODs
    // [lod_min, lod_max) it is listed at (an instance below its max_lod, a moment blob at its layer's alone).
    struct EditRegion {
        VoxelBox box;
        float k = 0.0f;
        float margin_c = 0.0f;
        int lod_min = 0;
        int lod_max = 0;
    };
    std::vector<EditRegion> edit_regions;
    Region whole_region;
    Region feature_region;
    std::vector<Climate::InstanceShare> instances;
    std::vector<Archetype::TreeSpan> trees; // the forest's trees, in id order
    std::vector<Archetype::Species> species;
    std::vector<Source> sources;
    bool stream_sources = true; // --stream
    std::vector<std::shared_ptr<const Archetype::ScatterSource>> streamed; // the valid sources, by their id
    std::shared_ptr<Archetype::InstanceStream> stream; // the gpu table shares it
    std::shared_ptr<FieldGPU> field_gpu; // attached; read by the drain tasks: std::atomic_load / atomic_store
    std::shared_ptr<FieldGPU> field_gpu_starting; // started, not ready yet (main thread)
    bool walk_verify = false;
    mutable std::atomic<int64_t> verified_points{ 0 };
    mutable std::atomic<int64_t> verify_differences{ 0 };
    std::vector<Climate::FeatureShare> features;
    bool has_features = false; // build_table: the lists name some, their library types added
    std::vector<Biome::LavaPool> lava_pools; // keeps trees and rocks out of the lakes (in_lake)
    double site_floor_m = -INFINITY; // set_site_floor_m
    Material::Library materials;
    bool instance_band = true; // false: a world's "instance_band" (the shapes gallery), tests
    Archetype::ErodeRule erode_rule; // tests vary it
    bool moment_blobs = true; // false: tests of the records alone
    // The moment layers of the records (add_moment_blobs; an item per record id), kept up to date by the edits, and each
    // kept blob's record id by (lod, cell, material).
    struct BlobKey {
        int lod = 0;
        int64_t x = 0, y = 0, z = 0;
        uint32_t material = 0;
        bool operator<(const BlobKey &p_o) const;
    };
    int moment_blob_type = -1; // -1: no moment layers
    int blob_type = -1; // the library's moment blob type (whether or not the table has layers)
    // The moment layers' parameters (the LOD range left to the caller).
    Archetype::MomentLayerParams moment_params() const;
    Archetype::MomentLayers moment_layers;
    std::map<BlobKey, uint32_t> blob_ids;

    // Makes p_snapshot the current one (readers and the GPU table) and p_region the last edit's region.
    void publish(const std::shared_ptr<const Archetype::InstanceSnapshot> &p_snapshot, const VoxelBox &p_region);
    // Makes p_snapshot the current one, the last edit's region kept (a new base: the same lists).
    void swap_snapshot(const std::shared_ptr<const Archetype::InstanceSnapshot> &p_snapshot);
    // The widest instance an edit may add (box plus twice k, voxels): 4 top-LOD chunk spans, so the index places a
    // handful of entries for it, and at most 2^29 so that box_lo minus the influence margin (clamped to INPUT_LIMIT
    // = 2^30) stays in int32 on the GPU (arch_in_box).
    int64_t max_instance_extent() const;
    // The band rule, unless set_instance_band(false) (the shapes gallery: its ladder is sized one cube per LOD, and
    // eroding the thin frames tears them).
    void finish_record(Archetype::TableRecord &r_rec) const;
    // The finished record add_instance would append (its inputs checked: the type an index of the library, edit_error,
    // record_fit_error), or why it is refused. The table's capacity is the caller's.
    ov::str edit_record(int p_type, const VoxelPos &p_anchor, const float *p_rotation, const float *p_params, int p_op, float p_k,
            Archetype::TableRecord &r_rec) const;
    std::vector<Edit> start_edits; // set_edits
    std::map<uint32_t, Edit> edits; // the live edits by record id (live_edits)
    // The changed record's margins and LODs, then a new snapshot (the previous base plus the edited records'
    // overlay) over the edit region.
    void finish_edit(const Archetype::TableRecord &p_before, const Archetype::TableRecord &p_after, const VoxelBox &p_region, uint32_t p_id);
    // A source's entries: a kind, a cell (LOD0 voxels) and the species named there in the world's order (its classes).
    struct Group {
        Archetype::Species::Kind kind = Archetype::Species::TREE;
        int64_t side = 0;
        std::vector<const Archetype::Species *> species;
    };
    // The instances' sources (the world's list and the biomes'), the rocks' then the trees', each kind's by cell.
    std::vector<Group> groups() const;
    // An entry's cell of p_cell_m metres in LOD0 voxels (forest_cell_side's rounding): its group's side.
    int64_t cell_side(double p_cell_m) const;
    // A source over the whole ground: one site per cell of p_side voxels where the ground crosses it (the walk only tries
    // the cells along the surface, so the empty air costs nothing), out of the lakes.
    Archetype::CellScatter scene_source(uint32_t p_salt, int64_t p_side) const;
    // A tree source: grown trees, a site's species its biome's (site_class).
    Archetype::CellScatter forest_source(const Archetype::TreeTypes &p_types, const Group &p_group, uint32_t p_salt) const;
    // A rock source: a rock (a bumpy ball sunk 0.4 of its radius) or a pillar (a capsule) per site, turned to the up
    // there, its species' radius across, the species its biome's (site_class).
    Archetype::CellScatter scatter_source(const Types &p_types, const Group &p_group, uint32_t p_salt) const;
    // A site's class among p_group's species: its biome's (BiomePick) shares of the group picked by the site's variant
    // (pick_share), -1 where its biome grows none of them; the variant modulo their count when every biome grows each
    // at share 1. Thread-safe (sources call it).
    std::function<int(const Archetype::Site &)> site_class(const Group &p_group) const;
    // Every site of p_source in the whole region's box, appended to the records (each a contiguous run, kept whole or left
    // out), and its stats under r_stats[p_name]; the sites' runs to r_spans if given.
    void add_source(const Archetype::CellScatter &p_source, const ov::str &p_name, ov::Value &r_stats, std::vector<Archetype::TreeSpan> *r_spans = nullptr);
    // The sources as keys of a stream whose slots start at p_slot_base, and their stats.
    void add_stream(uint32_t p_slot_base, ov::Value &r_stats);
    // A region's box: its square round the ground under its centre (p_on_base: the base's point there), 4 half sides up
    // and down. Invalid for side 0.
    VoxelBox region_box(const Region &p_region, bool p_on_base = false) const;
    // The biome features' sites in the feature region: each kind and cell the lists name, its candidates (Biome::sites)
    // where their biome's entry keeps them (BiomePick; the world's list for a biome that lists none), at its radius.
    std::vector<Biome::Site> feature_sites() const;
    // The biome features in the feature region, appended to the records, and their stats.
    void add_biomes(const Types &p_types, ov::Value &r_stats);
    // The moment blobs of the records placed so far (each listed at its layer's LOD only), appended, and their stats.
    void add_moment_blobs(int p_blob_type, ov::Value &r_stats);
    // After record p_id changed: its item updated in the moment layers (MomentLayers::update) and the kept blobs that
    // changed applied to the table (added, replaced, removed): their ids appended to r_ids, their regions to edit_regions.
    void update_moment_blobs(uint32_t p_id, std::vector<uint32_t> &r_ids);
    // Whether a blob's record fits the table (record_fit_error, the list id range): the build and the edits refuse the same.
    bool blob_fits(const Archetype::TableRecord &p_rec) const;
    // A new blob record for p_key, appended.
    void add_blob_record(const BlobKey &p_key, const Archetype::TableRecord &p_rec);
    // A blob record changed from p_old to p_new (removed: absent): its id and region recorded for the edit.
    void blob_changed(uint32_t p_id, const Archetype::TableRecord &p_old, const Archetype::TableRecord &p_new, int p_lod, std::vector<uint32_t> &r_ids);
    // Finishes and appends r_records (one feature) if every one fits the table (record_fit_error, the limits of a
    // runtime edit), else none (ERR naming p_what).
    bool admit_records(std::vector<Archetype::TableRecord> &r_records, const char *p_what);
    // Whether p_voxel is in or near a lava lake (Biome::LavaPool::near): no tree or rock there (the sources' accept).
    bool in_lake(const VoxelPos &p_voxel) const;
    // The biome sites listed in the table's stats (the counts cover all of them).
    static constexpr int SITES_LISTED_MAX = 256;
};
