#pragma once

#include "archetype_gpu_table.hpp"
#include "archetype_table.hpp"
#include "biome_features.hpp"
#include "cell_scatter.hpp"
#include "instance_table.hpp"
#include "moment_layers.hpp"
#include "scatter_source.hpp"
#include "sdf_base.hpp"
#include "tree_archetype.hpp"

#include <atomic>
#include <memory>

// The instance half of the terrain: every point folds the instances near it onto the ground's register
// (instance_table.hpp, archetype_table.hpp fold_onto; the ground is SDFCascade's). This class owns everything about
// the instances: the library, the records and their snapshot (read from other threads), the GPU table, the runtime
// edits, and the scenes placed on the ground (the shapes gallery, the --instances scatter, the --trees forest),
// which only need the subclass's ground_y.
class SDFTable : public SDFBase {
public:
    static constexpr uint32_t ROCK_SEED = 0x20C4u;
    // The scatter's cell sites (cell_sites.hpp; the forest's is Archetype::TREE_SITE_SALT).
    static constexpr uint32_t SCATTER_SITE_SALT = 0x5CA77u;
    // The scatter's and the forest's centre (metres, x and z), in front of the main_scene spawn.
    static constexpr double SCENE_X = -50.0;
    static constexpr double SCENE_Z = -150.0;
    // The shapes gallery's centre (metres, x and z): the shapes_scene spawn.
    static constexpr double SHAPES_X = -52.0;
    static constexpr double SHAPES_Z = -95.0;

    // What build_table places on the ground. PLACED: the scatter and the forest (none by default); SHAPES: the shapes
    // gallery first (hard unions, subtractions and intersections, sharp creases for dual contouring: a ladder of
    // eight cubes holed through every face, each twice the size and twice the distance of the last, i.e. about one
    // LOD coarser each on screen, turned to show the camera the same two faces; a die, a staircase and a cube
    // tunnelled along its three axes behind the spawn), then the same.
    enum class Scene { PLACED,
        SHAPES };
    // Set before configure().
    void set_scene(Scene p_scene) { scene = p_scene; }

    // A scatter of about p_count rocks and pillars in a square of side p_area (metres) around (SCENE_X, SCENE_Z), one
    // per cell of the ground (--instances / --instance_area). Set before configure().
    void set_scatter(int p_count, double p_area) {
        scatter_count = p_count;
        scatter_area = p_area;
    }
    // A forest of about p_count grown trees (tree_archetype.hpp) in a square of side p_area (metres) around
    // (SCENE_X, SCENE_Z) (--trees / --tree_area). Set before configure().
    void set_forest(int p_count, double p_area) {
        forest_count = p_count;
        forest_area = p_area;
    }
    // Whether the scatter and the forest stream (instance_stream.hpp: their keys computed as the chunks ask for them, their
    // records in GPU slots past the table's), else placed whole in the table at configure (the reference; the tests that
    // edit their records). Set before configure().
    void set_stream_sources(bool p_enabled) { stream_sources = p_enabled; }
    // The forest's trees, in id order (placed whole only).
    const std::vector<Archetype::TreeSpan> &get_trees() const { return trees; }
    // The procedural sources of the scene (cell_scatter.hpp), by name: "scatter", "trees" (empty without them).
    struct Source {
        const char *name = "";
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
    // The biome features (biome_features.hpp: lava lakes, floating islands) of the biome cells within p_radius metres
    // of (SCENE_X, SCENE_Z) (--biomes, kilometres there; 0: none). Set before configure().
    void set_biome_radius(double p_radius) { biome_radius = p_radius; }
    // The materials the records name (TableRecord::material): the page shader's material_tex.
    const Material::Library &get_materials() const override { return materials; }

    // The band rule for the instances (Archetype::set_band); false keeps them at every LOD. Set before configure().
    void set_instance_band(bool p_enabled) { instance_band = p_enabled; }
    // The band rule's erosion rate in cells per LOD (Archetype::set_band). Set before configure().
    void set_instance_erode(float p_cells) { erode_rule.rate = p_cells; }
    // The band rule's resolution: a part is kept while its half thickness holds this many cells (Archetype::set_band).
    void set_instance_resolve(float p_cells) { erode_rule.resolve = p_cells; }
    // The far field of the dropped instances (moment_layers.hpp; with the band rule only, default on): the blobs of their
    // moments, one per LOD layer, cell and material, placed after the scene's records. Set before configure().
    void set_moment_blobs(bool p_enabled) { moment_blobs = p_enabled; }

    // An instance erodes by at most this many cells between LOD L and L+1 (the band rule; 0 without it).
    float instance_erode_cells() const { return instance_band ? erode_rule.rate : 0.0f; }
    // The skirts must hide the band rule's step, and the moment blobs' when the scene places instances that can have
    // some (asked before configure: the blobs aren't built yet): each layer's are listed at its LOD alone, so the
    // far-field surfaces of a LOD L and a LOD L+1 chunk differ by up to MOMENT_SKIRT_CELLS cells of L (0.93 of the
    // skirt on a 4,000-rock scatter at LOD 3 | 4, the worst measured: test_cascade_table).
    static constexpr float MOMENT_SKIRT_CELLS = 2.5f;
    // The locality margins (skirt_depth) take it whole; the skirt walls only where a blob touches the sample
    // (SDFCascade's moment skirt), so the rest of the terrain keeps the band rule's.
    float min_skirt_depth(const ChunkParam &, int) const override {
        const bool blobs = instance_band && moment_blobs && (scatter_count > 0 || forest_count > 0 || biome_radius > 0.0);
        return instance_erode_cells() + (blobs ? MOMENT_SKIRT_CELLS : 0.0f);
    }
    // The moment skirt in cells: MOMENT_SKIRT_CELLS when the table has moment layers (edits may add blobs where the build
    // placed none; the skirt only reaches the samples a blob touches), else 0.
    float moment_skirt_cells() const { return moment_blob_type >= 0 ? MOMENT_SKIRT_CELLS : 0.0f; }

    SDFGpuTable *get_gpu_table() override { return &gpu; }
    using SDFBase::get_gpu_table; // the const form

    // The node's world box grown by the cull band (2 chunk sizes at its LOD) against every instance's box, band or
    // not: a culled coarse node hides its subtree, where the instance is (instance_table.hpp). A streamed source's records
    // stand on the ground: a node of its zone is kept while its corners are within the records' ground_reach of it.
    bool cull_exempt(const ChunkId &p_node, int p_lod, float p_past_band_m) const override;

    // The streamed sources' gating (InstanceStream::admit; every chunk READY without them).
    Admit admit(const ChunkId &p_chunk, uint64_t p_owner, uint64_t p_tag) override;
    void take_ready(uint64_t p_owner, std::vector<std::pair<ChunkId, uint64_t>> &r_out) override;
    size_t parked_count(uint64_t p_owner) const override { return stream ? stream->parked_count(p_owner) : 0; }
    // poll_rebuild(), then the stream's poll.
    void poll() override;
    // The streamed sources (null when none streams).
    const Archetype::InstanceStream *get_stream() const { return stream.get(); }
    // The records the CPU folds at p_voxel for a LOD-p_lod sample, in fold order (compose_list): r_list indexes the
    // returned view, valid until the next call on this thread. Any thread.
    // p_view: the streamed keys not landed yet are left out instead of computed (InstanceStream::view_records; main thread).
    Archetype::RecordView point_list(const Archetype::InstanceSnapshot &p_snap, const VoxelPos &p_voxel, int p_lod, std::vector<uint32_t> &r_list,
            bool p_view = false) const;

    // Runtime edits: rebuild the snapshot (copy on write) and bump the version; the GPU re-uploads the records when
    // it changes. last_edit_box() / edit_affects_chunk() tell the terrain what to re-mesh.
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

    // The absolute LOD0 voxel y of the ground's top solid voxel in column (p_x, p_z), instances left out: where the
    // scenes stand.
    virtual int64_t ground_y(int64_t p_x, int64_t p_z) const = 0;
    // Why an edit's inputs are refused (empty if they are fine): the GPU and the CPU only agree on finite, normal
    // floats, the box and the fold assume an orthonormal rotation, the table folds the four top-level operators, and
    // a smooth radius is >= 0.
    static String edit_error(const float *p_rotation, const float *p_params, int p_op, float p_k);

protected:
    // The library, the scene's records and the first snapshot, its values in LOD0 voxels times the voxel size
    // (metres). Call from configure(), after SDFBase's and once ground_y works.
    void build_table(int p_lod_count);
    // Waits for the stream's tasks and drops it: they call the subclass's ground_field, so its destructor calls this first.
    void stop_streaming();

    // The base palette (SDFGpuTable::base_palette): GROUND, then p_ground's materials, registered by name (their
    // parameters replaced on a reconfigure). Call from configure(); empty: {GROUND}.
    void set_ground_materials(const std::vector<std::pair<String, Material::Params>> &p_ground) {
        std::vector<uint32_t> base{ Material::GROUND };
        for (const std::pair<String, Material::Params> &m : p_ground) {
            base.push_back(materials.set(m.first, m.second));
        }
        gpu.set_base_palette(base);
    }

private:
    // The library indices the scenes place.
    struct Types {
        int capsule = -1;
        int rock = -1;
        int holed_cube = -1;
        int die = -1;
        int stairs = -1;
        int tunnel_cube = -1;
        Biome::Types biome;
    };

    Scene scene = Scene::PLACED;
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
    int scatter_count = 0; // --instances
    double scatter_area = 200.0; // --instance_area (metres)
    int forest_count = 0; // --trees
    double forest_area = 60.0; // --tree_area (metres)
    std::vector<Archetype::TreeSpan> trees; // the forest's trees, in id order
    std::vector<Source> sources;
    bool stream_sources = true; // --stream
    std::vector<std::shared_ptr<const Archetype::ScatterSource>> streamed; // the valid sources, by their id
    std::shared_ptr<Archetype::InstanceStream> stream; // the gpu table shares it
    double biome_radius = 0.0; // --biomes (metres)
    std::vector<Biome::LavaPool> lava_pools; // keeps trees and rocks out of the lakes (in_lake)
    Material::Library materials;
    bool instance_band = true; // --instance_band
    Archetype::ErodeRule erode_rule; // --instance_erode, --instance_resolve
    bool moment_blobs = true; // --moments
    bool has_moment_blobs = false; // add_moment_blobs placed some
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
    // The band rule, unless set_instance_band(false) (VoxelTerrain: --instance_band=0, and the shapes gallery: its ladder
    // is sized one cube per LOD, and eroding the thin frames tears them).
    void finish_record(Archetype::TableRecord &r_rec) const;
    // The changed record's margins and LODs, then a new snapshot (the previous base plus the edited records'
    // overlay) over the edit region.
    void finish_edit(const Archetype::TableRecord &p_before, const Archetype::TableRecord &p_after, const VoxelBox &p_region, uint32_t p_id);
    float voxels(double p_metres) const;
    // A record of type p_type at (p_x, p_z) metres, p_dy_m above the ground there, turned by p_yaw about y, smooth
    // radius p_k_m metres, appended.
    void place(Archetype::Combine p_op, double p_k_m, int p_type, const float *p_params, double p_x, double p_z, double p_dy_m, double p_yaw);
    void add_shapes(const Types &p_types);
    // The sources the scene places: a box of side p_area_m around the scene's centre, over every height the ground can
    // reach in it (4 times its half side up and down from the ground at the centre; the walk only tries the cells along
    // the surface, so the empty air costs nothing), out of the lakes.
    Archetype::CellScatter scene_source(uint32_t p_salt, int p_count, double p_area_m) const;
    // The forest's (--trees): grown trees, the species from the site's variant.
    Archetype::CellScatter forest_source(const Archetype::TreeTypes &p_types) const;
    // The scatter's (--instances): a rock (even variant: a bumpy ball sunk 0.4 of its radius) or a pillar (a capsule)
    // per site, turned to the up there, 1 to 4 m across.
    Archetype::CellScatter scatter_source(const Types &p_types) const;
    // Every site of p_source in its zone, appended to the records (each a contiguous run, kept whole or left out), and
    // its stats under r_stats[p_name]; the sites' runs to r_spans if given.
    void add_source(const Archetype::CellScatter &p_source, const char *p_name, Dictionary &r_stats, std::vector<Archetype::TreeSpan> *r_spans = nullptr);
    // The sources as keys of a stream whose slots start at p_slot_base, and their stats.
    void add_stream(uint32_t p_slot_base, Dictionary &r_stats);
    // The biome features within biome_radius of the scene's centre, appended to the records, and their stats.
    void add_biomes(const Types &p_types, Dictionary &r_stats);
    // The moment blobs of the records placed so far (each listed at its layer's LOD only), appended, and their stats.
    void add_moment_blobs(int p_blob_type, Dictionary &r_stats);
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
    // Whether column (p_x, p_z) is in or near a lava lake (Biome::LAKE_KEEP_OUT radii): no tree or rock there (the
    // sources' accept). Column-based: the climate stack's per-cell picks replace it.
    bool in_lake(int64_t p_x, int64_t p_z) const;
    // The biome sites listed in the table's stats (the counts cover all of them).
    static constexpr int SITES_LISTED_MAX = 256;
};
