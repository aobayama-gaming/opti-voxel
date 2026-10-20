#pragma once

#include "archetype_gpu_table.hpp"
#include "archetype_table.hpp"
#include "biome_features.hpp"
#include "instance_table.hpp"
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

    // A deterministic scatter of p_count rocks and pillars in a square of side p_area (metres) around
    // (SCENE_X, SCENE_Z) (--instances / --instance_area). Set before configure().
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
    const std::vector<Archetype::TreeSpan> &get_trees() const { return trees; }
    // The biome features (biome_features.hpp: lava lakes, floating islands) of the biome cells within p_radius metres
    // of (SCENE_X, SCENE_Z) (--biomes, kilometres there; 0: none). Set before configure().
    void set_biome_radius(double p_radius) { biome_radius = p_radius; }
    // The lava lakes' surfaces (the page shader colours them).
    const std::vector<Biome::LavaPool> &get_lava_pools() const { return lava_pools; }

    // The band rule for the instances (Archetype::set_band); false keeps them at every LOD. Set before configure().
    void set_instance_band(bool p_enabled) { instance_band = p_enabled; }
    // The band rule's erosion rate in cells per LOD (Archetype::set_band). Set before configure().
    void set_instance_erode(float p_cells) { erode_rule.rate = p_cells; }
    // The band rule's onset: a part erodes once its half thickness holds fewer than this many cells (0: from LOD 1).
    void set_instance_resolve(float p_cells) { erode_rule.resolve = p_cells; }

    // An instance erodes by erode_rule.rate cells between LOD L and L+1 (the band rule): the skirts must hide that step.
    float min_skirt_depth(const ChunkParam &, int) const override { return instance_band ? erode_rule.rate : 0.0f; }

    SDFGpuTable *get_gpu_table() override { return &gpu; }
    using SDFBase::get_gpu_table; // the const form

    // The node's world box grown by the cull band (2 chunk sizes at its LOD) against every instance's box, band or
    // not: a culled coarse node hides its subtree, where the instance is (instance_table.hpp).
    bool cull_exempt(const ChunkId &p_node, int p_lod) const override {
        const std::shared_ptr<const Archetype::InstanceSnapshot> snap = current();
        return snap && snap->near_instance(p_node, p_lod);
    }

    // Runtime edits: rebuild the snapshot (copy on write) and bump the version; the GPU re-uploads the records when
    // it changes. last_edit_box() / edit_affects_chunk() tell the terrain what to re-mesh.
    int add_instance(int p_type, const VoxelPos &p_anchor, const float *p_rotation, const float *p_params, int p_op, float p_k) override;
    bool remove_instance(int p_id) override;
    bool move_instance(int p_id, const VoxelPos &p_anchor, const float *p_rotation) override;
    // Whether a chunk's samples can be changed by the last edit (its box grown by M(lod) meets the edit's region):
    // the chunks to re-mesh.
    bool edit_affects_chunk(const ChunkId &p_chunk) const override;

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
    Archetype::GpuTable gpu; // the meshers read and upload through get_gpu_table()
    float edit_k = 0.0f; // the changed records' largest margin inputs (edit_affects_chunk)
    float edit_margin_c = 0.0f;
    int edit_lod_max = 0; // the changed records are listed below this LOD (edit_affects_chunk)
    int scatter_count = 0; // --instances
    double scatter_area = 200.0; // --instance_area (metres)
    int forest_count = 0; // --trees
    double forest_area = 60.0; // --tree_area (metres)
    std::vector<Archetype::TreeSpan> trees; // the forest's trees, in id order
    double biome_radius = 0.0; // --biomes (metres)
    std::vector<Biome::LavaPool> lava_pools;
    bool instance_band = true; // --instance_band
    Archetype::ErodeRule erode_rule; // --instance_erode, --instance_resolve

    // Makes p_snapshot the current one (readers and the GPU table) and p_region the last edit's region.
    void publish(const std::shared_ptr<const Archetype::InstanceSnapshot> &p_snapshot, const VoxelBox &p_region);
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
    static VoxelBox record_box(const Archetype::TableRecord &p_rec);
    float voxels(double p_metres) const;
    // A record of type p_type at (p_x, p_z) metres, p_dy_m above the ground there, turned by p_yaw about y, smooth
    // radius p_k_m metres, appended.
    void place(Archetype::Combine p_op, double p_k_m, int p_type, const float *p_params, double p_x, double p_z, double p_dy_m, double p_yaw);
    void add_shapes(const Types &p_types);
    void add_scatter(const Types &p_types);
    // The forest's trees, appended to the records (each a contiguous run), and its stats for the table's report.
    void add_forest(const Archetype::TreeTypes &p_types, double p_voxel, double p_centre_x, double p_centre_z, Dictionary &r_stats);
    // The biome features within biome_radius of the scene's centre, appended to the records, and their stats.
    void add_biomes(const Types &p_types, Dictionary &r_stats);
    // Finishes and appends r_records (one feature) if every one fits the table (record_fit_error, the limits of a
    // runtime edit), else none (ERR naming p_what).
    bool admit(std::vector<Archetype::TableRecord> &r_records, const char *p_what);
    // Whether column (p_x, p_z) is in or near a lava lake (Biome::LAKE_KEEP_OUT radii): no tree or rock there.
    bool in_lake(int64_t p_x, int64_t p_z) const;
    // The biome sites listed in the table's stats (the counts cover all of them).
    static constexpr int SITES_LISTED_MAX = 256;
};
