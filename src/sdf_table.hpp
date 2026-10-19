#pragma once

#include "archetype_gpu_table.hpp"
#include "archetype_table.hpp"
#include "instance_table.hpp"
#include "sdf_base.hpp"
#include "tree_archetype.hpp"

#include <atomic>
#include <memory>

// A terrain with an archetype instance table: the ground is the subclass's (SDFCascade: the Hermite cascade;
// SDFArchetype: a compiled Program), and every point folds the instances near it onto it (instance_table.hpp,
// archetype_table.hpp fold / fold_onto). This class owns everything about the instances, the same for every
// ground: the library, the records and their snapshot (read from other threads), the GPU table, the runtime edits,
// and the scenes placed on the ground (the demo objects, the --instances scatter, the --trees forest), which only
// need the subclass's ground_y.
class SDFTable : public SDFBase {
public:
    static constexpr uint32_t ROCK_SEED = 0x20C4u;

    // What build_table places on the ground. PLACED: the scatter and the forest; DEMO: the demo objects (rocks,
    // tree trunks and a cave mouth in front of the main_scene spawn) first, then the same.
    enum class Scene { EMPTY,
        PLACED,
        DEMO };

    // A deterministic scatter of p_count rocks and pillars in a square of side p_area (metres) around the demo
    // objects (--instances / --instance_area). Set before configure().
    void set_scatter(int p_count, double p_area) {
        scatter_count = p_count;
        scatter_area = p_area;
    }
    // A forest of about p_count grown trees (tree_archetype.hpp) in a square of side p_area (metres) around the demo
    // objects (--trees / --tree_area). Set before configure().
    void set_forest(int p_count, double p_area) {
        forest_count = p_count;
        forest_area = p_area;
    }
    const std::vector<Archetype::TreeSpan> &get_trees() const { return trees; }

    // The band rule for the instances (Archetype::set_band); false keeps them at every LOD. Set before configure().
    void set_instance_band(bool p_enabled) { instance_band = p_enabled; }
    // The band rule's erosion rate in cells per LOD (Archetype::set_band). Set before configure().
    void set_instance_erode(float p_cells) { erode_rule.rate = p_cells; }
    // The band rule's onset: a part erodes once its half thickness holds fewer than this many cells (0: from LOD 1).
    void set_instance_resolve(float p_cells) { erode_rule.resolve = p_cells; }

    // An instance erodes by erode_rule.rate cells between LOD L and L+1: the skirts must hide that step.
    float min_skirt_depth(const ChunkParam &, int) const override { return erode_rule.rate; }

    SDFGpuTable *get_gpu_table() const override { return &gpu; }

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
    // The library, the records of p_scene and the first snapshot over p_ground (an empty Program for a ground that
    // isn't one), its values in LOD0 voxels times p_voxel (metres). Call from configure(), after SDFBase's.
    void build_table(const Archetype::Program &p_ground, double p_voxel, int p_lod_count, Scene p_scene);

private:
    std::shared_ptr<const Archetype::DefLibrary> library;
    Archetype::Program ground_program;
    float out_scale = 1.0f;
    int lod_count = 24;
    uint32_t version = 0;
    Archetype::RecordStore records; // editable (paged, copy on write): each snapshot shares its pages
    std::shared_ptr<const Archetype::InstanceSnapshot> snapshot;
    mutable Archetype::GpuTable gpu; // the meshers read and upload through get_gpu_table()
    float edit_k = 0.0f; // the changed records' largest margin inputs (edit_affects_chunk)
    float edit_margin_c = 0.0f;
    int edit_lod_max = 0; // the changed records are listed below this LOD (edit_affects_chunk)
    int scatter_count = 0; // --instances
    double scatter_area = 200.0; // --instance_area (metres)
    int forest_count = 0; // --trees
    double forest_area = 60.0; // --tree_area (metres)
    std::vector<Archetype::TreeSpan> trees; // the forest's trees, in id order
    bool instance_band = true; // --instance_band
    Archetype::ErodeRule erode_rule; // --instance_erode, --instance_resolve

    // A new snapshot: a whole new index (p_changed < 0: the scene was rebuilt) or the previous base plus the edited
    // records' overlay.
    void rebuild(const VoxelBox &p_region, int64_t p_changed = -1);
    // The widest instance an edit may add (box plus twice k, voxels): 4 top-LOD chunk spans, so the index places a
    // handful of entries for it, and at most 2^29 so that box_lo minus the influence margin (clamped to INPUT_LIMIT
    // = 2^30) stays in int32 on the GPU (arch_in_box).
    int64_t max_instance_extent() const;
    void finish_record(Archetype::TableRecord &r_rec) const;
    // The LODs an edit can change: below the larger of the record's max_lod before and after it.
    void set_edit_lods(const Archetype::TableRecord &p_before, const Archetype::TableRecord &p_after);
    // The changed records' margins and LODs, then a new snapshot over the edit region.
    void finish_edit(const Archetype::TableRecord &p_before, const Archetype::TableRecord &p_after, const VoxelBox &p_region, uint32_t p_id);
    static VoxelBox record_box(const Archetype::TableRecord &p_rec);
    // The forest's trees, appended to the records (each a contiguous run), and its stats for the table's report.
    void add_forest(const Archetype::TreeTypes &p_types, double p_voxel, double p_centre_x, double p_centre_z);
};
