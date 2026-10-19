#pragma once

#include "instance_table.hpp"
#include "sdf_base.hpp"

#include <atomic>
#include <memory>

// The archetype instance table's data for the GPU (SDFGpuTable): the DefRecord array and the per-batch
// list of instance indices. Each mesher owns the actual buffers (on its own RenderingDevice) and uploads
// record_bytes() when version() changes; archetype_table.cpp generates terrain_sdf reading them.
namespace Archetype {

class GpuTable : public SDFGpuTable {
public:
    // Main thread: the snapshot the GPU mirrors. Its version drives the record upload.
    // Read from the render thread (direct mesher) and the legacy worker: atomic, and each reader works on
    // one copy.
    void set_snapshot(const std::shared_ptr<const InstanceSnapshot> &p_snapshot) { std::atomic_store(&snap, p_snapshot); }
    std::shared_ptr<const InstanceSnapshot> snapshot() const { return std::atomic_load(&snap); }

    String glsl_buffers(int p_record_binding, int p_list_binding) const override;
    String glsl_set_chunk() const override;

    uint32_t version() const override {
        const std::shared_ptr<const InstanceSnapshot> s = snapshot();
        return s ? s->version : 0u;
    }
    uint32_t instance_count() const override {
        const std::shared_ptr<const InstanceSnapshot> s = snapshot();
        return s ? uint32_t(s->records.size()) : 0u;
    }
    // The buffers are sized for this many instances up front (runtime edits only append). Raised by
    // --instance_capacity for a large scatter.
    uint32_t instance_capacity() const override { return capacity; }
    void set_capacity(uint32_t p_capacity) { capacity = p_capacity; }
    PackedByteArray record_bytes(uint32_t p_have_version, uint32_t &r_version) const override;
    uint32_t record_size() const override { return RECORD_BYTES; }

    void build_lists(const std::vector<ChunkId> &p_chunks, const ChunkParam &p_param, PackedByteArray &r_chunk_ids, PackedByteArray &r_list) const override;

    Dictionary get_stats() const override;
    // The scene's own numbers (SDFArchetype's forest), merged into get_stats(). Main thread, before the meshers start.
    void set_scene_stats(const Dictionary &p_stats) { scene_stats = p_stats; }

private:
    // Main thread only (build_lists is called by the meshers' dispatch on the main thread).
    struct ListStats {
        int64_t chunks[ChunkMath::MAX_LOD_NUMBERS] = {};
        int64_t entries[ChunkMath::MAX_LOD_NUMBERS] = {};
        int64_t longest[ChunkMath::MAX_LOD_NUMBERS] = {};
        int64_t build_usec = 0;
        int64_t batches = 0;
        int64_t truncated = 0;
    };
    mutable ListStats list_stats;
    std::shared_ptr<const InstanceSnapshot> snap;
    uint32_t capacity = 16384; // the GPU record/list buffers hold this many instances
    Dictionary scene_stats;
};

} // namespace Archetype
