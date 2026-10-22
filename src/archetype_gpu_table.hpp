#pragma once

#include "instance_table.hpp"
#include "sdf_base.hpp"

#include <atomic>
#include <memory>

// The archetype instance table's data for the GPU (SDFGpuTable): the DefRecord array and the per-batch
// list of instance indices. Each mesher owns the actual buffers (on its own RenderingDevice) and uploads
// record_bytes() when version() changes; archetype_table.cpp generates the arch_fold reading them.
namespace Archetype {

// The DefRecord struct, the sdf_records / sdf_list buffers at these bindings and the global g_sdf_chunk (the
// chunk being sampled): what the generated fold reads (generate_fold_glsl).
String glsl_table_buffers(int p_record_binding, int p_list_binding);

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
    void set_capacity(uint32_t p_capacity) {
        // A list word holds the record id in LIST_ID_BITS: past it, clamped (the table then refuses more records).
        if (p_capacity > LIST_ID_MASK + 1) {
            ERR_PRINT(vformat("GpuTable: a capacity of %d instances is over the list word's %d: clamped.", p_capacity, LIST_ID_MASK + 1));
            p_capacity = LIST_ID_MASK + 1;
        }
        capacity = p_capacity;
    }
    PackedByteArray record_bytes(uint32_t p_have_version, uint32_t &r_version) const override;
    uint32_t record_size() const override { return RECORD_BYTES; }

    uint32_t build_lists(const std::vector<ChunkId> &p_chunks, const ChunkParam &p_param, PackedByteArray &r_chunk_ids, PackedByteArray &r_list,
            uint32_t p_capacity_words, std::vector<std::vector<uint32_t>> *r_palettes = nullptr) const override;

    const std::vector<uint32_t> &base_palette() const override { return base; }
    // Configure only, before the meshers start.
    void set_base_palette(const std::vector<uint32_t> &p_base) { base = p_base; }

    Dictionary get_stats() const override;
    // The scene's own numbers (the forest), merged into get_stats(). Main thread, before the meshers start.
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
        int64_t split = 0; // batches ended early: their next chunk's list didn't fit (queued again)
        int64_t palette_truncated = 0; // chunks naming more than ChunkPalette::MAX_ENTRIES materials
        int64_t palette_entries[ChunkMath::MAX_LOD_NUMBERS] = {};
        int64_t palette_longest[ChunkMath::MAX_LOD_NUMBERS] = {};
    };
    mutable ListStats list_stats;
    std::shared_ptr<const InstanceSnapshot> snap;
    uint32_t capacity = 16384; // the GPU record/list buffers hold this many instances
    std::vector<uint32_t> base{ Material::GROUND };
    Dictionary scene_stats;
};

} // namespace Archetype
