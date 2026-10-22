#pragma once

#include "instance_stream.hpp"
#include "instance_table.hpp"
#include "sdf_base.hpp"

#include <algorithm>
#include <atomic>
#include <memory>

// The archetype instance table's data for the GPU (SDFGpuTable): the DefRecord array and the per-batch
// list of instance indices. Each mesher owns the actual buffers (on its own RenderingDevice) and uploads
// record_bytes() when version() changes; archetype_table.cpp generates the arch_fold reading them.
namespace Archetype {

// The DefRecord struct, the sdf_records / sdf_list buffers at these bindings and the global g_sdf_chunk (the
// chunk being sampled): what the generated fold reads (generate_fold_glsl).
String glsl_table_buffers(int p_record_binding, int p_list_binding);

// The fold order of a list, the GPU's (build_lists) and the CPU's (SDFTable::point_list) alike: the scene's records
// (snapshot ids below p_scene_end), the streamed ones, then the edits' (added since the scene), so an edit carves the
// streamed records too. p_ids: a snapshot list, in id order; p_scene(id) takes a snapshot id, p_streamed() appends the
// streamed records.
template <typename Scene, typename Streamed>
void compose_list(const std::vector<uint32_t> &p_ids, uint32_t p_scene_end, Scene p_scene, Streamed p_streamed) {
    const auto edits = std::lower_bound(p_ids.begin(), p_ids.end(), p_scene_end);
    std::for_each(p_ids.begin(), edits, p_scene);
    p_streamed();
    std::for_each(edits, p_ids.end(), p_scene);
}

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
    // The buffers are sized for the table's records (runtime edits only append) and the streamed slots past them.
    uint32_t instance_capacity() const override { return capacity + slots; }
    // The table's records at most (the streamed slots start there).
    uint32_t table_capacity() const { return capacity; }
    // Main thread, before the meshers start: the table's capacity and the streamed sources (null: none), whose slots
    // start at p_capacity. A list word holds the record id in LIST_ID_BITS: past it, the slots then the table are
    // clamped (the table then refuses more records).
    void set_capacity(uint32_t p_capacity, std::shared_ptr<const InstanceStream> p_stream = nullptr);
    UploadCursor upload_cursor() const override;
    PackedByteArray record_bytes(const UploadCursor &p_have, UploadCursor &r_now) const override;
    uint32_t record_size() const override { return RECORD_BYTES; }

    uint32_t build_lists(const std::vector<ChunkId> &p_chunks, const ChunkParam &p_param, PackedByteArray &r_chunk_ids, PackedByteArray &r_list,
            uint32_t p_capacity_words, std::vector<std::vector<uint32_t>> *r_palettes = nullptr) const override;

    Dictionary get_stats() const override;
    // The scene's own numbers (the forest), merged into get_stats(), and the snapshot ids below p_scene_end the scene's
    // (compose_list). Main thread, before the meshers start.
    void set_scene(const Dictionary &p_stats, uint32_t p_scene_end) {
        scene_stats = p_stats;
        scene_end = p_scene_end;
    }
    uint32_t get_scene_end() const { return scene_end; }

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
    uint32_t capacity = 16384; // the table's records the GPU buffer holds
    uint32_t slots = 0; // the streamed records' slots, past capacity
    std::shared_ptr<const InstanceStream> stream;
    uint32_t scene_end = 0;
    Dictionary scene_stats;
};

} // namespace Archetype
