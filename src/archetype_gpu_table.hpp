#pragma once

#include "instance_table.hpp"
#include "sdf_base.hpp"

#include <memory>

// The archetype instance table's data for the GPU (SDFGpuTable): the DefRecord array and the per-batch
// list of instance indices. Each mesher owns the actual buffers (on its own RenderingDevice) and uploads
// record_bytes() when version() changes; archetype_table.cpp generates terrain_sdf reading them.
namespace Archetype {

class GpuTable : public SDFGpuTable {
public:
    // Main thread: the snapshot the GPU mirrors. Its version drives the record upload.
    void set_snapshot(const std::shared_ptr<const InstanceSnapshot> &p_snapshot) { snap = p_snapshot; }
    const std::shared_ptr<const InstanceSnapshot> &snapshot() const { return snap; }

    String glsl_buffers(int p_record_binding, int p_list_binding) const override;
    String glsl_set_chunk() const override;

    uint32_t version() const override { return snap ? snap->version : 0u; }
    uint32_t instance_count() const override { return snap ? uint32_t(snap->records.size()) : 0u; }
    PackedByteArray record_bytes() const override;

    void build_lists(const std::vector<ChunkId> &p_chunks, const ChunkParam &p_param, PackedByteArray &r_chunk_ids, PackedByteArray &r_list) const override;

private:
    std::shared_ptr<const InstanceSnapshot> snap;
};

} // namespace Archetype
