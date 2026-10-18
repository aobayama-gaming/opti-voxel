#include "archetype_gpu_table.hpp"

#include "gpu_frame.hpp"

#include <algorithm>

namespace Archetype {

String GpuTable::glsl_buffers(int p_record_binding, int p_list_binding) const {
    return String(GLSL_DEF_RECORD_STRUCT) +
            vformat("\nlayout(set = 0, binding = %d, std430) restrict readonly buffer DefRecords { DefRecord sdf_records[]; };\n", p_record_binding) +
            vformat("layout(set = 0, binding = %d, std430) restrict readonly buffer DefList { uint sdf_list[]; };\n", p_list_binding);
}

String GpuTable::glsl_set_chunk() const {
    return "\n    g_sdf_chunk = input_chunk[gl_GlobalInvocationID.z / chunk_param.chunk_size];";
}

PackedByteArray GpuTable::record_bytes(uint32_t &r_version) const {
    const std::shared_ptr<const InstanceSnapshot> s = snapshot();
    PackedByteArray bytes;
    r_version = s ? s->version : 0u;
    if (!s) {
        return bytes;
    }
    // The used records only (the buffer is sized for capacity() up front): an edit uploads its table, not
    // the whole capacity.
    const uint32_t count = uint32_t(s->records.size());
    ERR_FAIL_COND_V_MSG(count > capacity, bytes, vformat("GpuTable: %d instances, over the GPU capacity of %d.", count, capacity));
    bytes.resize(int64_t(count) * RECORD_BYTES);
    for (uint32_t i = 0; i < count; i++) {
        encode_record(bytes, i, s->records[i]);
    }
    return bytes;
}

void GpuTable::build_lists(const std::vector<ChunkId> &p_chunks, const ChunkParam &p_param, PackedByteArray &r_chunk_ids, PackedByteArray &r_list) const {
    (void)p_param;
    const std::shared_ptr<const InstanceSnapshot> snap = snapshot();
    std::vector<uint32_t> words;
    std::vector<uint32_t> indices;
    for (size_t i = 0; i < p_chunks.size(); i++) {
        snap->chunk_list(p_chunks[i], indices);
        const int64_t base = int64_t(i) * GPUFrame::CHUNK_INPUT_BYTES;
        r_chunk_ids.encode_u32(base + 12, uint32_t(words.size()));
        r_chunk_ids.encode_s32(base + 28, int32_t(indices.size()));
        words.insert(words.end(), indices.begin(), indices.end());
    }
    r_list.resize(int64_t(words.size()) * 4);
    for (size_t i = 0; i < words.size(); i++) {
        r_list.encode_u32(int64_t(i) * 4, words[i]);
    }
}

} // namespace Archetype
