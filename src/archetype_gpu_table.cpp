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

PackedByteArray GpuTable::record_bytes() const {
    PackedByteArray bytes;
    const uint32_t count = MAX(instance_count(), 1u);
    bytes.resize(count * RECORD_BYTES);
    bytes.fill(0);
    if (snap) {
        for (uint32_t i = 0; i < snap->records.size(); i++) {
            encode_record(bytes, i, snap->records[i]);
        }
    }
    return bytes;
}

void GpuTable::build_lists(const std::vector<ChunkId> &p_chunks, const ChunkParam &p_param, PackedByteArray &r_chunk_ids, PackedByteArray &r_list) const {
    (void)p_param;
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
