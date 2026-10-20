#include "archetype_gpu_table.hpp"

#include "chunk_palette.hpp"
#include "debug_log.hpp"
#include "gpu_frame.hpp"
#include "page_format.hpp"
#include <godot_cpp/classes/worker_thread_pool.hpp>

#include <algorithm>
#include <cstring>

namespace Archetype {

String glsl_table_buffers(int p_record_binding, int p_list_binding) {
    return String(GLSL_DEF_RECORD_STRUCT) +
            vformat("\nlayout(set = 0, binding = %d, std430) restrict readonly buffer DefRecords { DefRecord sdf_records[]; };\n", p_record_binding) +
            vformat("layout(set = 0, binding = %d, std430) restrict readonly buffer DefList { uint sdf_list[]; };\n", p_list_binding) +
            "ChunkInput g_sdf_chunk; // the chunk being sampled: its list offset (id.w) and length (anchor_hi.w)\n";
}

// The virtual's implementation: glsl_table_buffers, the one source (FrameSpike splices it directly).
String GpuTable::glsl_buffers(int p_record_binding, int p_list_binding) const {
    return glsl_table_buffers(p_record_binding, p_list_binding);
}

String GpuTable::glsl_set_chunk() const {
    return "\n    g_sdf_chunk = input_chunk[gl_GlobalInvocationID.z / chunk_param.chunk_size];";
}

PackedByteArray GpuTable::record_bytes(uint32_t p_have_version, uint32_t &r_version) const {
    const std::shared_ptr<const InstanceSnapshot> s = snapshot();
    PackedByteArray bytes;
    r_version = s ? s->version : 0u;
    if (!s) {
        return bytes;
    }
    // The buffer is sized for capacity() up front; only the pages changed since the GPU's version go up.
    ERR_FAIL_COND_V_MSG(s->records.size() > capacity, bytes, vformat("GpuTable: %d instances, over the GPU capacity of %d.", s->records.size(), capacity));
    std::vector<std::pair<uint32_t, uint32_t>> runs;
    s->records.changed_since(p_have_version, runs);
    int64_t total = 4;
    for (const std::pair<uint32_t, uint32_t> &r : runs) {
        total += 8 + int64_t(r.second) * RECORD_BYTES;
    }
    bytes.resize(total);
    bytes.encode_u32(0, uint32_t(runs.size()));
    int64_t at = 4;
    for (const std::pair<uint32_t, uint32_t> &r : runs) {
        bytes.encode_u32(at, r.first * RECORD_BYTES);
        bytes.encode_u32(at + 4, r.second * RECORD_BYTES);
        PackedByteArray span;
        span.resize(int64_t(r.second) * RECORD_BYTES);
        for (uint32_t i = 0; i < r.second; i++) {
            encode_record(span, i, s->records[r.first + i]);
        }
        std::memcpy(bytes.ptrw() + at + 8, span.ptr(), span.size());
        at += 8 + span.size();
    }
    return bytes;
}

void GpuTable::build_lists(const std::vector<ChunkId> &p_chunks, const ChunkParam &p_param, PackedByteArray &r_chunk_ids, PackedByteArray &r_list,
        std::vector<std::vector<uint32_t>> *r_palettes) const {
    (void)p_param;
    const uint64_t t0 = now_usec();
    const std::shared_ptr<const InstanceSnapshot> snap = snapshot();
    if (r_palettes) {
        r_palettes->assign(p_chunks.size(), std::vector<uint32_t>{ Material::GROUND });
    }
    if (!snap || snap->empty()) {
        r_list.clear(); // every list empty: encode_chunk_input left the offsets and counts at 0
        for (const ChunkId &c : p_chunks) {
            list_stats.chunks[MIN(MAX(ChunkMath::get_lod(c), 0), ChunkMath::MAX_LOD_NUMBERS - 1)]++;
        }
        list_stats.batches++;
        list_stats.build_usec += int64_t(now_usec() - t0);
        return;
    }
    const uint32_t cap = max_chunk_list();
    // The chunks' lists and palettes on the worker pool (a dense scatter makes thousands of candidates per chunk): the
    // list words come out packed (SDFGpuTable::list_word).
    struct Job {
        const InstanceSnapshot *snap;
        const ChunkId *chunks;
        std::vector<uint32_t> *lists;
        std::vector<uint32_t> *palettes;
        uint8_t *fits;
        static void run(void *p_job, uint32_t p_i) {
            const Job &job = *static_cast<const Job *>(p_job);
            std::vector<uint32_t> &list = job.lists[p_i];
            job.snap->chunk_list(job.chunks[p_i], list);
            thread_local std::vector<uint32_t> materials;
            thread_local std::vector<uint16_t> indices;
            materials.resize(list.size());
            for (size_t k = 0; k < list.size(); k++) {
                materials[k] = job.snap->records[list[k]].material;
            }
            job.fits[p_i] = ChunkPalette::build(materials.data(), uint32_t(list.size()), job.palettes[p_i], indices);
            for (size_t k = 0; k < list.size(); k++) {
                list[k] = list_word(list[k], indices[k]);
            }
        }
    };
    std::vector<std::vector<uint32_t>> lists(p_chunks.size());
    std::vector<std::vector<uint32_t>> palettes(p_chunks.size());
    std::vector<uint8_t> fits(p_chunks.size(), 1);
    Job job{ snap.get(), p_chunks.data(), lists.data(), palettes.data(), fits.data() };
    if (p_chunks.size() > 1) {
        WorkerThreadPool *pool = WorkerThreadPool::get_singleton();
        pool->wait_for_group_task_completion(pool->add_native_group_task(&Job::run, &job, int(p_chunks.size()), -1, true, "GpuTable::build_lists"));
    } else if (p_chunks.size() == 1) {
        Job::run(&job, 0);
    }
    const size_t budget = p_chunks.size() * size_t(cap);
    std::vector<uint32_t> words;
    for (size_t i = 0; i < p_chunks.size(); i++) {
        std::vector<uint32_t> &indices = lists[i];
        const int lod = MIN(MAX(ChunkMath::get_lod(p_chunks[i]), 0), ChunkMath::MAX_LOD_NUMBERS - 1);
        list_stats.chunks[lod]++;
        list_stats.entries[lod] += int64_t(indices.size());
        list_stats.longest[lod] = MAX(list_stats.longest[lod], int64_t(indices.size()));
        list_stats.palette_entries[lod] += int64_t(palettes[i].size());
        list_stats.palette_longest[lod] = MAX(list_stats.palette_longest[lod], int64_t(palettes[i].size()));
        if (!fits[i]) {
            list_stats.palette_truncated++;
            ERR_PRINT(vformat("GpuTable: a LOD %d chunk names more than %d materials: the rest draw as ground.", lod, ChunkPalette::MAX_ENTRIES));
        }
        // The buffer holds batch size * cap words: one long list fits beside short ones. Only a batch over the
        // whole budget is cut (its chunk gets a wrong surface, and the CPU's evaluate disagrees there).
        if (words.size() + indices.size() > budget) {
            list_stats.truncated++;
            ERR_PRINT(vformat("GpuTable: a batch's instance lists are over the %d words of the buffer: a LOD %d chunk's list of %d truncated (wrong surface there).",
                    int64_t(budget), lod, int64_t(indices.size())));
            indices.resize(budget - words.size());
            // Its palette came from the whole list: it draws as ground (the cascade writes no material for a cut list
            // it may not hold, and both meshers then read none).
            palettes[i].assign(1, Material::GROUND);
        }
        const int64_t base = int64_t(i) * GPUFrame::CHUNK_INPUT_BYTES;
        r_chunk_ids.encode_u32(base + 12, uint32_t(words.size()));
        r_chunk_ids.encode_s32(base + 28, int32_t(indices.size()));
        words.insert(words.end(), indices.begin(), indices.end());
        if (r_palettes) {
            (*r_palettes)[i] = std::move(palettes[i]);
        }
    }
    r_list.resize(int64_t(words.size()) * 4);
    std::memcpy(r_list.ptrw(), words.data(), words.size() * 4); // little-endian, like the GPU
    list_stats.batches++;
    list_stats.build_usec += int64_t(now_usec() - t0);
}

Dictionary GpuTable::get_stats() const {
    Dictionary d;
    const std::shared_ptr<const InstanceSnapshot> s = snapshot();
    d["records"] = s ? int64_t(s->records.size()) : int64_t(0);
    d["capacity"] = int64_t(capacity);
    d["max_chunk_list"] = int64_t(max_chunk_list());
    d["list_batches"] = list_stats.batches;
    d["list_build_ms_avg"] = list_stats.batches > 0 ? double(list_stats.build_usec) / 1000.0 / double(list_stats.batches) : 0.0;
    d["list_truncated"] = list_stats.truncated;
    d["palette_truncated"] = list_stats.palette_truncated;
    Dictionary per_lod;
    for (int lod = 0; lod < ChunkMath::MAX_LOD_NUMBERS; lod++) {
        if (list_stats.chunks[lod] == 0) {
            continue;
        }
        Dictionary l;
        l["chunks"] = list_stats.chunks[lod];
        l["avg"] = double(list_stats.entries[lod]) / double(list_stats.chunks[lod]);
        l["max"] = list_stats.longest[lod];
        l["palette_avg"] = double(list_stats.palette_entries[lod]) / double(list_stats.chunks[lod]);
        l["palette_max"] = list_stats.palette_longest[lod];
        per_lod[itos(lod)] = l;
    }
    d["list_len_per_lod"] = per_lod;
    d.merge(scene_stats.duplicate(true)); // a copy: a Dictionary is shared by reference
    return d;
}

} // namespace Archetype
