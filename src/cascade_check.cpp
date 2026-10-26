#include "cascade_check.hpp"

#include "cascade.hpp"
#include "cascade_check_points.hpp"
#include "cascade_check_sets.hpp"
#include "cascade_measure.hpp"
#include "cascade_passes.hpp"
#include "debug_log.hpp"
#include "gpu_frame.hpp"
#include "sdf_cascade.hpp"

#include "ov/clock.hpp"

#include <algorithm>
#include <memory>
#include <vector>

using namespace Cascade;
using namespace CascadeCheckSets;
using ExactNoise::bits_float;
using ExactNoise::exact_pow2;
using ExactNoise::float_bits;

namespace {

// A buffer's bytes (p_size 0: to its end).
ov::bytes read_bytes(ov::gpu::Device *p_device, ov::gpu::Buffer p_buffer, uint64_t p_offset = 0, uint64_t p_size = 0) {
    return p_device->read_buffer(p_buffer, p_offset, p_size);
}

constexpr uint32_t MAX_BATCH = 128;
constexpr int CHUNKS_PER_PLACE = 14;
constexpr int CHILDREN = 8;
constexpr uint32_t TIMING_CHUNKS = 128;

struct TestChunk {
    ChunkId id;
    int lod = 0;
    ov::str place;
};

struct Pair {
    size_t child;
    size_t parent;
};

// The batch of LOD L: 14 chunks near each of 4 places (0, +2^60, -2^60, straddling a 2^32-voxel boundary on
// every axis), 8 more (the "children"), and the children's parents (LOD L + 1) when that LOD exists.
std::vector<TestChunk> batch_chunks(int p_lod, int p_lods, const ChunkParam &p_param, std::vector<Pair> &r_pairs) {
    const int64_t span = (p_param.half_chunk_voxels() * 2) << p_lod;
    const int64_t P32 = int64_t(1) << 32;
    struct Place {
        const char *name;
        int64_t base;
        int64_t range;
    };
    const Place places[] = {
        { "zero", 0, 64 * span },
        { "+2^60", (int64_t(1) << 60) - 123457, 64 * span },
        { "-2^60", -(int64_t(1) << 60) + 54321, 64 * span },
        { "2^32", P32, (int64_t(20) << p_lod) },
    };
    const int64_t boundaries[] = { P32, -P32, 3 * P32, -5 * P32 };
    std::vector<TestChunk> chunks;
    uint32_t n = 0;
    auto add = [&](const Place &p_place, int p_index) {
        int64_t base = p_place.base;
        if (p_place.base == P32) {
            base = boundaries[p_index % 4];
        }
        Vector3i64 v;
        for (int a = 0; a < 3; a++) {
            v[a] = base + ExactNoise::random_offset(uint32_t(p_lod), n, uint32_t(a) + 0xCA5Cu, p_place.range);
        }
        n++;
        chunks.push_back({ ChunkMath::chunk_at_voxel(v, p_lod, p_param), p_lod, p_place.name });
    };
    for (const Place &place : places) {
        for (int i = 0; i < CHUNKS_PER_PLACE; i++) {
            add(place, i);
        }
    }
    for (int i = 0; i < CHILDREN; i++) {
        add(places[i % 4], i);
        if (p_lod + 1 < p_lods) {
            const size_t child = chunks.size() - 1;
            r_pairs.push_back({ child, child + 1 });
            chunks.push_back({ ChunkMath::get_parent_from_child(chunks[child].id), p_lod + 1, chunks[child].place + " parent" });
        }
    }
    return chunks;
}

struct Checks {
    int64_t values = 0;
    int64_t differing = 0;
    int64_t samples = 0;
    int64_t samples_differing = 0;
    int64_t relation_points = 0;
    int64_t relation_differing = 0;
    int64_t relation_modulated = 0; // G3 pairs at a modulated level: no prediction (check_relation)
    int64_t chunks = 0;
    int64_t levels = 0;
    std::vector<int64_t> chunks_per_lod;
    std::vector<int64_t> differing_per_lod;
    ov::Value examples = ov::Value::array();

    // Callers test wants_example() first: formatting every mismatch of a broken run would take hours.
    bool wants_example() const { return examples.size() < 24; }
    void example(const ov::str &p_text) {
        if (wants_example()) {
            examples.push_back(p_text);
        }
    }
};

// One channel set: every batch through the GPU passes level by level, compared with Cascade::run.
class SetChecker {
public:
    SetChecker(ov::gpu::Device *p_device, SpikeSet p_set, int p_lods, const ChunkParam &p_param, Checks &r_checks) :
            device(p_device), set(p_set), lods(p_lods), param(p_param), checks(r_checks) {
        configure_variant(sdf, p_set, p_lods, param);
        // The channels the GPU carries (an empty FieldRange: the ground's only): the chunk passes' range, every level with
        // the biome pick (CascadeGLSL::field_range); the CPU's runs carry the same.
        range = CascadeGLSL::field_range(&sdf, false);
        all_fields = range.everywhere();
        nch = range.empty() ? channel_count(sdf.get_spec().set) : sdf.get_spec().gpu_channels();
        nv = CascadeGLSL::vec4s_per_point(sdf.get_spec(), range);
        cs = p_param.chunk_size;
        cells = uint64_t(cs) * cs * cs;
        input = device->create_buffer(ov::gpu::BufferKind::STORAGE, MAX_BATCH * GPUFrame::CHUNK_INPUT_BYTES);
        const ov::bytes param_bytes = (GPUFrame::chunk_param_bytes(param));
        param_ubo = device->create_buffer(ov::gpu::BufferKind::UNIFORM, param_bytes.size(), param_bytes.data());
        samples = device->create_buffer(ov::gpu::BufferKind::STORAGE, MAX_BATCH * cells * 16);
        debug = device->create_buffer(ov::gpu::BufferKind::STORAGE, MAX_BATCH * cells * nv * 16);
        debug_samples = device->create_buffer(ov::gpu::BufferKind::STORAGE, MAX_BATCH * cells * 16);
    }

    ~SetChecker() {
        debug_passes.free();
        passes.free();
        for (const ov::gpu::Buffer rid : { input, param_ubo, samples, debug, debug_samples }) {
            device->free(rid);
        }
    }

    void run() {
        // The meshers' variant for every level and the samples; the debug variant (same scratch) re-runs
        // level 0 to read its channels.
        if (!passes.create(device, &sdf, param, MAX_BATCH, input, param_ubo, samples) ||
                !debug_passes.create(device, &sdf, param, MAX_BATCH, input, param_ubo, debug_samples, debug, &passes)) {
            checks.example("set " + set_label(set) + ": the cascade passes failed to compile");
            checks.differing++;
            return;
        }
        checks.chunks_per_lod.resize(lods);
        checks.differing_per_lod.resize(lods);
        for (int lod = 0; lod < lods; lod++) {
            run_batch(lod);
        }
    }

private:
    ov::gpu::Device *device;
    SpikeSet set;
    int lods;
    ChunkParam param;
    Checks &checks;
    SDFCascade sdf;
    int nch = 0;
    uint32_t nv = 0;
    FieldRange range;
    bool all_fields = false;
    int cs = 0;
    uint64_t cells = 0;
    ov::gpu::Buffer input;
    ov::gpu::Buffer param_ubo;
    ov::gpu::Buffer samples;
    ov::gpu::Buffer debug;
    ov::gpu::Buffer debug_samples;
    CascadePasses passes;
    CascadePasses debug_passes;

    // The batch being checked: its chunks, their Λ_L boxes and every CPU level (top first).
    std::vector<TestChunk> chunks;
    std::vector<Box> boxes;
    std::vector<std::vector<Level>> cpu;

    void run_batch(int p_lod) {
        std::vector<Pair> pairs;
        chunks = batch_chunks(p_lod, lods, param, pairs);
        const uint32_t count = uint32_t(chunks.size());
        cpu.assign(count, std::vector<Level>());
        boxes.assign(count, Box());
        ov::bytes ids;
        ids.resize(count * GPUFrame::CHUNK_INPUT_BYTES);
        int min_lod = 64;
        int max_lod = 0;
        for (uint32_t c = 0; c < count; c++) {
            boxes[c] = chunk_box(chunks[c].id, param);
            Cascade::run(sdf.get_spec(), boxes[c], chunks[c].lod, &cpu[c], Source(), all_fields);
            GPUFrame::encode_chunk_input(ids, c, chunks[c].id, param);
            min_lod = ov::min(min_lod, chunks[c].lod);
            max_lod = ov::max(max_lod, chunks[c].lod);
            checks.chunks_per_lod[size_t(chunks[c].lod)]++;
        }
        checks.chunks += count;
        device->update_buffer(input, 0, ids.size(), ids.data());
        ov::bytes final_channels;
        for (int j = passes.level_count(min_lod) - 1; j >= 0; j--) {
            passes.record_level(j, 0, count, ov::ivec2(min_lod, max_lod));
            if (j == 0) {
                debug_passes.record_level(0, 0, count, ov::ivec2(min_lod, max_lod));
            }
            device->sync();
            const uint64_t n = uint64_t(level_bound(cs, j));
            const uint64_t bytes = count * n * n * n * nv * 16;
            if (j > 0) {
                // The scratch is planar (vec4 v of every slot together): the whole buffer.
                const ov::bytes scratch = read_bytes(device, (j & 1) ? passes.get_scratch_a() : passes.get_scratch_b());
                compare_level(j, scratch, ov::bytes(), uint64_t(scratch.size()) / 16 / nv);
                continue;
            }
            final_channels = read_bytes(device, debug, 0, bytes);
            const ov::bytes final_samples = read_bytes(device, samples, 0, count * cells * 16);
            // The debug variant's samples must be the same bits as the meshers' variant's.
            if (read_bytes(device, debug_samples, 0, count * cells * 16) != final_samples) {
                checks.samples_differing++;
                checks.example("set " + set_label(set) + ov::format(" LOD %d batch: the debug variant's samples differ from the meshers' variant's", p_lod));
            }
            compare_level(0, final_channels, final_samples);
        }
        for (const Pair &pair : pairs) {
            check_relation(pair, reinterpret_cast<const float *>(final_channels.data()));
        }
    }

    // Level j of every chunk of the batch (GPU layout: x fastest, stride level_bound(j)) against the CPU's
    // (x slowest); at j = 0 also the samples against SDFCascade::sdf + skirt of the CPU channels. p_plane: the planar
    // scratch's plane (vec4s), 0 for the per-point debug channels.
    void compare_level(int p_j, const ov::bytes &p_data, const ov::bytes &p_samples, uint64_t p_plane = 0) {
        const float *gpu = reinterpret_cast<const float *>(p_data.data());
        const uint64_t n = uint64_t(level_bound(cs, p_j));
        const uint64_t points = n * n * n;
        const Spec &spec = sdf.get_spec();
        // The ground everywhere, the fields where the range carries them, the carry groups the last level reads on it and on
        // the level above, which derives them (the GPU stores no other).
        const int ground_ch = channel_count(spec.set);
        auto compared = [&](int p_lod, int p_m, int p_ch) {
            if (p_ch < ground_ch) {
                return true;
            }
            if (p_ch < spec.gpu_carry(0)) {
                return range.carried(p_lod, p_m);
            }
            return range.carries(p_lod, p_m) && (p_ch - spec.gpu_carry(0)) / FIELD_CHANNELS < spec.modulation->carry_used(p_lod);
        };
        for (uint32_t c = 0; c < uint32_t(chunks.size()); c++) {
            const int lod_c = chunks[c].lod;
            const int top_c = spec.run_top(lod_c, all_fields);
            const int m = lod_c + p_j;
            if (m > top_c) {
                continue;
            }
            const Level &level = cpu[c][size_t(top_c - m)];
            OV_ERR_CONTINUE(level.m != m);
            checks.levels++;
            const Box &box = level.box;
            int64_t differing = 0;
            for_each_point(box, [&](const Vector3i64 &p_p, int64_t p_cpu_i) {
                const Vector3i64 i = p_p - Vector3i64(box.lo[0], box.lo[1], box.lo[2]);
                const uint64_t slot = c * points + uint64_t(i.x) + n * (uint64_t(i.y) + n * uint64_t(i.z));
                for (int ch = 0; ch < nch; ch++) {
                    if (!compared(lod_c, m, ch)) {
                        continue;
                    }
                    const float want = level.ch[spec.slot_of_gpu(ch)][size_t(p_cpu_i)];
                    const float got = p_plane > 0 ? gpu[(uint64_t(ch / 4) * p_plane + slot) * 4 + ch % 4] : gpu[slot * nv * 4 + ch];
                    checks.values++;
                    if (float_bits(want) != float_bits(got)) {
                        differing++;
                        if (checks.wants_example()) {
                            checks.example(ov::format("set %s LOD %d chunk %s (%s) level m=%d point %s channel %d: gpu %s cpu %s (bits %x / %x)", set_label(set), lod_c,
                                    WorldCoord::to_string(chunks[c].id), chunks[c].place, m, WorldCoord::to_string(p_p), ch, got, want, float_bits(got), float_bits(want)));
                        }
                    }
                }
                if (p_j == 0) {
                    compare_sample(c, level, p_p, i, size_t(p_cpu_i), reinterpret_cast<const float *>(p_samples.data()) + slot * 4);
                }
            });
            checks.differing += differing;
            checks.differing_per_lod[size_t(lod_c)] += differing;
        }
    }

    void compare_sample(uint32_t p_chunk, const Level &p_level, const Vector3i64 &p_p, const Vector3i64 &p_grid, size_t p_cpu_i, const float *p_got) {
        // The sample's ground channels (the climate's ground terms on the slots, SDFCascade::ground_channels).
        float slots[SLOTS], ch[CHANNELS];
        p_level.slots_at(int64_t(p_cpu_i), slots);
        sdf.ground_channels(slots, ch);
        const int lod = chunks[p_chunk].lod;
        const ExactNoise::Float4 want = sdf.with_skirt(sdf.sdf(ch, p_p * (int64_t(1) << lod), lod), lod, Float3{ float(p_grid.x), float(p_grid.y), float(p_grid.z) });
        const float w[4] = { want.x, want.y, want.z, want.w };
        checks.samples++;
        for (int k = 0; k < 4; k++) {
            if (float_bits(w[k]) != float_bits(p_got[k])) {
                checks.samples_differing++;
                if (checks.wants_example()) {
                    checks.example(ov::format("set %s LOD %d chunk %s sample %s: gpu (%s %s %s %s) cpu (%s %s %s %s)", set_label(set), lod,
                            WorldCoord::to_string(chunks[p_chunk].id), WorldCoord::to_string(p_grid), p_got[0], p_got[1], p_got[2], p_got[3], w[0], w[1], w[2], w[3]));
                }
                return;
            }
        }
    }

    // G3: the GPU's Λ_L channels of a child against its parent's Λ_{L+1} channels at the points they share:
    // V_L = flush(V_{L+1} * 2^-order + O_L).
    void check_relation(const Pair &p_pair, const float *p_channels) {
        const Spec &spec = sdf.get_spec();
        const Box &cb = boxes[p_pair.child];
        const Box &pb = boxes[p_pair.parent];
        const int lod = chunks[p_pair.child].lod;
        // A child modulated at its own level reads the carry its level L + 1 derives from the fields, which the parent
        // chunk's samples (LOD L + 1) don't carry: no prediction from its readback there.
        if (spec.modulation && spec.modulation->modulated(lod) != 0u) {
            checks.relation_modulated++;
            return;
        }
        auto at = [&](size_t p_chunk, const Box &p_box, const Vector3i64 &p_p, int p_ch) {
            const uint64_t slot = p_chunk * cells + uint64_t(p_p.x - p_box.lo[0]) + uint64_t(cs) * (uint64_t(p_p.y - p_box.lo[1]) + uint64_t(cs) * uint64_t(p_p.z - p_box.lo[2]));
            return p_channels[slot * nv * 4 + p_ch];
        };
        for_each_point(cb, [&](const Vector3i64 &p_p, int64_t) {
            const Vector3i64 q(p_p.x / 2, p_p.y / 2, p_p.z / 2);
            if (((p_p.x | p_p.y | p_p.z) & 1) != 0 || !pb.contains(q)) {
                return;
            }
            // H(V_{L+1}) at a copy point (every slot, the fields' too), then Λ_L's insertions (Cascade::insert_at).
            float expect[SLOTS] = {};
            const int n0 = channel_count(spec.set);
            for (int ch = 0; ch < nch; ch++) {
                const int local = ch < n0 ? ch : (ch - n0) % 4;
                expect[spec.slot_of_gpu(ch)] = at(p_pair.parent, pb, q, ch) * exact_pow2(-channel_order(local));
            }
            insert_at(spec, lod, p_p, expect);
            checks.relation_points++;
            // The ground: a run's last level carries no field, and its carry is derived from the level above's.
            for (int ch = 0; ch < n0; ch++) {
                if (float_bits(flush_value(expect[spec.slot_of_gpu(ch)])) != float_bits(at(p_pair.child, cb, p_p, ch))) {
                    checks.relation_differing++;
                    if (checks.wants_example()) {
                        checks.example(ov::format("set %s G3 LOD %d child %s point %s channel %d", set_label(set), lod, WorldCoord::to_string(chunks[p_pair.child].id),
                                WorldCoord::to_string(p_p), ch));
                    }
                    return;
                }
            }
        });
    }
};

void check_set(ov::gpu::Device *p_device, SpikeSet p_set, int p_lods, const ChunkParam &p_param, Checks &r_checks) {
    SetChecker checker(p_device, p_set, p_lods, p_param, r_checks);
    checker.run();
}

// G9 (instance table): a cascade terrain with a scatter of rocks and pillars, LODs 0 to TABLE_LODS - 1, a batch of the
// chunks holding the records per LOD with their instance lists and records on the GPU: every sample of the meshers'
// fold variant against SDFCascade::sample_sdf + skirt with the chunk's list (fold_onto + finish_table where an
// instance passes its box test, the bare cascade elsewhere), bit for bit.
constexpr int TABLE_LODS = 7;

// The LOD-p_lod chunks holding the records' anchors (at most MAX_BATCH), encoded with their lists.
std::vector<ChunkId> table_batch(const Archetype::InstanceSnapshot &p_snap, const SDFGpuTable &p_table, int p_lod, const ChunkParam &p_param,
        ov::bytes &r_inputs, ov::bytes &r_words) {
    std::vector<ChunkId> ids;
    for (uint32_t r = 0; r < p_snap.records.size() && ids.size() < MAX_BATCH; r++) {
        const ChunkId id = ChunkMath::chunk_at_voxel(p_snap.records[r].anchor, p_lod, p_param);
        if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
            ids.push_back(id);
        }
    }
    r_inputs.resize(ids.size() * GPUFrame::CHUNK_INPUT_BYTES);
    for (size_t c = 0; c < ids.size(); c++) {
        GPUFrame::encode_chunk_input(r_inputs.data() + c * GPUFrame::CHUNK_INPUT_BYTES, ids[c], p_param);
    }
    // The batch keeps the chunks whose lists fit the buffer (the same capacity as the list buffer below).
    ids.resize(p_table.build_lists(ids, p_param, r_inputs, r_words, MAX_BATCH * p_table.max_chunk_list()));
    r_inputs.resize(ids.size() * GPUFrame::CHUNK_INPUT_BYTES);
    return ids;
}

// One chunk's GPU samples p_got (32^3 vec4) against the CPU's; counts the samples an instance changed in r_folded.
void compare_table_chunk(const SDFCascade &p_sdf, const Archetype::InstanceSnapshot &p_snap, const ChunkId &p_id, int p_lod, const ChunkParam &p_param,
        const float *p_got, Checks &r_checks, int64_t &r_folded) {
    std::vector<uint32_t> list;
    p_snap.chunk_list(p_id, list);
    const Box box = chunk_box(p_id, p_param);
    std::vector<Level> levels;
    Cascade::run(p_sdf.get_spec(), box, p_lod, &levels);
    const Level &level = levels.back();
    const uint64_t cs = uint64_t(p_param.chunk_size);
    for_each_point(box, [&](const Vector3i64 &p_p, int64_t p_cpu_i) {
        const Vector3i64 i = p_p - Vector3i64(box.lo[0], box.lo[1], box.lo[2]);
        float ch[CHANNELS] = {};
        for (int k = 0; k < channel_count(p_sdf.get_spec().set); k++) {
            ch[k] = level.ch[k][size_t(p_cpu_i)];
        }
        WorldPos p;
        p.voxel = VoxelPos(p_p.x << p_lod, p_p.y << p_lod, p_p.z << p_lod);
        const Float3 grid{ float(i.x), float(i.y), float(i.z) };
        float extra = 0.0f;
        const ExactNoise::Float4 folded = p_sdf.sample_sdf(p_snap, list.data(), uint32_t(list.size()), ch, p, p_lod, &extra);
        const ExactNoise::Float4 want = p_sdf.with_skirt(folded, p_lod, grid, extra);
        const ExactNoise::Float4 bare = p_sdf.with_skirt(p_sdf.sdf(ch, p.voxel, p_lod), p_lod, grid);
        r_folded += float_bits(want.w) != float_bits(bare.w) ? 1 : 0;
        const float *g = p_got + (uint64_t(i.x) + cs * (uint64_t(i.y) + cs * uint64_t(i.z))) * 4;
        r_checks.samples++;
        if (!ExactNoise::same_bits(want, ExactNoise::Float4{ g[0], g[1], g[2], g[3] })) {
            r_checks.samples_differing++;
            if (r_checks.wants_example()) {
                r_checks.example(ov::format("LOD %d chunk %s sample %s: gpu (%s %s %s %s) cpu (%s %s %s %s)", p_lod, WorldCoord::to_string(p_id),
                        WorldCoord::to_string(i), g[0], g[1], g[2], g[3], want.x, want.y, want.z, want.w));
            }
        }
    });
}

ov::Value check_table(ov::gpu::Device *p_device, int p_lods, const ChunkParam &p_param) {
    ov::Value d = ov::Value::object();
    ChunkParam param = p_param;
    SDFCascade sdf;
    sdf.set_scatter(600, 160.0);
    sdf.set_stream_sources(false); // placed whole: the chunks' lists are the snapshot's (the streamed ones: unit tests)
    param.skirt_depth = ov::max(param.skirt_depth, sdf.min_skirt_depth(param, p_lods));
    sdf.configure(param, p_lods);
    const std::shared_ptr<const Archetype::InstanceSnapshot> snap = sdf.get_snapshot();
    SDFGpuTable *table = sdf.get_gpu_table();
    const int cs = param.chunk_size;
    const uint64_t cells = uint64_t(cs) * cs * cs;
    const ov::gpu::Buffer input = p_device->create_buffer(ov::gpu::BufferKind::STORAGE, MAX_BATCH * GPUFrame::CHUNK_INPUT_BYTES);
    const ov::bytes param_bytes = (GPUFrame::chunk_param_bytes(param));
    const ov::gpu::Buffer param_ubo = p_device->create_buffer(ov::gpu::BufferKind::UNIFORM, param_bytes.size(), param_bytes.data());
    const ov::gpu::Buffer samples = p_device->create_buffer(ov::gpu::BufferKind::STORAGE, MAX_BATCH * cells * 16);
    const ov::gpu::Buffer records = p_device->create_buffer(ov::gpu::BufferKind::STORAGE, uint64_t(table->instance_capacity()) * table->record_size());
    const ov::gpu::Buffer list = p_device->create_buffer(ov::gpu::BufferKind::STORAGE, uint64_t(MAX_BATCH) * table->max_chunk_list() * 4);
    SDFGpuTable::UploadCursor sent;
    SDFGpuTable::upload_records(*p_device, records, table->record_bytes(sent, sent));
    CascadePasses passes;
    Checks checks;
    int64_t folded = 0;
    if (!passes.create(p_device, &sdf, param, MAX_BATCH, input, param_ubo, samples, ov::gpu::Buffer(), nullptr, records, list)) {
        checks.example("the cascade passes failed to compile");
        checks.samples_differing++;
    }
    for (int lod = 0; lod < TABLE_LODS && passes.is_valid(); lod++) {
        ov::bytes inputs, words;
        const std::vector<ChunkId> ids = table_batch(*snap, *table, lod, param, inputs, words);
        p_device->update_buffer(input, 0, inputs.size(), inputs.data());
        if (!words.empty()) {
            p_device->update_buffer(list, 0, words.size(), words.data());
        }
        passes.record_all(ov::ivec2(lod, lod), uint32_t(ids.size()), true); // the fold variant, empty lists included (the bare bits there)
        p_device->sync();
        const ov::bytes got_bytes = read_bytes(p_device, samples, 0, ids.size() * cells * 16);
        const float *got = reinterpret_cast<const float *>(got_bytes.data());
        for (size_t c = 0; c < ids.size(); c++) {
            compare_table_chunk(sdf, *snap, ids[c], lod, param, got + c * cells * 4, checks, folded);
        }
        checks.chunks += int64_t(ids.size());
    }
    passes.free();
    for (const ov::gpu::Buffer rid : { input, param_ubo, samples, records, list }) {
        p_device->free(rid);
    }
    d["records"] = int64_t(snap->records.size());
    d["chunks"] = checks.chunks;
    d["samples"] = checks.samples;
    d["samples_folded"] = folded;
    d["samples_differing"] = checks.samples_differing;
    d["examples"] = checks.examples;
    d["ok"] = checks.samples_differing == 0 && folded > 0 && checks.samples > 0;
    return d;
}

// G10 (biome materials): the meshers' cascade passes on a climate set (4c and the other sets asked), every LOD's batch: each sample's
// sdf_material (no instance: the ground's pick, cascade_ground_material, at the solid samples) against SDFCascade::ground_material of the CPU
// run's last level slots (the pick derived on the level above, like the chunk passes), bit for bit; and how many biomes showed.
// Empty for a set without a climate (8, 4) or with fewer than 3 biomes (nothing for ok to count); a climate set whose
// table was refused fails.
ov::Value check_biomes(ov::gpu::Device *p_device, SpikeSet p_set, int p_lods, const ChunkParam &p_param) {
    ov::Value d = ov::Value::object();
    if (p_set == SpikeSet::H8 || p_set == SpikeSet::L4) {
        return d;
    }
    ChunkParam param = p_param;
    SDFCascade sdf;
    configure_variant(sdf, p_set, p_lods, param);
    if (sdf.get_climate() == nullptr) {
        d["ok"] = false;
        d["error"] = "the climate was refused: " + sdf.get_climate_error();
        return d;
    }
    if (sdf.get_climate()->biomes.size() < 3) {
        return d;
    }
    const int cs = param.chunk_size;
    const uint64_t cells = uint64_t(cs) * cs * cs;
    const ov::gpu::Buffer input = p_device->create_buffer(ov::gpu::BufferKind::STORAGE, MAX_BATCH * GPUFrame::CHUNK_INPUT_BYTES);
    const ov::bytes param_bytes = (GPUFrame::chunk_param_bytes(param));
    const ov::gpu::Buffer param_ubo = p_device->create_buffer(ov::gpu::BufferKind::UNIFORM, param_bytes.size(), param_bytes.data());
    const ov::gpu::Buffer samples = p_device->create_buffer(ov::gpu::BufferKind::STORAGE, MAX_BATCH * cells * 16);
    const ov::gpu::Buffer materials = p_device->create_buffer(ov::gpu::BufferKind::STORAGE, MAX_BATCH * cells * 4);
    CascadePasses passes;
    Checks checks;
    std::vector<int64_t> seen(Climate::MAX_BIOMES + 1, 0);
    if (!CascadeGLSL::ground_materials(&sdf) ||
            !passes.create(p_device, &sdf, param, MAX_BATCH, input, param_ubo, samples, ov::gpu::Buffer(), nullptr, ov::gpu::Buffer(), ov::gpu::Buffer(), materials)) {
        checks.example("no ground materials, or the cascade passes failed to compile");
        checks.samples_differing++;
    }
    for (int lod = 0; lod < p_lods && passes.is_valid(); lod++) {
        std::vector<Pair> pairs;
        const std::vector<TestChunk> chunks = batch_chunks(lod, p_lods, param, pairs);
        ov::bytes ids;
        ids.resize(int64_t(chunks.size()) * GPUFrame::CHUNK_INPUT_BYTES);
        ov::ivec2 lods(64, 0);
        for (size_t c = 0; c < chunks.size(); c++) {
            GPUFrame::encode_chunk_input(ids, int64_t(c), chunks[c].id, param);
            lods = ov::ivec2(ov::min(lods.x, chunks[c].lod), ov::max(lods.y, chunks[c].lod));
        }
        p_device->update_buffer(input, 0, ids.size(), ids.data());
        passes.record_all(lods, uint32_t(chunks.size()));
        p_device->sync();
        const ov::bytes got_bytes = read_bytes(p_device, materials, 0, chunks.size() * cells * 4);
        const uint32_t *got = reinterpret_cast<const uint32_t *>(got_bytes.data());
        for (size_t c = 0; c < chunks.size(); c++) {
            const int l = chunks[c].lod;
            const Box box = chunk_box(chunks[c].id, param);
            const Level level = Cascade::run(sdf.get_spec(), box, l, nullptr, Source(), true);
            for_each_point(box, [&](const Vector3i64 &p_p, int64_t p_cpu_i) {
                const Vector3i64 i = p_p - Vector3i64(box.lo[0], box.lo[1], box.lo[2]);
                float slots[SLOTS];
                level.slots_at(p_cpu_i, slots);
                // Air samples (the sample with its skirt above 0) keep 0: no vertex reads them.
                float ch[CHANNELS];
                sdf.ground_channels(slots, ch);
                const ExactNoise::Float4 r = sdf.with_skirt(sdf.sdf(ch, p_p * (int64_t(1) << l), l), l, Float3{ float(i.x), float(i.y), float(i.z) });
                const uint32_t want = r.w > 0.0f ? 0u : sdf.ground_material(slots);
                const uint32_t g = got[c * cells + uint64_t(i.x) + uint64_t(cs) * (uint64_t(i.y) + uint64_t(cs) * uint64_t(i.z))];
                seen[ov::min(want, uint32_t(Climate::MAX_BIOMES))]++;
                checks.samples++;
                if (g != want) {
                    checks.samples_differing++;
                    if (checks.wants_example()) {
                        checks.example(ov::format("LOD %d chunk %s (%s) sample %s: gpu %d cpu %d", l, WorldCoord::to_string(chunks[c].id), chunks[c].place,
                                WorldCoord::to_string(i), g, want));
                    }
                }
            });
        }
        checks.chunks += int64_t(chunks.size());
    }
    passes.free();
    for (const ov::gpu::Buffer rid : { input, param_ubo, samples, materials }) {
        p_device->free(rid);
    }
    ov::Value per_biome = ov::Value::object();
    int biomes = 0;
    for (size_t m = 1; m < seen.size(); m++) {
        if (seen[m] > 0) {
            per_biome[sdf.get_climate()->biomes[m - 1].name] = seen[m];
            biomes++;
        }
    }
    d["chunks"] = checks.chunks;
    d["samples"] = checks.samples;
    d["samples_differing"] = checks.samples_differing;
    d["samples_per_biome"] = per_biome;
    d["examples"] = checks.examples;
    d["samples_air"] = seen[0]; // no pick there
    d["ok"] = checks.samples_differing == 0 && seen[0] < checks.samples && biomes >= 3;
    return d;
}

struct BatchTime {
    double wall_ms = 0.0; // submit + sync
    double gpu_ms = -1.0; // timestamps, -1 if they did not come back
    std::vector<double> levels; // GPU ms per level, j = top ... 0
    int timestamp_frames = -1; // extra frames before the timestamps came back
};

// Every level of one batch (input already uploaded), timed.
BatchTime time_batch(ov::gpu::Device *p_device, CascadePasses &p_passes, int p_lod, const ov::str &p_prefix) {
    BatchTime out;
    const int levels = p_passes.level_count(p_lod);
    p_device->timestamp(p_prefix + "b");
    for (int j = levels - 1; j >= 0; j--) {
        p_passes.record_level(j, 0, TIMING_CHUNKS, ov::ivec2(p_lod, p_lod));
        p_device->timestamp(p_prefix + ov::itos(j));
    }
    const uint64_t t0 = now_usec();
    p_device->sync();
    out.wall_ms = double(now_usec() - t0) / 1000.0;
    // A local device returns a frame's timestamps when that frame slot is begun again: look after the
    // sync, then after up to 3 empty frames.
    int64_t t_begin = -1;
    std::vector<int64_t> t_level(levels, -1);
    for (int extra = 0; extra <= 3 && t_begin < 0; extra++) {
        if (extra > 0) {
            p_device->sync();
        }
        for (const ov::gpu::Timestamp &ts : p_device->timestamps()) {
            const ov::str &name = ts.name;
            if (name.compare(0, p_prefix.size(), p_prefix) != 0) {
                continue;
            }
            const ov::str tail = name.substr(p_prefix.length());
            const int64_t t = int64_t(ts.gpu_ns);
            if (tail == "b") {
                t_begin = t;
            } else if (std::atoi(tail.c_str()) < levels) {
                t_level[size_t(std::atoi(tail.c_str()))] = t;
            }
        }
        if (t_begin >= 0) {
            out.timestamp_frames = extra;
        }
    }
    if (t_begin >= 0 && t_level[0] >= t_begin) {
        out.gpu_ms = double(t_level[0] - t_begin) / 1e6;
        int64_t prev = t_begin;
        for (int j = levels - 1; j >= 0; j--) {
            out.levels.push_back(t_level[j] >= prev ? double(t_level[j] - prev) / 1e6 : -1.0);
            prev = ov::max(prev, t_level[j]);
        }
    }
    return out;
}

// G6 / G7: GPU time of every level for single-LOD batches of 128 chunks.
ov::Value timing(ov::gpu::Device *p_device, const std::vector<SpikeSet> &p_sets, int p_lods, const ChunkParam &p_param, int p_runs) {
    const int cs = p_param.chunk_size;
    const uint64_t cells = uint64_t(cs) * cs * cs;
    const ov::gpu::Buffer input = p_device->create_buffer(ov::gpu::BufferKind::STORAGE, TIMING_CHUNKS * GPUFrame::CHUNK_INPUT_BYTES);
    const ov::bytes param_bytes = (GPUFrame::chunk_param_bytes(p_param));
    const ov::gpu::Buffer param_ubo = p_device->create_buffer(ov::gpu::BufferKind::UNIFORM, param_bytes.size(), param_bytes.data());
    const ov::gpu::Buffer samples = p_device->create_buffer(ov::gpu::BufferKind::STORAGE, TIMING_CHUNKS * cells * 16);
    std::vector<std::unique_ptr<SDFCascade>> sdfs;
    std::vector<std::unique_ptr<CascadePasses>> passes;
    for (const SpikeSet set : p_sets) {
        sdfs.push_back(std::make_unique<SDFCascade>());
        ChunkParam param = p_param;
        configure_variant(*sdfs.back(), set, p_lods, param); // the same chunks, only the metres differ
        passes.push_back(std::make_unique<CascadePasses>());
        passes.back()->create(p_device, sdfs.back().get(), p_param, TIMING_CHUNKS, input, param_ubo, samples);
    }
    std::vector<int> lods;
    for (int lod = 0; lod < p_lods; lod += 4) {
        lods.push_back(lod);
    }
    // [set][lod index] -> per-run times (ms): wall clock and GPU timestamps; per level GPU ms (last runs).
    std::vector<std::vector<std::vector<double>>> wall(p_sets.size(), std::vector<std::vector<double>>(lods.size()));
    std::vector<std::vector<std::vector<double>>> gpu(p_sets.size(), std::vector<std::vector<double>>(lods.size()));
    std::vector<std::vector<std::vector<double>>> per_level(p_sets.size(), std::vector<std::vector<double>>(lods.size()));
    int timestamp_frames = -1; // extra frames before the timestamps came back (-1: never)
    for (int r = -2; r < p_runs; r++) { // two warm-up rounds
        for (size_t li = 0; li < lods.size(); li++) {
            const int lod = lods[li];
            ov::bytes ids;
            ids.resize(TIMING_CHUNKS * GPUFrame::CHUNK_INPUT_BYTES);
            for (uint32_t c = 0; c < TIMING_CHUNKS; c++) {
                GPUFrame::encode_chunk_input(ids, c, CascadeMeasure::random_chunk(lod, 0x7133u + uint32_t(r + 2), int(c)), p_param);
            }
            p_device->update_buffer(input, 0, ids.size(), ids.data());
            p_device->sync();
            for (size_t si = 0; si < p_sets.size(); si++) {
                const BatchTime t = time_batch(p_device, *passes[si], lod, ov::format("cs%d_%d_%d_", r + 2, lod, int(si)));
                if (r < 0) {
                    continue;
                }
                wall[si][li].push_back(t.wall_ms);
                timestamp_frames = ov::max(timestamp_frames, t.timestamp_frames);
                if (t.gpu_ms >= 0.0) {
                    gpu[si][li].push_back(t.gpu_ms);
                    per_level[si][li] = t.levels;
                }
            }
        }
    }
    ov::Value out = ov::Value::object();
    ov::Value rows = ov::Value::array();
    for (size_t si = 0; si < p_sets.size(); si++) {
        for (size_t li = 0; li < lods.size(); li++) {
            ov::Value row = ov::Value::object();
            row["set"] = set_label(p_sets[si]);
            row["lod"] = lods[li];
            row["levels"] = passes[si]->level_count(lods[li]);
            row["wall_ms_median"] = median(wall[si][li]);
            row["wall_ms_min"] = wall[si][li].empty() ? 0.0 : *std::min_element(wall[si][li].begin(), wall[si][li].end());
            row["gpu_ms_median"] = median(gpu[si][li]);
            row["gpu_ms_min"] = gpu[si][li].empty() ? 0.0 : *std::min_element(gpu[si][li].begin(), gpu[si][li].end());
            row["gpu_runs"] = int64_t(gpu[si][li].size());
            std::vector<double> lv;
            for (const double v : per_level[si][li]) {
                lv.push_back(v);
            }
            row["gpu_ms_per_level_last"] = ov::Value::array_of(lv); // j = levels - 1 ... 0
            rows.push_back(row);
        }
        out["scratch_mb_" + set_label(p_sets[si])] = double(passes[si]->scratch_bytes()) / 1048576.0;
    }
    out["rows"] = rows;
    out["chunks_per_batch"] = TIMING_CHUNKS;
    out["runs"] = p_runs;
    out["timestamp_frames"] = timestamp_frames;
    for (std::unique_ptr<CascadePasses> &p : passes) {
        p->free();
    }
    for (const ov::gpu::Buffer rid : { input, param_ubo, samples }) {
        p_device->free(rid);
    }
    return out;
}

} // namespace

ov::Value CascadeCheck::run(ov::gpu::Device &p_device, const ov::Value &p_options) {
    ov::Value report = ov::Value::object();
    report["ok"] = false;
    const int lods = int(p_options.get("lods", 25).as_int());
    ChunkParam param;
    param.voxel_size = float(p_options.get("voxel", 0.5).as_double());
    std::vector<SpikeSet> sets;
    const ov::str labels = p_options.get("sets", "8,4,4c").as_str();
    for (size_t from = 0; from <= labels.size();) {
        const size_t comma = std::min(labels.find(',', from), labels.size());
        SpikeSet set;
        if (parse_variant(labels.substr(from, comma - from), set)) {
            sets.push_back(set);
        }
        from = comma + 1;
    }
    ov::gpu::Device *device = &p_device;
    bool ok = true;
    if (p_options.get("check", true).as_bool()) {
        ov::Value checks_report = ov::Value::object();
        for (const SpikeSet set : sets) {
            Checks checks;
            const uint64_t t0 = now_usec();
            check_set(device, set, lods, param, checks);
            ov::Value d = ov::Value::object();
            d["chunks"] = checks.chunks;
            d["levels"] = checks.levels;
            d["values"] = checks.values;
            d["differing"] = checks.differing;
            d["samples"] = checks.samples;
            d["samples_differing"] = checks.samples_differing;
            d["relation_points"] = checks.relation_points;
            d["relation_differing"] = checks.relation_differing;
            d["relation_modulated"] = checks.relation_modulated;
            d["chunks_per_lod"] = ov::Value::array_of(checks.chunks_per_lod);
            d["differing_per_lod"] = ov::Value::array_of(checks.differing_per_lod);
            d["examples"] = checks.examples;
            d["seconds"] = double(now_usec() - t0) / 1e6;
            int64_t min_chunks = 1 << 30;
            for (int64_t v : checks.chunks_per_lod) {
                min_chunks = ov::min(min_chunks, v);
            }
            d["min_chunks_per_lod"] = min_chunks;
            const bool set_ok = checks.values > 0 && checks.samples > 0 && (checks.relation_points > 0 || checks.relation_modulated > 0) && min_chunks >= 64 && checks.differing == 0 &&
                    checks.samples_differing == 0 && checks.relation_differing == 0;
            d["ok"] = set_ok;
            ok = ok && set_ok;
            checks_report[set_label(set)] = d;
        }
        report["check"] = checks_report;
    }
    if (p_options.get("table", true).as_bool()) {
        const uint64_t t0 = now_usec();
        ov::Value d = check_table(device, lods, param);
        d["seconds"] = double(now_usec() - t0) / 1e6;
        ok = ok && d["ok"].as_bool();
        report["table"] = d;
    }
    if (p_options.get("biomes", true).as_bool()) {
        // The example climate always, and every other set asked whose climate has biomes to pick.
        ov::Value biomes = ov::Value::object();
        std::vector<SpikeSet> climates = { SpikeSet::L4C };
        for (const SpikeSet set : sets) {
            if (std::find(climates.begin(), climates.end(), set) == climates.end()) {
                climates.push_back(set);
            }
        }
        for (const SpikeSet set : climates) {
            const uint64_t t0 = now_usec();
            ov::Value d = check_biomes(device, set, lods, param);
            if (d.is_empty()) {
                continue;
            }
            d["seconds"] = double(now_usec() - t0) / 1e6;
            ok = ok && d["ok"].as_bool();
            biomes[set_label(set)] = d;
        }
        report["biomes"] = biomes;
    }
    if (p_options.get("points", true).as_bool()) {
        ov::Value points_report = ov::Value::object();
        for (const SpikeSet set : sets) {
            const uint64_t t0 = now_usec();
            ov::Value d = CascadeCheckPoints::check_points(device, set, lods, param);
            d["seconds"] = double(now_usec() - t0) / 1e6;
            ok = ok && d["ok"].as_bool();
            points_report[set_label(set)] = d;
        }
        points_report["timing"] = CascadeCheckPoints::point_timing(device, sets, lods, param, int(p_options.get("runs", 10).as_int()));
        report["points"] = points_report;
    }
    if (p_options.get("timing", true).as_bool()) {
        report["timing"] = timing(device, sets, lods, param, int(p_options.get("runs", 10).as_int()));
    }
    report["lods"] = lods;
    report["ok"] = ok;
    return report;
}
