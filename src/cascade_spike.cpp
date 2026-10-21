#include "cascade_spike.hpp"

#include "cascade.hpp"
#include "cascade_glsl.hpp"
#include "cascade_measure.hpp"
#include "gpu_frame.hpp"
#include "sdf_cascade.hpp"

#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <algorithm>
#include <memory>
#include <vector>

using namespace Cascade;
using ExactNoise::bits_float;
using ExactNoise::exact_pow2;
using ExactNoise::float_bits;

namespace {

constexpr uint32_t MAX_BATCH = 128;
constexpr int CHUNKS_PER_PLACE = 14;
constexpr int CHILDREN = 8;
constexpr uint32_t TIMING_CHUNKS = 128;

uint64_t now_usec() {
    return Time::get_singleton()->get_ticks_usec();
}

struct TestChunk {
    ChunkId id;
    int lod = 0;
    String place;
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
    int64_t chunks = 0;
    int64_t levels = 0;
    PackedInt64Array chunks_per_lod;
    PackedInt64Array differing_per_lod;
    Array examples;

    // Callers test wants_example() first: formatting every mismatch of a broken run would take hours.
    bool wants_example() const { return examples.size() < 24; }
    void example(const String &p_text) {
        if (wants_example()) {
            examples.push_back(p_text);
        }
    }
};

String set_label(ChannelSet p_set) {
    return p_set == ChannelSet::HERMITE8 ? "8" : "4";
}

// One channel set: every batch through the GPU passes level by level, compared with Cascade::run.
class SetChecker {
public:
    SetChecker(RenderingDevice *p_rd, ChannelSet p_set, int p_lods, const ChunkParam &p_param, Checks &r_checks) :
            rd(p_rd), set(p_set), lods(p_lods), param(p_param), checks(r_checks) {
        sdf.channel_set = p_set;
        sdf.configure(p_param, p_lods);
        nch = channel_count(p_set);
        nv = CascadeGLSL::vec4s_per_point(p_set);
        cs = p_param.chunk_size;
        cells = uint64_t(cs) * cs * cs;
        input = rd->storage_buffer_create(MAX_BATCH * GPUFrame::CHUNK_INPUT_BYTES);
        const PackedByteArray param_bytes = GPUFrame::chunk_param_bytes(p_param);
        param_ubo = rd->uniform_buffer_create(param_bytes.size(), param_bytes);
        samples = rd->storage_buffer_create(MAX_BATCH * cells * 16);
        debug = rd->storage_buffer_create(MAX_BATCH * cells * nv * 16);
        debug_samples = rd->storage_buffer_create(MAX_BATCH * cells * 16);
    }

    ~SetChecker() {
        debug_passes.free();
        passes.free();
        for (const RID &rid : { input, param_ubo, samples, debug, debug_samples }) {
            rd->free_rid(rid);
        }
    }

    void run() {
        // The meshers' variant for every level and the samples; the debug variant (same scratch) re-runs
        // level 0 to read its channels.
        if (!passes.create(rd, &sdf, param, MAX_BATCH, input, param_ubo, samples) ||
                !debug_passes.create(rd, &sdf, param, MAX_BATCH, input, param_ubo, debug_samples, debug, &passes)) {
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
    RenderingDevice *rd;
    ChannelSet set;
    int lods;
    ChunkParam param;
    Checks &checks;
    SDFCascade sdf;
    int nch = 0;
    uint32_t nv = 0;
    int cs = 0;
    uint64_t cells = 0;
    RID input;
    RID param_ubo;
    RID samples;
    RID debug;
    RID debug_samples;
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
        PackedByteArray ids;
        ids.resize(count * GPUFrame::CHUNK_INPUT_BYTES);
        int min_lod = 64;
        for (uint32_t c = 0; c < count; c++) {
            boxes[c] = chunk_box(chunks[c].id, param);
            Cascade::run(sdf.get_spec(), boxes[c], chunks[c].lod, &cpu[c]);
            GPUFrame::encode_chunk_input(ids, c, chunks[c].id, param);
            min_lod = MIN(min_lod, chunks[c].lod);
            checks.chunks_per_lod.set(chunks[c].lod, checks.chunks_per_lod[chunks[c].lod] + 1);
        }
        checks.chunks += count;
        rd->buffer_update(input, 0, ids.size(), ids);
        PackedByteArray final_channels;
        for (int j = passes.level_count(min_lod) - 1; j >= 0; j--) {
            passes.record_level(j, 0, count);
            if (j == 0) {
                debug_passes.record_level(0, 0, count);
            }
            rd->submit();
            rd->sync();
            const uint64_t n = uint64_t(level_bound(cs, j));
            const uint64_t bytes = count * n * n * n * nv * 16;
            if (j > 0) {
                compare_level(j, rd->buffer_get_data((j & 1) ? passes.get_scratch_a() : passes.get_scratch_b(), 0, bytes), PackedByteArray());
                continue;
            }
            final_channels = rd->buffer_get_data(debug, 0, bytes);
            const PackedByteArray final_samples = rd->buffer_get_data(samples, 0, count * cells * 16);
            // The debug variant's samples must be the same bits as the meshers' variant's.
            if (rd->buffer_get_data(debug_samples, 0, count * cells * 16) != final_samples) {
                checks.samples_differing++;
                checks.example("set " + set_label(set) + vformat(" LOD %d batch: the debug variant's samples differ from the meshers' variant's", p_lod));
            }
            compare_level(0, final_channels, final_samples);
        }
        for (const Pair &pair : pairs) {
            check_relation(pair, reinterpret_cast<const float *>(final_channels.ptr()));
        }
    }

    // Level j of every chunk of the batch (GPU layout: x fastest, stride level_bound(j)) against the CPU's
    // (x slowest); at j = 0 also the samples against SDFCascade::sdf + skirt of the CPU channels.
    void compare_level(int p_j, const PackedByteArray &p_data, const PackedByteArray &p_samples) {
        const float *gpu = reinterpret_cast<const float *>(p_data.ptr());
        const uint64_t n = uint64_t(level_bound(cs, p_j));
        const uint64_t points = n * n * n;
        const int top = sdf.get_spec().top_level();
        for (uint32_t c = 0; c < uint32_t(chunks.size()); c++) {
            const int lod_c = chunks[c].lod;
            const int top_c = MAX(top, lod_c);
            const int m = lod_c + p_j;
            if (m > top_c) {
                continue;
            }
            const Level &level = cpu[c][size_t(top_c - m)];
            ERR_CONTINUE(level.m != m);
            checks.levels++;
            const Box &box = level.box;
            int64_t differing = 0;
            for_each_point(box, [&](const Vector3i64 &p_p, int64_t p_cpu_i) {
                const Vector3i64 i = p_p - Vector3i64(box.lo[0], box.lo[1], box.lo[2]);
                const uint64_t slot = c * points + uint64_t(i.x) + n * (uint64_t(i.y) + n * uint64_t(i.z));
                for (int ch = 0; ch < nch; ch++) {
                    const float want = level.ch[ch][size_t(p_cpu_i)];
                    const float got = gpu[slot * nv * 4 + ch];
                    checks.values++;
                    if (float_bits(want) != float_bits(got)) {
                        differing++;
                        if (checks.wants_example()) {
                            checks.example(vformat("set %s LOD %d chunk %s (%s) level m=%d point %s channel %d: gpu %s cpu %s (bits %x / %x)", set_label(set), lod_c,
                                    WorldCoord::to_string(chunks[c].id), chunks[c].place, m, WorldCoord::to_string(p_p), ch, got, want, float_bits(got), float_bits(want)));
                        }
                    }
                }
                if (p_j == 0) {
                    compare_sample(c, level, p_p, i, size_t(p_cpu_i), reinterpret_cast<const float *>(p_samples.ptr()) + slot * 4);
                }
            });
            checks.differing += differing;
            checks.differing_per_lod.set(lod_c, checks.differing_per_lod[lod_c] + differing);
        }
    }

    void compare_sample(uint32_t p_chunk, const Level &p_level, const Vector3i64 &p_p, const Vector3i64 &p_grid, size_t p_cpu_i, const float *p_got) {
        float ch[CHANNELS] = {};
        for (int k = 0; k < nch; k++) {
            ch[k] = p_level.ch[k][p_cpu_i];
        }
        const int lod = chunks[p_chunk].lod;
        const ExactNoise::Float4 want = sdf.with_skirt(sdf.sdf(ch, p_p.y * (int64_t(1) << lod), lod), lod, Float3{ float(p_grid.x), float(p_grid.y), float(p_grid.z) });
        const float w[4] = { want.x, want.y, want.z, want.w };
        checks.samples++;
        for (int k = 0; k < 4; k++) {
            if (float_bits(w[k]) != float_bits(p_got[k])) {
                checks.samples_differing++;
                if (checks.wants_example()) {
                    checks.example(vformat("set %s LOD %d chunk %s sample %s: gpu (%s %s %s %s) cpu (%s %s %s %s)", set_label(set), lod,
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
        auto at = [&](size_t p_chunk, const Box &p_box, const Vector3i64 &p_p, int p_ch) {
            const uint64_t slot = p_chunk * cells + uint64_t(p_p.x - p_box.lo[0]) + uint64_t(cs) * (uint64_t(p_p.y - p_box.lo[1]) + uint64_t(cs) * uint64_t(p_p.z - p_box.lo[2]));
            return p_channels[slot * nv * 4 + p_ch];
        };
        for_each_point(cb, [&](const Vector3i64 &p_p, int64_t) {
            const Vector3i64 q(p_p.x / 2, p_p.y / 2, p_p.z / 2);
            if (((p_p.x | p_p.y | p_p.z) & 1) != 0 || !pb.contains(q)) {
                return;
            }
            float expect[CHANNELS] = {};
            for (int ch = 0; ch < nch; ch++) {
                expect[ch] = at(p_pair.parent, pb, q, ch) * exact_pow2(-channel_order(ch));
            }
            for (const Band &band : spec.bands) {
                if (spec.insert_level(band) == lod) {
                    float o[CHANNELS];
                    band_channels(spec, band, p_p, o);
                    for (int ch = 0; ch < nch; ch++) {
                        expect[ch] = expect[ch] + o[ch];
                    }
                }
            }
            checks.relation_points++;
            for (int ch = 0; ch < nch; ch++) {
                if (float_bits(flush_value(expect[ch])) != float_bits(at(p_pair.child, cb, p_p, ch))) {
                    checks.relation_differing++;
                    if (checks.wants_example()) {
                        checks.example(vformat("set %s G3 LOD %d child %s point %s channel %d", set_label(set), lod, WorldCoord::to_string(chunks[p_pair.child].id),
                                WorldCoord::to_string(p_p), ch));
                    }
                    return;
                }
            }
        });
    }
};

void check_set(RenderingDevice *p_rd, ChannelSet p_set, int p_lods, const ChunkParam &p_param, Checks &r_checks) {
    SetChecker checker(p_rd, p_set, p_lods, p_param, r_checks);
    checker.run();
}

// G9 (instance table): a cascade terrain with a scatter of rocks and pillars, LODs 0 to TABLE_LODS - 1, a batch of the
// chunks holding the records per LOD with their instance lists and records on the GPU: every sample of the meshers'
// fold variant against SDFCascade::sample_sdf + skirt with the chunk's list (fold_onto + finish_table where an
// instance passes its box test, the bare cascade elsewhere), bit for bit.
constexpr int TABLE_LODS = 7;

// The LOD-p_lod chunks holding the records' anchors (at most MAX_BATCH), encoded with their lists.
std::vector<ChunkId> table_batch(const Archetype::InstanceSnapshot &p_snap, const SDFGpuTable &p_table, int p_lod, const ChunkParam &p_param,
        PackedByteArray &r_inputs, PackedByteArray &r_words) {
    std::vector<ChunkId> ids;
    for (uint32_t r = 0; r < p_snap.records.size() && ids.size() < MAX_BATCH; r++) {
        const ChunkId id = ChunkMath::chunk_at_voxel(p_snap.records[r].anchor, p_lod, p_param);
        if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
            ids.push_back(id);
        }
    }
    r_inputs.resize(int64_t(ids.size()) * GPUFrame::CHUNK_INPUT_BYTES);
    for (size_t c = 0; c < ids.size(); c++) {
        GPUFrame::encode_chunk_input(r_inputs, int64_t(c), ids[c], p_param);
    }
    p_table.build_lists(ids, p_param, r_inputs, r_words);
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
        const ExactNoise::Float4 bare = p_sdf.with_skirt(p_sdf.sdf(ch, p.voxel.y, p_lod), p_lod, grid);
        r_folded += float_bits(want.w) != float_bits(bare.w) ? 1 : 0;
        const float *g = p_got + (uint64_t(i.x) + cs * (uint64_t(i.y) + cs * uint64_t(i.z))) * 4;
        r_checks.samples++;
        if (!ExactNoise::same_bits(want, ExactNoise::Float4{ g[0], g[1], g[2], g[3] })) {
            r_checks.samples_differing++;
            if (r_checks.wants_example()) {
                r_checks.example(vformat("LOD %d chunk %s sample %s: gpu (%s %s %s %s) cpu (%s %s %s %s)", p_lod, WorldCoord::to_string(p_id),
                        WorldCoord::to_string(i), g[0], g[1], g[2], g[3], want.x, want.y, want.z, want.w));
            }
        }
    });
}

Dictionary check_table(RenderingDevice *p_rd, int p_lods, const ChunkParam &p_param) {
    Dictionary d;
    ChunkParam param = p_param;
    SDFCascade sdf;
    sdf.set_scatter(600, 160.0);
    param.skirt_depth = MAX(param.skirt_depth, sdf.min_skirt_depth(param, p_lods));
    sdf.configure(param, p_lods);
    const std::shared_ptr<const Archetype::InstanceSnapshot> snap = sdf.get_snapshot();
    SDFGpuTable *table = sdf.get_gpu_table();
    const int cs = param.chunk_size;
    const uint64_t cells = uint64_t(cs) * cs * cs;
    const RID input = p_rd->storage_buffer_create(MAX_BATCH * GPUFrame::CHUNK_INPUT_BYTES);
    const PackedByteArray param_bytes = GPUFrame::chunk_param_bytes(param);
    const RID param_ubo = p_rd->uniform_buffer_create(param_bytes.size(), param_bytes);
    const RID samples = p_rd->storage_buffer_create(MAX_BATCH * cells * 16);
    const RID records = p_rd->storage_buffer_create(uint64_t(table->instance_capacity()) * table->record_size());
    const RID list = p_rd->storage_buffer_create(uint64_t(MAX_BATCH) * table->max_chunk_list() * 4);
    uint32_t version = 0;
    SDFGpuTable::upload_records(p_rd, records, table->record_bytes(UINT32_MAX, version));
    CascadePasses passes;
    Checks checks;
    int64_t folded = 0;
    if (!passes.create(p_rd, &sdf, param, MAX_BATCH, input, param_ubo, samples, RID(), nullptr, records, list)) {
        checks.example("the cascade passes failed to compile");
        checks.samples_differing++;
    }
    for (int lod = 0; lod < TABLE_LODS && passes.is_valid(); lod++) {
        PackedByteArray inputs, words;
        const std::vector<ChunkId> ids = table_batch(*snap, *table, lod, param, inputs, words);
        p_rd->buffer_update(input, 0, inputs.size(), inputs);
        if (!words.is_empty()) {
            p_rd->buffer_update(list, 0, words.size(), words);
        }
        passes.record_all(lod, uint32_t(ids.size()), true); // the fold variant, empty lists included (the bare bits there)
        p_rd->submit();
        p_rd->sync();
        const PackedByteArray got_bytes = p_rd->buffer_get_data(samples, 0, ids.size() * cells * 16);
        const float *got = reinterpret_cast<const float *>(got_bytes.ptr());
        for (size_t c = 0; c < ids.size(); c++) {
            compare_table_chunk(sdf, *snap, ids[c], lod, param, got + c * cells * 4, checks, folded);
        }
        checks.chunks += int64_t(ids.size());
    }
    passes.free();
    for (const RID &rid : { input, param_ubo, samples, records, list }) {
        p_rd->free_rid(rid);
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

double median(std::vector<double> p_v) {
    if (p_v.empty()) {
        return 0.0;
    }
    std::sort(p_v.begin(), p_v.end());
    return p_v[p_v.size() / 2];
}

struct BatchTime {
    double wall_ms = 0.0; // submit + sync
    double gpu_ms = -1.0; // timestamps, -1 if they did not come back
    std::vector<double> levels; // GPU ms per level, j = top ... 0
    int timestamp_frames = -1; // extra frames before the timestamps came back
};

// Every level of one batch (input already uploaded), timed.
BatchTime time_batch(RenderingDevice *p_rd, CascadePasses &p_passes, int p_lod, const String &p_prefix) {
    BatchTime out;
    const int levels = p_passes.level_count(p_lod);
    p_rd->capture_timestamp(p_prefix + String("b"));
    for (int j = levels - 1; j >= 0; j--) {
        p_passes.record_level(j, 0, TIMING_CHUNKS);
        p_rd->capture_timestamp(p_prefix + itos(j));
    }
    const uint64_t t0 = now_usec();
    p_rd->submit();
    p_rd->sync();
    out.wall_ms = double(now_usec() - t0) / 1000.0;
    // A local device returns a frame's timestamps when that frame slot is begun again: look after the
    // sync, then after up to 3 empty frames.
    int64_t t_begin = -1;
    std::vector<int64_t> t_level(levels, -1);
    for (int extra = 0; extra <= 3 && t_begin < 0; extra++) {
        if (extra > 0) {
            p_rd->submit();
            p_rd->sync();
        }
        for (uint32_t i = 0; i < p_rd->get_captured_timestamps_count(); i++) {
            const String name = p_rd->get_captured_timestamp_name(i);
            if (!name.begins_with(p_prefix)) {
                continue;
            }
            const String tail = name.substr(p_prefix.length());
            const int64_t t = p_rd->get_captured_timestamp_gpu_time(i);
            if (tail == "b") {
                t_begin = t;
            } else if (tail.to_int() < levels) {
                t_level[tail.to_int()] = t;
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
            prev = MAX(prev, t_level[j]);
        }
    }
    return out;
}

// G6 / G7: GPU time of every level for single-LOD batches of 128 chunks.
Dictionary timing(RenderingDevice *p_rd, const std::vector<ChannelSet> &p_sets, int p_lods, const ChunkParam &p_param, int p_runs) {
    const int cs = p_param.chunk_size;
    const uint64_t cells = uint64_t(cs) * cs * cs;
    const RID input = p_rd->storage_buffer_create(TIMING_CHUNKS * GPUFrame::CHUNK_INPUT_BYTES);
    const PackedByteArray param_bytes = GPUFrame::chunk_param_bytes(p_param);
    const RID param_ubo = p_rd->uniform_buffer_create(param_bytes.size(), param_bytes);
    const RID samples = p_rd->storage_buffer_create(TIMING_CHUNKS * cells * 16);
    std::vector<std::unique_ptr<SDFCascade>> sdfs;
    std::vector<std::unique_ptr<CascadePasses>> passes;
    for (const ChannelSet set : p_sets) {
        sdfs.push_back(std::make_unique<SDFCascade>());
        sdfs.back()->channel_set = set;
        sdfs.back()->configure(p_param, p_lods);
        passes.push_back(std::make_unique<CascadePasses>());
        passes.back()->create(p_rd, sdfs.back().get(), p_param, TIMING_CHUNKS, input, param_ubo, samples);
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
            PackedByteArray ids;
            ids.resize(TIMING_CHUNKS * GPUFrame::CHUNK_INPUT_BYTES);
            for (uint32_t c = 0; c < TIMING_CHUNKS; c++) {
                GPUFrame::encode_chunk_input(ids, c, CascadeMeasure::random_chunk(lod, 0x7133u + uint32_t(r + 2), int(c)), p_param);
            }
            p_rd->buffer_update(input, 0, ids.size(), ids);
            p_rd->submit();
            p_rd->sync();
            for (size_t si = 0; si < p_sets.size(); si++) {
                const BatchTime t = time_batch(p_rd, *passes[si], lod, vformat("cs%d_%d_%d_", r + 2, lod, int(si)));
                if (r < 0) {
                    continue;
                }
                wall[si][li].push_back(t.wall_ms);
                timestamp_frames = MAX(timestamp_frames, t.timestamp_frames);
                if (t.gpu_ms >= 0.0) {
                    gpu[si][li].push_back(t.gpu_ms);
                    per_level[si][li] = t.levels;
                }
            }
        }
    }
    Dictionary out;
    Array rows;
    for (size_t si = 0; si < p_sets.size(); si++) {
        for (size_t li = 0; li < lods.size(); li++) {
            Dictionary row;
            row["set"] = set_label(p_sets[si]);
            row["lod"] = lods[li];
            row["levels"] = passes[si]->level_count(lods[li]);
            row["wall_ms_median"] = median(wall[si][li]);
            row["wall_ms_min"] = wall[si][li].empty() ? 0.0 : *std::min_element(wall[si][li].begin(), wall[si][li].end());
            row["gpu_ms_median"] = median(gpu[si][li]);
            row["gpu_ms_min"] = gpu[si][li].empty() ? 0.0 : *std::min_element(gpu[si][li].begin(), gpu[si][li].end());
            row["gpu_runs"] = int64_t(gpu[si][li].size());
            PackedFloat64Array lv;
            for (const double v : per_level[si][li]) {
                lv.push_back(v);
            }
            row["gpu_ms_per_level_last"] = lv; // j = levels - 1 ... 0
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
    for (const RID &rid : { input, param_ubo, samples }) {
        p_rd->free_rid(rid);
    }
    return out;
}

// D1 (GPU discoverer): point items through the cascade passes (CascadePasses::create_points) against
// Cascade::point + SDFCascade::sdf, the bits of the metres value. Per LOD 0-24, 5 places x 1024 corners of that
// LOD's corner grid (multiples of 28 * 2^L voxels): near 0, +2^60, -2^60, straddling 2^32-voxel boundaries, and
// around lattice index 2^32 (voxel 2^(32 + L)). One batch per LOD, like a discoverer wave.
Dictionary check_points(RenderingDevice *p_rd, ChannelSet p_set, int p_lods, const ChunkParam &p_param) {
    constexpr int PER_PLACE = 1024;
    constexpr int PLACES = 5;
    constexpr uint32_t MAX_POINTS = PER_PLACE * PLACES;
    SDFCascade sdf;
    sdf.channel_set = p_set;
    sdf.configure(p_param, p_lods);
    const ChunkParam pp = CascadeGLSL::point_param(p_param);
    const RID input = p_rd->storage_buffer_create(MAX_POINTS * GPUFrame::CHUNK_INPUT_BYTES);
    const PackedByteArray param_bytes = GPUFrame::chunk_param_bytes(pp);
    const RID param_ubo = p_rd->uniform_buffer_create(param_bytes.size(), param_bytes);
    const RID values = p_rd->storage_buffer_create(MAX_POINTS * sizeof(float));
    CascadePasses passes;
    Dictionary d;
    int64_t compared = 0;
    int64_t differing = 0;
    PackedInt64Array per_lod;
    Array examples;
    if (!passes.create_points(p_rd, &sdf, pp, MAX_POINTS, input, param_ubo, values)) {
        examples.push_back("the point passes failed to compile");
        differing++;
    }
    const int64_t P32 = int64_t(1) << 32;
    const int64_t boundaries[] = { P32, -P32, 3 * P32, -5 * P32 };
    for (int lod = 0; lod < p_lods && passes.is_valid(); lod++) {
        const int64_t span = pp.half_chunk_voxels() << (lod + 1);
        const int64_t bases[PLACES] = { 0, (int64_t(1) << 60) - 123457, -(int64_t(1) << 60) + 54321, 0, int64_t(1) << (32 + lod) };
        std::vector<VoxelPos> points;
        PackedByteArray bytes;
        bytes.resize(MAX_POINTS * GPUFrame::CHUNK_INPUT_BYTES);
        for (int place = 0; place < PLACES; place++) {
            for (int i = 0; i < PER_PLACE; i++) {
                VoxelPos v;
                for (int a = 0; a < 3; a++) {
                    const int64_t base = place == 3 ? boundaries[(i + a) % 4] : bases[place];
                    v[a] = (WorldCoord::floor_div(base, span) + ExactNoise::random_offset(uint32_t(lod), uint32_t(place * PER_PLACE + i), uint32_t(a) + 0xD1u, 32)) * span;
                }
                CascadeGLSL::encode_point_input(bytes, int64_t(points.size()), v, lod, pp);
                points.push_back(v);
            }
        }
        p_rd->buffer_update(input, 0, bytes.size(), bytes);
        passes.record_all(lod, uint32_t(points.size()));
        p_rd->submit();
        p_rd->sync();
        const PackedByteArray got_bytes = p_rd->buffer_get_data(values, 0, points.size() * sizeof(float));
        const float *got = reinterpret_cast<const float *>(got_bytes.ptr());
        int64_t lod_differing = 0;
        for (size_t i = 0; i < points.size(); i++) {
            const VoxelPos &v = points[i];
            float ch[CHANNELS];
            point(sdf.get_spec(), Vector3i64(v.x >> lod, v.y >> lod, v.z >> lod), lod, ch);
            const float want = sdf.sdf(ch, v.y, lod).w;
            compared++;
            if (float_bits(want) != float_bits(got[i])) {
                lod_differing++;
                if (examples.size() < 24) {
                    examples.push_back(vformat("set %s LOD %d point %s: gpu %s cpu %s (bits %x / %x)", set_label(p_set), lod, WorldCoord::to_string(v), got[i], want,
                            float_bits(got[i]), float_bits(want)));
                }
            }
        }
        differing += lod_differing;
        per_lod.push_back(lod_differing);
    }
    d["points"] = compared;
    d["differing"] = differing;
    d["differing_per_lod"] = per_lod;
    d["points_per_lod"] = int64_t(MAX_POINTS);
    d["scratch_mb"] = double(passes.scratch_bytes()) / 1048576.0;
    d["examples"] = examples;
    d["ok"] = differing == 0 && compared == int64_t(p_lods) * MAX_POINTS;
    passes.free();
    for (const RID &rid : { input, param_ubo, values }) {
        p_rd->free_rid(rid);
    }
    return d;
}

} // namespace

void CascadeSpike::_bind_methods() {
    ClassDB::bind_method(D_METHOD("run", "options"), &CascadeSpike::run);
}

Dictionary CascadeSpike::run(const Dictionary &p_options) {
    Dictionary report;
    report["ok"] = false;
    const int lods = int(p_options.get("lods", 25));
    ChunkParam param;
    param.voxel_size = float(p_options.get("voxel", 0.5));
    std::vector<ChannelSet> sets;
    for (const String &s : String(p_options.get("sets", "8,4")).split(",")) {
        sets.push_back(s == "4" ? ChannelSet::LINEAR4 : ChannelSet::HERMITE8);
    }
    RenderingDevice *rd = RenderingServer::get_singleton()->create_local_rendering_device();
    if (!rd) {
        report["error"] = "no local RenderingDevice (headless or Compatibility renderer)";
        return report;
    }
    bool ok = true;
    if (bool(p_options.get("check", true))) {
        Dictionary checks_report;
        for (const ChannelSet set : sets) {
            Checks checks;
            const uint64_t t0 = now_usec();
            check_set(rd, set, lods, param, checks);
            Dictionary d;
            d["chunks"] = checks.chunks;
            d["levels"] = checks.levels;
            d["values"] = checks.values;
            d["differing"] = checks.differing;
            d["samples"] = checks.samples;
            d["samples_differing"] = checks.samples_differing;
            d["relation_points"] = checks.relation_points;
            d["relation_differing"] = checks.relation_differing;
            d["chunks_per_lod"] = checks.chunks_per_lod;
            d["differing_per_lod"] = checks.differing_per_lod;
            d["examples"] = checks.examples;
            d["seconds"] = double(now_usec() - t0) / 1e6;
            int64_t min_chunks = 1 << 30;
            for (int64_t v : checks.chunks_per_lod) {
                min_chunks = MIN(min_chunks, v);
            }
            d["min_chunks_per_lod"] = min_chunks;
            const bool set_ok = checks.values > 0 && checks.samples > 0 && checks.relation_points > 0 && min_chunks >= 64 && checks.differing == 0 &&
                    checks.samples_differing == 0 && checks.relation_differing == 0;
            d["ok"] = set_ok;
            ok = ok && set_ok;
            checks_report[set_label(set)] = d;
        }
        report["check"] = checks_report;
    }
    if (bool(p_options.get("table", true))) {
        const uint64_t t0 = now_usec();
        Dictionary d = check_table(rd, lods, param);
        d["seconds"] = double(now_usec() - t0) / 1e6;
        ok = ok && bool(d["ok"]);
        report["table"] = d;
    }
    if (bool(p_options.get("points", true))) {
        Dictionary points_report;
        for (const ChannelSet set : sets) {
            const uint64_t t0 = now_usec();
            Dictionary d = check_points(rd, set, lods, param);
            d["seconds"] = double(now_usec() - t0) / 1e6;
            ok = ok && bool(d["ok"]);
            points_report[set_label(set)] = d;
        }
        report["points"] = points_report;
    }
    if (bool(p_options.get("timing", true))) {
        report["timing"] = timing(rd, sets, lods, param, int(p_options.get("runs", 10)));
    }
    memdelete(rd);
    report["lods"] = lods;
    report["ok"] = ok;
    return report;
}
