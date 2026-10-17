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

// The LOD-L chunk whose interior holds LOD0 voxel p_voxel.
ChunkId chunk_holding(const Vector3i64 &p_voxel, int p_lod, const ChunkParam &p_param) {
    const int64_t span = (p_param.half_chunk_voxels() * 2) << p_lod;
    auto axis = [&](int64_t p_v) { return WorldCoord::floor_div(p_v, span) * span / p_param.half_chunk_voxels() + (int64_t(1) << p_lod); };
    return ChunkId(axis(p_voxel.x), axis(p_voxel.y), axis(p_voxel.z));
}

int64_t random_offset(uint32_t p_a, uint32_t p_b, uint32_t p_c, int64_t p_range) {
    const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ p_a, p_b, p_c });
    const uint64_t r = (uint64_t(h.x) << 32) | h.y;
    return int64_t(r % uint64_t(2 * p_range + 1)) - p_range;
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
            v[a] = base + random_offset(uint32_t(p_lod), n, uint32_t(a) + 0xCA5Cu, p_place.range);
        }
        n++;
        chunks.push_back({ chunk_holding(v, p_lod, p_param), p_lod, p_place.name });
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
void check_set(RenderingDevice *p_rd, ChannelSet p_set, int p_lods, const ChunkParam &p_param, Checks &r_checks) {
    SDFCascade sdf;
    sdf.channel_set = p_set;
    sdf.configure(p_param, p_lods);
    const Spec &spec = sdf.get_spec();
    const int top = spec.top_level();
    const int nch = channel_count(p_set);
    const uint32_t nv = CascadeGLSL::vec4s_per_point(p_set);
    const int cs = p_param.chunk_size;
    const uint64_t cells = uint64_t(cs) * cs * cs;

    const RID input = p_rd->storage_buffer_create(MAX_BATCH * GPUFrame::CHUNK_INPUT_BYTES);
    const PackedByteArray param_bytes = GPUFrame::chunk_param_bytes(p_param);
    const RID param_ubo = p_rd->uniform_buffer_create(param_bytes.size(), param_bytes);
    const RID samples = p_rd->storage_buffer_create(MAX_BATCH * cells * 16);
    const RID debug = p_rd->storage_buffer_create(MAX_BATCH * cells * nv * 16);
    const RID debug_samples = p_rd->storage_buffer_create(MAX_BATCH * cells * 16);
    // The meshers' variant for every level and the samples; the debug variant (same scratch) re-runs level 0
    // to read its channels.
    CascadePasses passes;
    CascadePasses debug_passes;
    if (!passes.create(p_rd, &sdf, p_param, MAX_BATCH, input, param_ubo, samples) ||
            !debug_passes.create(p_rd, &sdf, p_param, MAX_BATCH, input, param_ubo, debug_samples, debug, &passes)) {
        r_checks.example("set " + set_label(p_set) + ": the cascade passes failed to compile");
        r_checks.differing++;
    } else {
        r_checks.chunks_per_lod.resize(p_lods);
        r_checks.differing_per_lod.resize(p_lods);
        for (int lod = 0; lod < p_lods; lod++) {
            std::vector<Pair> pairs;
            const std::vector<TestChunk> chunks = batch_chunks(lod, p_lods, p_param, pairs);
            const uint32_t count = uint32_t(chunks.size());
            // CPU oracle: every level of every chunk, top first.
            std::vector<std::vector<Level>> cpu(count);
            std::vector<Box> boxes(count);
            PackedByteArray ids;
            ids.resize(count * GPUFrame::CHUNK_INPUT_BYTES);
            int min_lod = 64;
            for (uint32_t c = 0; c < count; c++) {
                boxes[c] = chunk_box(chunks[c].id, p_param);
                run(spec, boxes[c], chunks[c].lod, &cpu[c]);
                GPUFrame::encode_chunk_input(ids, c, chunks[c].id, p_param);
                min_lod = MIN(min_lod, chunks[c].lod);
                r_checks.chunks_per_lod.set(chunks[c].lod, r_checks.chunks_per_lod[chunks[c].lod] + 1);
            }
            r_checks.chunks += count;
            p_rd->buffer_update(input, 0, ids.size(), ids);
            PackedByteArray final_channels;
            PackedByteArray final_samples;
            for (int j = passes.level_count(min_lod) - 1; j >= 0; j--) {
                passes.record_level(j, 0, count);
                if (j == 0) {
                    debug_passes.record_level(0, 0, count);
                }
                p_rd->submit();
                p_rd->sync();
                const uint32_t n = uint32_t(level_bound(cs, j));
                const uint64_t points = uint64_t(n) * n * n;
                PackedByteArray data;
                if (j > 0) {
                    data = p_rd->buffer_get_data((j & 1) ? passes.get_scratch_a() : passes.get_scratch_b(), 0, count * points * nv * 16);
                } else {
                    data = p_rd->buffer_get_data(debug, 0, count * points * nv * 16);
                    final_channels = data;
                    final_samples = p_rd->buffer_get_data(samples, 0, count * points * 16);
                    // The debug variant's samples must be the same bits as the meshers' variant's.
                    if (p_rd->buffer_get_data(debug_samples, 0, count * points * 16) != final_samples) {
                        r_checks.samples_differing++;
                        r_checks.example("set " + set_label(p_set) + vformat(" LOD %d batch: the debug variant's samples differ from the meshers' variant's", lod));
                    }
                }
                const float *gpu = reinterpret_cast<const float *>(data.ptr());
                for (uint32_t c = 0; c < count; c++) {
                    const int lod_c = chunks[c].lod;
                    const int top_c = MAX(top, lod_c);
                    const int m = lod_c + j;
                    if (m > top_c) {
                        continue;
                    }
                    const Level &level = cpu[c][size_t(top_c - m)];
                    ERR_CONTINUE(level.m != m);
                    r_checks.levels++;
                    const Box &box = level.box;
                    int64_t differing = 0;
                    for (int64_t ix = 0; ix < box.size(0); ix++) {
                        for (int64_t iy = 0; iy < box.size(1); iy++) {
                            for (int64_t iz = 0; iz < box.size(2); iz++) {
                                const int64_t cpu_i = (ix * box.size(1) + iy) * box.size(2) + iz;
                                const uint64_t slot = c * points + uint64_t(ix) + n * (uint64_t(iy) + n * uint64_t(iz));
                                for (int ch = 0; ch < nch; ch++) {
                                    const float want = level.ch[ch][size_t(cpu_i)];
                                    const float got = gpu[slot * nv * 4 + ch];
                                    r_checks.values++;
                                    if (float_bits(want) != float_bits(got)) {
                                        differing++;
                                        if (r_checks.wants_example()) {
                                            r_checks.example(vformat("set %s LOD %d chunk %s (%s) level m=%d point (%d %d %d) channel %d: gpu %s cpu %s (bits %x / %x)",
                                                    set_label(p_set), lod_c, WorldCoord::to_string(chunks[c].id), chunks[c].place, m, box.lo[0] + ix, box.lo[1] + iy,
                                                    box.lo[2] + iz, ch, got, want, float_bits(got), float_bits(want)));
                                        }
                                    }
                                }
                                if (j == 0) {
                                    // The sample: SDFCascade::sample of the CPU channels.
                                    float chv[CHANNELS] = {};
                                    for (int ch = 0; ch < nch; ch++) {
                                        chv[ch] = level.ch[ch][size_t(cpu_i)];
                                    }
                                    const int64_t voxel_y = (box.lo[1] + iy) * (int64_t(1) << lod_c);
                                    const ExactNoise::Float4 want = sdf.sample(chv, voxel_y, lod_c, Float3{ float(ix), float(iy), float(iz) });
                                    const float *got = reinterpret_cast<const float *>(final_samples.ptr()) + (c * points + uint64_t(ix) + n * (uint64_t(iy) + n * uint64_t(iz))) * 4;
                                    const float w[4] = { want.x, want.y, want.z, want.w };
                                    r_checks.samples++;
                                    bool same = true;
                                    for (int k = 0; k < 4; k++) {
                                        same = same && float_bits(w[k]) == float_bits(got[k]);
                                    }
                                    if (!same) {
                                        r_checks.samples_differing++;
                                        if (r_checks.wants_example()) {
                                            r_checks.example(vformat("set %s LOD %d chunk %s sample (%d %d %d): gpu (%s %s %s %s) cpu (%s %s %s %s)", set_label(p_set), lod_c,
                                                    WorldCoord::to_string(chunks[c].id), ix, iy, iz, got[0], got[1], got[2], got[3], w[0], w[1], w[2], w[3]));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    r_checks.differing += differing;
                    r_checks.differing_per_lod.set(lod_c, r_checks.differing_per_lod[lod_c] + differing);
                }
            }
            // G3: the GPU's Λ_L channels of each child against its parent's Λ_{L+1} channels.
            const uint64_t points = cells;
            const float *gpu = reinterpret_cast<const float *>(final_channels.ptr());
            for (const Pair &pair : pairs) {
                const Box &cb = boxes[pair.child];
                const Box &pb = boxes[pair.parent];
                const int lod_c = chunks[pair.child].lod;
                auto at = [&](size_t p_chunk, const Box &p_box, const Vector3i64 &p_p, int p_ch) {
                    const uint64_t slot = p_chunk * points + uint64_t(p_p.x - p_box.lo[0]) + uint64_t(cs) * (uint64_t(p_p.y - p_box.lo[1]) + uint64_t(cs) * uint64_t(p_p.z - p_box.lo[2]));
                    return gpu[slot * nv * 4 + p_ch];
                };
                for_each_point(cb, [&](const Vector3i64 &p_p, int64_t) {
                    if (((p_p.x | p_p.y | p_p.z) & 1) != 0) {
                        return;
                    }
                    const Vector3i64 q(p_p.x / 2, p_p.y / 2, p_p.z / 2);
                    if (!pb.contains(q)) {
                        return;
                    }
                    float expect[CHANNELS] = {};
                    for (int ch = 0; ch < nch; ch++) {
                        expect[ch] = at(pair.parent, pb, q, ch) * exact_pow2(-channel_order(ch));
                    }
                    for (const Band &band : spec.bands) {
                        if (spec.insert_level(band) == lod_c) {
                            float o[CHANNELS];
                            band_channels(spec, band, p_p, o);
                            for (int ch = 0; ch < nch; ch++) {
                                expect[ch] = expect[ch] + o[ch];
                            }
                        }
                    }
                    r_checks.relation_points++;
                    for (int ch = 0; ch < nch; ch++) {
                        if (float_bits(flush_value(expect[ch])) != float_bits(at(pair.child, cb, p_p, ch))) {
                            r_checks.relation_differing++;
                            if (r_checks.wants_example()) {
                                r_checks.example(vformat("set %s G3 LOD %d child %s point %s channel %d", set_label(p_set), lod_c, WorldCoord::to_string(chunks[pair.child].id),
                                        WorldCoord::to_string(p_p), ch));
                            }
                            break;
                        }
                    }
                });
            }
        }
    }
    debug_passes.free();
    passes.free();
    for (const RID &rid : { input, param_ubo, samples, debug, debug_samples }) {
        p_rd->free_rid(rid);
    }
}

double median(std::vector<double> p_v) {
    if (p_v.empty()) {
        return 0.0;
    }
    std::sort(p_v.begin(), p_v.end());
    return p_v[p_v.size() / 2];
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
                CascadePasses &p = *passes[si];
                const int levels = p.level_count(lod);
                const String prefix = vformat("cs%d_%d_%d_", r + 2, lod, int(si));
                p_rd->capture_timestamp(prefix + String("b"));
                for (int j = levels - 1; j >= 0; j--) {
                    p.record_level(j, 0, TIMING_CHUNKS);
                    p_rd->capture_timestamp(prefix + itos(j));
                }
                const uint64_t t0 = now_usec();
                p_rd->submit();
                p_rd->sync();
                const uint64_t t1 = now_usec();
                if (r < 0) {
                    continue;
                }
                wall[si][li].push_back(double(t1 - t0) / 1000.0);
                // A local device returns a frame's timestamps when that frame slot is begun again: look after
                // the sync, then after up to 3 empty frames.
                int64_t t_begin = -1;
                std::vector<int64_t> t_level(levels, -1);
                for (int extra = 0; extra <= 3 && t_begin < 0; extra++) {
                    if (extra > 0) {
                        p_rd->submit();
                        p_rd->sync();
                    }
                    for (uint32_t i = 0; i < p_rd->get_captured_timestamps_count(); i++) {
                        const String name = p_rd->get_captured_timestamp_name(i);
                        if (!name.begins_with(prefix)) {
                            continue;
                        }
                        const String tail = name.substr(prefix.length());
                        const int64_t t = p_rd->get_captured_timestamp_gpu_time(i);
                        if (tail == "b") {
                            t_begin = t;
                        } else if (tail.to_int() < levels) {
                            t_level[tail.to_int()] = t;
                        }
                    }
                    if (t_begin >= 0) {
                        timestamp_frames = MAX(timestamp_frames, extra);
                    }
                }
                if (t_begin >= 0 && t_level[0] >= t_begin) {
                    gpu[si][li].push_back(double(t_level[0] - t_begin) / 1e6);
                    std::vector<double> lv;
                    int64_t prev = t_begin;
                    for (int j = levels - 1; j >= 0; j--) {
                        lv.push_back(t_level[j] >= prev ? double(t_level[j] - prev) / 1e6 : -1.0);
                        prev = MAX(prev, t_level[j]);
                    }
                    per_level[si][li] = lv;
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
    if (bool(p_options.get("timing", true))) {
        report["timing"] = timing(rd, sets, lods, param, int(p_options.get("runs", 10)));
    }
    memdelete(rd);
    report["lods"] = lods;
    report["ok"] = ok;
    return report;
}
