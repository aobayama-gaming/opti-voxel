#include "cascade_check_points.hpp"

#include "cascade_points_gpu.hpp"

using namespace CascadeCheckSets;
using ExactNoise::Float4;
using ExactNoise::float_bits;
using ExactNoise::same_bits;

namespace CascadeCheckPoints {

namespace {

constexpr int PER_PLACE = 1024;
constexpr int PLACES = 5;
constexpr int POINT_LODS[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24 };
constexpr uint32_t TIMING_SIZES[] = { 1024, 4096, 16384, 65536 };
constexpr int TIMING_LODS[] = { 0, 5 };

ov::str text(const Float4 &p_f) {
    return ov::format("(%s %s %s %s)", p_f.x, p_f.y, p_f.z, p_f.w);
}

// Voxel p_index of place p_place at LOD p_lod: within 64 LOD cells of its base (any voxel, so any lattice index).
VoxelPos test_voxel(int p_lod, int p_place, int p_index) {
    const int64_t P32 = int64_t(1) << 32;
    const int64_t boundaries[] = { P32, -P32, 3 * P32, -5 * P32 };
    const int64_t bases[PLACES] = { 0, (int64_t(1) << 60) - 123457, -(int64_t(1) << 60) + 54321, 0, int64_t(1) << ov::min(32 + p_lod, 60) };
    VoxelPos v;
    for (int a = 0; a < 3; a++) {
        const int64_t base = p_place == 3 ? boundaries[(p_index + a) % 4] : bases[p_place];
        v[a] = base + ExactNoise::random_offset(uint32_t(p_lod), uint32_t(p_place * PER_PLACE + p_index), uint32_t(a) + 0xF1E1u, int64_t(64) << p_lod);
    }
    return v;
}

} // namespace

ov::Value check_points(ov::gpu::Device *p_device, SpikeSet p_set, int p_lods, const ChunkParam &p_param) {
    SDFCascade sdf;
    ChunkParam param = p_param;
    configure_variant(sdf, p_set, p_lods, param);
    std::vector<VoxelPos> voxels;
    std::vector<VoxelPos> lattice;
    std::vector<int> lods;
    for (const int lod : POINT_LODS) {
        if (lod >= p_lods) {
            continue;
        }
        for (int place = 0; place < PLACES; place++) {
            for (int i = 0; i < PER_PLACE; i++) {
                voxels.push_back(test_voxel(lod, place, i));
                lattice.push_back(SDFBase::ground_lattice(voxels.back(), lod));
                lods.push_back(lod);
            }
        }
    }
    ov::Value d = ov::Value::object();
    ov::Value examples = ov::Value::array();
    int64_t differing = 0;
    int64_t compared = 0;
    std::vector<int64_t> per_lod;
    CascadePointsGPU gpu;
    CascadePointsGPU::BatchStats stats;
    std::vector<Float4> units(voxels.size());
    std::vector<Float4> fields(voxels.size());
    // One batch of every LOD: items of different LODs side by side; the registers, then the site walk's entry point.
    const bool ran = gpu.create(p_device, &sdf, param, uint32_t(voxels.size())) && gpu.units(lattice.data(), lods.data(), int(voxels.size()), units.data(), stats) &&
            gpu.ground_field(voxels.data(), lods.data(), int(voxels.size()), fields.data(), stats);
    if (!ran) {
        examples.push_back("the point passes failed (compile or run)");
        differing++;
    }
    for (size_t first = 0; ran && first < voxels.size(); first += PLACES * PER_PLACE) {
        int64_t lod_differing = 0;
        for (size_t i = first; i < first + PLACES * PER_PLACE; i++) {
            // The register, the site walk's reading (CascadePointsGPU::ground_field) and the discoverer's value made of the
            // register (cull_values' conversion), each against the CPU's own entry point.
            const int lod = lods[i];
            const Float4 want_unit = sdf.ground_unit(lattice[i], lod);
            const Float4 &field = fields[i];
            const Float4 want_field = sdf.ground_field(voxels[i], lod);
            const float cull = SDFCascade::to_metres(units[i], lod, param.voxel_size).w;
            const float want_cull = sdf.evaluate_cull(WorldPos{ lattice[i], Float3() }, lod);
            compared++;
            if (!same_bits(units[i], want_unit) || !same_bits(field, want_field) || float_bits(cull) != float_bits(want_cull)) {
                lod_differing++;
                if (examples.size() < 24) {
                    examples.push_back(ov::format("set %s LOD %d voxel %s: unit gpu %s cpu %s, field %s / %s, cull %s / %s", set_label(p_set), lod, WorldCoord::to_string(voxels[i]),
                            text(units[i]), text(want_unit), text(field), text(want_field), cull, want_cull));
                }
            }
        }
        differing += lod_differing;
        per_lod.push_back(lod_differing);
    }
    d["points"] = compared;
    d["differing"] = differing;
    d["differing_per_lod"] = ov::Value::array_of(per_lod);
    d["points_per_lod"] = int64_t(PLACES * PER_PLACE);
    d["batches"] = stats.batches;
    d["gpu_ms"] = stats.gpu_ms;
    d["scratch_mb"] = double(gpu.scratch_bytes()) / 1048576.0;
    d["examples"] = examples;
    d["ok"] = ran && differing == 0 && compared == int64_t(voxels.size());
    gpu.free();
    return d;
}

ov::Value point_timing(ov::gpu::Device *p_device, const std::vector<SpikeSet> &p_sets, int p_lods, const ChunkParam &p_param, int p_runs) {
    ov::Value out = ov::Value::object();
    ov::Value rows = ov::Value::array();
    for (const SpikeSet set : p_sets) {
        SDFCascade sdf;
        ChunkParam param = p_param;
        configure_variant(sdf, set, p_lods, param);
        CascadePointsGPU gpu;
        const uint32_t most = TIMING_SIZES[std::size(TIMING_SIZES) - 1];
        if (!gpu.create(p_device, &sdf, param, most)) {
            out["error_" + set_label(set)] = "the point passes failed to compile";
            continue;
        }
        out["scratch_mb_" + set_label(set)] = double(gpu.scratch_bytes()) / 1048576.0;
        std::vector<VoxelPos> points(most);
        std::vector<int> lods(most);
        std::vector<Float4> values(most);
        for (const int lod : TIMING_LODS) {
            for (uint32_t i = 0; i < most; i++) {
                points[i] = SDFBase::ground_lattice(test_voxel(lod, int(i % 3), int(i % PER_PLACE)) + VoxelPos(int64_t(i / PER_PLACE) << (lod + 7), 0, 0), lod);
                lods[i] = lod;
            }
            for (const uint32_t size : TIMING_SIZES) {
                std::vector<double> gpu_ms;
                // One warm-up run, then p_runs timed ones.
                for (int r = -1; r < p_runs; r++) {
                    CascadePointsGPU::BatchStats stats;
                    if (!gpu.units(points.data(), lods.data(), int(size), values.data(), stats)) {
                        break;
                    }
                    if (r >= 0 && stats.gpu_ms >= 0.0) {
                        gpu_ms.push_back(stats.gpu_ms);
                    }
                }
                ov::Value row = ov::Value::object();
                row["set"] = set_label(set);
                row["lod"] = lod;
                row["points"] = int64_t(size);
                row["gpu_ms_median"] = median(gpu_ms);
                row["us_per_point"] = median(gpu_ms) * 1000.0 / double(size);
                row["gpu_runs"] = int64_t(gpu_ms.size());
                rows.push_back(row);
            }
        }
        gpu.free();
    }
    out["rows"] = rows;
    out["runs"] = p_runs;
    return out;
}

} // namespace CascadeCheckPoints
