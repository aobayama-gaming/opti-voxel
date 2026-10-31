#include "test_common.hpp"

#include "base_regions.hpp"
#include "cascade.hpp"
#include "climate.hpp"
#include "gpu_frame.hpp"
#include "sdf_cascade.hpp"

#include "ov/bytes.hpp"

#include <cstring>

namespace {

// Earth's radius at 5 cm voxels.
constexpr double EARTH_R = 6371000.0 / 0.05;

const Vector3i64 CENTRES[] = { Vector3i64(0, 0, 0), Vector3i64(int64_t(1) << 40, -(int64_t(1) << 40), 0),
    Vector3i64(-(int64_t(1) << 59), int64_t(1) << 59, -(int64_t(1) << 58)) };

// A chunk of LOD p_lod holding a voxel near the sphere's surface, p_i-th of a spread over its +y cap.
ChunkId surface_chunk(const BaseSurface &p_base, int p_lod, int p_i, const ChunkParam &p_param) {
    const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ uint32_t(p_i), uint32_t(p_lod), 0x5E6Du });
    const int64_t spread = int64_t(1) << (p_lod + 9);
    const VoxelPos at = p_base.tangent_point(int64_t(h.x % uint32_t(spread)) - spread / 2, int64_t(h.y % uint32_t(spread)) - spread / 2);
    return ChunkMath::chunk_at_voxel(at, p_lod, p_param);
}

// Whether an item's slot table gives a point the region BaseSurface::at uses for it: the slot from the point's low
// word as the GLSL base_at reads it (not BaseRegions' own slot of a cell), the region's bytes as encode() lays them out.
bool reads_own_region(const BaseSurface &p_base, const BaseRegions::Tables &p_tables, uint32_t p_item, const VoxelPos &p_point, int p_lod) {
    const int s = p_lod + BaseSurface::REGION_SHIFT;
    const int slot = int((WorldCoord::lo32(p_point.x) >> s) & 1u) | int((WorldCoord::lo32(p_point.y) >> s) & 1u) << 1 |
            int((WorldCoord::lo32(p_point.z) >> s) & 1u) << 2;
    const uint32_t index = p_tables.index[size_t(p_item) * BaseRegions::SLOTS + size_t(slot)];
    if (index >= p_tables.region_count()) {
        return false;
    }
    uint8_t want[BaseRegions::REGION_BYTES] = {};
    BaseRegions::encode(want, p_base.region(BaseSurface::region_cell(p_point, p_lod), p_lod));
    return std::memcmp(want, p_tables.regions.data() + size_t(index) * BaseRegions::REGION_BYTES, BaseRegions::REGION_BYTES) == 0;
}

} // namespace

TEST_CASE("GPUFrame: decode_item reads back a chunk's anchor and LOD, and a point item") {
    ChunkParam param;
    const Vector3i64 bases[] = { Vector3i64(0, 0, 0), Vector3i64(-(int64_t(1) << 50), int64_t(1) << 45, 12345678901), Vector3i64(int64_t(1) << 58, -(int64_t(1) << 58), 7) };
    int bad = 0;
    for (const Vector3i64 &b : bases) {
        for (int lod = 0; lod < 24; lod++) {
            for (int i = -3; i <= 3; i++) {
                const ChunkId id = ChunkMath::chunk_at_voxel(b + Vector3i64(i * 1000003, -i * 777, i * 31), lod, param);
                uint8_t item[GPUFrame::CHUNK_INPUT_BYTES];
                GPUFrame::encode_chunk_input(item, id, param);
                VoxelPos anchor;
                int got_lod = -1;
                GPUFrame::decode_item(item, param, false, anchor, got_lod);
                bad += got_lod != lod || anchor != ChunkMath::chunk_min_voxel(id, param);
                const VoxelPos point = ChunkMath::chunk_min_voxel(id, param) + Vector3i64(int64_t(i + 3) << lod, 0, 0);
                CHECK(GPUFrame::encode_point_item(item, point, lod));
                GPUFrame::decode_item(item, param, true, anchor, got_lod);
                bad += got_lod != lod || anchor != point;
            }
        }
    }
    CHECK(bad == 0);
}

// Every sample of a chunk's box finds, through its slot, the region BaseSurface::at uses for it (the 8 slots name the
// box's cells: at most two per axis), at every LOD below the switch, near 0 and far centres.
TEST_CASE("BaseRegions: every chunk sample reads its own region through its slot") {
    ChunkParam param;
    int wrong = 0;
    int checked = 0;
    for (const Vector3i64 &c : CENTRES) {
        const BaseSurface b = BaseSurface::sphere(c, EARTH_R);
        CHECK_OR_RETURN(b.kind() == BaseSurface::Kind::SPHERE);
        for (int lod = 0; lod < b.get_switch_lod(); lod++) {
            constexpr uint32_t CHUNKS = 12;
            ov::bytes items(CHUNKS * GPUFrame::CHUNK_INPUT_BYTES);
            ChunkId ids[CHUNKS];
            for (uint32_t i = 0; i < CHUNKS; i++) {
                ids[i] = surface_chunk(b, lod, int(i), param);
                GPUFrame::encode_chunk_input(items, i, ids[i], param);
            }
            BaseRegions::Tables tables;
            BaseRegions::build(b, items.data(), CHUNKS, param, false, tables);
            for (uint32_t i = 0; i < CHUNKS; i++) {
                const Cascade::Box box = Cascade::chunk_box(ids[i], param);
                const Vector3i64 lo(box.lo[0], box.lo[1], box.lo[2]);
                // The box's corners and a stride through it: every cell boundary is crossed.
                for (int z = 0; z < param.chunk_size; z += 3) {
                    for (int y = 0; y < param.chunk_size; y += 3) {
                        for (int x = 0; x < param.chunk_size; x += 3) {
                            const int last = param.chunk_size - 1;
                            const Vector3i64 l = lo + Vector3i64(x + 3 > last ? last : x, y + 3 > last ? last : y, z + 3 > last ? last : z);
                            const VoxelPos p(l.x << lod, l.y << lod, l.z << lod);
                            wrong += !reads_own_region(b, tables, i, p, lod);
                            checked++;
                        }
                    }
                }
            }
        }
    }
    CHECK(checked > 0);
    CHECK(wrong == 0);
}

TEST_CASE("BaseRegions: point items, the switch LOD, deduplication") {
    ChunkParam param;
    const BaseSurface b = BaseSurface::sphere(Vector3i64(0, 0, 0), EARTH_R);
    const int sw = b.get_switch_lod();

    SUBCASE("a point item fills its own cell's slot") {
        const ChunkParam point_param = [&] {
            ChunkParam p = param;
            p.skirt_size = 0;
            return p;
        }();
        constexpr uint32_t POINTS = 64;
        ov::bytes items(POINTS * GPUFrame::CHUNK_INPUT_BYTES);
        VoxelPos points[POINTS];
        int lods[POINTS];
        for (uint32_t i = 0; i < POINTS; i++) {
            lods[i] = int(i % uint32_t(sw + 3));
            const int64_t step = int64_t(1) << lods[i];
            const VoxelPos t = b.tangent_point(int64_t(i) * 999331 - 32000000, -int64_t(i) * 77777);
            points[i] = VoxelPos(WorldCoord::floor_div(t.x, step) * step, WorldCoord::floor_div(t.y, step) * step, WorldCoord::floor_div(t.z, step) * step);
            CHECK(GPUFrame::encode_point_item(items.data() + size_t(i) * GPUFrame::CHUNK_INPUT_BYTES, points[i], lods[i]));
        }
        BaseRegions::Tables tables;
        BaseRegions::build(b, items.data(), POINTS, point_param, true, tables);
        int wrong = 0;
        for (uint32_t i = 0; i < POINTS; i++) {
            wrong += lods[i] < sw && !reads_own_region(b, tables, i, points[i], lods[i]);
        }
        CHECK(wrong == 0);
        CHECK(tables.region_count() <= POINTS + 1); // CascadePasses' buffer: one region per point item, plus the zero one
    }

    SUBCASE("items at or above the switch LOD get no region") {
        constexpr uint32_t CHUNKS = 4;
        ov::bytes items(CHUNKS * GPUFrame::CHUNK_INPUT_BYTES);
        for (uint32_t i = 0; i < CHUNKS; i++) {
            GPUFrame::encode_chunk_input(items, i, surface_chunk(b, sw + int(i), int(i), param), param);
        }
        BaseRegions::Tables tables;
        BaseRegions::build(b, items.data(), CHUNKS, param, false, tables);
        CHECK(tables.region_count() == 1); // the zero region alone
        for (const uint32_t v : tables.index) {
            CHECK(v == 0);
        }
    }

    SUBCASE("neighbouring chunks share their regions") {
        // A 4^3 block of LOD-3 chunks spans 4 * 28 = 112 cells per axis: 2 or 3 region cells per axis, not 8 per chunk.
        const int lod = 3;
        const ChunkId first = surface_chunk(b, lod, 0, param);
        const int64_t step = int64_t(2) << lod;
        ov::bytes items(64 * GPUFrame::CHUNK_INPUT_BYTES);
        for (uint32_t i = 0; i < 64; i++) {
            GPUFrame::encode_chunk_input(items, i, first + Vector3i64(int64_t(i % 4) * step, int64_t(i / 4 % 4) * step, int64_t(i / 16) * step), param);
        }
        BaseRegions::Tables tables;
        BaseRegions::build(b, items.data(), 64, param, false, tables);
        CHECK(tables.region_count() <= 27);
        CHECK(tables.region_count() >= 8);
    }
}

TEST_CASE("BaseRegions: the sphere's base_at GLSL names the bindings and the switch") {
    const BaseSurface b = BaseSurface::sphere(Vector3i64(0, 0, 0), EARTH_R);
    const ov::str g = BaseRegions::glsl(b);
    CHECK(ov::contains(g, ov::format("binding = %d", BaseRegions::REGION_BINDING)));
    CHECK(ov::contains(g, ov::format("binding = %d", BaseRegions::INDEX_BINDING)));
    CHECK(ov::contains(g, ov::format("lod >= %d", b.get_switch_lod())));
    CHECK(!ov::contains(g, "SLOTS"));
}

TEST_CASE("SDFCascade::planet_base: a large planet raises its switch LOD instead of being refused") {
    const BaseSurface earth = SDFCascade::planet_base(6371000.0, 0.05);
    CHECK(earth.kind() == BaseSurface::Kind::SPHERE);
    CHECK(earth.get_switch_lod() == BaseSurface::SWITCH_LOD);
    const BaseSurface jupiter = SDFCascade::planet_base(69911000.0, 0.05);
    CHECK(jupiter.kind() == BaseSurface::Kind::SPHERE);
    CHECK(jupiter.get_switch_lod() > BaseSurface::SWITCH_LOD);
    CHECK(SDFCascade::planet_base(6371000.0, 0.005).kind() == BaseSurface::Kind::SPHERE); // Earth at 5 mm
}

TEST_CASE("SDFCascade::planet_base: a small planet lowers its switch LOD, down to the radius form alone") {
    // 50 km at 5 cm: 10^6 voxels, the curvature form below LOD 5 (its regions 64 * 2^4 voxels, 51 m).
    const BaseSurface moon = SDFCascade::planet_base(50000.0, 0.05);
    CHECK(moon.kind() == BaseSurface::Kind::SPHERE);
    CHECK(moon.get_switch_lod() == 5);
    // 2 km: 40,000 voxels, the curvature form at LOD 0 only; 100 m: the radius form alone.
    for (const auto &[radius_m, switch_lod] : { std::pair{ 2000.0, 1 }, std::pair{ 100.0, 0 } }) {
        const BaseSurface tiny = SDFCascade::planet_base(radius_m, 0.05);
        CHECK(tiny.kind() == BaseSurface::Kind::SPHERE);
        CHECK(tiny.get_switch_lod() == switch_lod);
        // The radius form at the switch next to the pole: |p| - R within a few float steps of the radius.
        const VoxelPos p = tiny.tangent_point(0, 0) + VoxelPos(0, 8, 0);
        const double alt = std::ldexp(tiny.altitude(p), -switch_lod);
        CHECK(std::fabs(double(tiny.at(p, switch_lod).w) - alt) < 1e-3);
    }
}

TEST_CASE("SDFCascade: a planet's relief is capped at PLANET_RELIEF of its radius, Earth's and the flat world's untouched") {
    ChunkParam param;
    param.voxel_size = 0.05f;
    auto max_amp = [&](double p_radius_m) {
        SDFCascade sdf;
        sdf.planet_radius_m = p_radius_m;
        sdf.configure(param, 16);
        float amp = 0.0f;
        for (const Cascade::Band &b : sdf.get_spec().bands) {
            amp = ov::max(amp, b.amp);
        }
        return double(amp);
    };
    const double flat = max_amp(0.0);
    CHECK(flat == doctest::Approx(SDFCascade::MAX_AMP_M / 0.05));
    CHECK(max_amp(6371000.0) == flat);
    CHECK(max_amp(2000.0) == doctest::Approx(SDFCascade::PLANET_RELIEF / SDFCascade::PLANET_HEADROOM * 2000.0 / 0.05));
}

TEST_CASE("SDFCascade: a climate's biome multipliers fit under a 2 km planet's relief; its curves in metres don't") {
    auto error = [](const char *p_path) {
        SDFCascade sdf;
        CHECK(Climate::load(p_path, sdf.climate_data) == ov::str());
        sdf.planet_radius_m = 2000.0;
        ChunkParam param;
        param.voxel_size = 0.05f;
        sdf.configure(param, 16);
        CHECK((sdf.get_climate() != nullptr) == sdf.get_climate_error().empty());
        return sdf.get_climate_error();
    };
    // Biome multipliers up to 2.5 (it was refused at 1: the octaves started at the relief itself).
    CHECK(error("res://climate/plain.json") == ov::str());
    // The mountains' 1.1 km curve: more than the planet's 62.5 m relief.
    CHECK(error("res://climate/example.json").find("its curve moves the ground") != ov::str::npos);
}

TEST_CASE("BaseSurface::on_base: the base's point along the up, on any side of a planet") {
    const BaseSurface flat = BaseSurface::flat(1234);
    CHECK(flat.on_base(VoxelPos(-5, 99999, 7)) == VoxelPos(-5, 1234, 7));
    const BaseSurface planet = SDFCascade::planet_base(2000.0, 0.05);
    for (const VoxelPos p : { VoxelPos(100000, 3, -2), VoxelPos(-7, -90000, 11), VoxelPos(5000, 6000, -39000), VoxelPos(0, 0, 0) }) {
        const VoxelPos b = planet.on_base(p);
        CHECK(std::fabs(planet.altitude(b)) <= 1.0);
        const UpAxis up = planet.up_dir(p), bu = planet.up_dir(b);
        if (!(p == VoxelPos(0, 0, 0))) {
            CHECK(up[0] * bu[0] + up[1] * bu[1] + up[2] * bu[2] > 0.9999);
        }
    }
}
