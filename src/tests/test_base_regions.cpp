#include "test_climate_common.hpp"

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

TEST_CASE("BaseSurface::planet: a large planet raises its switch LOD instead of being refused") {
    const BaseSurface earth = BaseSurface::planet(6371000.0, 0.05);
    CHECK(earth.kind() == BaseSurface::Kind::SPHERE);
    CHECK(earth.get_switch_lod() == BaseSurface::SWITCH_LOD);
    const BaseSurface jupiter = BaseSurface::planet(69911000.0, 0.05);
    CHECK(jupiter.kind() == BaseSurface::Kind::SPHERE);
    CHECK(jupiter.get_switch_lod() > BaseSurface::SWITCH_LOD);
    CHECK(BaseSurface::planet(6371000.0, 0.005).kind() == BaseSurface::Kind::SPHERE); // Earth at 5 mm
}

TEST_CASE("BaseSurface::planet: a small planet lowers its switch LOD, down to the radius form alone") {
    // 50 km at 5 cm: 10^6 voxels, the curvature form below LOD 5 (its regions 64 * 2^4 voxels, 51 m).
    const BaseSurface moon = BaseSurface::planet(50000.0, 0.05);
    CHECK(moon.kind() == BaseSurface::Kind::SPHERE);
    CHECK(moon.get_switch_lod() == 5);
    // 2 km: 40,000 voxels, the curvature form at LOD 0 only; 100 m: the radius form alone.
    for (const auto &[radius_m, switch_lod] : { std::pair{ 2000.0, 1 }, std::pair{ 100.0, 0 } }) {
        const BaseSurface tiny = BaseSurface::planet(radius_m, 0.05);
        CHECK(tiny.kind() == BaseSurface::Kind::SPHERE);
        CHECK(tiny.get_switch_lod() == switch_lod);
        // The radius form at the switch next to the pole: |p| - R within a few float steps of the radius.
        const VoxelPos p = tiny.tangent_point(0, 0) + VoxelPos(0, 8, 0);
        const double alt = std::ldexp(tiny.altitude(p), -switch_lod);
        CHECK(std::fabs(double(tiny.at(p, switch_lod).w) - alt) < 1e-3);
    }
}

TEST_CASE("SDFCascade: a note plays at its written amplitude, on the flat world and on any planet") {
    ChunkParam param;
    param.voxel_size = 0.05f;
    auto max_amp = [&](double p_radius_m, const char *p_world) {
        GroundSDF sdf;
        if (p_world != nullptr) {
            CHECK(load_table(p_world, sdf.climate_data) == ov::str());
        }
        sdf.planet_radius_m = p_radius_m;
        sdf.configure(param, 16);
        CHECK(sdf.get_world_error() == ov::str());
        return double(ClimateTest::note_amp(sdf.program()));
    };
    const double flat = max_amp(0.0, nullptr);
    CHECK(flat == doctest::Approx(3276.8 / 0.05)); // the template's top note (ground.biomes.json)
    CHECK(max_amp(6371000.0, nullptr) == flat);
    // A 2 km planet takes tiny.world.json's 409.6 m note as written (a cap held every note under 8 m before).
    CHECK(max_amp(2000.0, "res://worlds/tiny.world.json") == doctest::Approx(409.6 / 0.05));
}

// p_sdf configured at 5 cm and 16 LODs on a planet of p_radius_m (a torus round p_ring_m > 0) with p_path's climate
// (none: null).
static void configure_planet(SDFCascade &r_sdf, const char *p_path, double p_radius_m, double p_ring_m = 0.0) {
    if (p_path != nullptr) {
        CHECK(load_table(p_path, r_sdf.climate_data) == ov::str());
    }
    r_sdf.planet_radius_m = p_radius_m;
    r_sdf.planet_ring_m = p_ring_m;
    ChunkParam param;
    param.voxel_size = 0.05f;
    r_sdf.configure(param, 16);
}

TEST_CASE("SDFCascade: a 2 km planet takes 400 m notes and refuses a ground under its centre, never another ground") {
    // tiny.world.json: the template's octaves up to 3.3 km wavelengths, 819 m of amplitudes in all: above the centre.
    SDFCascade tiny;
    configure_planet(tiny, "res://worlds/tiny.world.json", 2000.0);
    CHECK(tiny.get_world_error() == ov::str());
    CHECK(tiny.get_climate() != nullptr);
    const Terrain::LevelProgram &prog = tiny.program();
    const size_t out = size_t(prog.output);
    CHECK(prog.lo[out] > -2000.0 / 0.05);
    CHECK(prog.hi[out] > 400.0 / 0.05);
    CHECK(ClimateTest::note_top_k(prog) == 16);
    CHECK(tiny.ground_top_m() == doctest::Approx(prog.hi[out] * 0.05));
    // The template ground itself: its 3.3 km notes reach through the centre. A world error and no program, not a flat
    // ground.
    GroundSDF bare;
    configure_planet(bare, nullptr, 2000.0);
    CHECK(ov::contains(bare.get_world_error(), "past the planet's centre"));
    CHECK(bare.program().nodes.empty());
    // A given ground 2.5 km down: refused likewise, no template instead (World::start refuses it); the flat world takes it.
    SDFCascade sunk;
    sunk.climate_data = ClimateTest::json(R"({"biomes": [{"name": "deep", "graph": {"nodes": [{"id": "c", "kind": "constant", "value_m": -2500},
        {"id": "o", "kind": "output", "height": "c"}]}}]})");
    configure_planet(sunk, nullptr, 2000.0);
    CHECK(sunk.get_climate() == nullptr);
    CHECK(ov::contains(sunk.get_world_error(), "past the planet's centre"));
    CHECK(sunk.program().nodes.empty());
    configure_planet(sunk, nullptr, 0.0);
    CHECK(sunk.get_world_error() == ov::str());
    CHECK(sunk.get_climate() != nullptr);
}

TEST_CASE("SDFCascade: a refused planet is a world error and no program, never a silent flat world") {
    // A torus whose ring is its tube (under BaseSurface::MIN_RING_TUBES): it used to open as the flat datum, ok.
    SDFCascade torus;
    configure_planet(torus, "res://worlds/tiny.world.json", 2000.0, 2000.0);
    CHECK_FALSE(torus.get_base().curved());
    CHECK(ov::contains(torus.get_world_error(), "the planet (radius 2000 m, ring 2000 m)"));
    CHECK(ov::contains(torus.get_world_error(), "ring must be at least"));
    CHECK(torus.program().nodes.empty());
    CHECK(torus.get_climate() == nullptr);
    ov::str refusal;
    CHECK(BaseSurface::planet(2000.0, 0.05, 2000.0, &refusal).kind() == BaseSurface::Kind::FLAT);
    CHECK(ov::contains(refusal, "ring must be at least"));
    // A sphere past 2^60 voxels.
    GroundSDF huge;
    configure_planet(huge, nullptr, 1e20);
    CHECK(ov::contains(huge.get_world_error(), "the radius must be in"));
    CHECK(huge.program().nodes.empty());
    // An accepted torus (its 2 km tube takes tiny's ground, kept from above), and a refused planet reconfigured flat: no
    // error left.
    configure_planet(torus, nullptr, 2000.0, 6000.0);
    CHECK(torus.get_base().kind() == BaseSurface::Kind::TORUS);
    CHECK(torus.get_world_error() == ov::str());
    configure_planet(huge, nullptr, 0.0);
    CHECK(huge.get_world_error() == ov::str());
    CHECK_FALSE(huge.program().nodes.empty());
}

TEST_CASE("BaseSurface::on_base: the base's point along the up, on any side of a planet") {
    const BaseSurface flat = BaseSurface::flat(1234);
    CHECK(flat.on_base(VoxelPos(-5, 99999, 7)) == VoxelPos(-5, 1234, 7));
    const BaseSurface planet = BaseSurface::planet(2000.0, 0.05);
    for (const VoxelPos p : { VoxelPos(100000, 3, -2), VoxelPos(-7, -90000, 11), VoxelPos(5000, 6000, -39000), VoxelPos(0, 0, 0) }) {
        const VoxelPos b = planet.on_base(p);
        CHECK(std::fabs(planet.altitude(b)) <= 1.0);
        const UpAxis up = planet.up_dir(p), bu = planet.up_dir(b);
        if (!(p == VoxelPos(0, 0, 0))) {
            CHECK(up[0] * bu[0] + up[1] * bu[1] + up[2] * bu[2] > 0.9999);
        }
    }
}
