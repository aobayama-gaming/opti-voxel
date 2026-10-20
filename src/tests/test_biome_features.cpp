#include "test_common.hpp"

#include "biome_features.hpp"

#include <algorithm>
#include <cmath>

using namespace Archetype;

namespace {

// The biome types (Biome::add_types, as SDFTable::build_table).
struct BiomeLibrary {
    DefLibrary lib;
    Biome::Types types;
    explicit BiomeLibrary(double p_voxel) :
            types(Biome::add_types(lib, p_voxel)) {}
};

bool same_site(const Biome::Site &p_a, const Biome::Site &p_b) {
    return p_a.feature == p_b.feature && p_a.x == p_b.x && p_a.z == p_b.z && p_a.radius == p_b.radius && p_a.seed == p_b.seed;
}

// A flat ground whose top solid voxel is y = p_top: the register the cascade gives (unit gradient, distance in voxels).
Float4 flat_ground(const VoxelPos &p_voxel, int64_t p_top) {
    return Float4{ 0.0f, 1.0f, 0.0f, float(double(p_voxel.y - p_top) - 0.5) };
}

// The value at p_voxel (LOD 0) of the ground with p_records folded in, finished as SDFTable does (no band).
float field(const BiomeLibrary &p_lib, std::vector<TableRecord> &r_records, const VoxelPos &p_voxel, int64_t p_top) {
    std::vector<uint32_t> list(r_records.size());
    for (uint32_t i = 0; i < list.size(); i++) {
        finish_record(p_lib.lib[r_records[i].def], r_records[i], 2.0f, false, ErodeRule());
        list[i] = i;
    }
    WorldPos p;
    p.voxel = p_voxel;
    return fold_onto(flat_ground(p_voxel, p_top), p_lib.lib, r_records.data(), list.data(), uint32_t(list.size()), p, 0, true).w;
}

} // namespace

TEST_CASE("Biome: a cell's site is the same from any square, in the middle half of its cell") {
    const double voxel = 0.5;
    for (int f = 0; f < Biome::FEATURE_COUNT; f++) {
        const Biome::Feature feature = Biome::Feature(f);
        const int64_t cell = int64_t(1) << Biome::cell_shift(feature, voxel);
        const int64_t far = (int64_t(1) << 60) - 40 * cell;
        for (const int64_t base : { int64_t(0), -far, far - 64 * cell, (int64_t(1) << 32) - 3 * cell }) {
            std::vector<Biome::Site> wide, narrow, again;
            Biome::sites(feature, base - 32 * cell, base - 32 * cell, base + 32 * cell, base + 32 * cell, voxel, wide);
            Biome::sites(feature, base - 5 * cell + 7, base - 9 * cell - 3, base + 6 * cell, base + 2 * cell + 11, voxel, narrow);
            Biome::sites(feature, base - 32 * cell, base - 32 * cell, base + 32 * cell, base + 32 * cell, voxel, again);
            CHECK(wide.size() == again.size());
            if (wide.size() != again.size()) {
                continue;
            }
            for (size_t i = 0; i < wide.size(); i++) {
                CHECK(same_site(wide[i], again[i]));
            }
            for (const Biome::Site &s : narrow) {
                CHECK(std::any_of(wide.begin(), wide.end(), [&](const Biome::Site &w) { return same_site(w, s); }));
            }
            // The chance: 65^2 cells, about 0.5-0.6 of them kept.
            const double kept = double(wide.size()) / (65.0 * 65.0);
            CHECK(std::fabs(kept - Biome::kind(feature).chance) < 0.05);
            for (const Biome::Site &s : wide) {
                const int64_t ix = s.x - WorldCoord::floor_div(s.x, cell) * cell;
                const int64_t iz = s.z - WorldCoord::floor_div(s.z, cell) * cell;
                CHECK(ix >= cell / 4);
                CHECK(ix < 3 * cell / 4);
                CHECK(iz >= cell / 4);
                CHECK(iz < 3 * cell / 4);
                CHECK(s.radius * voxel >= Biome::kind(feature).radius_lo_m);
                CHECK(s.radius * voxel < Biome::kind(feature).radius_hi_m);
                // The other spots a lake tries: in the middle half of the same cell, the rest unchanged.
                for (int attempt = 0; attempt < Biome::LAKE_ATTEMPTS; attempt++) {
                    const Biome::Site m = Biome::moved(s, attempt, voxel);
                    CHECK(WorldCoord::floor_div(m.x, cell) == WorldCoord::floor_div(s.x, cell));
                    CHECK(WorldCoord::floor_div(m.z, cell) == WorldCoord::floor_div(s.z, cell));
                    CHECK(m.x - WorldCoord::floor_div(m.x, cell) * cell >= cell / 4);
                    CHECK(m.x - WorldCoord::floor_div(m.x, cell) * cell < 3 * cell / 4);
                    CHECK(m.z - WorldCoord::floor_div(m.z, cell) * cell >= cell / 4);
                    CHECK(m.z - WorldCoord::floor_div(m.z, cell) * cell < 3 * cell / 4);
                    CHECK((m.radius == s.radius && m.seed == s.seed && m.feature == s.feature));
                    CHECK((attempt != 0 || same_site(m, s)));
                }
            }
        }
    }
}

TEST_CASE("Biome: a lava lake on flat ground is a flat lava surface below the rim, the ground untouched far away") {
    for (const double voxel : { 0.5, 0.05 }) {
        const BiomeLibrary lib(voxel);
        const int64_t top = 1000;
        Biome::Site site;
        site.x = 40000;
        site.z = -7000;
        site.radius = 30.0 / voxel;
        site.seed = 77;
        Biome::Placed out;
        const bool placed = Biome::place(site, [&](int64_t, int64_t) { return top; }, lib.lib, lib.types, voxel, out);
        CHECK(placed);
        CHECK(out.lava);
        CHECK(out.records.size() == 2);
        if (!placed || out.records.size() != 2) {
            continue;
        }
        CHECK(same_site(out.at, site)); // flat: the first spot
        std::vector<TableRecord> &records = out.records;
        const Biome::LavaPool &pool = out.pool;
        CHECK(pool.top.y < top);
        CHECK(pool.radius == site.radius); // the colour: the basin, not the slab past its wall
        // The lake's centre column: air above the lava (the ground cut away), lava at and just below its surface.
        CAPTURE(voxel);
        CHECK(field(lib, records, VoxelPos(site.x, top - 1, site.z), top) > 0.0f);
        CHECK(field(lib, records, VoxelPos(site.x, pool.top.y + 1, site.z), top) > 0.0f);
        CHECK(field(lib, records, VoxelPos(site.x, pool.top.y, site.z), top) <= 0.0f);
        CHECK(field(lib, records, VoxelPos(site.x, pool.top.y - 3, site.z), top) < 0.0f);
        // Half way to the shore, the same lava level.
        const int64_t half = std::llround(0.5 * site.radius);
        CHECK(field(lib, records, VoxelPos(site.x + half, pool.top.y + 1, site.z), top) > 0.0f);
        CHECK(field(lib, records, VoxelPos(site.x + half, pool.top.y, site.z), top) <= 0.0f);
        // Far away, the bare ground bit for bit.
        const VoxelPos away(site.x + std::llround(3.0 * site.radius), top + 2, site.z);
        CHECK(field(lib, records, away, top) == flat_ground(away, top).w);
    }
}

TEST_CASE("Biome: no lava lake on a slope") {
    const BiomeLibrary lib(0.5);
    Biome::Site site;
    site.radius = 60.0;
    Biome::Placed out;
    CHECK_FALSE(Biome::place(site, [](int64_t p_x, int64_t) { return p_x; }, lib.lib, lib.types, 0.5, out));
    CHECK(out.records.empty());
}

TEST_CASE("Biome: a floating island floats, its underside 25-70 m above the highest ground under it") {
    for (const double voxel : { 0.5, 0.05 }) {
        const BiomeLibrary lib(voxel);
        // A ground rising to the east: the island clears its highest point.
        auto ground = [](int64_t p_x, int64_t) { return 500 + p_x / 8; };
        for (uint32_t seed = 0; seed < 8; seed++) {
            Biome::Site site;
            site.feature = Biome::FLOATING_ISLAND;
            site.x = 1000 + 37 * int64_t(seed);
            site.radius = (15.0 + 3.0 * seed) / voxel;
            site.seed = seed * 7919u;
            Biome::Placed out;
            CHECK(Biome::place(site, ground, lib.lib, lib.types, voxel, out));
            std::vector<TableRecord> &records = out.records;
            CHECK(records.size() == 1);
            if (records.size() != 1) {
                continue;
            }
            const TableRecord &rec = records[0];
            const double bottom = double(rec.anchor.y) - double(rec.params[1]); // the apex, without the bumps
            const int64_t top = ground(site.x, 0);
            CAPTURE(voxel);
            CAPTURE(seed);
            // Above the ground under its centre by the altitude at least, and at most the altitude plus the rise of
            // the ground across the island (1/8 per voxel).
            CHECK(bottom - double(top) >= 25.0 / voxel);
            CHECK(bottom - double(top) <= 70.0 / voxel + site.radius);
            std::vector<TableRecord> folded = records;
            // Inside the island, half its underside's depth below the rim plane: solid; half way down to the ground: air.
            CHECK(field(lib, folded, VoxelPos(site.x, rec.anchor.y - std::llround(0.5 * rec.params[1]), site.z), top) < 0.0f);
            CHECK(field(lib, folded, VoxelPos(site.x, (top + std::llround(bottom)) / 2 - 1, site.z), top) > 0.0f);
        }
    }
}
