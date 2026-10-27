#include "test_common.hpp"

#include "biome_features.hpp"
#include "sdf_cascade.hpp"

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
    return p_a.feature == p_b.feature && p_a.at == p_b.at && p_a.radius == p_b.radius && p_a.seed == p_b.seed;
}

// The flat base the lake tests stand on: its datum at y = 0.
const BaseSurface FLAT0 = BaseSurface::flat(0);

// A ground under every point from p_top_y(x) (the top solid voxel of the column), on FLAT0.
template <typename F>
Biome::Ground column_ground(F p_top_y) {
    return Biome::Ground{ &FLAT0, [p_top_y](const VoxelPos &p_v) { return VoxelPos(p_v.x, p_top_y(p_v.x), p_v.z); } };
}

// A flat ground whose top solid voxel is y = p_top: the register the cascade gives (unit gradient, distance in voxels).
Float4 flat_ground(const VoxelPos &p_voxel, int64_t p_top) {
    return Float4{ 0.0f, 1.0f, 0.0f, float(double(p_voxel.y - p_top) - 0.5) };
}

// The value at p_voxel (LOD 0) of the ground p_ground (a register there) with p_records folded in, finished as SDFTable
// does (no band).
float fold(const BiomeLibrary &p_lib, std::vector<TableRecord> &r_records, const VoxelPos &p_voxel, const Float4 &p_ground) {
    std::vector<uint32_t> list(r_records.size());
    for (uint32_t i = 0; i < list.size(); i++) {
        finish_record(p_lib.lib[r_records[i].def], r_records[i], 2.0f, false, ErodeRule());
        list[i] = i;
    }
    WorldPos p;
    p.voxel = p_voxel;
    return fold_onto(p_ground, p_lib.lib, r_records.data(), list.data(), uint32_t(list.size()), p, 0, true).w;
}

float field(const BiomeLibrary &p_lib, std::vector<TableRecord> &r_records, const VoxelPos &p_voxel, int64_t p_top) {
    return fold(p_lib, r_records, p_voxel, flat_ground(p_voxel, p_top));
}

Vector3i64 cell_of(const VoxelPos &p_v, int64_t p_cell) {
    return Vector3i64(WorldCoord::floor_div(p_v.x, p_cell), WorldCoord::floor_div(p_v.y, p_cell), WorldCoord::floor_div(p_v.z, p_cell));
}

} // namespace

TEST_CASE("Biome: a cell's site is the same from any box, on the flat datum in the middle half of its cell") {
    const double voxel = 0.5;
    for (int f = 0; f < Biome::FEATURE_COUNT; f++) {
        const Biome::Feature feature = Biome::Feature(f);
        const int64_t cell = int64_t(1) << Biome::cell_shift(feature, voxel);
        const int64_t far = (int64_t(1) << 60) - 40 * cell;
        for (const int64_t base : { int64_t(0), -far, far - 64 * cell, (int64_t(1) << 32) - 3 * cell }) {
            const BaseSurface flat = BaseSurface::flat(base / 3);
            const int64_t d = flat.datum_y();
            std::vector<Biome::Site> wide, narrow, again;
            Biome::sites(feature, VoxelPos(base - 32 * cell, d - 3 * cell, base - 32 * cell), VoxelPos(base + 32 * cell, d + 2 * cell, base + 32 * cell), flat, voxel, wide);
            Biome::sites(feature, VoxelPos(base - 5 * cell + 7, d - 1, base - 9 * cell - 3), VoxelPos(base + 6 * cell, d, base + 2 * cell + 11), flat, voxel, narrow);
            Biome::sites(feature, VoxelPos(base - 32 * cell, d - 3 * cell, base - 32 * cell), VoxelPos(base + 32 * cell, d + 2 * cell, base + 32 * cell), flat, voxel, again);
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
            // The chance: 65^2 columns (one cell of each on the datum), about 0.5-0.6 of them kept.
            const double kept = double(wide.size()) / (65.0 * 65.0);
            CHECK(std::fabs(kept - Biome::kind(feature).chance) < 0.05);
            for (const Biome::Site &s : wide) {
                CHECK(s.at.y == d);
                const int64_t ix = s.at.x - WorldCoord::floor_div(s.at.x, cell) * cell;
                const int64_t iz = s.at.z - WorldCoord::floor_div(s.at.z, cell) * cell;
                CHECK(ix >= cell / 4);
                CHECK(ix < 3 * cell / 4);
                CHECK(iz >= cell / 4);
                CHECK(iz < 3 * cell / 4);
                CHECK(s.radius * voxel >= Biome::kind(feature).radius_lo_m);
                CHECK(s.radius * voxel < Biome::kind(feature).radius_hi_m);
                // The other spots a lake tries: in the middle half of the same cell, the rest unchanged.
                for (int attempt = 0; attempt < Biome::LAKE_ATTEMPTS; attempt++) {
                    const Biome::Site m = Biome::moved(s, attempt, flat, voxel);
                    CHECK(cell_of(m.at, cell) == cell_of(s.at, cell));
                    CHECK(m.at.x - WorldCoord::floor_div(m.at.x, cell) * cell >= cell / 4);
                    CHECK(m.at.x - WorldCoord::floor_div(m.at.x, cell) * cell < 3 * cell / 4);
                    CHECK(m.at.z - WorldCoord::floor_div(m.at.z, cell) * cell >= cell / 4);
                    CHECK(m.at.z - WorldCoord::floor_div(m.at.z, cell) * cell < 3 * cell / 4);
                    CHECK((m.radius == s.radius && m.seed == s.seed && m.feature == s.feature));
                    CHECK((attempt != 0 || same_site(m, s)));
                }
            }
        }
    }
}

TEST_CASE("Biome: on a planet, sites are on the base, one per cell, the same from any box, all round") {
    const double voxel = 0.5;
    const BaseSurface planet = SDFCascade::planet_base(20000.0, voxel); // 20 km: 40,000 voxels
    CHECK(planet.kind() == BaseSurface::Kind::SPHERE);
    if (!(planet.kind() == BaseSurface::Kind::SPHERE)) {
        return;
    }
    const int64_t r = std::llround(planet.get_radius());
    for (int f = 0; f < Biome::FEATURE_COUNT; f++) {
        const Biome::Feature feature = Biome::Feature(f);
        const int64_t cell = int64_t(1) << Biome::cell_shift(feature, voxel);
        std::vector<Biome::Site> all, half;
        Biome::sites(feature, VoxelPos(-r - cell, -r - cell, -r - cell), VoxelPos(r + cell, r + cell, r + cell), planet, voxel, all);
        Biome::sites(feature, VoxelPos(0, -r - cell, -r - cell), VoxelPos(r + cell, r + cell, r + cell), planet, voxel, half);
        // The sphere's area over a cell's face, times the chance: within 30 % (cells cut by the sphere count once).
        const double expected = 4.0 * M_PI * double(r) * double(r) / double(cell * cell) * Biome::kind(feature).chance;
        CAPTURE(all.size());
        CHECK(double(all.size()) > 0.7 * expected);
        CHECK(double(all.size()) < 1.3 * expected);
        auto key = [](const VoxelPos &p_v) { return std::array<int64_t, 3>{ p_v.x, p_v.y, p_v.z }; };
        std::vector<std::array<int64_t, 3>> cells, half_at;
        for (const Biome::Site &s : half) {
            half_at.push_back(key(s.at));
        }
        std::sort(half_at.begin(), half_at.end());
        int sides[6] = {};
        int off_base = 0, not_in_half = 0;
        for (const Biome::Site &s : all) {
            off_base += std::fabs(planet.altitude(s.at)) > 1.0;
            cells.push_back(key(VoxelPos(cell_of(s.at, cell).x, cell_of(s.at, cell).y, cell_of(s.at, cell).z)));
            for (int a = 0; a < 3; a++) {
                const int64_t v = a == 0 ? s.at.x : (a == 1 ? s.at.y : s.at.z);
                sides[2 * a] += v > r / 2;
                sides[2 * a + 1] += v < -r / 2;
            }
            not_in_half += s.at.x >= 0 && !std::binary_search(half_at.begin(), half_at.end(), key(s.at));
        }
        CHECK(off_base == 0);
        CHECK(not_in_half == 0);
        std::sort(cells.begin(), cells.end());
        CHECK(std::adjacent_find(cells.begin(), cells.end()) == cells.end()); // one site per cell
        for (int k = 0; k < 6; k++) {
            CHECK(sides[k] > 0); // all round, not a pole's plane
        }
    }
}

TEST_CASE("Biome: a lava lake on flat ground is a flat lava surface below the rim, the ground untouched far away") {
    for (const double voxel : { 0.5, 0.05 }) {
        const BiomeLibrary lib(voxel);
        const int64_t top = 1000;
        Biome::Site site;
        site.at = VoxelPos(40000, 0, -7000);
        site.radius = 30.0 / voxel;
        site.seed = 77;
        Biome::Placed out;
        const bool placed = Biome::place(site, column_ground([&](int64_t) { return top; }), lib.lib, lib.types, voxel, out);
        CHECK(placed);
        CHECK(out.lava);
        CHECK(out.records.size() == 2);
        if (!placed || out.records.size() != 2) {
            continue;
        }
        CHECK(same_site(out.at, site)); // flat: the first spot
        std::vector<TableRecord> &records = out.records;
        const Biome::LavaPool &pool = out.pool;
        const int64_t x = site.at.x, z = site.at.z;
        CHECK(pool.top.y < top);
        CHECK(pool.radius == site.radius); // the colour: the basin, not the slab past its wall
        // The lake's centre column: air above the lava (the ground cut away), lava at and just below its surface.
        CAPTURE(voxel);
        CHECK(field(lib, records, VoxelPos(x, top - 1, z), top) > 0.0f);
        CHECK(field(lib, records, VoxelPos(x, pool.top.y + 1, z), top) > 0.0f);
        CHECK(field(lib, records, VoxelPos(x, pool.top.y, z), top) <= 0.0f);
        CHECK(field(lib, records, VoxelPos(x, pool.top.y - 3, z), top) < 0.0f);
        // Half way to the shore, the same lava level.
        const int64_t half = std::llround(0.5 * site.radius);
        CHECK(field(lib, records, VoxelPos(x + half, pool.top.y + 1, z), top) > 0.0f);
        CHECK(field(lib, records, VoxelPos(x + half, pool.top.y, z), top) <= 0.0f);
        // Far away, the bare ground bit for bit.
        const VoxelPos away(x + std::llround(3.0 * site.radius), top + 2, z);
        CHECK(field(lib, records, away, top) == flat_ground(away, top).w);
        // The keep-out: the lake's axis, not its height.
        CHECK(pool.near(VoxelPos(x + half, pool.top.y + 5000, z)));
        CHECK_FALSE(pool.near(VoxelPos(x + std::llround(2.0 * site.radius), pool.top.y, z)));
    }
}

TEST_CASE("Biome: a lava lake 90 degrees round a planet stands on its up, the y axis level there") {
    const double voxel = 0.05;
    const BiomeLibrary lib(voxel);
    const BaseSurface planet = SDFCascade::planet_base(2000.0, voxel); // 2 km
    CHECK(planet.kind() == BaseSurface::Kind::SPHERE);
    if (!(planet.kind() == BaseSurface::Kind::SPHERE)) {
        return;
    }
    // The ground is the base itself: its top solid voxel along the up, the register its altitude.
    const Biome::Ground ground{ &planet, [&](const VoxelPos &p_v) { return planet.on_base(p_v); } };
    auto sphere_ground = [&](const VoxelPos &p_v) {
        const UpAxis n = planet.up_dir(p_v);
        return Float4{ float(n[0]), float(n[1]), float(n[2]), float(planet.altitude(p_v) - 0.5) };
    };
    Biome::Site site;
    site.at = planet.on_base(VoxelPos(int64_t(1) << 20, 0, 0)); // the +x equator: up = +x
    site.radius = 20.0 / voxel;
    site.seed = 5;
    Biome::Placed out;
    const bool placed = Biome::place(site, ground, lib.lib, lib.types, voxel, out);
    CHECK(placed);
    CHECK(out.records.size() == 2);
    if (!placed || out.records.size() != 2) {
        return;
    }
    const Biome::LavaPool &pool = out.pool;
    CHECK(pool.up[0] > 0.999);
    for (const TableRecord &rec : out.records) {
        CHECK(rec.rotation[3] > 0.999f); // local y on +x
    }
    std::vector<TableRecord> &records = out.records;
    auto f = [&](const VoxelPos &p_v) { return fold(lib, records, p_v, sphere_ground(p_v)); };
    const VoxelPos t = pool.top;
    CHECK(planet.altitude(t) < 0.0);
    CHECK(f(t + VoxelPos(1, 0, 0)) > 0.0f);
    CHECK(f(t) <= 0.0f);
    CHECK(f(t - VoxelPos(3, 0, 0)) < 0.0f);
    // Half way to the shore along a tangent (z, or y: level here), the same lava level.
    const int64_t half = std::llround(0.5 * site.radius);
    for (const VoxelPos side : { VoxelPos(0, 0, half), VoxelPos(0, -half, 0) }) {
        CHECK(f(t + side + VoxelPos(1, 0, 0)) > 0.0f);
        CHECK(f(t + side) <= 0.0f);
    }
    // Far away round the planet, the bare ground bit for bit.
    const VoxelPos away = planet.on_base(t + VoxelPos(0, 0, std::llround(3.0 * site.radius)));
    CHECK(f(away) == sphere_ground(away).w);
    CHECK(pool.near(t + VoxelPos(400, half, 0)));
    CHECK_FALSE(pool.near(t + VoxelPos(0, 2 * std::llround(site.radius), 0)));
}

TEST_CASE("Biome: no lava lake on a slope") {
    const BiomeLibrary lib(0.5);
    Biome::Site site;
    site.radius = 60.0;
    Biome::Placed out;
    CHECK_FALSE(Biome::place(site, column_ground([](int64_t p_x) { return p_x; }), lib.lib, lib.types, 0.5, out));
    CHECK(out.records.empty());
}

TEST_CASE("Biome: a floating island floats, its underside 25-70 m above the highest ground under it") {
    for (const double voxel : { 0.5, 0.05 }) {
        const BiomeLibrary lib(voxel);
        // A ground rising to the east: the island clears its highest point.
        auto top_y = [](int64_t p_x) { return 500 + p_x / 8; };
        for (uint32_t seed = 0; seed < 8; seed++) {
            Biome::Site site;
            site.feature = Biome::FLOATING_ISLAND;
            site.at = VoxelPos(1000 + 37 * int64_t(seed), 0, 0);
            site.radius = (15.0 + 3.0 * seed) / voxel;
            site.seed = seed * 7919u;
            Biome::Placed out;
            CHECK(Biome::place(site, column_ground(top_y), lib.lib, lib.types, voxel, out));
            std::vector<TableRecord> &records = out.records;
            CHECK(records.size() == 1);
            if (records.size() != 1) {
                continue;
            }
            const TableRecord &rec = records[0];
            const double bottom = double(rec.anchor.y) - double(rec.params[1]); // the apex, without the bumps
            const int64_t top = top_y(site.at.x);
            CAPTURE(voxel);
            CAPTURE(seed);
            // Above the ground under its centre by the altitude at least, and at most the altitude plus the rise of
            // the ground across the island (1/8 per voxel).
            CHECK(bottom - double(top) >= 25.0 / voxel);
            CHECK(bottom - double(top) <= 70.0 / voxel + site.radius);
            std::vector<TableRecord> folded = records;
            // Inside the island, half its underside's depth below the rim plane: solid; half way down to the ground: air.
            CHECK(field(lib, folded, VoxelPos(site.at.x, rec.anchor.y - std::llround(0.5 * rec.params[1]), 0), top) < 0.0f);
            CHECK(field(lib, folded, VoxelPos(site.at.x, (top + std::llround(bottom)) / 2 - 1, 0), top) > 0.0f);
        }
    }
}
