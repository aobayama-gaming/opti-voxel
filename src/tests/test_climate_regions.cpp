#include "test_climate_common.hpp"

#include <cmath>
#include <memory>

// Regions (Climate::Region): several landforms in one table, each behind a part of the fields' space, its height terms
// gated by it. project/climate/region_test.json holds valley.json's fields, stretch, height term and biomes behind a
// 'cold' region and the cliff's and the mangrove's terms behind a 'warm' one: 9 fields and the multifractal state, past
// the old cap of 8 field groups.

using namespace Climate;
using ClimateTest::ClimateSDF;
using ClimateTest::lattice_pos;
using ExactNoise::float_bits;

namespace {

// Region p_region's value at lattice point p_p of Λ_m (the fields carried there).
Float4 region_at(const Table &p_table, const Cascade::Spec &p_spec, int p_region, const Vector3i64 &p_p, int p_m) {
    Float4 f[MAX_FIELDS];
    fields_at(p_table, p_spec, p_p, p_m, f);
    return edges_value(p_table.regions[size_t(p_region)].edges, f);
}

// Whether p_r is the constant p_v: value p_v, gradient 0.
bool is_constant(const Float4 &p_r, float p_v) {
    return p_r.w == p_v && p_r.x == 0.0f && p_r.y == 0.0f && p_r.z == 0.0f;
}

// A point near the datum, spread over 64 km (5 cm voxels: several periods of the 13 km temperature axis), on Λ_m.
Vector3i64 sample_point(const SDFCascade &p_sdf, int p_i, int p_m) {
    const int64_t x = int64_t(p_i % 97) * 13219 - 640000;
    const int64_t z = int64_t(p_i / 97) * 12893 - 640000;
    const int64_t y = p_sdf.get_datum_y() + int64_t(p_i % 13) * 997 - 6000;
    return Vector3i64((x * 7) >> p_m, y >> p_m, (z * 7) >> p_m);
}

} // namespace

TEST_CASE("Climate regions: a table of two landforms parses into more field groups than the old cap") {
    ClimateSDF r("res://climate/region_test.json", 15);
    CHECK(r.sdf.get_climate_error() == String());
    CLIMATE_REQUIRE(r.sdf.get_climate() != nullptr);
    const Table &t = *r.sdf.get_climate();
    CHECK(t.fields.size() == 9);
    CHECK(t.regions.size() == 2);
    CHECK(r.sdf.get_spec().group_count() == 11); // the ground, 9 fields, the multifractal state
    // A biome's region edges come before its own.
    CHECK(t.biomes[0].edges.size() == 2);
    CHECK(t.biomes[0].edges[0].field == t.regions[0].edges[0].field);
    CHECK(t.biomes[0].edges[0].anchor == t.regions[0].edges[0].anchor);
    CHECK(t.heights[0].region == 0);
    CHECK(t.heights[1].region == 1);
    int64_t bad = 0;
    for (const int lod : { 0, 4, 9 }) {
        for (const int64_t base : { int64_t(11), (int64_t(1) << 32) - 2, -(int64_t(1) << 40) }) {
            bad += ClimateTest::box_point_mismatches(r.sdf.get_spec(), Vector3i64(base, (r.sdf.get_datum_y() >> lod) - 1, base / 5), 3, lod, true);
        }
    }
    CHECK(bad == 0);
}

// In the cold region's core every edge is exactly 1: the valley's gated term is its ungated one, the cliff's and the
// mangrove's add exactly 0, the warm biomes weigh exactly 0. The SDF and the pick are valley.json's, bit for bit. The
// core (Climate::core_regions) is deeper than the region's edge: the bands read the weights derived at the cascade's
// coarser lattices (km apart for a 13 km axis), whose temperature can be in the fade while the point's is not.
TEST_CASE("Climate regions: a region's core is its standalone landform, bit for bit") {
    ClimateSDF world("res://climate/region_test.json", 15);
    ClimateSDF valley("res://climate/landforms/valley.json", 15);
    CLIMATE_REQUIRE(world.sdf.get_climate() != nullptr);
    CLIMATE_REQUIRE(valley.sdf.get_climate() != nullptr);
    const Table &wt = *world.sdf.get_climate();
    const Table &vt = *valley.sdf.get_climate();
    int core = 0, differ = 0, picks = 0;
    for (const int lod : { 0, 3, 7 }) {
        for (int i = 0; i < 97 * 40; i++) {
            const Vector3i64 p = sample_point(world.sdf, i, lod);
            Float4 f[MAX_FIELDS];
            fields_at(wt, world.sdf.get_spec(), p, lod, f);
            if (!is_constant(edges_value(wt.regions[0].edges, f), 1.0f) || (core_regions(wt, world.sdf.get_spec(), p, lod) & 1u) == 0u) {
                continue;
            }
            core++;
            differ += float_bits(world.sdf.evaluate(lattice_pos(p, lod), lod)) != float_bits(valley.sdf.evaluate(lattice_pos(p, lod), lod));
            float ws[Cascade::SLOTS], vs[Cascade::SLOTS];
            Cascade::point_slots(world.sdf.get_spec(), p, lod, ws, true);
            Cascade::point_slots(valley.sdf.get_spec(), p, lod, vs, true);
            // The world's biomes: floor, plateau, lowland, upland; the valley's: floor, upland.
            const int w = pick_slots(wt, ws);
            const int v = pick_slots(vt, vs);
            picks += (w == 0 && v == 0) || (w == 3 && v == 1);
        }
    }
    MESSAGE("cold core: ", core, " points, ", differ, " SDF values differ, ", picks, " picks agree");
    CHECK(core > 1000);
    CHECK(differ == 0);
    CHECK(picks == core);
}

// Outside a region's fades its gated terms add exactly 0: the warm core's ground is the table's without the valley's
// term.
TEST_CASE("Climate regions: a gated term is absent outside its region") {
    ClimateSDF world("res://climate/region_test.json", 15);
    CLIMATE_REQUIRE(world.sdf.get_climate() != nullptr);
    Dictionary data;
    CHECK(load("res://climate/region_test.json", data) == String());
    Array heights = Array(data["heights"]).duplicate();
    heights.remove_at(0);
    data["heights"] = heights;
    ClimateSDF without(data, 15);
    CLIMATE_REQUIRE(without.sdf.get_climate() != nullptr);
    const Table &wt = *world.sdf.get_climate();
    int outside = 0, differ = 0;
    for (const int lod : { 0, 5 }) {
        for (int i = 0; i < 97 * 20; i++) {
            const Vector3i64 p = sample_point(world.sdf, i, lod);
            if (!is_constant(region_at(wt, world.sdf.get_spec(), 0, p, lod), 0.0f)) {
                continue;
            }
            outside++;
            differ += float_bits(world.sdf.evaluate(lattice_pos(p, lod), lod)) != float_bits(without.sdf.evaluate(lattice_pos(p, lod), lod));
        }
    }
    CHECK(outside > 500);
    CHECK(differ == 0);
}

// The skirts cover the step a gated term makes between two LODs where its region's own field changes: at LODs 15 and
// 16 the temperature axis (13 km, octave 18 at 5 cm) loses its octave, so the regions jump with the valley's 2.1 km
// trough behind them. Without the gate's share in ground_step the bound is the bare term's.
TEST_CASE("Climate regions: ground_step bounds a gated term's step between LODs") {
    ClimateSDF world("res://climate/region_test.json", 20);
    CLIMATE_REQUIRE(world.sdf.get_climate() != nullptr);
    const Table &t = *world.sdf.get_climate();
    const Cascade::Spec &spec = world.sdf.get_spec();
    Table ungated = t;
    for (HeightTerm &h : ungated.heights) {
        h.region = -1;
    }
    for (const int lod : { 14, 15, 16 }) {
        const double amp = world.sdf.octave_step(lod);
        const double bound = ground_step(t, lod, amp);
        double worst = 0.0;
        for (int i = 0; i < 97 * 30; i++) {
            const Vector3i64 p = sample_point(world.sdf, i, lod + 1);
            float fine[Cascade::SLOTS], coarse[Cascade::SLOTS];
            Cascade::point_slots(spec, p * int64_t(2), lod, fine, true);
            Cascade::point_slots(spec, p, lod + 1, coarse, true);
            worst = std::max(worst, std::fabs(double(ground_value(t, fine).w) - double(ground_value(t, coarse).w)));
        }
        MESSAGE("LOD ", lod, " | ", lod + 1, ": worst step ", worst, " voxels, bound ", bound, ", ungated bound ", ground_step(ungated, lod, amp));
        CHECK(worst <= 2.0 * bound); // the skirts take twice the step (SDFCascade::configure)
        CHECK(bound >= ground_step(ungated, lod, amp));
    }
    CHECK(ground_step(t, 15, 0.0) > ground_step(ungated, 15, 0.0));
}

// The landform world (project/climate/world.json, tools/landforms/world.py): every landform's region core
// (Climate::core_regions) is its own table, its SDF and its pick (by name). Points over 800 km at LODs 0 and 6.
TEST_CASE("Climate regions: the world's landform cores are their own tables") {
    ClimateSDF world("res://climate/world.json", 25);
    CLIMATE_REQUIRE(world.sdf.get_climate() != nullptr);
    const Table &wt = *world.sdf.get_climate();
    CLIMATE_REQUIRE(wt.regions.size() == 8);
    std::vector<std::unique_ptr<ClimateSDF>> own;
    for (const Region &r : wt.regions) {
        own.push_back(std::make_unique<ClimateSDF>(("res://climate/landforms/" + r.name + ".json").utf8().get_data(), 25));
        CLIMATE_REQUIRE(own.back()->sdf.get_climate() != nullptr);
    }
    int core[8] = {}, differ[8] = {}, picks[8] = {};
    for (const int lod : { 0, 6 }) {
        for (int i = 0; i < 80 * 80; i++) {
            const int64_t x = int64_t(i % 80) * 200003 - 8000000;
            const int64_t z = int64_t(i / 80) * 199999 - 8000000;
            const int64_t y = world.sdf.get_datum_y() + int64_t(i % 17) * 1999 - 16000;
            const Vector3i64 p(x >> lod, y >> lod, z >> lod);
            float ws[Cascade::SLOTS];
            Cascade::point_slots(world.sdf.get_spec(), p, lod, ws, true);
            Float4 f[MAX_FIELDS];
            apply_terms(wt, ws, f);
            // A region exactly 1 at the point first (the cheap test), then its core.
            const bool any = std::any_of(wt.regions.begin(), wt.regions.end(), [&](const Region &p_r) { return is_constant(edges_value(p_r.edges, f), 1.0f); });
            const uint32_t cores = any ? core_regions(wt, world.sdf.get_spec(), p, lod) : 0u;
            for (int r = 0; r < 8; r++) {
                if (((cores >> r) & 1u) == 0u) {
                    continue;
                }
                const SDFCascade &o = own[size_t(r)]->sdf;
                core[r]++;
                differ[r] += float_bits(world.sdf.evaluate(lattice_pos(p, lod), lod)) != float_bits(o.evaluate(lattice_pos(p, lod), lod));
                float os[Cascade::SLOTS];
                Cascade::point_slots(o.get_spec(), p, lod, os, true);
                picks[r] += wt.biomes[size_t(pick_slots(wt, ws))].name == o.get_climate()->biomes[size_t(pick_slots(*o.get_climate(), os))].name;
            }
        }
    }
    for (int r = 0; r < 8; r++) {
        MESSAGE(std::string(wt.regions[size_t(r)].name.utf8().get_data()), " core: ", core[r], " points, ", differ[r], " SDF values differ, ", picks[r],
                " picks agree");
        CHECK(core[r] >= 10);
        CHECK(differ[r] == 0);
        CHECK(picks[r] == core[r]);
    }
}
