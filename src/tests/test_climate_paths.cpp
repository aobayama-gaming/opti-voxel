#include "test_climate_common.hpp"

#include "cascade_glsl.hpp"
#include "climate_path_hints.hpp"

// The region paths (Climate::RegionPath, the GPU's climate code per region) and the modes the meshers learn for them
// (ClimatePathHints).

using namespace Climate;
using ClimateTest::ClimateSDF;
using ExactNoise::float_bits;

namespace {

// glsl_biome_weights' op order on the CPU: the live biomes' weights, the others +0 (no rest update).
void path_weights(const Table &p_table, const Float4 p_fields[MAX_FIELDS], uint32_t p_live, Float4 r_w[MAX_BIOMES]) {
    Float4 rest{ 0.0f, 0.0f, 0.0f, 1.0f };
    const size_t n = p_table.biomes.size();
    for (size_t b = 0; b + 1 < n; b++) {
        if ((p_live & (1u << b)) == 0u) {
            r_w[b] = Float4{ 0.0f, 0.0f, 0.0f, 0.0f };
            continue;
        }
        r_w[b] = Cascade::dmul(rest, edges_value(p_table.biomes[b].edges, p_fields));
        rest = Cascade::dsub(rest, r_w[b]);
    }
    r_w[n - 1] = rest;
}

// cl_edge_off on the CPU: whether some edge of the region returns +0.
bool region_off(const Table &p_table, int p_region, const Float4 p_fields[MAX_FIELDS]) {
    for (const Edge &e : p_table.regions[size_t(p_region)].edges) {
        if (!(Cascade::flush_value((p_fields[e.field].w - e.anchor) * e.scale) > 0.0f)) {
            return true;
        }
    }
    return false;
}

} // namespace

TEST_CASE("Climate paths: one per region of the world, none without two regions") {
    ClimateSDF world("res://climate/world.json", 25);
    CLIMATE_REQUIRE(world.sdf.get_climate() != nullptr);
    const Table &t = *world.sdf.get_climate();
    const std::vector<RegionPath> paths = region_paths(t);
    CHECK(paths.size() == t.regions.size());
    for (const RegionPath &p : paths) {
        CHECK((p.biomes & (1u << (t.biomes.size() - 1))) != 0u); // the default is always live
        CHECK((p.off & (1u << p.region)) == 0u);
        const uint32_t groups = path_field_groups(t, p);
        CHECK((groups & 1u) != 0u); // the ground
        CHECK(groups != (1u << (t.fields.size() + 1 + (t.has_state() ? 1 : 0))) - 1u); // fewer than every group
        for (const Edge &e : t.regions[size_t(p.region)].edges) {
            CHECK((groups & (2u << e.field)) != 0u); // its axes
        }
    }
    ClimateSDF cliff("res://climate/landforms/cliff.json", 25);
    CLIMATE_REQUIRE(cliff.sdf.get_climate() != nullptr);
    CHECK(region_paths(*cliff.sdf.get_climate()).empty());
    // No paths: the level shaders have no climate mode (the text before the paths).
    const ov::str text = CascadeGLSL::level_shader(&cliff.sdf, false, false, false, true, CascadeGLSL::View::FIELDS);
    CHECK(!ov::contains(text, "climate_items"));
    const ov::str wtext = CascadeGLSL::level_shader(&world.sdf, false, false, false, true, CascadeGLSL::View::FIELDS);
    CHECK(ov::contains(wtext, "climate_items"));
    CHECK(ov::contains(wtext, "cl_edge_off"));
}

TEST_CASE("Climate paths: where every other region is off, a path's weights and height sum are the whole table's, bit for bit") {
    ClimateSDF world("res://climate/world.json", 25);
    CLIMATE_REQUIRE(world.sdf.get_climate() != nullptr);
    const Table &t = *world.sdf.get_climate();
    const std::vector<RegionPath> paths = region_paths(t);
    int64_t held = 0, differ = 0;
    for (int i = 0; i < 4000; i++) {
        // Spread over ~3,000 km at 5 cm (the axes' octaves are 105 - 420 km) on Λ_10.
        const int m = 10;
        const Vector3i64 p(int64_t(i % 61) * 977 - 30000, (world.sdf.get_datum_y() >> m) + (i % 7) - 3, int64_t(i / 61) * 919 - 30000);
        Float4 f[MAX_FIELDS];
        fields_at(t, world.sdf.get_spec(), p, m, f);
        Float4 full[MAX_BIOMES];
        weights(t, f, full);
        Float4 regions[MAX_REGIONS] = {};
        for (const int r : t.gated) {
            regions[r] = edges_value(t.regions[size_t(r)].edges, f);
        }
        const Float4 hs = height_sum(t, f, regions);
        for (const RegionPath &path : paths) {
            bool off = true;
            for (int r = 0; r < MAX_REGIONS && off; r++) {
                off = (path.off & (1u << r)) == 0u || region_off(t, r, f);
            }
            if (!off) {
                continue;
            }
            held++;
            Float4 w[MAX_BIOMES];
            path_weights(t, f, path.biomes, w);
            for (size_t b = 0; b < t.biomes.size(); b++) {
                differ += same_bits(w[b], full[b]) ? 0 : 1;
            }
            Float4 live[MAX_REGIONS] = {};
            for (const int r : t.gated) {
                live[r] = (path.gated & (1u << r)) != 0u ? regions[r] : Float4{ 0.0f, 0.0f, 0.0f, 0.0f };
            }
            differ += same_bits(height_sum(t, f, live), hs) ? 0 : 1;
        }
    }
    MESSAGE("points where a path held: " << held);
    CHECK(held > 1000);
    CHECK(differ == 0);
}

TEST_CASE("Climate paths: a biome's region must start its edges") {
    ClimateSDF world("res://climate/world.json", 25);
    CLIMATE_REQUIRE(world.sdf.get_climate() != nullptr);
    Table t = *world.sdf.get_climate();
    Ground g;
    g.spec = &world.sdf.get_spec();
    g.voxel_size = 0.05;
    g.k_max = 1000;
    g.amp_cap = 1e30f;
    size_t b = 0;
    while (t.biomes[b].region < 0) {
        b++;
    }
    t.biomes[b].edges.erase(t.biomes[b].edges.begin()); // its region's first edge gone
    CHECK(ov::contains(validate(t, g), "start with its region"));
}

TEST_CASE("ClimatePathHints: a chunk takes its nearest reported ancestor's clean path, the whole table where none was") {
    ClimatePathHints h;
    h.configure(3, ClimatePathHints::HINTS);
    const ChunkId fine(1, 1, 1); // LOD 0
    const ChunkId parent = ChunkMath::get_parent_from_child(fine);
    const ChunkId other(201, 1, 1);
    std::vector<uint32_t> modes;
    h.modes({ fine }, modes);
    CHECK(modes[0] == Cascade::CLIMATE_UNKNOWN);

    // The parent reported paths 1 and 2 clean, `other` none.
    ov::bytes words(2 * CascadeGLSL::CLIMATE_ITEM_BYTES, 0);
    ov::put_u32(words.data() + 4, 0b110u);
    ov::put_u32(words.data() + CascadeGLSL::CLIMATE_ITEM_BYTES + 4, 0u);
    h.learn({ parent, other }, { Cascade::CLIMATE_UNKNOWN, Cascade::CLIMATE_UNKNOWN }, words);
    h.modes({ fine, other }, modes);
    CHECK(modes[0] == 2u); // path 1, the first clean one
    CHECK(modes[1] == Cascade::CLIMATE_BORDER);

    // A report of ~0 (no climate at its level 1) teaches nothing; a failed path makes the chunk a border.
    ov::bytes none(CascadeGLSL::CLIMATE_ITEM_BYTES, 0);
    ov::put_u32(none.data() + 4, ~0u);
    const ChunkId far(4001, 1, 1);
    h.learn({ far }, { Cascade::CLIMATE_UNKNOWN }, none);
    h.modes({ far }, modes);
    CHECK(modes[0] == Cascade::CLIMATE_UNKNOWN);
    ov::bytes failed(CascadeGLSL::CLIMATE_ITEM_BYTES, 0);
    ov::put_u32(failed.data() + 8, 1u);
    h.learn({ fine }, { 2u }, failed);
    h.modes({ fine }, modes);
    CHECK(modes[0] == Cascade::CLIMATE_BORDER);
    CHECK(h.get_stats().failed == 1);

    // REDUCE marks the hinted ones; OFF leaves every chunk unknown.
    h.configure(3, ClimatePathHints::REDUCE);
    h.learn({ parent }, { Cascade::CLIMATE_UNKNOWN }, words);
    h.modes({ fine }, modes);
    CHECK(modes[0] == (2u | Cascade::CLIMATE_REDUCE));
    h.configure(3, ClimatePathHints::OFF);
    h.learn({ parent }, { Cascade::CLIMATE_UNKNOWN }, words);
    h.modes({ fine }, modes);
    CHECK(modes[0] == Cascade::CLIMATE_UNKNOWN);
}
