#include "test_climate_common.hpp"

#include "cascade_glsl.hpp"
#include "climate_chunk_modes.hpp"

// The region guards (Climate::guarded_regions: the GPU's climate code skips a region's biomes and gated terms where the
// region is off) and the chunk modes the direct mesher learns for them (ClimateChunkModes).

using namespace Climate;
using ClimateTest::ClimateSDF;

namespace {

// cl_edge_off on the CPU: whether some edge of the region returns +0.
bool region_off(const Table &p_table, int p_region, const Float4 p_fields[MAX_FIELDS]) {
    for (const Edge &e : p_table.regions[size_t(p_region)].edges) {
        if (!(Cascade::flush_value((p_fields[e.field].w - e.anchor) * e.scale) > 0.0f)) {
            return true;
        }
    }
    return false;
}

// glsl_biome_weights' op order with the guards, on the CPU: a biome whose region is off is +0, with no rest update.
void guarded_weights(const Table &p_table, const Float4 p_fields[MAX_FIELDS], Float4 r_w[MAX_BIOMES]) {
    Float4 rest{ 0.0f, 0.0f, 0.0f, 1.0f };
    const size_t n = p_table.biomes.size();
    for (size_t b = 0; b + 1 < n; b++) {
        const int r = p_table.biomes[b].region;
        if (r >= 0 && region_off(p_table, r, p_fields)) {
            r_w[b] = Float4{ 0.0f, 0.0f, 0.0f, 0.0f };
            continue;
        }
        r_w[b] = Cascade::dmul(rest, edges_value(p_table.biomes[b].edges, p_fields));
        rest = Cascade::dsub(rest, r_w[b]);
    }
    r_w[n - 1] = rest;
}

// The guarded regions on at lattice point p_p of Λ_m (region_on_glsl's mask).
uint32_t regions_on(const Table &p_table, const Cascade::Spec &p_spec, const Vector3i64 &p_p, int p_m, Float4 r_fields[MAX_FIELDS]) {
    fields_at(p_table, p_spec, p_p, p_m, r_fields);
    uint32_t on = 0u;
    for (int r = 0; r < int(p_table.regions.size()); r++) {
        on |= (guarded_regions(p_table) & (1u << r)) != 0u && !region_off(p_table, r, r_fields) ? 1u << r : 0u;
    }
    return on;
}

} // namespace

TEST_CASE("Climate guards: the world's regions are guarded, a single landform keeps its text") {
    ClimateSDF world("res://climate/world.json", 25);
    CLIMATE_REQUIRE(world.sdf.get_climate() != nullptr);
    const Table &t = *world.sdf.get_climate();
    CHECK(guarded_regions(t) == (1u << t.regions.size()) - 1u);
    const uint32_t base = base_field_groups(t);
    CHECK((base & 1u) != 0u); // the ground
    for (int r = 0; r < int(t.regions.size()); r++) {
        for (const Edge &e : t.regions[size_t(r)].edges) {
            CHECK((base & (2u << e.field)) != 0u); // every region's axes: the guards read them
        }
    }
    uint32_t any = base;
    for (int r = 0; r < int(t.regions.size()); r++) {
        any |= region_field_groups(t, r);
    }
    CHECK(base != any); // some fields only a region reads

    ClimateSDF cliff("res://climate/landforms/cliff.json", 25);
    CLIMATE_REQUIRE(cliff.sdf.get_climate() != nullptr);
    CHECK(guarded_regions(*cliff.sdf.get_climate()) == 0u);
    const ov::str text = CascadeGLSL::level_shader(&cliff.sdf, false, false, false, true, CascadeGLSL::View::FIELDS);
    CHECK(!ov::contains(text, "climate_items"));
    CHECK(!ov::contains(text, "cl_edge_off"));
    // The world: guards in every climate shader, the climate words in the chunk items' only.
    const ov::str wtext = CascadeGLSL::level_shader(&world.sdf, false, false, false, true, CascadeGLSL::View::FIELDS);
    CHECK(ov::contains(wtext, "climate_items"));
    CHECK(ov::contains(wtext, "cl_edge_off"));
    const ov::str ptext = CascadeGLSL::level_shader(&world.sdf, false, true, false, false, CascadeGLSL::View::FIELDS);
    CHECK(!ov::contains(ptext, "climate_items"));
    CHECK(ov::contains(ptext, "cl_edge_off"));
}

TEST_CASE("Climate guards: where a region is off, its biomes and gated terms change nothing, bit for bit") {
    for (const char *path : { "res://climate/world.json", "res://climate/world_video.json", "res://climate/region_test.json" }) {
        ClimateSDF world(path, 25);
        CLIMATE_REQUIRE(world.sdf.get_climate() != nullptr);
        const Table &t = *world.sdf.get_climate();
        int64_t skipped = 0, differ = 0;
        for (int i = 0; i < 1000; i++) {
            // Spread over ~3,000 km at 5 cm (the axes' octaves are 105 - 420 km) on Λ_10.
            const int m = 10;
            const Vector3i64 p(int64_t(i % 31) * 1931 - 30000, (world.sdf.get_datum_y() >> m) + (i % 7) - 3, int64_t(i / 31) * 1873 - 30000);
            Float4 f[MAX_FIELDS];
            const uint32_t on = regions_on(t, world.sdf.get_spec(), p, m, f);
            skipped += __builtin_popcount(guarded_regions(t) & ~on);
            Float4 full[MAX_BIOMES], w[MAX_BIOMES];
            weights(t, f, full);
            guarded_weights(t, f, w);
            for (size_t b = 0; b < t.biomes.size(); b++) {
                differ += same_bits(w[b], full[b]) ? 0 : 1;
            }
            Float4 regions[MAX_REGIONS] = {}, guarded[MAX_REGIONS] = {};
            for (const int r : t.gated) {
                regions[r] = edges_value(t.regions[size_t(r)].edges, f);
                guarded[r] = (on & (1u << r)) != 0u ? regions[r] : Float4{ 0.0f, 0.0f, 0.0f, 0.0f };
            }
            differ += same_bits(height_sum(t, f, guarded), height_sum(t, f, regions)) ? 0 : 1;
        }
        MESSAGE(std::string(path) << ": regions skipped over 1000 points: " << skipped);
        CHECK(skipped > 500);
        CHECK(differ == 0);
    }
}

// The video scene's spawn (world_video.json) lies in the mangrove | desert fade for kilometres: two regions on, six off.
TEST_CASE("Climate guards: around the video spawn, two regions are on and six skipped") {
    ClimateSDF world("res://climate/world_video.json", 25);
    CLIMATE_REQUIRE(world.sdf.get_climate() != nullptr);
    const Table &t = *world.sdf.get_climate();
    const int mangrove = t.region_index("mangrove"), desert = t.region_index("desert");
    const bool named = mangrove >= 0 && desert >= 0;
    CLIMATE_REQUIRE(named);
    for (int i = 0; i < 400; i++) {
        const int m = 6; // 3.2 m at 5 cm: 400 points over ~2 km
        const Vector3i64 p(int64_t(i % 20) * 31 - 300, (world.sdf.get_datum_y() >> m) + (i % 5) - 2, int64_t(i / 20) * 29 - 300);
        Float4 f[MAX_FIELDS];
        CHECK(regions_on(t, world.sdf.get_spec(), p, m, f) == ((1u << mangrove) | (1u << desert)));
    }
}

TEST_CASE("Climate guards: a biome's region must start its edges") {
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

TEST_CASE("ClimateChunkModes: a reduced chunk takes the regions its nearest reported ancestor had on") {
    ClimateChunkModes h;
    h.configure(true, true);
    const ChunkId fine(1, 1, 1); // LOD 0
    const ChunkId parent = ChunkMath::get_parent_from_child(fine);
    std::vector<uint32_t> modes;
    h.modes({ fine }, modes);
    CHECK(modes[0] == Cascade::CLIMATE_FULL);

    ov::bytes words(CascadeGLSL::CLIMATE_ITEM_BYTES, 0);
    ov::put_u32(words.data() + 4, 0b0110u); // regions 1 and 2 on
    h.learn({ parent }, { Cascade::CLIMATE_FULL }, words);
    h.modes({ fine }, modes);
    CHECK(modes[0] == (Cascade::CLIMATE_REDUCE | 0b0110u));

    // A reduced chunk with another region on is counted, and its own report then rules.
    ov::bytes failed(CascadeGLSL::CLIMATE_ITEM_BYTES, 0);
    ov::put_u32(failed.data() + 4, 0b1110u);
    ov::put_u32(failed.data() + 8, 1u);
    h.learn({ fine }, { Cascade::CLIMATE_REDUCE | 0b0110u }, failed);
    CHECK(h.get_stats().failed == 1);
    h.modes({ fine }, modes);
    CHECK(modes[0] == (Cascade::CLIMATE_REDUCE | 0b1110u));

    // Without reduce, every chunk is full; without guards, no words at all.
    h.configure(true, false);
    h.learn({ parent }, { Cascade::CLIMATE_FULL }, words);
    h.modes({ fine }, modes);
    CHECK(modes[0] == Cascade::CLIMATE_FULL);
    h.configure(false, true);
    h.modes({ fine }, modes);
    CHECK(modes.empty());
}
