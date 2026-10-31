#include "test_climate_common.hpp"

#include "cloud_climate.hpp"
#include "clouds.hpp"

#include "ov/clock.hpp"
#include "ov/half.hpp"

#include <algorithm>
#include <cmath>

// The clouds' climate (CloudClimate): biomes' "clouds" parsed, blended by their weights, and the map round the camera.

using ClimateTest::ClimateSDF;
using ClimateTest::json;

namespace {

// The example climate with "clouds" p_clouds (JSON text) on its first biome.
ov::Value example_with_clouds(const char *p_clouds) {
    ov::Value data;
    CHECK(Climate::load("res://climate/example.json", data) == ov::str());
    data["biomes"].at(0)["clouds"] = json(p_clouds);
    return data;
}

} // namespace

TEST_CASE("ov::float_to_half: the nearest half, back through half_to_float") {
    for (const float v : { 0.0f, 1.0f, 0.5f, 0.25f, 0.3f, 0.65f, 0.999f, 2.0f, 1e-6f }) {
        const float back = ov::half_to_float(ov::float_to_half(v));
        CHECK(std::abs(back - v) <= std::ldexp(std::abs(v), -11) + 6.2e-5f); // half an ulp; 0 under the normals
    }
    CHECK(ov::float_to_half(1.0f) == 0x3C00u);
    CHECK(ov::float_to_half(-2.0f) == 0xC000u);
    CHECK(ov::float_to_half(1e6f) == 0x7C00u);
}

TEST_CASE("Climate: a biome's clouds parse, out of range or unknown keys are refused") {
    {
        ClimateSDF ok(example_with_clouds(R"({"cover": 0.7, "type": 0.25})"), 15);
        CHECK_OR_RETURN(ok.sdf.get_climate() != nullptr);
        const Climate::BiomeClouds &c = ok.sdf.get_climate()->biomes[0].clouds;
        CHECK(c.set);
        CHECK(c.cover == 0.7f);
        CHECK(c.type == 0.25f);
        CHECK_FALSE(ok.sdf.get_climate()->biomes[1].clouds.set);
        CHECK(CloudClimate::any(*ok.sdf.get_climate()));
    }
    for (const char *bad : { R"({"cover": 1.5, "type": 0.5})", R"({"cover": 0.5})", R"({"cover": 0.5, "type": 0.5, "height": 1})" }) {
        ClimateSDF refused(example_with_clouds(bad), 15);
        CHECK(refused.sdf.get_climate() == nullptr);
        CHECK_FALSE(refused.sdf.get_climate_error().empty());
    }
    ClimateSDF none("res://climate/example.json", 15);
    CHECK_OR_RETURN(none.sdf.get_climate() != nullptr);
    CHECK_FALSE(CloudClimate::any(*none.sdf.get_climate()));
}

TEST_CASE("CloudClimate::of_weights: linear in the weights, over the biomes with clouds only") {
    Climate::Table t;
    t.biomes.resize(3);
    t.biomes[0].clouds = Climate::BiomeClouds{ true, 0.8f, 1.0f };
    t.biomes[2].clouds = Climate::BiomeClouds{ true, 0.1f, 0.0f };
    Climate::Float4 w[Climate::MAX_BIOMES] = {};
    w[0].w = 0.25f;
    w[1].w = 0.5f;
    w[2].w = 0.25f;
    const CloudClimate::Texel x = CloudClimate::of_weights(t, w);
    CHECK(x.cover == doctest::Approx(0.25 * 0.8 + 0.25 * 0.1));
    CHECK(x.type == doctest::Approx(0.25));
    CHECK(x.weight == doctest::Approx(0.5));
}

TEST_CASE("CloudClimate::Map: cells round the layer only, one sample per base point, reused when it moves") {
    const BaseSurface flat = BaseSurface::flat(0);
    const CloudLayer layer = CloudLayer::for_base(INFINITY, 3300.0, 0.05);
    CloudClimate::Map map(flat, layer.climate_cell_shift, double(layer.bottom_m), double(layer.top_m), 0.05);
    const int64_t cell = int64_t(1) << layer.climate_cell_shift;
    int samples = 0;
    // Distinct per column: the base point's x and z.
    const CloudClimate::Map::SampleFn sample = [&](const VoxelPos &p_v, int p_m) {
        samples++;
        CHECK(p_m == layer.climate_cell_shift);
        CHECK(p_v.y == cell / 2); // the centre of the base cell over the datum
        return CloudClimate::Texel{ float(std::abs(WorldCoord::floor_div(p_v.x, cell)) % 7) / 8.0f, 0.5f, 1.0f };
    };
    const Vector3i64 centre = map.cell_of(VoxelPos(0, 400, 0)); // a camera 20 m up
    float max_cover = 0.0f;
    const std::vector<uint16_t> texels = map.build(centre, sample, max_cover);
    const int n = CloudClimate::Map::SIZE;
    CHECK_OR_RETURN(texels.size() == size_t(n) * n * n * 4);
    CHECK(samples == n * n); // one per column
    CHECK(max_cover == doctest::Approx(6.0 / 8.0));
    // The evaluated layers: the cells within MARGIN cells of the layer, the same in every column.
    const VoxelPos corner = map.corner(centre);
    int layers = 0;
    for (int j = 0; j < n; j++) {
        const double altitude = double(corner.y + j * cell + cell / 2) * 0.05;
        const bool in = altitude >= double(layer.bottom_m) - CloudClimate::Map::MARGIN * cell * 0.05 &&
                altitude <= double(layer.top_m) + CloudClimate::Map::MARGIN * cell * 0.05;
        const uint16_t weight = texels[((size_t(5) * n + j) * n + 9) * 4 + 2];
        CHECK((weight == ov::float_to_half(1.0f)) == in);
        layers += in;
    }
    CHECK(layers >= 3);
    // A camera a few cells away keeps the map; one SIZE / 8 + 1 away rebuilds it, sampling the new columns only.
    CHECK(CloudClimate::Map::covers(centre, centre + Vector3i64(n / 8, 0, -n / 8)));
    CHECK_FALSE(CloudClimate::Map::covers(centre, centre + Vector3i64(n / 8 + 1, 0, 0)));
    samples = 0;
    const Vector3i64 moved = centre + Vector3i64(n / 8 + 1, 0, 0);
    const std::vector<uint16_t> again = map.build(moved, sample, max_cover);
    CHECK(samples == (n / 8 + 1) * n);
    // The same absolute cell holds the same texel in both maps.
    const size_t a = ((size_t(7) * n + 34) * n + 20) * 4;
    const size_t b = ((size_t(7) * n + 34) * n + 20 - (n / 8 + 1)) * 4;
    CHECK(texels[a] == again[b]);
}

TEST_CASE("CloudClimate::Stream: the first build lands, a camera within SIZE / 8 cells keeps it, one past rebuilds round it") {
    const CloudLayer layer = CloudLayer::for_base(INFINITY, 3300.0, 0.05);
    const CloudClimate::Map map(BaseSurface::flat(0), layer.climate_cell_shift, double(layer.bottom_m), double(layer.top_m), 0.05);
    CloudClimate::Stream stream(map, [](const VoxelPos &, int) { return CloudClimate::Texel{ 0.5f, 1.0f, 1.0f }; });
    const int64_t cell = int64_t(1) << layer.climate_cell_shift;
    const int n = CloudClimate::Map::SIZE;
    std::vector<uint16_t> texels;
    CHECK_FALSE(stream.update(VoxelPos(0, 400, 0), texels)); // starts the first build
    CHECK_FALSE(stream.live());
    stream.finish();
    CHECK(stream.update(VoxelPos(0, 400, 0), texels));
    CHECK(stream.live());
    CHECK(texels.size() == size_t(n) * n * n * 4);
    CHECK(stream.max_cover() == 0.5f);
    CHECK(stream.corner() == VoxelPos(-n / 2 * cell, -n / 2 * cell, -n / 2 * cell));
    CHECK(stream.extent_vox() == double(n * cell));
    // n / 8 cells over: the same map, no build.
    stream.finish();
    CHECK_FALSE(stream.update(VoxelPos(n / 8 * cell, 400, 0), texels));
    // One more: a build round the new cell, live once collected.
    const VoxelPos far((n / 8 + 1) * cell, 400, 0);
    CHECK_FALSE(stream.update(far, texels));
    stream.finish();
    CHECK(stream.update(far, texels));
    CHECK(stream.corner() == VoxelPos((n / 8 + 1 - n / 2) * cell, -n / 2 * cell, -n / 2 * cell));
    // 1000 km up: centred on the base under it (the layer stays in the map), so this map still holds it.
    const VoxelPos high((n / 8 + 1) * cell, 20000000000, 0);
    CHECK(map.centre_for(high) == map.cell_of(VoxelPos((n / 8 + 1) * cell, 0, 0)));
    CHECK_FALSE(stream.update(high, texels));
    CHECK(stream.live());
}

TEST_CASE("CloudClimate::at on the world climate: every biome sets its clouds, the cover varies by region") {
    ClimateSDF world("res://climate/world.json", 25);
    const Climate::Table *t = world.sdf.get_climate();
    CHECK_OR_RETURN(t != nullptr);
    for (const Climate::Biome &b : t->biomes) {
        CHECK_MESSAGE(b.clouds.set, b.name);
    }
    const CloudLayer layer = CloudLayer::for_base(INFINITY, 3300.0, 0.05);
    const int64_t datum = world.sdf.get_datum_y();
    // A 2000 km line: several periods of the 419 km axes.
    float lo = 1.0f;
    float hi = 0.0f;
    const int n = 200;
    const uint64_t t0 = ov::now_usec();
    for (int i = 0; i < n; i++) {
        const VoxelPos v(int64_t(i) * 200000, datum, int64_t(i) * 70000 - 7000000);
        const CloudClimate::Texel x = CloudClimate::at(*t, world.sdf.get_spec(), v, layer.climate_cell_shift);
        CHECK(x.weight == doctest::Approx(1.0).epsilon(1e-3));
        CHECK(x.cover >= 0.0f);
        CHECK(x.cover <= 0.8f + 1e-4f);
        lo = std::min(lo, x.cover);
        hi = std::max(hi, x.cover);
    }
    const double us = double(ov::now_usec() - t0) / n;
    MESSAGE("CloudClimate::at: " << us << " us per point; cover " << lo << " .. " << hi);
    CHECK(lo < 0.2f); // a desert somewhere
    CHECK(hi > 0.55f); // and a cloudy region
}
