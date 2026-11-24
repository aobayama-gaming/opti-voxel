#include "test_common.hpp"

#include "clouds.hpp"

#include <cmath>
#include <cstring>

namespace {

bool power_of_two(int p_v) {
    return p_v > 0 && (p_v & (p_v - 1)) == 0;
}

} // namespace

TEST_CASE("CloudLayer::of_weather: the ground to the levels' top, Earth's optical depth, the noises the cells'") {
    const CloudLayer earth;
    const CloudLayer ocean = CloudLayer::of_weather(200.0, 45.0, 0.05);
    CHECK(ocean.bottom_m == 0.0f);
    CHECK(ocean.top_m == 200.0f);
    CHECK(double(ocean.extinction) * 200.0 == doctest::Approx(double(earth.extinction) * double(earth.top_m - earth.bottom_m)).epsilon(1e-5));
    CHECK(double(ocean.shape_period_vox) * 0.05 <= CloudLayer::SHAPE_CELLS * 45.0);
    CHECK(double(ocean.shape_period_vox) * 0.05 > 0.5 * CloudLayer::SHAPE_CELLS * 45.0);
    // The noises' periods: powers of two of voxels, the warp a multiple of the others (the wind's offset wraps at it)
    // and longer than the weather (the field repeats at the warp's).
    const CloudLayer fine = CloudLayer::of_weather(200.0, 45.0, 0.001); // the frame's longest period clamps them all
    const CloudLayer earthlike = CloudLayer::of_weather(4000.0, 20000.0, 0.5);
    for (const CloudLayer &l : { ocean, fine, earthlike }) {
        CHECK(power_of_two(l.shape_period_vox));
        CHECK(power_of_two(l.detail_period_vox));
        CHECK(power_of_two(l.weather_period_vox));
        CHECK(power_of_two(l.warp_period_vox));
        CHECK(l.warp_period_vox >= 4 * l.weather_period_vox);
        CHECK(l.weather_period_vox % l.shape_period_vox == 0);
        CHECK(l.shape_period_vox % l.detail_period_vox == 0);
    }
}

TEST_CASE("Clouds: every constant of the passes' and the hosts' GLSL substituted") {
    const CubeSphereGrid grid(16, 2000.0);
    for (const ov::str &glsl : { Clouds::glsl_pass_noise(), Clouds::glsl_pass_march(grid, 4), Clouds::glsl_pass_shadow(grid, 4), Clouds::glsl_lookup() }) {
        CHECK(glsl.find('$') == ov::str::npos);
    }
    CHECK(Clouds::glsl_pass_march(grid, 4).find("vec3 cloud_camera_m;") != ov::str::npos);
    CHECK(Clouds::glsl_pass_shadow(grid, 4).find("float cloud_shadow_extent_m;") != ov::str::npos);
}

TEST_CASE("Clouds::block_bytes: std140, the base first, padded to 16") {
    BaseSurface::Render base;
    base.kind = BaseSurface::Kind::SPHERE;
    base.point_m = Float3{ 1.0f, 2.0f, 3.0f };
    const ov::bytes b = Clouds::block_bytes(base, Atmosphere::HostConstants(), 7.0f, CloudLayer(), Clouds::Frame());
    CHECK(b.size() % 16 == 0);
    int kind;
    std::memcpy(&kind, b.data(), 4);
    CHECK(kind == int(BaseSurface::Kind::SPHERE));
    float x;
    std::memcpy(&x, b.data() + 16, 4);
    CHECK(x == 1.0f);
}

TEST_CASE("Clouds::make_frame: the shadow map's axes perpendicular to the sun, its origin on its texels, the camera inside; the reach grows with altitude") {
    const CloudLayer layer;
    BaseSurface::Render flat;
    flat.point_m = Float3{ 0.0f, -20.0f, 0.0f }; // the camera 20 m above the datum
    const float k = 1.0f / std::sqrt(3.0f);
    const Float3 sun{ k, k, k };
    const Float3 camera{ 1234.5f, 0.0f, -987.25f };
    const Clouds::Frame f = Clouds::make_frame(layer, AtmosphereMedium::EARTH_RADIUS_M, camera, sun, flat);
    CHECK(std::abs(dot(f.shadow_u, sun)) < 1e-6f);
    CHECK(std::abs(dot(f.shadow_v, sun)) < 1e-6f);
    CHECK(std::abs(dot(f.shadow_u, f.shadow_v)) < 1e-6f);
    CHECK(double(dot(f.shadow_u, f.shadow_u)) == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(double(dot(f.shadow_v, f.shadow_v)) == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(f.shadow_extent_m == layer.shadow_min_extent_m);
    const double texel = double(f.shadow_extent_m) / Clouds::SHADOW_SIZE;
    const Float3 d = camera - f.shadow_origin_m;
    for (const Float3 &axis : { f.shadow_u, f.shadow_v }) {
        const double along = double(dot(d, axis));
        CHECK(along > 0.0);
        CHECK(along < double(f.shadow_extent_m));
        const double corner = double(dot(f.shadow_origin_m, axis)) / texel;
        CHECK(std::abs(corner - std::round(corner)) < 1e-3);
    }
    // From 100 km up: a longer reach, a shadow map of 4 altitudes.
    flat.point_m.y = -100000.0f;
    const Clouds::Frame high = Clouds::make_frame(layer, AtmosphereMedium::EARTH_RADIUS_M, Float3(), sun, flat);
    CHECK(high.reach_m > f.reach_m);
    CHECK(double(high.shadow_extent_m) == doctest::Approx(400000.0).epsilon(1e-5));
}
