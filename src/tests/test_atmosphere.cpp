#include "test_common.hpp"

#include "atmosphere.hpp"
#include "glsl_uniforms.hpp"

#include <cmath>
#include <cstring>

TEST_CASE("AtmosphereMedium::for_base: Earth and the flat world keep Earth's air; a small planet's is scaled, its optical depths the same") {
    const AtmosphereMedium earth;
    const AtmosphereMedium flat = AtmosphereMedium::for_base(INFINITY, 3300.0);
    const AtmosphereMedium planet = AtmosphereMedium::for_base(6371000.0, 3300.0);
    CHECK(flat.top_m == earth.top_m);
    CHECK(planet.top_m == earth.top_m);
    CHECK(planet.rayleigh.z == earth.rayleigh.z);
    // A 2 km planet: R / R_earth would put the top at 31 m, under its 62.5 m relief; RELIEF_TOPS reliefs set it.
    const AtmosphereMedium tiny = AtmosphereMedium::for_base(2000.0, 62.5);
    CHECK(double(tiny.top_m) == doctest::Approx(AtmosphereMedium::RELIEF_TOPS * 62.5).epsilon(1e-5));
    CHECK(double(tiny.rayleigh.z) * double(tiny.rayleigh_height_m) == doctest::Approx(double(earth.rayleigh.z) * double(earth.rayleigh_height_m)).epsilon(1e-5));
    CHECK(double(tiny.mie) * double(tiny.mie_height_m) == doctest::Approx(double(earth.mie) * double(earth.mie_height_m)).epsilon(1e-5));
    // An explicit scale wins.
    CHECK(double(AtmosphereMedium::for_base(2000.0, 62.5, 0.5).top_m) == doctest::Approx(0.5 * double(earth.top_m)));
}

TEST_CASE("Atmosphere::block_bytes: std140, a vec3 on 16 bytes, a scalar after it in its fourth word") {
    BaseSurface::Render base;
    base.kind = BaseSurface::Kind::SPHERE;
    base.point_m = Float3{ 1.0f, 2.0f, 3.0f };
    base.radius_m = 4.0f;
    base.k_m2 = 5.0f;
    const ov::bytes b = Atmosphere::block_bytes(base, AtmosphereMedium(), Atmosphere::Frame());
    CHECK(b.size() % 16 == 0);
    auto word = [&](size_t p_at) {
        float f;
        std::memcpy(&f, b.data() + p_at, 4);
        return f;
    };
    int kind;
    std::memcpy(&kind, b.data(), 4);
    CHECK(kind == int(BaseSurface::Kind::SPHERE));
    CHECK(word(16) == 1.0f); // base_point_m, aligned to 16
    CHECK(word(24) == 3.0f);
    CHECK(word(28) == 4.0f); // base_radius_m in the vec3's fourth word
    CHECK(word(32) == 5.0f);
    CHECK(word(48) == AtmosphereMedium().rayleigh.x); // the medium's first vec3, aligned to 16
}

TEST_CASE("Atmosphere: every constant of the passes' and the hosts' GLSL substituted") {
    for (const ov::str &glsl : { Atmosphere::glsl_pass_multiscatter(), Atmosphere::glsl_pass_volume(), Atmosphere::glsl_lookup() }) {
        CHECK(glsl.find('$') == ov::str::npos);
    }
    CHECK(Atmosphere::glsl_pass_volume().find("vec3 atmo_camera_m;") != ov::str::npos);
    CHECK(Atmosphere::glsl_medium_uniforms().find("uniform float atmo_max_path_m;") != ov::str::npos);
}

TEST_CASE("Atmosphere::make_frame: the slices reach past the camera's horizon, farther from higher up") {
    const AtmosphereMedium m;
    BaseSurface::Render flat;
    flat.point_m = Float3{ 0.0f, -10.0f, 0.0f }; // the camera 10 m above the datum
    const Atmosphere::Frame low = Atmosphere::make_frame(m, AtmosphereMedium::EARTH_RADIUS_M, Float3(), Float3{ 0.0f, 1.0f, 0.0f }, flat);
    flat.point_m.y = -10000.0f;
    const Atmosphere::Frame high = Atmosphere::make_frame(m, AtmosphereMedium::EARTH_RADIUS_M, Float3(), Float3{ 0.0f, 1.0f, 0.0f }, flat);
    const double horizon_10m = std::sqrt(10.0 * (2.0 * AtmosphereMedium::EARTH_RADIUS_M + 10.0));
    CHECK(double(low.range_m) == doctest::Approx(horizon_10m + Atmosphere::AERIAL_RANGE_M).epsilon(1e-5));
    CHECK(high.range_m > low.range_m);
    const double top_horizon = std::sqrt(double(m.top_m) * (2.0 * AtmosphereMedium::EARTH_RADIUS_M + double(m.top_m)));
    CHECK(double(low.max_path_m) == doctest::Approx(2.0 * top_horizon).epsilon(1e-5));
}
