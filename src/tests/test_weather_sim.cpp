#include "test_climate_common.hpp"

#include "weather_sim.hpp"

#include <cmath>
#include <cstring>
#include <vector>

// The weather's simulation (WeatherSim, docs/weather.md): its local laws, its stability and its determinism, and a
// biome's "weather" block.

using ClimateTest::ClimateSDF;
using ClimateTest::json;

namespace {

double length3(const double p_v[3]) {
    return std::sqrt(p_v[0] * p_v[0] + p_v[1] * p_v[1] + p_v[2] * p_v[2]);
}

void unit(double r_v[3]) {
    const double l = length3(r_v);
    for (int k = 0; k < 3; k++) {
        r_v[k] /= l;
    }
}

WeatherSim::Params small_planet(int p_levels = 1) {
    WeatherSim::Params p;
    p.radius_m = 2000.0;
    p.prevailing_mps = 8.0;
    p.seed = 7;
    p.levels = p_levels;
    p.top_m = 300.0;
    return p;
}

// Still air: no prevailing wind, no heat, a set moisture, no storms.
Climate::BiomeWeather still(float p_moisture) {
    Climate::BiomeWeather w;
    w.wind = 0.0f;
    w.heat = 0.0f;
    w.moisture = p_moisture;
    w.storms = 0.0f;
    return w;
}

// The largest |value| at p_offset of each interior cell's record (a level's field: level_offset + Field; the column's
// ETA: eta_offset), and whether every value is finite.
double largest(const WeatherSim &p_sim, int p_offset, bool &r_finite) {
    const auto &g = static_cast<const CubeSphereGrid &>(p_sim.grid());
    double m = 0.0;
    for (int f = 0; f < CubeSphereGrid::FACES; f++) {
        for (int j = 0; j < g.size(); j++) {
            for (int i = 0; i < g.size(); i++) {
                const float v = p_sim.state()[g.index(f, i, j) * size_t(p_sim.stride()) + size_t(p_offset)];
                r_finite = r_finite && std::isfinite(v);
                m = std::max(m, double(std::fabs(v)));
            }
        }
    }
    return m;
}

double mean_diag(const WeatherSim &p_sim, int p_diag) {
    const auto &g = static_cast<const CubeSphereGrid &>(p_sim.grid());
    double sum = 0.0;
    for (int f = 0; f < CubeSphereGrid::FACES; f++) {
        for (int j = 0; j < g.size(); j++) {
            for (int i = 0; i < g.size(); i++) {
                sum += p_sim.diagnostics()[g.index(f, i, j) * WeatherSim::DIAGS + p_diag];
            }
        }
    }
    return sum / double(g.cells());
}

// A level's field averaged over the interior cells.
double mean_level(const WeatherSim &p_sim, int p_level, int p_field) {
    const auto &g = static_cast<const CubeSphereGrid &>(p_sim.grid());
    double sum = 0.0;
    for (int f = 0; f < CubeSphereGrid::FACES; f++) {
        for (int j = 0; j < g.size(); j++) {
            for (int i = 0; i < g.size(); i++) {
                sum += p_sim.state()[g.index(f, i, j) * size_t(p_sim.stride()) + size_t(WeatherSim::level_offset(p_level) + p_field)];
            }
        }
    }
    return sum / double(g.cells());
}

// The circulation round a centre: the mean of ((up - centre) x wind) . up over the cells within p_radius radians,
// positive counterclockwise seen from above.
double circulation(const WeatherSim &p_sim, const double p_centre[3], double p_radius) {
    const auto &g = static_cast<const CubeSphereGrid &>(p_sim.grid());
    double sum = 0.0;
    int count = 0;
    for (int f = 0; f < CubeSphereGrid::FACES; f++) {
        for (int j = 0; j < g.size(); j++) {
            for (int i = 0; i < g.size(); i++) {
                double up[3];
                g.cell_direction(f, i, j, up);
                const double r[3] = { up[0] - p_centre[0], up[1] - p_centre[1], up[2] - p_centre[2] };
                if (length3(r) > p_radius) {
                    continue;
                }
                const float *s = &p_sim.state()[g.index(f, i, j) * size_t(p_sim.stride())]; // the ground level's wind
                const double c[3] = { r[1] * s[2] - r[2] * s[1], r[2] * s[0] - r[0] * s[2], r[0] * s[1] - r[1] * s[0] };
                sum += c[0] * up[0] + c[1] * up[1] + c[2] * up[2];
                count++;
            }
        }
    }
    return count > 0 ? sum / count : 0.0;
}

} // namespace

TEST_CASE("WeatherSim: the Coriolis turns the wind right in the north, left in the south, and keeps its speed") {
    for (const double y : { 0.8, -0.8 }) {
        double up[3] = { 0.6, y, 0.0 };
        unit(up);
        double wind[3] = { 0.0, 0.0, 5.0 }; // tangent: up has no z
        const double before[3] = { wind[0], wind[1], wind[2] };
        WeatherSim::coriolis(up, 2.0 * 0.03 * up[1], 1.0, wind);
        CHECK(std::fabs(length3(wind) - 5.0) < 1e-9);
        // (before x after) . up < 0: clockwise seen from above, a right turn.
        const double c[3] = { before[1] * wind[2] - before[2] * wind[1], before[2] * wind[0] - before[0] * wind[2], before[0] * wind[1] - before[1] * wind[0] };
        const double turn = c[0] * up[0] + c[1] * up[1] + c[2] * up[2];
        CHECK((y > 0.0 ? turn < 0.0 : turn > 0.0));
    }
}

TEST_CASE("WeatherSim: the prevailing wind, easterly trades, westerlies, calm at the equator and the poles") {
    auto east_speed = [](double p_lat_deg) {
        const double lat = p_lat_deg * 3.141592653589793 / 180.0;
        const double up[3] = { std::cos(lat), std::sin(lat), 0.0 };
        double wind[3];
        WeatherSim::prevailing(up, 10.0, wind);
        CHECK(std::fabs(wind[0] * up[0] + wind[1] * up[1] + wind[2] * up[2]) < 1e-9); // tangent
        // East at (cos, sin, 0) is pole x up = (0, 0, -cos): its speed along -z.
        return -wind[2];
    };
    CHECK(east_speed(15.0) < -9.9); // trades blow from the east
    CHECK(east_speed(45.0) > 9.9); // westerlies
    CHECK(east_speed(-45.0) > 9.9);
    CHECK(east_speed(75.0) < -9.9); // polar easterlies
    CHECK(std::fabs(east_speed(0.0)) < 1e-9);
    CHECK(std::fabs(east_speed(90.0)) < 1e-6);
}

TEST_CASE("WeatherSim: still air stays still") {
    const auto grid = std::make_shared<CubeSphereGrid>(16, 2000.0);
    WeatherSim sim(grid, small_planet(4), std::vector<Climate::BiomeWeather>(grid->cells(), still(0.5f)));
    for (int t = 0; t < 50; t++) {
        sim.step(0.1);
    }
    bool finite = true;
    for (int k = 0; k < sim.levels(); k++) {
        CHECK(largest(sim, WeatherSim::level_offset(k) + WeatherSim::WIND_X, finite) == 0.0);
        CHECK(largest(sim, WeatherSim::level_offset(k) + WeatherSim::WIND_Y, finite) == 0.0);
    }
    CHECK(largest(sim, sim.eta_offset(), finite) == 0.0);
    CHECK(finite);
    CHECK(mean_diag(sim, WeatherSim::COVER) == 0.0);
}

// A storm's low pulls the air in and the Coriolis turns it: counterclockwise seen from above in the north, clockwise in
// the south (a cyclone); the pressure waves it sends round stay bounded.
TEST_CASE("WeatherSim: a storm spins cyclonically, each hemisphere its way, and stays bounded") {
    const auto grid = std::make_shared<CubeSphereGrid>(32, 2000.0);
    for (const double y : { 0.7, -0.7 }) {
        WeatherSim sim(grid, small_planet(), std::vector<Climate::BiomeWeather>(grid->cells(), still(0.5f)));
        double centre[3] = { 0.5, y, 0.3 };
        unit(centre);
        sim.inject_storm(centre, 1.0);
        for (int t = 0; t < 600; t++) { // 60 s
            sim.step(0.1);
        }
        const double spin = circulation(sim, centre, 0.25);
        CHECK((y > 0.0 ? spin > 0.0 : spin < 0.0));
        bool finite = true;
        CHECK(largest(sim, sim.eta_offset(), finite) <= std::fabs(WeatherSim::STORM_ETA));
        CHECK(largest(sim, WeatherSim::WIND_X, finite) < 30.0);
        CHECK(finite);
    }
}

TEST_CASE("WeatherSim: moist ground makes cloud, dry ground none") {
    const auto grid = std::make_shared<CubeSphereGrid>(16, 2000.0);
    WeatherSim wet(grid, small_planet(), std::vector<Climate::BiomeWeather>(grid->cells(), still(1.3f)));
    WeatherSim dry(grid, small_planet(), std::vector<Climate::BiomeWeather>(grid->cells(), still(0.5f)));
    for (int t = 0; t < 3000; t++) { // 300 s
        wet.step(0.1);
        dry.step(0.1);
    }
    CHECK(mean_diag(wet, WeatherSim::COVER) > 0.5);
    CHECK(mean_diag(dry, WeatherSim::COVER) == 0.0);
    CHECK(mean_diag(dry, WeatherSim::RAIN) == 0.0);
}

// Storms and the prevailing wind on: the same seed gives the same bits, another seed other storms.
TEST_CASE("WeatherSim: deterministic for a seed") {
    const auto grid = std::make_shared<CubeSphereGrid>(16, 2000.0);
    Climate::BiomeWeather stormy;
    stormy.storms = 1000.0f;
    std::vector<Climate::BiomeWeather> w(grid->cells(), stormy);
    WeatherSim::Params other = small_planet();
    other.seed = 8;
    WeatherSim a(grid, small_planet(3), w);
    WeatherSim b(grid, small_planet(3), w);
    other.levels = 3;
    WeatherSim c(grid, other, w);
    for (int t = 0; t < 200; t++) {
        a.step(0.25);
        b.step(0.25);
        c.step(0.25);
    }
    CHECK(a.storms_started() > 0);
    CHECK(a.storms_started() == b.storms_started());
    CHECK(std::memcmp(a.state().data(), b.state().data(), a.state().size() * sizeof(float)) == 0);
    CHECK(std::memcmp(a.diagnostics().data(), b.diagnostics().data(), a.diagnostics().size() * sizeof(float)) == 0);
    CHECK(std::memcmp(a.state().data(), c.state().data(), a.state().size() * sizeof(float)) != 0);
}

// A long step is cut into substeps the gravity waves' CFL allows: 60 game seconds at once stay finite and bounded.
TEST_CASE("WeatherSim: a long step takes substeps and stays stable") {
    const auto grid = std::make_shared<CubeSphereGrid>(32, 2000.0);
    WeatherSim sim(grid, small_planet(6), std::vector<Climate::BiomeWeather>(grid->cells(), Climate::BiomeWeather()));
    CHECK(sim.max_substep_s() > 0.0);
    const double dir[3] = { 0.0, 1.0, 0.0 };
    sim.inject_storm(dir, 1.0);
    for (int t = 0; t < 20; t++) {
        sim.step(60.0);
    }
    bool finite = true;
    for (int k = 0; k < sim.levels(); k++) {
        CHECK(largest(sim, WeatherSim::level_offset(k) + WeatherSim::WIND_X, finite) < 40.0);
        CHECK(largest(sim, WeatherSim::level_offset(k) + WeatherSim::TEMP, finite) < 2.0);
    }
    CHECK(finite);
    CHECK(sim.time_s() == doctest::Approx(1200.0));
}

// Warm moist ground under the levels (no wind, no storms): the warm air overturns and carries the vapour up, where the
// saturation is lower: the cloud forms aloft, its base above the ground level, which stays clear.
TEST_CASE("WeatherSim: warm moist ground convects, its cloud forms aloft") {
    const auto grid = std::make_shared<CubeSphereGrid>(8, 2000.0);
    Climate::BiomeWeather warm = still(1.3f);
    warm.heat = 0.6f;
    WeatherSim sim(grid, small_planet(6), std::vector<Climate::BiomeWeather>(grid->cells(), warm));
    for (int t = 0; t < 60; t++) { // 600 s
        sim.step(10.0);
    }
    double aloft = 0.0;
    for (int k = 2; k < sim.levels(); k++) {
        aloft = std::max(aloft, mean_level(sim, k, WeatherSim::CLOUD));
    }
    CHECK(aloft > 0.25 * WeatherSim::FULL_CLOUD);
    CHECK(mean_level(sim, 0, WeatherSim::CLOUD) < 0.1 * aloft); // the base above the ground level
    CHECK(mean_level(sim, sim.levels() - 1, WeatherSim::VAPOUR) > 0.3); // lifted
    CHECK(mean_diag(sim, WeatherSim::COVER) > 0.25);
}

// Cold moist ground (stable air: no overturning): the lowest level saturates, the cloud sits on the ground (fog) and the
// levels above stay drier.
TEST_CASE("WeatherSim: cold moist ground makes fog in the lowest level") {
    const auto grid = std::make_shared<CubeSphereGrid>(8, 2000.0);
    Climate::BiomeWeather cold = still(1.4f);
    cold.heat = -0.8f;
    WeatherSim sim(grid, small_planet(6), std::vector<Climate::BiomeWeather>(grid->cells(), cold));
    for (int t = 0; t < 60; t++) {
        sim.step(10.0);
    }
    const double fog = mean_level(sim, 0, WeatherSim::CLOUD);
    CHECK(fog > 0.25 * WeatherSim::FULL_CLOUD);
    CHECK(mean_level(sim, sim.levels() - 1, WeatherSim::CLOUD) < 0.5 * fog);
}

TEST_CASE("Climate: a biome's weather parses, out of range or unknown keys are refused") {
    ov::Value data;
    CHECK(load_table("res://climate/example.json", data) == ov::str());
    data["biomes"].at(0)["weather"] = json(R"({"wind": 2, "drag": 0.2, "heat": 0.5, "moisture": 1.1, "storms": 3})");
    {
        ClimateSDF ok(data, 15);
        CHECK_OR_RETURN(ok.sdf.get_climate() != nullptr);
        const Climate::BiomeWeather &w = ok.sdf.get_climate()->biomes[0].weather;
        CHECK(w.wind == 2.0f);
        CHECK(w.drag == 0.2f);
        CHECK(w.heat == 0.5f);
        CHECK(w.moisture == 1.1f);
        CHECK(w.gust == Climate::BiomeWeather().gust); // its default
        CHECK(w.storms == 3.0f);
        const Climate::BiomeWeather &neutral = ok.sdf.get_climate()->biomes[1].weather;
        CHECK(neutral.moisture == Climate::BiomeWeather().moisture);
    }
    for (const char *bad : { R"({"heat": 2})", R"({"wind": -1})", R"({"rain": 1})", R"({"moisture": "wet"})" }) {
        data["biomes"].at(0)["weather"] = json(bad);
        ClimateSDF refused(data, 15);
        CHECK(refused.sdf.get_climate() == nullptr);
        CHECK_FALSE(refused.sdf.get_world_error().empty());
    }
}
