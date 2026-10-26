// Biome features: biome cells, lava lakes and floating islands (biome_features.hpp).
#include "biome_features.hpp"
#include "ov/math.hpp"

#include <cmath>
#include <initializer_list>

namespace Biome {

using namespace Archetype;
using ExactNoise::hash3;
using ExactNoise::unit_of;

namespace {

const Kind KINDS[FEATURE_COUNT] = {
    { "lava_lake", 640.0, 0.6, 18.0, 45.0, 0x1A7Au },
    { "floating_island", 640.0, 0.5, 15.0, 40.0, 0x15E7u },
};

// The shapes, in units of the feature's radius unless named in metres.
constexpr double BUMP_M = 16.0; // the basin's and the island's bump wavelength
constexpr double BASIN_BUMP_M = 1.0; // the bumps' amp
constexpr double LAVA_DROP = 0.04; // the lava below the rim's lowest ground (at least 3 voxels)
constexpr double LAVA_DEPTH = 0.15; // the basin's bottom below the lava (shallow: an eroded slab leaves a shallow pit)
constexpr double BASIN_CLEAR = 0.25; // the basin's top above the rim's highest ground
constexpr double BASIN_K = 0.2; // the smooth subtraction's radius: the rounded shore
constexpr double MAX_RIM_SPAN = 0.5; // a steeper rim: no lake there
constexpr double ISLAND_DEPTH = 1.1; // the underside's cone
constexpr double ISLAND_DOME = 0.3; // the top cap's height
constexpr double ISLAND_BUMP_M = 1.5;
constexpr double ISLAND_ALT_LO_M = 25.0; // the box's bottom above the highest ground, uniform in [lo, lo + range)
constexpr double ISLAND_ALT_RANGE_M = 45.0;
constexpr int RING_POINTS = 24;

// The cell's middle half along one axis: p_cell_index * p_cell + [p_cell / 4, 3 p_cell / 4), placed by p_bits.
int64_t middle_half(int64_t p_cell_index, int64_t p_cell, uint32_t p_bits) {
    return p_cell_index * p_cell + p_cell / 4 + int64_t(std::floor(unit_of(p_bits) * double(p_cell / 2)));
}

// The lowest and highest ground on p_rings rings of RING_POINTS columns out to radius p_r (voxels), the outer ring
// first, then the centre. False as soon as they span more than p_limit voxels (the rest is not sampled).
bool ground_span(const Site &p_site, const GroundFn &p_ground_y, double p_r, int p_rings, double p_limit, int64_t &r_lo, int64_t &r_hi) {
    r_lo = INT64_MAX;
    r_hi = INT64_MIN;
    auto sample = [&](int64_t p_x, int64_t p_z) {
        const int64_t y = p_ground_y(p_x, p_z);
        r_lo = ov::min(r_lo, y);
        r_hi = ov::max(r_hi, y);
        return double(r_hi - r_lo) <= p_limit;
    };
    for (int ring = p_rings; ring >= 1; ring--) {
        const double r = p_r * double(ring) / double(p_rings);
        for (int i = 0; i < RING_POINTS; i++) {
            const double a = 2.0 * M_PI * (double(i) + 0.5 * double(ring)) / double(RING_POINTS);
            if (!sample(p_site.x + std::llround(r * std::cos(a)), p_site.z + std::llround(r * std::sin(a)))) {
                return false;
            }
        }
    }
    return sample(p_site.x, p_site.z);
}

TableRecord record(int p_type, Combine p_op, double p_k, const VoxelPos &p_anchor, std::initializer_list<float> p_params, uint32_t p_salt) {
    TableRecord rec;
    rec.def = uint8_t(p_type);
    rec.op = p_op;
    rec.set_k(float(p_k));
    rec.anchor = p_anchor;
    std::copy(p_params.begin(), p_params.end(), rec.params);
    rec.salt = hash3(p_salt, 0xB10Eu, 0u).x;
    return rec;
}

bool lava_lake(const Site &p_site, const GroundFn &p_ground_y, const DefLibrary &p_library, const Types &p_types, double p_voxel, Placed &r_out) {
    const double r = p_site.radius;
    const float amp = float(BASIN_BUMP_M / p_voxel);
    // The basin's box at half height 0: its x reach is radius + bumps, its y reach the bumps alone. The slab reaches 2
    // voxels past that wall, so the lava meets it all round.
    const float flat[3] = { 0.0f, float(r), amp };
    Float3 lo{}, hi{};
    float lipschitz = 1.0f;
    p_library[p_types.basin].bounds(flat, lo, hi, lipschitz);
    const double bump = double(hi.y);
    const double slab_r = double(hi.x) + 2.0;
    int64_t rim_lo = 0, rim_hi = 0;
    if (!ground_span(p_site, p_ground_y, slab_r, 3, MAX_RIM_SPAN * r, rim_lo, rim_hi)) {
        return false;
    }
    const int64_t lava = rim_lo - int64_t(std::ceil(ov::max(3.0, LAVA_DROP * r)));
    const int64_t bottom = lava - int64_t(std::ceil(LAVA_DEPTH * r));
    const int64_t top = rim_hi + int64_t(std::ceil(BASIN_CLEAR * r + bump));
    const int64_t half = (top - bottom + 1) / 2;
    r_out.records.push_back(record(p_types.basin, Combine::SMOOTH_SUBTRACT, BASIN_K * r, VoxelPos(p_site.x, bottom + half, p_site.z),
            { float(half), float(r), amp }, p_site.seed));
    // The slab: its top face is the lava, from 4 voxels under the basin's bottom.
    const int64_t slab_half = (lava - bottom + 4 + 1) / 2;
    r_out.records.push_back(record(p_types.cylinder, Combine::UNION, 0.0, VoxelPos(p_site.x, lava - slab_half, p_site.z),
            { float(slab_half), float(slab_r) }, p_site.seed + 1u));
    r_out.records.back().material = Material::LAVA; // the cylinder type is shared: the slab says it is lava
    r_out.top_y = lava;
    r_out.lava = true;
    r_out.pool.top = VoxelPos(p_site.x, lava, p_site.z);
    r_out.pool.radius = r;
    return true;
}

void floating_island(const Site &p_site, const GroundFn &p_ground_y, const DefLibrary &p_library, const Types &p_types, double p_voxel, Placed &r_out) {
    const double r = p_site.radius;
    const double height = ISLAND_DEPTH * r;
    const double dome = ISLAND_DOME * r;
    const double dome_radius = (r * r + dome * dome) / (2.0 * dome);
    const double len = std::sqrt(height * height + r * r);
    const std::initializer_list<float> params = { float(r), float(height), float(height / len), float(r / len), float(dome_radius),
        float(dome_radius - dome), float(ISLAND_BUMP_M / p_voxel) };
    Float3 lo{}, hi{};
    float lipschitz = 1.0f;
    p_library[p_types.island].bounds(params.begin(), lo, hi, lipschitz);
    int64_t ground_lo = 0, ground_hi = 0;
    ground_span(p_site, p_ground_y, double(hi.x), 3, INFINITY, ground_lo, ground_hi);
    const double altitude = (ISLAND_ALT_LO_M + ISLAND_ALT_RANGE_M * unit_of(hash3(p_site.seed, 0xA17u, 0u).x)) / p_voxel;
    const int64_t y = ground_hi + std::llround(altitude) - int64_t(std::floor(lo.y)); // the box's bottom `altitude` up
    r_out.records.push_back(record(p_types.island, Combine::UNION, 0.0, VoxelPos(p_site.x, y, p_site.z), params, p_site.seed));
    r_out.top_y = y;
}

} // namespace

const Kind &kind(Feature p_feature) {
    return KINDS[p_feature];
}

int cell_shift(Feature p_feature, double p_voxel) {
    return ov::max(4, ExactNoise::lattice_k(kind(p_feature).cell_m, p_voxel));
}

Types add_types(DefLibrary &r_library, double p_voxel) {
    const int k = ExactNoise::lattice_k(BUMP_M, p_voxel);
    Types t;
    t.basin = r_library.add(make_basin_type(k, BASIN_SEED), Material::SCORCHED_ROCK);
    t.cylinder = r_library.add(make_cylinder_type());
    t.island = r_library.add(make_island_type(k, ISLAND_SEED), Material::ISLAND_ROCK);
    return t;
}

Site moved(const Site &p_site, int p_attempt, double p_voxel) {
    if (p_attempt == 0) {
        return p_site;
    }
    const int64_t cell = int64_t(1) << cell_shift(p_site.feature, p_voxel);
    const ExactNoise::UVec3 h = hash3(p_site.seed, uint32_t(p_attempt), 0x40EEu);
    Site s = p_site;
    s.x = middle_half(WorldCoord::floor_div(p_site.x, cell), cell, h.x);
    s.z = middle_half(WorldCoord::floor_div(p_site.z, cell), cell, h.y);
    return s;
}

void sites(Feature p_feature, int64_t p_x0, int64_t p_z0, int64_t p_x1, int64_t p_z1, double p_voxel, std::vector<Site> &r_out) {
    const Kind &k = kind(p_feature);
    const int64_t cell = int64_t(1) << cell_shift(p_feature, p_voxel);
    for (int64_t cz = WorldCoord::floor_div(p_z0, cell); cz <= WorldCoord::floor_div(p_z1, cell); cz++) {
        for (int64_t cx = WorldCoord::floor_div(p_x0, cell); cx <= WorldCoord::floor_div(p_x1, cell); cx++) {
            const ExactNoise::UVec3 h = hash3(ExactNoise::hash_cell(Vector3i64(cx, int64_t(p_feature), cz), k.seed), 0xCE11u, 0u);
            if (unit_of(h.x) >= k.chance) {
                continue;
            }
            Site s;
            s.feature = p_feature;
            s.x = middle_half(cx, cell, h.y);
            s.z = middle_half(cz, cell, h.z);
            if (s.x < p_x0 || s.x > p_x1 || s.z < p_z0 || s.z > p_z1) {
                continue;
            }
            const ExactNoise::UVec3 g = hash3(h.x, h.y, h.z);
            s.radius = (k.radius_lo_m + (k.radius_hi_m - k.radius_lo_m) * unit_of(g.x)) / p_voxel;
            s.seed = g.y;
            r_out.push_back(s);
        }
    }
}

bool place(const Site &p_site, const GroundFn &p_ground_y, const DefLibrary &p_library, const Types &p_types, double p_voxel, Placed &r_out) {
    r_out = Placed();
    if (p_site.feature == FLOATING_ISLAND) {
        r_out.at = p_site;
        floating_island(p_site, p_ground_y, p_library, p_types, p_voxel, r_out);
        return true;
    }
    for (int attempt = 0; attempt < LAKE_ATTEMPTS; attempt++) {
        r_out.at = moved(p_site, attempt, p_voxel);
        if (lava_lake(r_out.at, p_ground_y, p_library, p_types, p_voxel, r_out)) {
            return true;
        }
    }
    return false;
}

} // namespace Biome
