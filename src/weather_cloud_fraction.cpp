#include "weather_cloud_fraction.hpp"

#include "parallel_for.hpp"

#include "ov/str.hpp"

#include <algorithm>
#include <numeric>

namespace WeatherCloud {

const char *glsl() {
    static const ov::str text = ov::replace_all(ov::str(R"glsl(
const float SHARE_SLOPE = $SLOPE;
const float SHARE_SCALE = $SCALE;
const float SHARE_REACH = $REACH;

// The spread (units of sigma_s) of a uniform noise value u (weather_cloud_fraction.hpp).
float cloud_spread(float u) {
    float half_turn = 0.5 / SHARE_SLOPE;
    return tan(clamp((u - 0.5) / SHARE_SLOPE, -half_turn, half_turn)) / SHARE_SCALE;
}

// The mean of max(q1 + n, 0) over the spread's distribution.
float cloud_mean_excess(float q1) {
    float l = max(-q1, -SHARE_REACH);
    if (l >= SHARE_REACH) {
        return 0.0;
    }
    float k = SHARE_SCALE;
    return SHARE_SLOPE * (q1 * (atan(k * SHARE_REACH) - atan(k * l)) + log((1.0 + k * k * SHARE_REACH * SHARE_REACH) / (1.0 + k * k * l * l)) / (2.0 * k));
}
)glsl"),
            { { "$SLOPE", ov::format("%.9g", SHARE_SLOPE) }, { "$SCALE", ov::format("%.9g", SHARE_SCALE) }, { "$REACH", ov::format("%.9g", reach()) } });
    return text.c_str();
}

namespace {

uint32_t hash(uint32_t p_x, uint32_t p_y, uint32_t p_z, uint32_t p_seed) {
    uint32_t h = p_x * 0x8DA6B343u ^ p_y * 0xD8163841u ^ p_z * 0xCB1AB31Fu ^ p_seed * 0x9E3779B1u;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    return h;
}

// The lattice point's gradient: one of the cube's 12 edge directions (Perlin 2002).
double grad(uint32_t p_h, double p_x, double p_y, double p_z) {
    switch (p_h % 12u) {
        case 0:
            return p_x + p_y;
        case 1:
            return -p_x + p_y;
        case 2:
            return p_x - p_y;
        case 3:
            return -p_x - p_y;
        case 4:
            return p_x + p_z;
        case 5:
            return -p_x + p_z;
        case 6:
            return p_x - p_z;
        case 7:
            return -p_x - p_z;
        case 8:
            return p_y + p_z;
        case 9:
            return -p_y + p_z;
        case 10:
            return p_y - p_z;
        default:
            return -p_y - p_z;
    }
}

double fade(double p_t) {
    return p_t * p_t * p_t * (p_t * (p_t * 6.0 - 15.0) + 10.0);
}

// Gradient noise at p (lattice units), its lattice wrapped every p_n cells.
double gradient_noise(double p_x, double p_y, double p_z, int p_n, uint32_t p_seed) {
    const int x0 = int(std::floor(p_x)), y0 = int(std::floor(p_y)), z0 = int(std::floor(p_z));
    const double fx = p_x - x0, fy = p_y - y0, fz = p_z - z0;
    const double u = fade(fx), v = fade(fy), w = fade(fz);
    auto corner = [&](int p_dx, int p_dy, int p_dz) {
        const uint32_t h = hash(uint32_t((x0 + p_dx) % p_n), uint32_t((y0 + p_dy) % p_n), uint32_t((z0 + p_dz) % p_n), p_seed);
        return grad(h, fx - p_dx, fy - p_dy, fz - p_dz);
    };
    auto lerp = [](double p_a, double p_b, double p_t) { return p_a + (p_b - p_a) * p_t; };
    const double x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), u), x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), u);
    const double x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), u), x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), u);
    return lerp(lerp(x00, x10, v), lerp(x01, x11, v), w);
}

} // namespace

std::vector<float> uniform_noise(int p_size, int p_lattice, int p_octaves, uint32_t p_seed) {
    const size_t count = size_t(p_size) * size_t(p_size) * size_t(p_size);
    std::vector<float> value(count);
    parallel_for(uint32_t(p_size) * uint32_t(p_size), "WeatherCloud::uniform_noise", [&](uint32_t p_row) {
        const int z = int(p_row) / p_size, y = int(p_row) % p_size;
        for (int x = 0; x < p_size; x++) {
            double sum = 0.0, amp = 1.0;
            for (int o = 0; o < p_octaves; o++) {
                const int n = p_lattice << o;
                const double s = double(n) / p_size;
                sum += amp * gradient_noise(x * s, y * s, z * s, n, p_seed + uint32_t(o));
                amp *= 0.5;
            }
            value[(size_t(z) * p_size + y) * p_size + x] = float(sum);
        }
    });
    std::vector<uint32_t> order(count);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t p_a, uint32_t p_b) { return value[p_a] < value[p_b]; });
    std::vector<float> out(count);
    for (size_t r = 0; r < count; r++) {
        out[order[r]] = float((double(r) + 0.5) / double(count));
    }
    return out;
}

} // namespace WeatherCloud
