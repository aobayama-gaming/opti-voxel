#include "test_climate_common.hpp"

#include <algorithm>
#include <cmath>
#include <random>

// The fold (Cascade::node_interp, Spec::node_level): below its node level a field group is read from the 8 nodes of the
// point's cell instead of being carried down.

using namespace Cascade;
using ClimateTest::ClimateSDF;
using ExactNoise::float_bits;

namespace {

// Random channels of a group's nodes, as the cascade stores them (flushed: 0 or at least 2^-64 in magnitude).
struct NodeBlock {
    float ch[3][2][2][4]; // [x][y][z][channel]: two cells sharing the face x = 1
};

NodeBlock random_block(std::mt19937 &p_rng) {
    std::uniform_real_distribution<float> v(-3.0f, 3.0f);
    NodeBlock b;
    for (auto &x : b.ch) {
        for (auto &y : x) {
            for (auto &z : y) {
                for (float &c : z) {
                    c = flush_value(v(p_rng));
                }
            }
        }
    }
    return b;
}

// node_interp over the block's cell p_cell_x (0 or 1) with axis bases p_axes.
void interp(const NodeBlock &p_b, int p_cell_x, const NodeAxis p_axes[3], float p_scale, float r_ch[4]) {
    NodeCell cell;
    for (int a = 0; a < 3; a++) {
        cell.axis[a] = p_axes[a];
    }
    cell.scale = p_scale;
    node_interp(cell, [&](int p_x, int p_y, int p_z, float r[4]) { std::copy(p_b.ch[p_cell_x + p_x][p_y][p_z], p_b.ch[p_cell_x + p_x][p_y][p_z] + 4, r); }, r_ch);
}

bool same(const float *p_a, const float *p_b) {
    for (int c = 0; c < 4; c++) {
        if (float_bits(p_a[c]) != float_bits(p_b[c])) {
            return false;
        }
    }
    return true;
}

} // namespace

TEST_CASE("Climate fold: at a node it is the node, slopes times 2^-d, bit for bit; at t = 1 the cell's far node") {
    std::mt19937 rng(91);
    int checked = 0;
    for (int i = 0; i < 2000; i++) {
        const NodeBlock b = random_block(rng);
        const int d = 1 + i % NODE_MAX;
        const float scale = ExactNoise::exact_pow2(-d);
        const NodeAxis zero[3] = { node_axis(0.0f), node_axis(0.0f), node_axis(0.0f) };
        float r[4];
        interp(b, 0, zero, scale, r);
        const float *n = b.ch[0][0][0];
        const float want[4] = { n[VALUE], flush_value(n[DX] * scale), flush_value(n[DY] * scale), flush_value(n[DZ] * scale) };
        CHECK(same(r, want));
        // t = 1 on every axis (node_axis's basis, not a point's offset: node_cell never makes it) reads the far node.
        const NodeAxis one[3] = { node_axis(1.0f), node_axis(1.0f), node_axis(1.0f) };
        interp(b, 0, one, scale, r);
        const float *f = b.ch[1][1][1];
        const float far[4] = { f[VALUE], flush_value(f[DX] * scale), flush_value(f[DY] * scale), flush_value(f[DZ] * scale) };
        CHECK(same(r, far));
        checked++;
    }
    CHECK(checked == 2000);
    // node_cell: t from the low bits, the floor node for negative indices too.
    const NodeCell c = node_cell(Vector3i64(-5, 12, 7), 3, 5);
    CHECK(c.q == Vector3i64(-2, 3, 1));
    CHECK(c.axis[0].t == 0.75f);
    CHECK_FALSE(c.axis[1].mid);
    CHECK(c.axis[2].t == 0.75f);
    CHECK(c.scale == 0.25f);
}

// Two cells sharing a face give the same bits on it in every channel (value and gradient: the normals read the gradient
// channels, so no seam there either): the low cell at t = 1, the high one at t = 0, any offset on the other axes.
TEST_CASE("Climate fold: two cells sharing a face give the same bits on it, every channel") {
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> off(0, 15);
    int64_t differ = 0;
    for (int i = 0; i < 4000; i++) {
        const NodeBlock b = random_block(rng);
        const float ty = float(off(rng)) * 0.0625f;
        const float tz = float(off(rng)) * 0.0625f;
        const NodeAxis low[3] = { node_axis(1.0f), node_axis(ty), node_axis(tz) };
        const NodeAxis high[3] = { node_axis(0.0f), node_axis(ty), node_axis(tz) };
        float a[4], c[4];
        interp(b, 0, low, 0.0625f, a);
        interp(b, 1, high, 0.0625f, c);
        differ += !same(a, c);
    }
    CHECK(differ == 0);
}

// Along one axis (the transverse slopes 0, the other axes at nodes) it is the cubic Hermite through the nodes: a cubic
// with its exact slopes comes back, value and derivative, within float rounding.
TEST_CASE("Climate fold: a cubic along an axis is reproduced") {
    const auto f = [](double p_x) { return ((0.7 * p_x - 1.3) * p_x + 0.4) * p_x - 2.0; };
    const auto df = [](double p_x) { return (2.1 * p_x - 2.6) * p_x + 0.4; };
    double worst_v = 0.0, worst_s = 0.0;
    for (int axis = 0; axis < 3; axis++) {
        NodeBlock b = {};
        for (int x = 0; x < 2; x++) {
            for (int y = 0; y < 2; y++) {
                for (int z = 0; z < 2; z++) {
                    const int s = axis == 0 ? x : (axis == 1 ? y : z);
                    b.ch[x][y][z][VALUE] = float(f(s));
                    b.ch[x][y][z][DX + axis] = float(df(s));
                }
            }
        }
        for (int o = 1; o < 64; o++) {
            const float t = float(o) / 64.0f;
            NodeAxis axes[3] = { node_axis(0.0f), node_axis(0.0f), node_axis(0.0f) };
            axes[axis] = node_axis(t);
            float r[4];
            interp(b, 0, axes, 1.0f, r);
            worst_v = std::max(worst_v, std::fabs(double(r[VALUE]) - f(t)));
            worst_s = std::max(worst_s, std::fabs(double(r[DX + axis]) - df(t)));
        }
    }
    MESSAGE("cubic: worst value error ", worst_v, ", slope error ", worst_s);
    CHECK(worst_v < 1e-5);
    CHECK(worst_s < 1e-5);
}

// world.json's node levels: each field group's finest insertion level (5 cm: 100 m is k = 11, Λ_8), the state and the
// field of terms only carried.
TEST_CASE("Climate fold: node levels are each field group's finest insertion level") {
    ClimateSDF world("res://climate/world.json", 25);
    CLIMATE_REQUIRE(world.sdf.get_climate() != nullptr);
    const Spec &spec = world.sdf.get_spec();
    const Climate::Table &t = *world.sdf.get_climate();
    CHECK(spec.node_level(0) == 0);
    CHECK(spec.node_level(1 + t.field_index("scarp")) == 8);
    CHECK(spec.node_level(1 + t.field_index("valley")) == 9);
    CHECK(spec.node_level(1 + t.field_index("temperature")) == 19);
    CHECK(spec.node_level(1 + t.field_index("elev")) == 0); // terms only
    CHECK(spec.node_level(spec.group_count() - 1) == 0); // the multifractal state
    const FieldRange all{ 0, MAX_LEVEL };
    CHECK(spec.folds(all));
    CHECK((spec.folded(all, 0, 8) >> (1 + t.field_index("scarp")) & 1u) == 0u);
    CHECK((spec.folded(all, 0, 7) >> (1 + t.field_index("scarp")) & 1u) == 1u);
    CHECK(spec.folded(FieldRange(), 0, 0) == 0u);
}

// What the fold changes against the carried cascade (the iterated subdivision of the same nodes, H level by level with
// nothing inserted): LINEAR4 drops the cross derivatives at every level, the one-shot interpolation once, so the two
// differ by those terms, not just by rounding. Measured on world.json's fields, per channel and per number of levels
// below the node level, in units of the field's largest band amplitude (gradients per lattice spacing of Λ_m times 2^d,
// the node spacing's units).
TEST_CASE("Climate fold: world.json's fields against the carried cascade, per channel and depth below the node level") {
    ClimateSDF world("res://climate/world.json", 25);
    CLIMATE_REQUIRE(world.sdf.get_climate() != nullptr);
    const Spec &spec = world.sdf.get_spec();
    constexpr int DEPTHS = 9;
    double worst[DEPTHS][4] = {};
    int64_t compared = 0;
    for (int g = 1; g < spec.group_count(); g++) {
        const int node = spec.node_level(g);
        if (node == 0) {
            continue;
        }
        float amp = 0.0f;
        for (const Band &b : spec.group_bands(g)) {
            amp = std::max(amp, b.amp);
        }
        for (int depth = 1; depth < DEPTHS && depth <= node; depth++) {
            const int m = node - depth;
            for (int i = 0; i < 24; i++) {
                const Vector3i64 p((int64_t(i) * 7919 - 90000) >> m, (world.sdf.get_datum_y() + i * 331) >> m, (int64_t(i) * -6007 + 4000) >> m);
                float slots[SLOTS];
                point_slots(spec, p, m, slots, true);
                // The carried cascade: Λ_node's box of the point, subdivided down to it with nothing inserted.
                const std::vector<Box> boxes = level_boxes(point_box(p), m, node);
                Level cur = run(spec, boxes.back(), node, nullptr, Source(), true);
                for (int k = depth - 1; k >= 0; k--) {
                    cur = subdivide(cur, boxes[size_t(k)]);
                    flush(cur);
                }
                for (int c = 0; c < 4; c++) {
                    const double scale = c == VALUE ? 1.0 : std::ldexp(1.0, depth);
                    const double diff = std::fabs(double(slots[group_slot(g) + c]) - double(cur.at(group_slot(g) + c, p))) * scale / double(amp);
                    worst[depth][c] = std::max(worst[depth][c], diff);
                }
                compared++;
            }
        }
    }
    for (int depth = 1; depth < DEPTHS; depth++) {
        MESSAGE("node level - ", depth, ": value ", worst[depth][0], ", dx ", worst[depth][1], ", dy ", worst[depth][2], ", dz ", worst[depth][3],
                " (of the field's largest amplitude)");
    }
    CHECK(compared > 1000);
    for (int depth = 1; depth < DEPTHS; depth++) {
        CHECK(worst[depth][0] < 0.05);
    }
}
