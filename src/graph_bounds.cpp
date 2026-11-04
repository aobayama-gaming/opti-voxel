#include "graph_compile.hpp"

#include "ov/format.hpp"
#include "ov/math.hpp"

#include <cmath>

// LevelProgram's bounds (graph_compile.hpp): each node's value interval, its step at each LOD, the validation.
namespace Terrain {

namespace {

// perlin3's range (Cascade's shapes and amplitudes take the same).
constexpr double NOISE_BOUND = 1.0;
// RCP's value at most 1 / (3/4) (the total S = sum k + (1 - sum k)^2 >= 3/4) and its slope 1 / S^2 at most 16/9.
constexpr double RCP_BOUND = 4.0 / 3.0;
constexpr double RCP_SLOPE = 16.0 / 9.0;

// p_a times p_b, 0 where either is (an unbounded value times 0).
double times(double p_a, double p_b) {
    return p_a == 0.0 || p_b == 0.0 ? 0.0 : p_a * p_b;
}

struct Bounds {
    const LevelProgram &prog;
    const Node &n;
    double lo(size_t p_j) const { return prog.lo[size_t(n.in[p_j])]; }
    double hi(size_t p_j) const { return prog.hi[size_t(n.in[p_j])]; }
    double peak(size_t p_j) const { return ov::max(std::fabs(lo(p_j)), std::fabs(hi(p_j))); }
};

// Node p_i's value within [r_lo, r_hi], from its inputs' (|smooth_abs(a)| <= |a|, a smooth min dips k / 4 under the min).
void value_bound(const LevelProgram &p_prog, size_t p_i, double &r_lo, double &r_hi) {
    const Node &n = p_prog.nodes[p_i];
    const Bounds b{ p_prog, n };
    const std::vector<float> &k = n.k;
    r_lo = 0.0;
    r_hi = 0.0;
    switch (n.op) {
        case Op::NOTE:
            r_lo = -NOISE_BOUND;
            r_hi = NOISE_BOUND;
            break;
        case Op::CONST:
            r_lo = r_hi = double(k[0]);
            break;
        case Op::UP:
            r_lo = -INFINITY;
            r_hi = INFINITY;
            break;
        case Op::LIN:
            for (size_t j = 0; j < n.in.size(); j++) {
                const double c = double(k[j]);
                r_lo += times(c, c >= 0.0 ? b.lo(j) : b.hi(j));
                r_hi += times(c, c >= 0.0 ? b.hi(j) : b.lo(j));
            }
            break;
        case Op::MUL: {
            const double p[4] = { times(b.lo(0), b.lo(1)), times(b.lo(0), b.hi(1)), times(b.hi(0), b.lo(1)), times(b.hi(0), b.hi(1)) };
            r_lo = ov::min(ov::min(p[0], p[1]), ov::min(p[2], p[3]));
            r_hi = ov::max(ov::max(p[0], p[1]), ov::max(p[2], p[3]));
            break;
        }
        case Op::MIN:
        case Op::SMIN:
            r_lo = ov::min(b.lo(0), b.lo(1)) - (n.op == Op::SMIN ? double(k[0]) / 4.0 : 0.0);
            r_hi = ov::min(b.hi(0), b.hi(1));
            break;
        case Op::MAX:
        case Op::SMAX:
            r_lo = ov::max(b.lo(0), b.lo(1));
            r_hi = ov::max(b.hi(0), b.hi(1)) + (n.op == Op::SMAX ? double(k[0]) / 4.0 : 0.0);
            break;
        case Op::CURVE:
            r_hi = double(Climate::curve_bound(p_prog.curves[size_t(n.curve)]));
            r_lo = -r_hi;
            break;
        case Op::SHAPE:
            if (k[2] == 0.0f && k[3] == 0.0f) {
                r_lo = double(k[0]) + times(double(k[1]), k[1] >= 0.0f ? b.lo(0) : b.hi(0));
                r_hi = double(k[0]) + times(double(k[1]), k[1] >= 0.0f ? b.hi(0) : b.lo(0));
            } else {
                const double p = b.peak(0);
                r_hi = std::fabs(double(k[0])) + times(p, std::fabs(double(k[1])) + std::fabs(double(k[2]))) + times(p * p, std::fabs(double(k[3])));
                r_lo = -r_hi;
            }
            break;
        case Op::AXIS:
        case Op::FILL:
        case Op::NORM:
            r_hi = 1.0;
            break;
        case Op::RCP:
            r_hi = RCP_BOUND;
            break;
        case Op::PICK:
            r_hi = double(p_prog.biomes - 1);
            break;
        case Op::COUNT:
            break;
    }
}

// Node p_i's largest change when the notes of level p_lod are dropped (p_d: its inputs' changes), through each op's
// Lipschitz bound in each input: what LOD p_lod + 1 lacks against LOD p_lod at their shared points (a node computed on
// a coarser level is the same bits in both, a finer note is absent in both).
double step(const LevelProgram &p_prog, const std::vector<double> &p_d, size_t p_i, int p_lod) {
    const Node &n = p_prog.nodes[p_i];
    const Bounds b{ p_prog, n };
    const auto d = [&](size_t p_j) { return p_d[size_t(n.in[p_j])]; };
    const std::vector<float> &k = n.k;
    switch (n.op) {
        case Op::NOTE:
            return n.level == p_lod ? NOISE_BOUND : 0.0; // its warp is coarser (the gate): the same in both
        case Op::LIN: {
            double s = 0.0;
            for (size_t j = 0; j < n.in.size(); j++) {
                s += times(std::fabs(double(k[j])), d(j));
            }
            return s;
        }
        case Op::MUL:
            return times(b.peak(0), d(1)) + times(b.peak(1), d(0)) + d(0) * d(1);
        case Op::MIN:
        case Op::MAX:
        case Op::SMIN:
        case Op::SMAX:
            return ov::max(d(0), d(1));
        case Op::CURVE:
            return times(double(Climate::curve_slope_bound(p_prog.curves[size_t(n.curve)])), d(0));
        case Op::SHAPE:
            return times(std::fabs(double(k[1])) + std::fabs(double(k[2])) + times(2.0 * std::fabs(double(k[3])), b.peak(0)), d(0)) + times(std::fabs(double(k[3])), d(0) * d(0));
        case Op::AXIS:
            return ov::min(1.0, times(Climate::KERNEL_SLOPE * std::fabs(double(k[1])), d(0)));
        case Op::FILL:
            return ov::min(1.0, 2.0 * d(0));
        case Op::RCP:
            return ov::min(RCP_BOUND, RCP_SLOPE * d(0));
        case Op::NORM:
            return ov::min(1.0, RCP_BOUND * d(0) + d(1) + d(0) * d(1));
        case Op::CONST:
        case Op::UP:
        case Op::PICK:
        case Op::COUNT:
            break;
    }
    return 0.0;
}

// Empty, or the first node out of the format's ranges: a note's amplitude in the noise's exact range, a warp's
// displacement within WARP_LIMIT.
ov::str validate_nodes(const LevelProgram &p_prog) {
    for (const Node &n : p_prog.nodes) {
        for (size_t j = 0; n.op == Op::LIN && j < n.in.size(); j++) {
            if (p_prog.nodes[size_t(n.in[j])].op == Op::NOTE && n.k[j] != 0.0f && !ExactNoise::in_exact_range(std::fabs(n.k[j]))) {
                return ov::format("climate: a note's amplitude %s is outside the noise's exact range [2^-16, 2^40] voxels.", n.k[j]);
            }
        }
        for (size_t j = 0; n.op == Op::NOTE && j < n.in.size(); j++) {
            const int w = n.in[j];
            if (ov::max(std::fabs(p_prog.lo[size_t(w)]), std::fabs(p_prog.hi[size_t(w)])) > double(Cascade::WARP_LIMIT)) {
                return ov::format("climate: a warp moves a note by up to %s voxels, over %s.", ov::max(std::fabs(p_prog.lo[size_t(w)]), std::fabs(p_prog.hi[size_t(w)])), Cascade::WARP_LIMIT);
            }
        }
    }
    return ov::str();
}

} // namespace

ov::str bound_program(const Climate::Ground &p_ground, LevelProgram &r_program) {
    const size_t n = r_program.nodes.size();
    r_program.lo.assign(n, 0.0);
    r_program.hi.assign(n, 0.0);
    std::vector<bool> up(n, false); // reads up: the output's lower side is a carve, unbounded
    for (size_t i = 0; i < n; i++) {
        value_bound(r_program, i, r_program.lo[i], r_program.hi[i]);
        up[i] = r_program.nodes[i].op == Op::UP;
        for (const int in : r_program.nodes[i].in) {
            up[i] = up[i] || up[size_t(in)];
        }
    }
    r_program.step.assign(size_t(r_program.top + 1), 0.0);
    std::vector<double> d(n, 0.0);
    for (int lod = 0; lod <= r_program.top; lod++) {
        for (size_t i = 0; i < n; i++) {
            d[i] = step(r_program, d, i, lod);
        }
        r_program.step[size_t(lod)] = d[size_t(r_program.output)];
    }
    const size_t o = size_t(r_program.output);
    const double reach = up[o] ? r_program.hi[o] : ov::max(std::fabs(r_program.lo[o]), std::fabs(r_program.hi[o]));
    if (!(reach <= double(p_ground.amp_cap))) {
        return ov::format("climate: the ground reaches %s voxels from its base, over the cap of %s.", reach, p_ground.amp_cap);
    }
    return validate_nodes(r_program);
}

} // namespace Terrain
