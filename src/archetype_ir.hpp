#pragma once

#include "exact_noise.hpp"
#include "world_coord.hpp"

#include <godot_cpp/variant/string.hpp>

#include <cstdint>
#include <vector>

// Archetype IR: an SDF as a linear SSA program, the one source of its GPU code (archetype_glsl.hpp,
// generated GLSL) and of its CPU evaluation (run(), an interpreter: archetypes are data authored at
// runtime, and C++ can't be compiled in the game). Both back-ends walk the same instruction list, one
// statement per instruction, so their operations come in the same order and give the same bits
// (FrameSpike probes it: arch_* in archetype_probes.cpp).
//
// Registers are dual numbers, Float4 (x, y, z: gradient, w: value), forward-mode derivatives: every
// program yields a value and its gradient, like terrain_sdf. Units: values in LOD0 voxels, gradients per
// voxel of the evaluated LOD (2^lod LOD0 voxels, so an INPUT's gradient is 2^lod). Every LOD factor is an
// exact power of two: values are the same bits at every LOD, gradients scale exactly (unit-tested).
//
// Exactness rules (exact_noise.hpp): only + - *, compares and selects, floor, abs, det_rsqrt and the exact
// noise; no division, no sqrt, no mix / dot / length / normalize; constants as raw bits; no denormal. The
// last one depends on the data: a point is in the exact domain when every register component is 0 or
// within [2^-60, 2^60] (any product of two is then a normal float) and every RSQRT reads a value > 0;
// outside it the GPU may flush or round differently (exact_domain()). Builders keep authored archetypes
// inside it.
//
// The sign of a zero is not kept: the GPU compiler folds x + 0.0 to x (no SignedZeroInfNanPreserve), so a
// zero may come out -0 on one side and +0 on the other. The IR never lets it reach a non-zero value (no
// division, compares see -0 == +0, RSQRT only on x > 0), and comparisons of results go through
// zero_canonical().
//
// A register whose gradient is statically zero (a constant, a floor) is "flat": both back-ends skip its
// gradient terms, the same way.
namespace Archetype {

using ExactNoise::Float4;
using Reg = int32_t;

enum class Op : uint8_t {
    CONST, // `value`
    PARAM, // params[index]: an instance parameter, a constant once flattened
    INPUT, // float(sat_diff(p[axis], anchors[index][axis], INPUT_LIMIT)) + p.frac[axis]; gradient 2^lod on axis
    ADD,
    SUB,
    MUL,
    NEG,
    ABS, // value |a|, gradient negated where a < 0
    MIN, // the whole register with the smaller value (b < a ? b : a)
    MAX, // b > a ? b : a
    SELECT, // a < b ? c : d
    FLOOR, // flat
    RSQRT, // det_rsqrt(a), a a normal float > 0 (builders guard it)
    DIST, // a / |grad a|: first-order distance, gradient normalised (to 2^lod); grad a must not be flat
    PERLIN3, // one exact-noise octave times `value` (amp): wavelength 2^k voxels, seed, lattice offset
    PERLIN2, // the same over (x, z)
};

struct Instr {
    Op op = Op::CONST;
    uint8_t axis = 0; // INPUT
    bool flat = false; // gradient statically zero (set by Builder::emit, checked by validate)
    Reg a = -1;
    Reg b = -1;
    Reg c = -1;
    Reg d = -1;
    int32_t index = 0; // INPUT anchor, PARAM index
    float value = 0.0f; // CONST value, noise amplitude
    // Noise: ExactNoise::Octave's fields (amp is `value`).
    int k = 0;
    uint32_t seed = 0;
    Vector3i64 offset;
    // Recorded for later steps, unused here: the band (wavelength in LOD0 voxels, 0 = none) of the band
    // rule, and the cost of cost-based batching (op_cost).
    float band = 0.0f;
};

// Saturation of INPUT's integer offset (voxels): exact below 2^24, rounded identically above.
inline constexpr int32_t INPUT_LIMIT = 1 << 30;
// Floor of |grad|^2 in DIST and the output normalisation, per LOD0 voxel (times 4^lod).
inline constexpr int GRAD_FLOOR_EXP = -40;
inline constexpr int MAX_REGISTERS = 4096;

struct Program {
    std::vector<Instr> code;
    std::vector<VoxelPos> anchors;
    std::vector<float> params;
    Reg output = -1;
    // Epilogue: value * out_scale (e.g. voxels to metres); gradient normalised to length 1 if `normalize`.
    float out_scale = 1.0f;
    bool normalize = false;

    // Empty if the program is well formed: operands before their use, register budget, flat flags,
    // constants finite and not denormal, noise amplitudes and gradient factors amp * 2^(lod - k) in
    // [2^-16, 2^40] at the LODs [0, p_lod_count) (ExactNoise::in_exact_range).
    String validate(int p_lod_count) const;
    // Sum of op_cost: a relative GPU cost (multiply-adds, a noise octave counted from its body).
    int cost() const;
};

int op_cost(Op p_op);

// Whether p_in's gradient is statically zero, given the instructions before it.
bool is_flat(const Instr &p_in, const std::vector<Instr> &p_code);

// Number of register operands of an op (a, b, c, d in that order).
int op_arity(Op p_op);

// The output of p_program at p_pos for a chunk of LOD p_lod: (gradient, value) after the epilogue.
Float4 run(const Program &p_program, const WorldPos &p_pos, int p_lod);

// Every register (before the epilogue), for tests and probes.
void run_registers(const Program &p_program, const WorldPos &p_pos, int p_lod, std::vector<Float4> &r_regs);
Float4 finish(const Program &p_program, const Float4 &p_out, int p_lod);

// Every component 0 or within [2^-60, 2^60], every RSQRT input > 0 (see the exactness rules above).
bool exact_domain(const Program &p_program, const std::vector<Float4> &p_regs);
// The bits of a float with -0 read as +0.
inline uint32_t zero_canonical(uint32_t p_bits) {
    return (p_bits << 1) == 0u ? 0u : p_bits;
}
// FNV-1a over every component of every register (zero_canonical): what the GLSL trace variant computes.
uint32_t trace_hash(const std::vector<Float4> &p_regs);

// Local frame: three registers holding the local x, y, z (LOD0 voxels).
struct Local {
    Reg x = -1;
    Reg y = -1;
    Reg z = -1;
};

// Appends instructions to a Program. The derived shapes are built from the ops above (no new op); sizes
// and positions are in LOD0 voxels, in the frame of the `Local` they get.
class Builder {
public:
    Program program;

    int anchor(const VoxelPos &p_anchor);
    Reg constant(float p_value);
    Reg param(int p_index);
    Reg input(int p_axis, int p_anchor = 0);
    // The position relative to anchor p_anchor (x, y, z inputs).
    Local local(int p_anchor = 0);
    Reg add(Reg p_a, Reg p_b);
    Reg sub(Reg p_a, Reg p_b);
    Reg mul(Reg p_a, Reg p_b);
    Reg neg(Reg p_a);
    Reg abs(Reg p_a);
    Reg min(Reg p_a, Reg p_b);
    Reg max(Reg p_a, Reg p_b);
    Reg select(Reg p_a, Reg p_b, Reg p_c, Reg p_d);
    Reg floor(Reg p_a);
    Reg rsqrt(Reg p_a);
    Reg dist(Reg p_a);
    Reg perlin3(int p_k, uint32_t p_seed, const Vector3i64 &p_offset, float p_amp);
    Reg perlin2(int p_k, uint32_t p_seed, const Vector3i64 &p_offset, float p_amp);

    // With a constant operand.
    Reg add(Reg p_a, float p_b) { return add(p_a, constant(p_b)); }
    Reg mul(Reg p_a, float p_b) { return mul(p_a, constant(p_b)); }

    Reg clamp(Reg p_a, float p_lo, float p_hi);
    // sqrt(x^2 + y^2 + z^2) as d2 * rsqrt(max(d2, 2^-40)) (p_z < 0: 2D).
    Reg length(Reg p_x, Reg p_y, Reg p_z = -1);
    Reg sphere(const Local &p_p, float p_radius);
    Reg box(const Local &p_p, float p_hx, float p_hy, float p_hz);
    // Vertical segment from y = 0 to y = p_height, radius p_radius.
    Reg capsule(const Local &p_p, float p_height, float p_radius);
    // Vertical, centred on y = 0, half height p_half_height.
    Reg cylinder(const Local &p_p, float p_half_height, float p_radius);
    // n . p - p_d (n of length 1).
    Reg plane(const Local &p_p, float p_nx, float p_ny, float p_nz, float p_d);
    // Polynomial smooth min of radius p_k (> 0): min(a, b) - p_k / 4 at worst, gradient h grad a + (1 - h) grad b.
    Reg smooth_min(Reg p_a, Reg p_b, float p_k);
    Reg smooth_max(Reg p_a, Reg p_b, float p_k);
    // Sum of the set's octaves from 0, like ExactNoise::fbm (same value bits).
    Reg fbm(const ExactNoise::OctaveSet &p_set);

    Program finish(Reg p_output);

    // Appends an instruction as is (operands already in this program), its flat flag recomputed.
    Reg emit(const Instr &p_instr);
};

// A fuzzed program for the probes: p_ops random instructions (every op, guarded RSQRT, inputs scaled down so
// values stay moderate), anchors drawn from p_anchors. Deterministic in p_seed.
Program random_program(uint32_t p_seed, const std::vector<VoxelPos> &p_anchors, int p_ops);

} // namespace Archetype
