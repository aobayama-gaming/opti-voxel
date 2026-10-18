// Archetype::Builder (the ops and the shapes derived from them) and the program fuzzer of the probes.
#include "archetype_ir.hpp"

#include <cmath>

namespace Archetype {

using ExactNoise::exact_pow2;

Reg Builder::emit(const Instr &p_instr) {
    Instr in = p_instr;
    in.flat = is_flat(in, program.code);
    program.code.push_back(in);
    return Reg(program.code.size() - 1);
}

int Builder::anchor(const VoxelPos &p_anchor) {
    for (int i = 0; i < int(program.anchors.size()); i++) {
        if (program.anchors[i] == p_anchor) {
            return i;
        }
    }
    program.anchors.push_back(p_anchor);
    return int(program.anchors.size() - 1);
}

Reg Builder::constant(float p_value) {
    Instr in;
    in.op = Op::CONST;
    in.value = p_value;
    return emit(in);
}

Reg Builder::param(int p_index) {
    Instr in;
    in.op = Op::PARAM;
    in.index = p_index;
    return emit(in);
}

Reg Builder::input(int p_axis, int p_anchor) {
    if (program.anchors.empty()) {
        anchor(VoxelPos());
    }
    Instr in;
    in.op = Op::INPUT;
    in.axis = uint8_t(p_axis);
    in.index = p_anchor;
    return emit(in);
}

Local Builder::local(int p_anchor) {
    return Local{ input(0, p_anchor), input(1, p_anchor), input(2, p_anchor) };
}

namespace {

Instr instr(Op p_op, Reg p_a, Reg p_b = -1, Reg p_c = -1, Reg p_d = -1) {
    Instr in;
    in.op = p_op;
    in.a = p_a;
    in.b = p_b;
    in.c = p_c;
    in.d = p_d;
    return in;
}

Instr noise(Op p_op, int p_k, uint32_t p_seed, const Vector3i64 &p_offset, float p_amp) {
    Instr in;
    in.op = p_op;
    in.k = p_k;
    in.seed = p_seed;
    in.offset = p_offset;
    in.value = p_amp;
    in.band = float(std::ldexp(1.0, p_k));
    return in;
}

} // namespace

Reg Builder::add(Reg p_a, Reg p_b) {
    return emit(instr(Op::ADD, p_a, p_b));
}
Reg Builder::sub(Reg p_a, Reg p_b) {
    return emit(instr(Op::SUB, p_a, p_b));
}
Reg Builder::mul(Reg p_a, Reg p_b) {
    return emit(instr(Op::MUL, p_a, p_b));
}
Reg Builder::neg(Reg p_a) {
    return emit(instr(Op::NEG, p_a));
}
Reg Builder::abs(Reg p_a) {
    return emit(instr(Op::ABS, p_a));
}
Reg Builder::min(Reg p_a, Reg p_b) {
    return emit(instr(Op::MIN, p_a, p_b));
}
Reg Builder::max(Reg p_a, Reg p_b) {
    return emit(instr(Op::MAX, p_a, p_b));
}
Reg Builder::select(Reg p_a, Reg p_b, Reg p_c, Reg p_d) {
    return emit(instr(Op::SELECT, p_a, p_b, p_c, p_d));
}
Reg Builder::floor(Reg p_a) {
    return emit(instr(Op::FLOOR, p_a));
}
Reg Builder::rsqrt(Reg p_a) {
    return emit(instr(Op::RSQRT, p_a));
}
Reg Builder::dist(Reg p_a) {
    return emit(instr(Op::DIST, p_a));
}
Reg Builder::perlin3(int p_k, uint32_t p_seed, const Vector3i64 &p_offset, float p_amp) {
    return emit(noise(Op::PERLIN3, p_k, p_seed, p_offset, p_amp));
}
Reg Builder::perlin2(int p_k, uint32_t p_seed, const Vector3i64 &p_offset, float p_amp) {
    return emit(noise(Op::PERLIN2, p_k, p_seed, p_offset, p_amp));
}

// Every builder emits one instruction per statement: C++ leaves the order of a call's arguments
// unspecified, and the instruction order (the GLSL text, its FNV pin) must not depend on the compiler.

Reg Builder::clamp(Reg p_a, float p_lo, float p_hi) {
    const Reg lo = max(p_a, constant(p_lo));
    return min(lo, constant(p_hi));
}

Reg Builder::length(Reg p_x, Reg p_y, Reg p_z) {
    const Reg xx = mul(p_x, p_x);
    Reg d2 = add(xx, mul(p_y, p_y));
    if (p_z >= 0) {
        d2 = add(d2, mul(p_z, p_z));
    }
    // The zero guard: below 2^-40 voxels^2 the length reads d2 * 2^20 (< 2^-20 voxels).
    return mul(d2, rsqrt(max(d2, constant(exact_pow2(GRAD_FLOOR_EXP)))));
}

Reg Builder::sphere(const Local &p_p, float p_radius) {
    const Reg len = length(p_p.x, p_p.y, p_p.z);
    return sub(len, constant(p_radius));
}

Reg Builder::box(const Local &p_p, float p_hx, float p_hy, float p_hz) {
    const Reg zero = constant(0.0f);
    const Reg q[3] = { abs(p_p.x), abs(p_p.y), abs(p_p.z) };
    const float h[3] = { p_hx, p_hy, p_hz };
    Reg d[3];
    Reg pos[3];
    for (int i = 0; i < 3; i++) {
        d[i] = sub(q[i], constant(h[i]));
        pos[i] = max(d[i], zero);
    }
    const Reg outside = length(pos[0], pos[1], pos[2]);
    const Reg inner = max(d[0], max(d[1], d[2]));
    return add(outside, min(inner, zero));
}

Reg Builder::capsule(const Local &p_p, float p_height, float p_radius) {
    const Reg y = sub(p_p.y, clamp(p_p.y, 0.0f, p_height));
    const Reg len = length(p_p.x, y, p_p.z);
    return sub(len, constant(p_radius));
}

Reg Builder::cylinder(const Local &p_p, float p_half_height, float p_radius) {
    const Reg zero = constant(0.0f);
    const Reg len = length(p_p.x, p_p.z);
    const Reg dr = sub(len, constant(p_radius));
    const Reg ay = abs(p_p.y);
    const Reg dy = sub(ay, constant(p_half_height));
    const Reg pr = max(dr, zero);
    const Reg outside = length(pr, max(dy, zero));
    const Reg inner = max(dr, dy);
    return add(outside, min(inner, zero));
}

Reg Builder::plane(const Local &p_p, float p_nx, float p_ny, float p_nz, float p_d) {
    Reg sum = -1;
    const Reg axes[3] = { p_p.x, p_p.y, p_p.z };
    const float n[3] = { p_nx, p_ny, p_nz };
    for (int i = 0; i < 3; i++) {
        if (n[i] != 0.0f) {
            const Reg term = n[i] == 1.0f ? axes[i] : mul(axes[i], n[i]);
            sum = sum < 0 ? term : add(sum, term);
        }
    }
    return sub(sum, constant(p_d));
}

Reg Builder::smooth_min(Reg p_a, Reg p_b, float p_k) {
    // h = clamp(0.5 + (b - a) / (2k), 0, 1); b + (a - b) h - k h (1 - h). 1 / (2k) is a constant.
    const Reg h = clamp(add(mul(sub(p_b, p_a), float(0.5 / double(p_k))), 0.5f), 0.0f, 1.0f);
    const Reg mixed = add(p_b, mul(sub(p_a, p_b), h));
    const Reg one = constant(1.0f);
    return sub(mixed, mul(mul(h, sub(one, h)), p_k));
}

Reg Builder::smooth_max(Reg p_a, Reg p_b, float p_k) {
    const Reg na = neg(p_a);
    return neg(smooth_min(na, neg(p_b), p_k));
}

Reg Builder::fbm(const ExactNoise::OctaveSet &p_set) {
    Reg sum = constant(0.0f);
    for (const ExactNoise::Octave &o : p_set.octaves) {
        sum = add(sum, p_set.is_2d ? perlin2(o.k, o.seed, o.offset, o.amp) : perlin3(o.k, o.seed, o.offset, o.amp));
    }
    return sum;
}

Program Builder::finish(Reg p_output) {
    program.output = p_output;
    return program;
}

namespace {

// Deterministic stream for the fuzzer.
struct Rng {
    uint32_t seed;
    uint32_t n = 0;
    uint32_t next() { return ExactNoise::pcg3d(ExactNoise::UVec3{ seed, n++, 0xF022u }).x; }
    int below(int p_n) { return int(next() % uint32_t(p_n)); }
    // +-(1 + u) * 2^e, e in [p_e0, p_e0 + p_span).
    float value(int p_e0, int p_span) {
        const float m = 1.0f + float(next() >> 16) * exact_pow2(-16);
        const int e = p_e0 + below(p_span);
        const float sign = (next() & 1u) ? -1.0f : 1.0f;
        return float(std::ldexp(double(m), e)) * sign;
    }
};

struct Fuzz {
    Builder b;
    Rng rng;
    std::vector<Reg> regs;
    std::vector<float> mag; // rough bound of |value|, keeps products moderate

    void push(Reg p_r, float p_mag) {
        regs.push_back(p_r);
        mag.push_back(p_mag);
    }
    // Mostly a recent register (deep chains), sometimes any.
    int pick() {
        const int n = int(regs.size());
        return rng.below(10) < 7 ? n - 1 - rng.below(MIN(n, 6)) : rng.below(n);
    }
    void input(int p_anchors) {
        const int e = rng.below(13);
        const int axis = rng.below(3);
        const int anchor = rng.below(p_anchors);
        push(b.mul(b.input(axis, anchor), exact_pow2(-e)), std::ldexp(1.0f, 18 - e));
    }
    void noise(bool p_3d) {
        const int k = 2 + rng.below(13);
        const float amp = std::fabs(rng.value(0, 3));
        Vector3i64 offset;
        offset.x = int64_t(rng.next()) << 8;
        offset.y = -int64_t(rng.next());
        offset.z = int64_t(rng.next()) * 3;
        const uint32_t seed = rng.next();
        push(p_3d ? b.perlin3(k, seed, offset, amp) : b.perlin2(k, seed, offset, amp), 2.0f * amp);
    }
    void step(int p_anchors);
};

void Fuzz::step(int p_anchors) {
    const int ia = pick();
    const int ib = pick();
    const Reg a = regs[ia];
    const Reg r_b = regs[ib];
    const float ma = mag[ia];
    const float mb = mag[ib];
    switch (rng.below(15)) {
        case 0:
            push(b.constant(rng.value(-4, 8)), 16.0f);
            break;
        case 1:
            input(p_anchors);
            break;
        case 2:
            push(b.add(a, r_b), ma + mb);
            break;
        case 3:
            push(b.sub(a, r_b), ma + mb);
            break;
        case 4:
            if (ma * mb < 1048576.0f) {
                push(b.mul(a, r_b), ma * mb);
            } else {
                push(b.sub(r_b, a), ma + mb);
            }
            break;
        case 5:
            push(b.neg(a), ma);
            break;
        case 6:
            push(b.abs(a), ma);
            break;
        case 7:
            push(b.min(a, r_b), MAX(ma, mb));
            break;
        case 8:
            push(b.max(a, r_b), MAX(ma, mb));
            break;
        case 9: {
            const int ic = pick();
            const int id = pick();
            push(b.select(a, r_b, regs[ic], regs[id]), MAX(mag[ic], mag[id]));
            break;
        }
        case 10:
            push(b.floor(a), ma + 1.0f);
            break;
        case 11: {
            const Reg m = b.abs(a);
            push(b.rsqrt(b.max(m, b.constant(exact_pow2(-8)))), 16.0f);
            break;
        }
        case 12:
            if (!b.program.code[a].flat) {
                push(b.dist(a), ma * 1024.0f);
            } else {
                push(b.add(r_b, a), ma + mb);
            }
            break;
        default:
            noise(rng.below(2) == 0);
            break;
    }
}

} // namespace

Program random_program(uint32_t p_seed, const std::vector<VoxelPos> &p_anchors, int p_ops) {
    Fuzz f;
    f.rng.seed = p_seed;
    for (const VoxelPos &a : p_anchors) {
        f.b.anchor(a);
    }
    const int anchors = MAX(int(p_anchors.size()), 1);
    f.input(anchors);
    f.input(anchors);
    f.noise(true);
    while (int(f.b.program.code.size()) < p_ops) {
        f.step(anchors);
    }
    return f.b.finish(f.regs.back());
}

} // namespace Archetype
