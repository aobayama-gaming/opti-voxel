#include "climate.hpp"
#include "ov/math.hpp"

#include <algorithm>
#include <cmath>

namespace Climate {

using Cascade::dadd;
using Cascade::dflush;
using Cascade::dmul;
using Cascade::dsub;
using Cascade::flush_value;

namespace {

// The default's (1 - p_x)^2 for p_x < 1, else 0 (p_x: the kernels' sum).
Float4 fill(const Float4 &p_x) {
    if (!(p_x.w < 1.0f)) {
        return Float4{ 0.0f, 0.0f, 0.0f, 0.0f };
    }
    const Float4 t{ -p_x.x, -p_x.y, -p_x.z, 1.0f - p_x.w };
    return dmul(t, t);
}

// One axis's factor: the smoothstep of clamp((1 - u^2) KERNEL_FADE, 0, 1), u = (x - centre) / span.
Float4 axis_factor(const Axis &p_axis, const Float4 &p_x) {
    const Float4 u = dflush(Float4{ p_x.x * p_axis.inv_span, p_x.y * p_axis.inv_span, p_x.z * p_axis.inv_span, (p_x.w - p_axis.centre) * p_axis.inv_span });
    const Float4 uu = dmul(u, u);
    const float raw = flush_value((1.0f - uu.w) * KERNEL_FADE);
    if (!(raw > 0.0f)) {
        return Float4{ 0.0f, 0.0f, 0.0f, 0.0f };
    }
    if (raw >= 1.0f) {
        return Float4{ 0.0f, 0.0f, 0.0f, 1.0f };
    }
    const Float4 s = dflush(Float4{ -uu.x * KERNEL_FADE, -uu.y * KERNEL_FADE, -uu.z * KERNEL_FADE, raw });
    // s^2 (3 - 2 s); the x2 scalings are exact.
    const Float4 t{ s.x * -2.0f, s.y * -2.0f, s.z * -2.0f, 3.0f - s.w * 2.0f };
    return dmul(dmul(s, s), t);
}

// Biome p_biome's kernel at the field channels p_fields: its axes' factors multiplied in order.
Float4 kernel(const Biome &p_biome, const Float4 p_fields[MAX_FIELDS]) {
    Float4 k{ 0.0f, 0.0f, 0.0f, 1.0f };
    for (const Axis &a : p_biome.at) {
        k = dmul(k, axis_factor(a, p_fields[a.field]));
    }
    return k;
}

} // namespace

int Table::field_index(const ov::str &p_name) const {
    for (size_t i = 0; i < fields.size(); i++) {
        if (fields[i].name == p_name) {
            return int(i);
        }
    }
    return -1;
}

void apply_terms(const Table &p_table, const float p_slots[Cascade::SLOTS], Float4 r_fields[MAX_FIELDS]) {
    const Float4 ground = to_float4(p_slots);
    for (size_t i = 0; i < p_table.fields.size(); i++) {
        Float4 f = to_float4(p_slots + Cascade::field_slot(int(i)));
        for (const Term &t : p_table.fields[i].terms) {
            const Float4 s = t.source == GROUND ? ground : r_fields[t.source];
            f = dflush(Float4{ f.x + s.x * t.coef, f.y + s.y * t.coef, f.z + s.z * t.coef, f.w + s.w * t.coef });
        }
        r_fields[i] = f;
    }
}

bool plays(const Amp &p_amp, const Cascade::Band &p_band) {
    return p_amp.k == p_band.k && p_amp.seed == p_band.seed && p_amp.stretch == p_band.stretch;
}

const Amp *find_amp(const Biome &p_biome, const Cascade::Band &p_band) {
    for (const Amp &a : p_biome.amps) {
        if (plays(a, p_band)) {
            return &a;
        }
    }
    return nullptr;
}

float biome_amp(const Biome &p_biome, const Cascade::Band &p_band) {
    const Amp *a = find_amp(p_biome, p_band);
    return a ? a->amp : p_band.amp;
}

bool Table::has_stretch() const {
    return std::any_of(notes.begin(), notes.end(), [](const Cascade::Band &p_b) { return p_b.stretch > 0; });
}

Shape band_shape(const Biome &p_biome, const Cascade::Band &p_band) {
    const Amp *a = find_amp(p_biome, p_band);
    return a && a->has_shape ? a->shape : Shape();
}

float max_amp(const Table &p_table, const Cascade::Band &p_band) {
    float m = p_table.biomes.empty() ? p_band.amp : 0.0f;
    for (const Climate::Biome &b : p_table.biomes) {
        m = ov::max(m, biome_amp(b, p_band) * shape_bound(band_shape(b, p_band)));
    }
    return m;
}

namespace {

// Field p_field's step at a LOD-L chunk's seams: its largest octave inserted on Λ_{L+1} or Λ_{L+2}, plus each term's
// |coef| times its source's step (the ground's p_ground_amp, or an earlier field's). A LOD-L chunk's samples read the
// climate of Λ_{L+1} before its insertions, the fields down to the octave inserted on Λ_{L+2}: a coarser neighbour
// lacks that one, a finer neighbour adds the one on Λ_{L+1} (octave_step's two, one level higher).
double field_step(const Table &p_table, int p_field, int p_lod, double p_ground_amp) {
    const Field &f = p_table.fields[size_t(p_field)];
    double amp = 0.0;
    for (const Cascade::Band &b : f.bands) {
        if (b.k - ExactNoise::BAND_SHIFT == p_lod + 1 || b.k - ExactNoise::BAND_SHIFT == p_lod + 2) { // inserted on Λ_{L+1} / Λ_{L+2}
            amp = ov::max(amp, double(b.amp));
        }
    }
    for (const Term &t : f.terms) {
        amp += std::fabs(double(t.coef)) * (t.source == GROUND ? p_ground_amp : field_step(p_table, t.source, p_lod, p_ground_amp));
    }
    return amp;
}

} // namespace

double ground_step(const Table &p_table, int p_lod, double p_ground_amp) {
    // ground = h + sum_b w_b C_b(h). The weights sum to 1, so the h part moves by at most (1 + max |C_b'|) times h's
    // step. The weights' own change moves it by |dw_b| |C_b|. A kernel's |dk/dq| <= 1.54 (4 q (1 - q^2) at q^2 = 1/3)
    // and |dq| <= 2 sum |du| (|u| <= 1 where k > 0), so |dk_b| <= KERNEL_SLOPE sum |1 / span| field_step = K_b; the
    // default's |dk_d| <= 2 sum K; w = k / S with S >= 3/4, w <= 1, |dS| <= 3 sum K: |dw| <= (4/3)(|dk| + |dS|) <=
    // BLEND_SLOPE sum K for every biome, the default too.
    double slope = 0.0, kernels = 0.0, curves = 0.0;
    for (const Biome &b : p_table.biomes) {
        for (const Axis &a : b.at) {
            kernels += KERNEL_SLOPE * std::fabs(double(a.inv_span)) * field_step(p_table, a.field, p_lod, p_ground_amp);
        }
        if (!b.curve.empty()) {
            slope = std::max(slope, double(curve_slope_bound(b.curve)));
            curves += double(curve_bound(b.curve));
        }
    }
    return (1.0 + slope) * p_ground_amp + BLEND_SLOPE * kernels * curves;
}

void fields_at(const Table &p_table, const Cascade::Spec &p_combined, const Vector3i64 &p_p, int p_m, Float4 r_fields[MAX_FIELDS]) {
    float slots[Cascade::SLOTS];
    Cascade::point_slots(p_combined, p_p, p_m, slots, true);
    apply_terms(p_table, slots, r_fields);
}

void weights(const Table &p_table, const Float4 p_fields[MAX_FIELDS], Float4 r_w[MAX_BIOMES]) {
    for (int b = 0; b < MAX_BIOMES; b++) {
        r_w[b] = Float4{ 0.0f, 0.0f, 0.0f, 0.0f };
    }
    const size_t n = p_table.biomes.size();
    if (n == 1) {
        r_w[0] = Float4{ 0.0f, 0.0f, 0.0f, 1.0f };
        return;
    }
    Float4 sum{ 0.0f, 0.0f, 0.0f, 0.0f };
    for (size_t b = 0; b + 1 < n; b++) {
        r_w[b] = kernel(p_table.biomes[b], p_fields);
        sum = dadd(sum, r_w[b]);
    }
    if (n > 0) {
        r_w[n - 1] = fill(sum);
        const Float4 s = dadd(sum, r_w[n - 1]);
        // 1 / s: det_rsqrt(s)^2 and one Newton step (1 / 1 comes out exactly 1: a biome alone at its centre is whole).
        const float r = ExactNoise::det_rsqrt(s.w);
        const float inv0 = r * r;
        const float inv = inv0 * (2.0f - s.w * inv0);
        const Float4 rcp = dflush(Float4{ -(s.x * inv) * inv, -(s.y * inv) * inv, -(s.z * inv) * inv, inv });
        for (size_t b = 0; b < n; b++) {
            r_w[b] = dmul(r_w[b], rcp);
            r_w[b].w = ov::min(r_w[b].w, 1.0f); // rounding's last bit, never over 1
        }
    }
}

void state(const Table &p_table, const float p_slots[Cascade::SLOTS], State &r_state) {
    apply_terms(p_table, p_slots, r_state.fields);
    weights(p_table, r_state.fields, r_state.w);
}

int pick(const Table &p_table, const Float4 p_w[MAX_BIOMES], const Vector3i64 &p_p, int p_m) {
    int best[MAX_BIOMES];
    int count = 0;
    for (int b = 0; b < int(p_table.biomes.size()); b++) {
        if (count > 0 && ExactNoise::float_bits(p_w[b].w) == ExactNoise::float_bits(p_w[best[0]].w)) {
            best[count++] = b;
        } else if (count == 0 || p_w[b].w > p_w[best[0]].w) {
            best[0] = b;
            count = 1;
        }
    }
    if (count == 0) {
        return -1;
    }
    // The absolute voxel: the same point seen from two LODs breaks a tie the same way.
    const int64_t step = int64_t(1) << p_m;
    const Vector3i64 voxel(p_p.x * step, p_p.y * step, p_p.z * step);
    return best[ExactNoise::hash_cell(voxel, PICK_SEED) % uint32_t(count)];
}

SampleCarry sample_carry(const Table &p_table) {
    SampleCarry s;
    std::fill(std::begin(s.curve), std::end(s.curve), -1);
    for (size_t b = 0; b < p_table.biomes.size(); b++) {
        if (!p_table.biomes[b].curve.empty()) {
            s.curve[b] = s.count++;
        }
    }
    // Without fields the weights are constant: the last biome takes everything.
    if (!p_table.fields.empty() && !p_table.biomes.empty()) {
        s.pick = s.count++;
    }
    return s;
}

int pick_slots(const Table &p_table, const float p_slots[Cascade::SLOTS]) {
    const int c = sample_carry(p_table).pick;
    return c >= 0 ? int(p_slots[Cascade::carry_slot(c)]) : int(p_table.biomes.size()) - 1;
}

ov::Color srgb_to_linear(uint32_t p_rgb) {
    return ov::Color::hex((p_rgb << 8) | 0xFFu).srgb_to_linear();
}

Material::Params biome_material(const Biome &p_biome) {
    Material::Params m;
    m.albedo = p_biome.surface.flat;
    m.steep = p_biome.surface.steep;
    m.roughness = p_biome.surface.roughness;
    m.pattern = Material::PATTERN_GROUND;
    m.debug = srgb_to_linear(p_biome.debug_color);
    return m;
}

} // namespace Climate
