#include "climate.hpp"

#include <algorithm>
#include <cmath>

namespace Climate {

using Cascade::dflush;
using Cascade::dmul;
using Cascade::dsub;
using Cascade::flush_value;

namespace {

Float4 edge_weight(const Edge &p_edge, const Float4 &p_x) {
    const float raw = flush_value((p_x.w - p_edge.anchor) * p_edge.scale);
    if (!(raw > 0.0f)) {
        return Float4{ 0.0f, 0.0f, 0.0f, 0.0f };
    }
    if (raw >= 1.0f) {
        return Float4{ 0.0f, 0.0f, 0.0f, 1.0f };
    }
    const Float4 s = dflush(Float4{ p_x.x * p_edge.scale, p_x.y * p_edge.scale, p_x.z * p_edge.scale, raw });
    // s^2 (3 - 2 s); the x2 scalings are exact.
    const Float4 q{ s.x * -2.0f, s.y * -2.0f, s.z * -2.0f, 3.0f - s.w * 2.0f };
    return dmul(dmul(s, s), q);
}

} // namespace

Float4 edges_value(const std::vector<Edge> &p_edges, const Float4 p_fields[MAX_FIELDS]) {
    Float4 r{ 0.0f, 0.0f, 0.0f, 1.0f };
    for (const Edge &e : p_edges) {
        r = dmul(r, edge_weight(e, p_fields[e.field]));
    }
    return r;
}

void apply_terms(const Table &p_table, const float p_slots[Cascade::SLOTS], Float4 r_fields[MAX_FIELDS]) {
    for (int i = 0; i < MAX_FIELDS; i++) {
        r_fields[i] = Float4{ 0.0f, 0.0f, 0.0f, 0.0f };
    }
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

float biome_amp(const Biome &p_biome, int p_field, const Cascade::Band &p_band) {
    for (const Amp &e : p_biome.amps) {
        if (e.field == p_field && e.k == p_band.k) {
            return float(double(e.mult) * double(p_band.amp));
        }
    }
    return p_band.amp;
}

const Amp *find_amp(const Biome &p_biome, int p_field, int p_k) {
    for (const Amp &a : p_biome.amps) {
        if (a.field == p_field && a.k == p_k) {
            return &a;
        }
    }
    return nullptr;
}

Shape band_shape(const Table &p_table, const Biome *p_biome, int p_field, int p_k) {
    const Amp *a = p_biome ? find_amp(*p_biome, p_field, p_k) : nullptr;
    if (a && a->has_shape) {
        return a->shape;
    }
    Shape own;
    if (p_field != GROUND) {
        const Field &f = p_table.fields[size_t(p_field)];
        for (size_t i = 0; i < f.bands.size(); i++) {
            own = f.bands[i].k == p_k ? f.shapes[i] : own;
        }
    }
    return own;
}

float max_amp(const Table &p_table, int p_field, const Cascade::Band &p_band) {
    float m = p_table.biomes.empty() ? p_band.amp * shape_bound(band_shape(p_table, nullptr, p_field, p_band.k)) : 0.0f;
    for (const Climate::Biome &b : p_table.biomes) {
        m = MAX(m, biome_amp(b, p_field, p_band) * shape_bound(band_shape(p_table, &b, p_field, p_band.k)));
    }
    return m;
}

namespace {

// Field p_field's step between LOD p_lod and p_lod + 1: its largest L+2 / L+3 octave, plus each term's |coef| times its
// source's step (the ground's p_ground_amp, or an earlier field's).
double field_step(const Table &p_table, int p_field, int p_lod, double p_ground_amp) {
    const Field &f = p_table.fields[size_t(p_field)];
    double amp = 0.0;
    for (const Cascade::Band &b : f.bands) {
        if (b.k == p_lod + 2 || b.k == p_lod + 3) {
            amp = MAX(amp, double(max_amp(p_table, p_field, b)));
        }
    }
    for (const Term &t : f.terms) {
        amp += std::fabs(double(t.coef)) * (t.source == GROUND ? p_ground_amp : field_step(p_table, t.source, p_lod, p_ground_amp));
    }
    return amp;
}

} // namespace

double ground_step(const Table &p_table, int p_lod, double p_ground_amp) {
    double amp = p_ground_amp;
    for (const HeightTerm &t : p_table.heights) {
        amp += std::fabs(double(t.coef)) * (t.has_curve ? double(curve_slope_bound(t.curve)) : 1.0) * field_step(p_table, t.field, p_lod, p_ground_amp);
        if (t.region >= 0) {
            // r v steps by at most |r| |dv| + |v| |dr|, |r| <= 1: the term's own step above, plus its bound times the
            // region's (a smoothstep's slope is at most 1.5 |scale|, a product of values in [0, 1] adds its factors', and
            // r stays in [0, 1]: at most 1).
            double region = 0.0;
            for (const Edge &e : p_table.regions[size_t(t.region)].edges) {
                region += 1.5 * std::fabs(double(e.scale)) * field_step(p_table, e.field, p_lod, p_ground_amp);
            }
            amp += std::fabs(double(t.coef)) * double(curve_value_bound(t.curve)) * std::min(region, 1.0);
        }
    }
    return p_table.has_terraces() ? 2.0 * amp : amp;
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
    Float4 rest{ 0.0f, 0.0f, 0.0f, 1.0f };
    const size_t n = p_table.biomes.size();
    for (size_t b = 0; b + 1 < n; b++) {
        r_w[b] = dmul(rest, edges_value(p_table.biomes[b].edges, p_fields));
        rest = dsub(rest, r_w[b]);
    }
    if (n > 0) {
        r_w[n - 1] = rest;
    }
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

int pick_slots(const Table &p_table, const float p_slots[Cascade::SLOTS], const Vector3i64 &p_p, int p_m) {
    Float4 f[MAX_FIELDS];
    apply_terms(p_table, p_slots, f);
    Float4 w[MAX_BIOMES];
    weights(p_table, f, w);
    return pick(p_table, w, p_p, p_m);
}

Color srgb_to_linear(uint32_t p_rgb) {
    return Color::hex((p_rgb << 8) | 0xFFu).srgb_to_linear();
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
