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

bool region_core(const Table &p_table, int p_region, const Float4 p_fields[MAX_FIELDS], float p_margin) {
    const std::vector<Edge> &edges = p_table.regions[size_t(p_region)].edges;
    if (!(edges_value(edges, p_fields).w == 1.0f)) {
        return false;
    }
    // An edge field's finest octave (-1 without bands: a field of terms only, no doubling), and the slowest of those (the
    // margin's reference).
    const auto finest = [&](const Edge &p_e) {
        int k = -1;
        for (const Cascade::Band &b : p_table.fields[size_t(p_e.field)].bands) {
            k = k < 0 ? b.k : std::min(k, b.k);
        }
        return k;
    };
    int slowest = -1;
    for (const Edge &e : edges) {
        slowest = std::max(slowest, finest(e));
    }
    for (size_t i = 0; i < edges.size(); i++) {
        const int k = finest(edges[i]);
        const float margin = std::ldexp(p_margin, k < 0 ? 0 : slowest - k);
        for (const float d : { -margin, margin }) {
            Float4 g[MAX_FIELDS];
            std::copy(p_fields, p_fields + MAX_FIELDS, g);
            g[edges[i].field].w += d;
            if (!(edges_value(edges, g).w == 1.0f)) {
                return false;
            }
        }
    }
    return true;
}

std::vector<int> gated_regions(const Table &p_table) {
    std::vector<int> r;
    for (int i = 0; i < int(p_table.regions.size()); i++) {
        if (std::any_of(p_table.heights.begin(), p_table.heights.end(), [&](const HeightTerm &p_t) { return p_t.region == i; })) {
            r.push_back(i);
        }
    }
    return r;
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

const Amp *find_amp(const Biome &p_biome, int p_field, int p_k) {
    for (const Amp &a : p_biome.amps) {
        if (a.field == p_field && a.k == p_k) {
            return &a;
        }
    }
    return nullptr;
}

float biome_amp(const Biome &p_biome, int p_field, const Cascade::Band &p_band) {
    const Amp *a = find_amp(p_biome, p_field, p_band.k);
    return a ? float(double(a->mult) * double(p_band.amp)) : p_band.amp;
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
    // A cell field's bands are plain (validate) but its values reach CELL_BOUND, past perlin3's 1.
    const bool cell = p_field != GROUND && p_table.fields[size_t(p_field)].cell;
    auto bound = [&](const Biome *p_biome) { return cell ? ExactNoise::CELL_BOUND : shape_bound(band_shape(p_table, p_biome, p_field, p_band.k)); };
    float m = p_table.biomes.empty() ? p_band.amp * bound(nullptr) : 0.0f;
    for (const Climate::Biome &b : p_table.biomes) {
        m = MAX(m, biome_amp(b, p_field, p_band) * bound(&b));
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

// The step of a product of edge smoothsteps (in [0, 1]) between LOD p_lod and p_lod + 1: each edge's slope, at most
// 1.5 |scale| per unit of its field, times its field's step, summed.
double edges_step(const Table &p_table, const std::vector<Edge> &p_edges, int p_lod, double p_ground_amp) {
    double step = 0.0;
    for (const Edge &e : p_edges) {
        step += 1.5 * std::fabs(double(e.scale)) * field_step(p_table, e.field, p_lod, p_ground_amp);
    }
    return step;
}

} // namespace

double ground_step(const Table &p_table, int p_lod, double p_ground_amp) {
    double amp = p_ground_amp;
    for (const HeightTerm &t : p_table.heights) {
        amp += std::fabs(double(t.coef)) * (t.has_curve ? double(curve_slope_bound(t.curve)) : 1.0) * field_step(p_table, t.field, p_lod, p_ground_amp);
        if (t.region >= 0) {
            // r v steps by at most |r| |dv| + |v| |dr|, r in [0, 1]: the term's own step above, plus its bound times the
            // region's (at most 1: r stays in [0, 1]).
            const double region = edges_step(p_table, p_table.regions[size_t(t.region)].edges, p_lod, p_ground_amp);
            amp += std::fabs(double(t.coef)) * double(curve_bound(t.curve)) * std::min(region, 1.0);
        }
    }
    if (!p_table.has_terraces()) {
        return amp;
    }
    // ground = h + sum_b w_b (T_b(h) - h). T's slope is in [0, 2] and the weights sum to 1: the h part moves by at most
    // 2 amp. The weights' own change moves it by |dw_b| |T_b(h) - h|, and |T_b(h) - h| = step |t^2 - t| <= step / 4.
    // w_b = rest_b r_b, each a product of edge smoothsteps in [0, 1] (slope 1.5 per unit of scale (field - anchor)), so
    // |dw_b| <= the sum over the edges of biomes 0..b (the last biome: every other one's) of 1.5 |scale| field_step.
    double edges = 0.0, blend = 0.0;
    for (const Biome &b : p_table.biomes) {
        edges += edges_step(p_table, b.edges, p_lod, p_ground_amp);
        blend += edges * 0.25 * double(b.terrace.step);
    }
    return 2.0 * amp + blend;
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

int pick_slots(const Table &p_table, const float p_slots[Cascade::SLOTS], const Vector3i64 &p_p, int p_m) {
    State s;
    state(p_table, p_slots, s);
    return pick(p_table, s.w, p_p, p_m);
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
