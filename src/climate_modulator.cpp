#include "climate.hpp"

#include "ov/math.hpp"
#include "ov/str.hpp"

#include <algorithm>

// The climate's Cascade::Modulation (combined_spec): per modulated band (group, level), how each biome shapes it, on
// the CPU and in generated GLSL, op for op. Above a run's last level each band reads the climate of its point's channels
// before the level's insertions; the level above the last also derives from that climate, once per point (derive): the
// BandMod parts of the last level's bands and its samples' terms (SampleCarry), into the carry groups H takes down,
// where the last level's bands read their parts back (modulate).
namespace Climate {

using Cascade::BandMod;
using Cascade::dadd;
using Cascade::dflush;
using Cascade::dmul;
using Cascade::dscale;
using Cascade::dsub;
using Cascade::flush_value;

namespace {

// The multifractal weight clamp(gain (1 - c), 0, 1) of the state's complement c.
Float4 multifractal_weight(float p_gain, const Float4 &p_c) {
    const Float4 u = dscale(dsub(Float4{ 0.0f, 0.0f, 0.0f, 1.0f }, p_c), p_gain);
    if (!(u.w > 0.0f)) {
        return Float4{ 0.0f, 0.0f, 0.0f, 0.0f };
    }
    if (u.w >= 1.0f) {
        return Float4{ 0.0f, 0.0f, 0.0f, 1.0f };
    }
    return u;
}

// multifractal_weight in GLSL (after the shared helpers, shared_helpers_glsl).
constexpr const char GLSL_MULTIFRACTAL[] = R"glsl(
vec4 cl_multifractal(float gain, vec4 c) {
    vec4 u = cl_scale(cl_sub(vec4(0.0, 0.0, 0.0, 1.0), c), gain);
    if (!(u.w > 0.0)) {
        return vec4(0.0);
    }
    if (u.w >= 1.0) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    return u;
}
)glsl";

// A BandMod member a band reads from the carry: amp[index], state[index] or warp[index].
enum PartKind {
    AMP,
    STATE,
    WARP,
};
struct Part {
    PartKind kind = AMP;
    int index = 0;
};

const Float4 &part_of(const BandMod &p_mod, const Part &p_part) {
    return p_part.kind == AMP ? p_mod.amp[p_part.index] : (p_part.kind == STATE ? p_mod.state[p_part.index] : p_mod.warp[p_part.index]);
}

ov::str part_glsl(const Part &p_part) {
    static const char *NAMES[] = { "amp", "state", "warp" };
    return ov::format("%s[%d]", NAMES[p_part.kind], p_part.index);
}

// Carry group p_c of a point's slots.
void put_carry(float r_slots[Cascade::SLOTS], int p_c, const Float4 &p_v) {
    r_slots[Cascade::carry_slot(p_c, Cascade::VALUE)] = p_v.w;
    r_slots[Cascade::carry_slot(p_c, Cascade::DX)] = p_v.x;
    r_slots[Cascade::carry_slot(p_c, Cascade::DY)] = p_v.y;
    r_slots[Cascade::carry_slot(p_c, Cascade::DZ)] = p_v.z;
}

// The same in GLSL, into `Ch pre` (carried_glsl reads it back).
ov::str put_carry_glsl(const Cascade::Spec &p_spec, int p_c, const ov::str &p_v, const ov::str &p_indent) {
    const int b = p_spec.gpu_carry(p_c);
    ov::str code;
    static const char *COMP = "wxyz";
    for (int c = 0; c < 4; c++) {
        code += p_indent + ov::format("pre.c[%d] = %s.%c;\n", b + c, p_v, COMP[c]);
    }
    return code;
}

class Modulator : public Cascade::Modulation {
public:
    Modulator(const Table &p_table, const Cascade::Spec &p_spec) :
            table(p_table), sample(sample_carry(p_table)) {
        state = p_table.has_state() ? p_spec.group_count() - 1 : -1;
        for (int g = 0; g < p_spec.group_count(); g++) {
            const std::vector<Cascade::Band> &bands = p_spec.group_bands(g);
            for (size_t i = 0; i < bands.size(); i++) {
                add_entry(p_spec, g, bands[i], g == 0 ? Shape() : table.fields[size_t(g - 1)].shapes[i]);
            }
        }
        // The carry: the samples' terms first, then each level's parts, the ground's first (entries are in group
        // order): without the fields a last level reads only those (used, the GPU's carry).
        carry = sample.count;
        for (int m = 0; m <= Cascade::MAX_LEVEL; m++) {
            int c = sample.count;
            used[m] = c;
            for (Entry &e : entries[m]) {
                e.carry = c;
                c += int(e.parts.size());
                used[m] = e.group == 0 ? c : used[m];
            }
            carry = std::max(carry, c);
        }
        range.sample = p_table.has_ground_terms();
    }

    // A band some biome modulates, or a sample term: something reads the derive.
    bool needed() const {
        return sample.count > 0 || std::any_of(std::begin(mask), std::end(mask), [](uint32_t p_m) { return p_m != 0u; });
    }

    uint32_t modulated(int p_m) const override { return p_m >= 0 && p_m <= Cascade::MAX_LEVEL ? mask[p_m] : 0u; }
    Cascade::FieldRange field_range() const override { return range; }
    int state_group() const override { return state; }
    bool shaped() const override { return shaped_; }
    uint32_t group_features(int p_group) const override { return p_group >= 0 && p_group < Cascade::MAX_GROUPS ? features[p_group] : 0u; }
    int carry_groups() const override { return carry; }
    int copied_carry() const override { return sample.pick; }
    int carry_used(int p_m) const override { return p_m >= 0 && p_m <= Cascade::MAX_LEVEL ? used[p_m] : sample.count; }

    void modulate(int p_m, bool p_carried, const float p_pre[Cascade::SLOTS], BandMod r_mod[Cascade::MAX_GROUPS]) const override {
        if (p_m < 0 || p_m > Cascade::MAX_LEVEL || entries[p_m].empty()) {
            return;
        }
        // Above a run's last level: the climate of the point's channels before the level's insertions.
        State st;
        Float4 c{ 0.0f, 0.0f, 0.0f, 0.0f };
        if (!p_carried) {
            Climate::state(table, p_pre, st);
            c = state >= 0 ? to_float4(p_pre + Cascade::field_slot(state - 1)) : c;
        }
        for (const Entry &e : entries[p_m]) {
            BandMod &o = r_mod[e.group];
            o = BandMod();
            o.features = e.features;
            if (e.constant) {
                for (int t = 0; t < 3; t++) {
                    o.amp[t] = Float4{ 0.0f, 0.0f, 0.0f, e.k[t] };
                }
                continue;
            }
            if (!p_carried) {
                o = biome_mod(e, st, c);
                continue;
            }
            for (size_t i = 0; i < e.parts.size(); i++) {
                const Part &pt = e.parts[i];
                Float4 *member = pt.kind == AMP ? o.amp : (pt.kind == STATE ? o.state : o.warp);
                member[pt.index] = to_float4(p_pre + Cascade::carry_slot(e.carry + int(i)));
            }
        }
    }

    void derive(int p_m, const Vector3i64 &p_p, float r_slots[Cascade::SLOTS]) const override {
        const int next = p_m - 1;
        const bool bands = next >= 0 && next <= Cascade::MAX_LEVEL && !entries[next].empty();
        const bool terms = sample.count > 0;
        if (!bands && !terms) {
            return;
        }
        State st;
        Climate::state(table, r_slots, st);
        if (bands) {
            const Float4 c = state >= 0 ? to_float4(r_slots + Cascade::field_slot(state - 1)) : Float4{ 0.0f, 0.0f, 0.0f, 0.0f };
            for (const Entry &e : entries[next]) {
                if (e.parts.empty()) {
                    continue;
                }
                const BandMod o = biome_mod(e, st, c);
                for (size_t i = 0; i < e.parts.size(); i++) {
                    put_carry(r_slots, e.carry + int(i), part_of(o, e.parts[i]));
                }
            }
        }
        if (!terms) {
            return;
        }
        if (sample.heights >= 0) {
            Float4 regions[MAX_REGIONS] = {};
            for (const int r : table.gated) {
                regions[r] = edges_value(table.regions[size_t(r)].edges, st.fields);
            }
            put_carry(r_slots, sample.heights, height_sum(table, st.fields, regions));
        }
        for (size_t b = 0; b < table.biomes.size(); b++) {
            if (sample.terrace[b] >= 0) {
                put_carry(r_slots, sample.terrace[b], st.w[b]);
            }
        }
        if (sample.pick >= 0) {
            put_carry(r_slots, sample.pick, Float4{ 0.0f, 0.0f, 0.0f, float(pick(table, st.w, p_p, p_m)) });
        }
    }

    ov::str glsl(const Cascade::Spec &p_spec, bool p_carried, bool p_derive) const override {
        return p_carried ? glsl_carried(p_spec) : glsl_weighted(p_spec, p_derive);
    }

private:
    // How one biome shapes one band: k = amplitude times the shape's n, |n|, n^2 coefficients (the inserted, centred
    // terms), the shape itself (the state's uncentred signal), and the Amp's weights.
    struct BiomeBand {
        float k[3] = {};
        Shape shape;
        float gain = 0.0f;
        float warp = 0.0f;
        int warp_fields[3] = { -1, -1, -1 };
    };
    // One modulated band: per biome, or constant (a field band's own shape, no biome naming it). parts: the BandMod
    // members some biome makes non-zero, carried from carry group `carry` on (the rest stay 0).
    struct Entry {
        int group = 0;
        uint32_t features = 0;
        bool constant = false;
        float k[3] = {};
        std::vector<BiomeBand> biomes;
        std::vector<Part> parts;
        int carry = 0;
    };

    static uint32_t shape_features(const float p_k[3]) {
        return (p_k[1] != 0.0f ? Cascade::MOD_ABS : 0u) | (p_k[2] != 0.0f ? Cascade::MOD_SQ : 0u);
    }

    // The band's BandMod from the climate p_st of the level above and the state's complement p_c.
    BandMod biome_mod(const Entry &p_e, const State &p_st, const Float4 &p_c) const {
        BandMod o;
        o.features = p_e.features;
        const Float4 *f = p_st.fields;
        const Float4 *w = p_st.w;
        Float4 a[3] = {};
        for (size_t b = 0; b < p_e.biomes.size(); b++) {
            const BiomeBand &bb = p_e.biomes[b];
            // The weight times the multifractal's.
            const Float4 wt = bb.gain > 0.0f ? dmul(w[b], multifractal_weight(bb.gain, p_c)) : w[b];
            for (int t = 0; t < 3; t++) {
                const float k = bb.k[t];
                if (k != 0.0f) {
                    a[t] = Float4{ a[t].x + wt.x * k, a[t].y + wt.y * k, a[t].z + wt.z * k, a[t].w + wt.w * k };
                }
            }
            if (bb.gain > 0.0f) {
                o.state[0] = dadd(o.state[0], dsub(w[b], dscale(wt, bb.shape.c[0])));
                for (int t = 1; t < 4; t++) {
                    if (bb.shape.c[t] != 0.0f) {
                        o.state[t] = dsub(o.state[t], dscale(wt, bb.shape.c[t]));
                    }
                }
            }
            if (bb.warp != 0.0f) {
                for (int ax = 0; ax < 3; ax++) {
                    o.warp[ax] = dadd(o.warp[ax], dscale(dmul(w[b], f[bb.warp_fields[ax]]), bb.warp));
                }
            }
        }
        for (int t = 0; t < 3; t++) {
            o.amp[t] = dflush(a[t]);
        }
        return o;
    }

    void add_entry(const Cascade::Spec &p_spec, int p_group, const Cascade::Band &p_band, const Shape &p_own) {
        const int field = p_group - 1; // the ground is GROUND
        const bool named = std::any_of(table.biomes.begin(), table.biomes.end(), [&](const Biome &p_b) { return find_amp(p_b, field, p_band.k) != nullptr; });
        if (!named && p_own.plain()) {
            return; // its constant amplitude, unmodulated
        }
        Entry e;
        e.group = p_group;
        if (!named) {
            e.constant = true;
            for (int t = 0; t < 3; t++) {
                e.k[t] = float(double(p_band.amp) * double(p_own.c[1 + t]));
            }
            e.features = shape_features(e.k);
        } else {
            for (const Biome &b : table.biomes) {
                BiomeBand bb;
                const Amp *a = find_amp(b, field, p_band.k);
                bb.shape = band_shape(table, &b, field, p_band.k);
                const double amp = biome_amp(b, field, p_band);
                for (int t = 0; t < 3; t++) {
                    bb.k[t] = float(amp * double(bb.shape.c[1 + t]));
                }
                if (a) {
                    bb.gain = a->gain;
                    bb.warp = a->warp;
                    std::copy(std::begin(a->warp_fields), std::end(a->warp_fields), std::begin(bb.warp_fields));
                }
                e.features |= shape_features(bb.k) | (bb.gain > 0.0f ? Cascade::MOD_STATE : 0u) | (bb.warp != 0.0f ? Cascade::MOD_WARP : 0u);
                e.biomes.push_back(bb);
            }
            add_parts(e);
        }
        shaped_ = shaped_ || e.constant || e.features != 0u;
        features[p_group] |= e.features;
        const int m = p_spec.insert_level(p_band);
        if (m < 0 || m > Cascade::MAX_LEVEL) {
            return;
        }
        entries[m].push_back(e);
        mask[m] |= 1u << p_group;
        // Only the ground reaches the mesh: the fields are carried for the ground's modulated levels.
        if (p_group == 0) {
            range.floor = ov::min(range.floor, m);
            range.ceiling = ov::max(range.ceiling, m);
        }
    }

    // The members biome_mod can make non-zero: an amplitude some biome's k names, the state's terms with a multifractal
    // (state[t > 0] where some multifractal biome's shape has c[t]), the warp.
    static void add_parts(Entry &r_e) {
        auto any = [&](auto p_pred) { return std::any_of(r_e.biomes.begin(), r_e.biomes.end(), p_pred); };
        for (int t = 0; t < 3; t++) {
            if (any([&](const BiomeBand &p_b) { return p_b.k[t] != 0.0f; })) {
                r_e.parts.push_back(Part{ AMP, t });
            }
        }
        if (any([](const BiomeBand &p_b) { return p_b.gain > 0.0f; })) {
            r_e.parts.push_back(Part{ STATE, 0 });
            for (int t = 1; t < 4; t++) {
                if (any([&](const BiomeBand &p_b) { return p_b.gain > 0.0f && p_b.shape.c[t] != 0.0f; })) {
                    r_e.parts.push_back(Part{ STATE, t });
                }
            }
        }
        if (any([](const BiomeBand &p_b) { return p_b.warp != 0.0f; })) {
            for (int ax = 0; ax < 3; ax++) {
                r_e.parts.push_back(Part{ WARP, ax });
            }
        }
    }

    ov::str glsl_biome_mod(const Entry &p_e, const ov::str &p_target) const;
    ov::str glsl_derive_entry(const Entry &p_e, const Cascade::Spec &p_spec) const;
    ov::str glsl_carried(const Cascade::Spec &p_spec) const;
    ov::str glsl_weighted(const Cascade::Spec &p_spec, bool p_derive) const;
    ov::str glsl_plain(const Cascade::Spec &p_spec, bool p_derive) const;
    ov::str glsl_signature(const Cascade::Spec &p_spec, bool p_derive) const;
    ov::str derive_glsl(const Cascade::Spec &p_spec) const;
    ov::str glsl_entry(const Entry &p_e) const;

    Table table;
    SampleCarry sample;
    int state = -1;
    int carry = 0;
    bool shaped_ = false;
    std::vector<Entry> entries[Cascade::MAX_LEVEL + 1];
    int used[Cascade::MAX_LEVEL + 1] = {}; // carry_used: the samples' terms and the ground's parts
    uint32_t mask[Cascade::MAX_LEVEL + 1] = {};
    uint32_t features[Cascade::MAX_GROUPS] = {};
    Cascade::FieldRange range;
};

ov::str mask_switch(const uint32_t p_mask[Cascade::MAX_LEVEL + 1], bool p_derive = false) {
    ov::str code = "    uint mask = 0u;\n    switch (m) {\n";
    for (int m = 0; m <= Cascade::MAX_LEVEL; m++) {
        if (p_mask[m] != 0u) {
            code += ov::format("    case %d: mask = %du; break;\n", m, int64_t(p_mask[m]));
        }
    }
    // With the derive (at the function's end): not on a level deriving, modulated band or not.
    return code + (p_derive ? "    default: break;\n    }\n    if (mask == 0u && g_carry_n == 0u) {\n" : "    default: break;\n    }\n    if (mask == 0u) {\n") + "        return 0u;\n    }\n";
}

// biome_mod's loop over the biomes, op for op: the amplitudes into a0 .. a2 (declared), the state and the warp into
// p_target's state[] and warp[] (after glsl_weights and state_c).
ov::str Modulator::glsl_biome_mod(const Entry &p_e, const ov::str &p_target) const {
    using ExactNoise::glsl_float;
    const ov::str &o = p_target;
    ov::str code = "        {\n            precise vec4 a0 = vec4(0.0);\n            precise vec4 a1 = vec4(0.0);\n            precise vec4 a2 = vec4(0.0);\n";
    for (size_t b = 0; b < p_e.biomes.size(); b++) {
        const BiomeBand &bb = p_e.biomes[b];
        const ov::str wb = ov::format("w%d", int64_t(b));
        const ov::str wt = bb.gain > 0.0f ? ov::format("cl_mul(%s, cl_multifractal(%s, state_c))", wb, glsl_float(bb.gain)) : wb;
        code += ov::format("            {\n                vec4 wt = %s;\n", wt);
        for (int t = 0; t < 3; t++) {
            if (bb.k[t] != 0.0f) {
                code += ov::format("                a%d = a%d + wt * %s;\n", t, t, glsl_float(bb.k[t]));
            }
        }
        if (bb.gain > 0.0f) {
            code += ov::format("                %sstate[0] = cl_add(%sstate[0], cl_sub(%s, cl_scale(wt, %s)));\n", o, o, wb, glsl_float(bb.shape.c[0]));
            for (int t = 1; t < 4; t++) {
                if (bb.shape.c[t] != 0.0f) {
                    code += ov::format("                %sstate[%d] = cl_sub(%sstate[%d], cl_scale(wt, %s));\n", o, t, o, t, glsl_float(bb.shape.c[t]));
                }
            }
        }
        if (bb.warp != 0.0f) {
            for (int ax = 0; ax < 3; ax++) {
                code += ov::format("                %swarp[%d] = cl_add(%swarp[%d], cl_scale(cl_mul(%s, f%d), %s));\n", o, ax, o, ax, wb, bb.warp_fields[ax], glsl_float(bb.warp));
            }
        }
        code += "            }\n";
    }
    return code;
}

// biome_mod for one entry into its carry groups (in derive_glsl).
ov::str Modulator::glsl_derive_entry(const Entry &p_e, const Cascade::Spec &p_spec) const {
    ov::str code = "        {\n            vec4 state[4] = vec4[4](vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0));\n";
    code += "            vec4 warp[3] = vec4[3](vec4(0.0), vec4(0.0), vec4(0.0));\n";
    code += glsl_biome_mod(p_e, "");
    code += "            vec4 amp[3] = vec4[3](cl_flush(a0), cl_flush(a1), cl_flush(a2));\n";
    for (size_t i = 0; i < p_e.parts.size(); i++) {
        code += put_carry_glsl(p_spec, p_e.carry + int(i), part_glsl(p_e.parts[i]), "            ");
    }
    return code + "        }\n        }\n"; // glsl_biome_mod's block, then this one
}

ov::str Modulator::glsl_carried(const Cascade::Spec &p_spec) const {
    // The last level's: what the level above derived (no weights, so no helpers).
    ov::str code;
    const int groups = p_spec.group_count();
    if (!shaped_) {
        code += "\n// Climate::Modulator::modulate: each modulated band's amplitude, carried from the level above.\n";
        code += glsl_signature(p_spec, false);
        code += ov::format("    for (int g = 0; g < %d; g++) {\n        amp[g] = vec4(0.0);\n    }\n", groups);
    } else {
        code += "\n// Climate::Modulator::modulate: each modulated band's BandMod, carried from the level above.\n";
        code += glsl_signature(p_spec, false);
        code += ov::format("    for (int g = 0; g < %d; g++) {\n        mod[g].features = 0u;\n", groups);
        code += "        for (int t = 0; t < 3; t++) {\n            mod[g].amp[t] = vec4(0.0);\n            mod[g].warp[t] = vec4(0.0);\n        }\n";
        code += "        for (int t = 0; t < 4; t++) {\n            mod[g].state[t] = vec4(0.0);\n        }\n    }\n";
    }
    code += mask_switch(mask);
    code += "    switch (m) {\n";
    for (int m = 0; m <= Cascade::MAX_LEVEL; m++) {
        if (entries[m].empty()) {
            continue;
        }
        code += ov::format("    case %d:\n", m);
        for (const Entry &e : entries[m]) {
            if (e.group >= groups) {
                continue; // a field band, in a view without the fields
            }
            if (!shaped_) {
                // Plain: every entry's one part is its amplitude on n (or none: all its k are 0).
                code += ov::format("        amp[%d] = %s;\n", e.group, e.parts.empty() ? ov::str("vec4(0.0)") : carried_glsl(p_spec, "pre", e.carry));
                continue;
            }
            const ov::str o = ov::format("mod[%d]", e.group);
            code += ov::format("        %s.features = %du;\n", o, int64_t(e.features));
            if (e.constant) {
                for (int t = 0; t < 3; t++) {
                    code += ov::format("        %s.amp[%d] = vec4(0.0, 0.0, 0.0, %s);\n", o, t, ExactNoise::glsl_float(e.k[t]));
                }
                continue;
            }
            for (size_t i = 0; i < e.parts.size(); i++) {
                code += ov::format("        %s.%s = %s;\n", o, part_glsl(e.parts[i]), carried_glsl(p_spec, "pre", e.carry + int(i)));
            }
        }
        code += "        break;\n";
    }
    code += "    default: break;\n    }\n    return mask;\n}\n";
    return code;
}

// The amplitudes alone (no band shaped anywhere): the text the climate had before shaped bands.
ov::str Modulator::glsl_plain(const Cascade::Spec &p_spec, bool p_derive) const {
    using ExactNoise::glsl_float;
    ov::str code = splices_helpers(table, p_spec) ? ov::str() : glsl_helpers();
    code += p_derive ? curves_glsl(table) : ov::str();
    code += "\n// Climate::Modulator::amplitudes: the field terms, the biome weights, then each modulated band's amplitude.\n";
    code += glsl_signature(p_spec, p_derive);
    code += ov::format("    for (int g = 0; g < %d; g++) {\n        amp[g] = vec4(0.0);\n    }\n", p_spec.group_count());
    code += mask_switch(mask, p_derive);
    code += glsl_weights(table, p_spec);
    code += "    switch (m) {\n";
    for (int m = 0; m <= Cascade::MAX_LEVEL; m++) {
        if (entries[m].empty()) {
            continue;
        }
        code += ov::format("    case %d: {\n", m);
        for (const Entry &e : entries[m]) {
            // A field band on a level inserting none (the derive's views only: FieldRange::inserted) reads no amplitude.
            code += p_derive && e.group > 0 ? "        if (g_field_bands) { precise vec4 a = vec4(0.0);" : "        { precise vec4 a = vec4(0.0);";
            for (size_t b = 0; b < e.biomes.size(); b++) {
                if (e.biomes[b].k[0] != 0.0f) {
                    code += ov::format(" a = a + w%d * %s;", int64_t(b), glsl_float(e.biomes[b].k[0]));
                }
            }
            code += ov::format(" amp[%d] = cl_flush(a); }\n", e.group);
        }
        code += "        break;\n    }\n";
    }
    code += "    default: break;\n    }\n";
    code += p_derive ? derive_glsl(p_spec) : ov::str();
    return code + "    return mask;\n}\n";
}

// cascade_amplitudes' or cascade_mods' first line (Cascade::Modulation::glsl).
ov::str Modulator::glsl_signature(const Cascade::Spec &p_spec, bool p_derive) const {
    const ov::str out = shaped_ ? ov::format("BandMod mod[%d]", p_spec.group_count()) : ov::format("vec4 amp[%d]", p_spec.group_count());
    return ov::format("uint %s(int m, %s, out %s%s) {\n", shaped_ ? "cascade_mods" : "cascade_amplitudes", p_derive ? "inout Ch pre" : "Ch pre", out,
            p_derive ? ", WorldPos p" : "");
}

// modulate()'s loop body for one entry, op for op.
ov::str Modulator::glsl_entry(const Entry &p_e) const {
    using ExactNoise::glsl_float;
    const ov::str o = ov::format("mod[%d]", p_e.group);
    ov::str code = ov::format("        %s.features = %du;\n", o, int64_t(p_e.features));
    if (p_e.constant) {
        for (int t = 0; t < 3; t++) {
            code += ov::format("        %s.amp[%d] = vec4(0.0, 0.0, 0.0, %s);\n", o, t, glsl_float(p_e.k[t]));
        }
        return code;
    }
    code += glsl_biome_mod(p_e, o + ov::str("."));
    code += ov::format("            %s.amp[0] = cl_flush(a0);\n            %s.amp[1] = cl_flush(a1);\n            %s.amp[2] = cl_flush(a2);\n        }\n", o, o, o);
    return code;
}

ov::str Modulator::glsl_weighted(const Cascade::Spec &p_spec, bool p_derive) const {
    if (!shaped_) {
        return glsl_plain(p_spec, p_derive);
    }
    using ExactNoise::glsl_float;
    ov::str code = GLSL_MULTIFRACTAL; // after the shared helpers, spliced by the level shader (splices_helpers)
    code += p_derive ? curves_glsl(table) : ov::str();
    code += "\n// Climate::Modulator::modulate: the field terms, the biome weights, then each modulated band's BandMod.\n";
    code += glsl_signature(p_spec, p_derive);
    code += ov::format("    for (int g = 0; g < %d; g++) {\n        mod[g].features = 0u;\n", p_spec.group_count());
    code += "        for (int t = 0; t < 3; t++) {\n            mod[g].amp[t] = vec4(0.0);\n            mod[g].warp[t] = vec4(0.0);\n        }\n";
    code += "        for (int t = 0; t < 4; t++) {\n            mod[g].state[t] = vec4(0.0);\n        }\n    }\n";
    code += mask_switch(mask, p_derive);
    code += glsl_weights(table, p_spec);
    // The state's channels (the multifractal's complement).
    if (state >= 0) {
        const int sb = p_spec.gpu_base(state);
        code += ov::format("    vec4 state_c = vec4(pre.c[%d], pre.c[%d], pre.c[%d], pre.c[%d]);\n", sb + 1, sb + 2, sb + 3, sb);
    }
    code += "    switch (m) {\n";
    for (int m = 0; m <= Cascade::MAX_LEVEL; m++) {
        if (entries[m].empty()) {
            continue;
        }
        code += ov::format("    case %d: {\n", m);
        for (const Entry &e : entries[m]) {
            // As in glsl_plain: no BandMod for a field band on a level inserting none.
            code += p_derive && e.group > 0 ? "        if (g_field_bands) {\n" + glsl_entry(e) + "        }\n" : glsl_entry(e);
        }
        code += "        break;\n    }\n";
    }
    code += "    default: break;\n    }\n";
    code += p_derive ? derive_glsl(p_spec) : ov::str();
    return code + "    return mask;\n}\n";
}

// Climate::Modulator::derive at the end of the level's modulation, from its weights (the channels before the level's
// insertions), where g_carry_n says (j = 1): into the carry of the level below, a run's last (its ground's BandMod parts,
// its samples' terms: the carry_used groups, all of them written).
ov::str Modulator::derive_glsl(const Cascade::Spec &p_spec) const {
    ov::str code = "    if (g_carry_n == 0u) {\n        return mask;\n    }\n";
    code += "    switch (m) {\n";
    for (int m = 1; m <= Cascade::MAX_LEVEL; m++) {
        if (used[m - 1] == sample.count) {
            continue;
        }
        code += ov::format("    case %d:\n", m);
        for (const Entry &e : entries[m - 1]) {
            if (e.group == 0 && !e.parts.empty()) {
                code += glsl_derive_entry(e, p_spec);
            }
        }
        code += "        break;\n";
    }
    code += "    default: break;\n    }\n";
    if (sample.count > 0) {
        code += "    {\n";
        if (sample.heights >= 0) {
            for (const int r : table.gated) {
                code += glsl_edges_value(ov::format("rg%d", r), table.regions[size_t(r)].edges);
            }
            code += height_sum_glsl(table, "hs");
            code += put_carry_glsl(p_spec, sample.heights, "hs", "        ");
        }
        for (size_t b = 0; b < table.biomes.size(); b++) {
            if (sample.terrace[b] >= 0) {
                code += put_carry_glsl(p_spec, sample.terrace[b], ov::format("w%d", int64_t(b)), "        ");
            }
        }
        if (sample.pick >= 0) {
            code += pick_statements_glsl(table, "pk");
            code += put_carry_glsl(p_spec, sample.pick, "vec4(0.0, 0.0, 0.0, pk)", "        ");
        }
        code += "    }\n";
    }
    return code;
}

} // namespace

Cascade::Spec combined_spec(const Table &p_table, const Cascade::Spec &p_ground) {
    Cascade::Spec spec = p_ground.ground_only();
    for (Cascade::Band &b : spec.bands) {
        for (const std::pair<int, int> &s : p_table.ground_stretch) {
            b.stretch = s.first == b.k ? s.second : b.stretch;
        }
    }
    spec.cell_groups = 0;
    for (size_t i = 0; i < p_table.fields.size(); i++) {
        spec.fields.push_back(p_table.fields[i].bands);
        spec.cell_groups |= p_table.fields[i].cell ? 1u << (1 + i) : 0u;
    }
    if (p_table.has_state()) {
        spec.fields.emplace_back(); // the multifractal state: band-less, the ground's band overwrites it
    }
    std::shared_ptr<Modulator> mod = std::make_shared<Modulator>(p_table, spec);
    if (mod->needed()) {
        spec.modulation = mod; // a LINEAR4 ground where a band is modulated: validate refuses multipliers on another
    }
    return spec;
}

} // namespace Climate
