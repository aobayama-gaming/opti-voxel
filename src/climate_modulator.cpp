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
        for (const Cascade::Band &band : p_spec.bands) {
            add_entry(p_spec, band, p_spec.note_mod(p_spec.note_rank(band))); // biomes play the ground's notes only
        }
        // The carry: the samples' terms first, then each level's parts (used, the GPU's carry).
        carry = sample.count;
        for (int m = 0; m <= Cascade::MAX_LEVEL; m++) {
            int c = sample.count;
            used[m] = c;
            for (Entry &e : entries[m]) {
                e.carry = c;
                c += int(e.parts.size());
                used[m] = c;
            }
            carry = std::max(carry, c);
        }
        range.sample = p_table.has_programs();
    }

    // A band some biome modulates, or a sample term: something reads the derive.
    bool needed() const {
        return sample.count > 0 || std::any_of(std::begin(entries), std::end(entries), [](const std::vector<Entry> &p_e) { return !p_e.empty(); });
    }

    uint32_t modulated(int p_m) const override { return p_m >= 0 && p_m <= Cascade::MAX_LEVEL ? mask[p_m] : 0u; }
    Cascade::FieldRange field_range() const override { return range; }
    int state_group() const override { return state; }
    uint32_t group_features(int p_group) const override { return p_group == 0 ? features : 0u; }
    int carry_groups() const override { return carry; }
    int copied_carry() const override { return sample.pick; }
    int carry_used(int p_m) const override { return p_m >= 0 && p_m <= Cascade::MAX_LEVEL ? used[p_m] : sample.count; }

    void modulate(int p_m, bool p_carried, const float p_pre[Cascade::SLOTS], BandMod r_mod[Cascade::MAX_MODS]) const override {
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
            BandMod &o = r_mod[e.slot];
            o = BandMod();
            o.features = e.features;
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
        for (size_t b = 0; b < table.biomes.size(); b++) {
            if (sample.weight[b] >= 0) {
                put_carry(r_slots, sample.weight[b], st.w[b]);
            }
        }
        if (sample.pick >= 0) {
            put_carry(r_slots, sample.pick, Float4{ 0.0f, 0.0f, 0.0f, float(pick(table, st.w, p_p, p_m)) });
        }
        for (size_t f = 0; f < table.fields.size(); f++) {
            if (sample.field[f] >= 0) {
                put_carry(r_slots, sample.field[f], st.fields[f]);
            }
        }
    }

    ov::str glsl(const Cascade::Spec &p_spec, bool p_carried, bool p_derive, Cascade::ShaderParams &r_params) const override {
        return p_carried ? glsl_carried(p_spec, r_params) : glsl_weighted(p_spec, p_derive, r_params);
    }

private:
    // How one biome shapes one band: k = amplitude times the shape's n, |n|, n^2 coefficients (the inserted, centred
    // terms), the shape itself (the state's uncentred signal), and the Amp's weights. *_live: the terms in the text (the
    // non-zero ones; generic: those a node may make non-zero, their values tested where biome_mod tests them).
    struct BiomeBand {
        float k[3] = {};
        Shape shape;
        float gain = 0.0f;
        float warp = 0.0f;
        int warp_fields[3] = { -1, -1, -1 };
        bool k_live[3] = {};
        bool c_live[4] = {};
        bool gain_live = false;
        bool warp_live = false;
    };
    // One modulated ground band, per biome. features: biome_mod's (the BandMod's; the text's are the live terms');
    // parts: the BandMod members some biome's live terms write, carried from carry group `carry` on (the rest stay 0);
    // slot: its BandMod's (Spec::note_mod, 0 for a level's first note).
    struct Entry {
        uint32_t features = 0;
        std::vector<BiomeBand> biomes;
        std::vector<Part> parts;
        int carry = 0;
        int slot = 0;
    };

    static uint32_t feature_bits(bool p_abs, bool p_sq, bool p_state, bool p_warp) {
        return (p_abs ? Cascade::MOD_ABS : 0u) | (p_sq ? Cascade::MOD_SQ : 0u) | (p_state ? Cascade::MOD_STATE : 0u) | (p_warp ? Cascade::MOD_WARP : 0u);
    }

    // A uint of the text: its literal in a specialised one, read from r_params in a generic one.
    ov::str uint_glsl(uint32_t p_v, Cascade::ShaderParams &r_params) const {
        return table.generic ? ov::format("uint(%s)", r_params(float(p_v))) : ov::format("%du", int64_t(p_v));
    }
    // Statements p_code under biome_mod's test p_test: as they are in a specialised text (the term is live: its test
    // holds), in a branch on the value in a generic one.
    ov::str guarded(const ov::str &p_test, const ov::str &p_code) const {
        return table.generic ? "                if (" + p_test + ov::str(") {\n") + p_code + ov::str("                }\n") : p_code;
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

    void add_entry(const Cascade::Spec &p_spec, const Cascade::Band &p_band, int p_slot) {
        if (std::none_of(table.biomes.begin(), table.biomes.end(), [&](const Biome &p_b) { return find_amp(p_b, p_band) != nullptr; })) {
            return; // its constant amplitude, unmodulated
        }
        Entry e;
        e.slot = p_slot;
        const bool generic = table.generic;
        for (const Biome &b : table.biomes) {
            BiomeBand bb;
            const Amp *a = find_amp(b, p_band);
            bb.shape = band_shape(b, p_band);
            const double amp = biome_amp(b, p_band);
            // A generic table's: the graph may make a coefficient non-zero where it has the node (a shape's any, else the
            // plain shape's), and the amplitude where it mixes the note.
            for (int t = 0; t < 4; t++) {
                bb.c_live[t] = generic ? (a && a->has_shape) || Shape().c[t] != 0.0f : bb.shape.c[t] != 0.0f;
            }
            for (int t = 0; t < 3; t++) {
                bb.k[t] = float(amp * double(bb.shape.c[1 + t]));
                bb.k_live[t] = generic ? a && a->mixed && bb.c_live[1 + t] : bb.k[t] != 0.0f;
            }
            if (a) {
                bb.gain = a->gain;
                bb.warp = a->warp;
                std::copy(std::begin(a->warp_fields), std::end(a->warp_fields), std::begin(bb.warp_fields));
            }
            bb.gain_live = a && a->multifractal;
            bb.warp_live = a && a->warped;
            e.features |= feature_bits(bb.k[1] != 0.0f, bb.k[2] != 0.0f, bb.gain > 0.0f, bb.warp != 0.0f);
            features |= feature_bits(bb.k_live[1], bb.k_live[2], bb.gain_live, bb.warp_live);
            e.biomes.push_back(bb);
        }
        add_parts(e);
        const int m = p_spec.insert_level(p_band);
        if (m < 0 || m > Cascade::MAX_LEVEL) {
            return;
        }
        entries[m].push_back(e);
        // Modulated unless it is a generic table's constant or silent note (the specialised table has no entry for it).
        const bool constant = std::find(table.constant.begin(), table.constant.end(), note_id(p_band)) != table.constant.end();
        mask[m] |= constant || p_band.amp == 0.0f ? 0u : 1u;
        // The fields are carried for the ground's modulated levels.
        range.floor = ov::min(range.floor, m);
        range.ceiling = ov::max(range.ceiling, m);
    }

    // The members biome_mod's live terms write: an amplitude some biome's k names, the state's terms with a multifractal
    // (state[t > 0] where some multifractal biome's shape has c[t]), the warp.
    static void add_parts(Entry &r_e) {
        auto any = [&](auto p_pred) { return std::any_of(r_e.biomes.begin(), r_e.biomes.end(), p_pred); };
        for (int t = 0; t < 3; t++) {
            if (any([&](const BiomeBand &p_b) { return p_b.k_live[t]; })) {
                r_e.parts.push_back(Part{ AMP, t });
            }
        }
        if (any([](const BiomeBand &p_b) { return p_b.gain_live; })) {
            r_e.parts.push_back(Part{ STATE, 0 });
            for (int t = 1; t < 4; t++) {
                if (any([&](const BiomeBand &p_b) { return p_b.gain_live && p_b.c_live[t]; })) {
                    r_e.parts.push_back(Part{ STATE, t });
                }
            }
        }
        if (any([](const BiomeBand &p_b) { return p_b.warp_live; })) {
            for (int ax = 0; ax < 3; ax++) {
                r_e.parts.push_back(Part{ WARP, ax });
            }
        }
    }

    ov::str glsl_biome_mod(const Entry &p_e, const ov::str &p_target, Cascade::ShaderParams &r_params) const;
    ov::str glsl_derive_entry(const Entry &p_e, const Cascade::Spec &p_spec, Cascade::ShaderParams &r_params) const;
    ov::str glsl_carried(const Cascade::Spec &p_spec, Cascade::ShaderParams &r_params) const;
    ov::str glsl_weighted(const Cascade::Spec &p_spec, bool p_derive, Cascade::ShaderParams &r_params) const;
    ov::str glsl_signature(const Cascade::Spec &p_spec, bool p_derive, Cascade::ShaderParams &r_params) const;
    ov::str derive_glsl(const Cascade::Spec &p_spec, Cascade::ShaderParams &r_params) const;
    ov::str glsl_entry(const Entry &p_e, Cascade::ShaderParams &r_params) const;

    Table table;
    SampleCarry sample;
    int state = -1;
    int carry = 0;
    std::vector<Entry> entries[Cascade::MAX_LEVEL + 1];
    int used[Cascade::MAX_LEVEL + 1] = {}; // carry_used: the samples' terms and the ground's parts
    uint32_t mask[Cascade::MAX_LEVEL + 1] = {};
    uint32_t features = 0; // the ground's (group 0)
    Cascade::FieldRange range;
};

// biome_mod's loop over the biomes, op for op: the amplitudes into a0 .. a2 (declared), the state and the warp into
// p_target's state[] and warp[] (after glsl_weights and state_c).
ov::str Modulator::glsl_biome_mod(const Entry &p_e, const ov::str &p_target, Cascade::ShaderParams &r_params) const {
    const ov::str &o = p_target;
    ov::str code = "        {\n            precise vec4 a0 = vec4(0.0);\n            precise vec4 a1 = vec4(0.0);\n            precise vec4 a2 = vec4(0.0);\n";
    for (size_t b = 0; b < p_e.biomes.size(); b++) {
        const BiomeBand &bb = p_e.biomes[b];
        const ov::str wb = ov::format("w%d", int64_t(b));
        const ov::str gain = bb.gain_live ? r_params(bb.gain) : ov::str();
        const ov::str mf = ov::format("cl_mul(%s, cl_multifractal(%s, state_c))", wb, gain);
        const ov::str wt = !bb.gain_live ? wb : (table.generic ? ov::format("%s > 0.0 ? %s : %s", gain, mf, wb) : mf);
        code += ov::format("            {\n                vec4 wt = %s;\n", wt);
        for (int t = 0; t < 3; t++) {
            if (bb.k_live[t]) {
                const ov::str k = r_params(bb.k[t]);
                code += guarded(k + ov::str(" != 0.0"), ov::format("                a%d = a%d + wt * %s;\n", t, t, k));
            }
        }
        if (bb.gain_live) {
            ov::str state = ov::format("                %sstate[0] = cl_add(%sstate[0], cl_sub(%s, cl_scale(wt, %s)));\n", o, o, wb, r_params(bb.shape.c[0]));
            for (int t = 1; t < 4; t++) {
                if (bb.c_live[t]) {
                    const ov::str c = r_params(bb.shape.c[t]);
                    state += guarded(c + ov::str(" != 0.0"), ov::format("                %sstate[%d] = cl_sub(%sstate[%d], cl_scale(wt, %s));\n", o, t, o, t, c));
                }
            }
            code += guarded(gain + ov::str(" > 0.0"), state);
        }
        if (bb.warp_live) {
            const ov::str warp = r_params(bb.warp);
            ov::str axes;
            for (int ax = 0; ax < 3; ax++) {
                axes += ov::format("                %swarp[%d] = cl_add(%swarp[%d], cl_scale(cl_mul(%s, f%d), %s));\n", o, ax, o, ax, wb, bb.warp_fields[ax], warp);
            }
            code += guarded(warp + ov::str(" != 0.0"), axes);
        }
        code += "            }\n";
    }
    return code;
}

// biome_mod for one entry into its carry groups (in derive_glsl).
ov::str Modulator::glsl_derive_entry(const Entry &p_e, const Cascade::Spec &p_spec, Cascade::ShaderParams &r_params) const {
    ov::str code = "        {\n            vec4 state[4] = vec4[4](vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0));\n";
    code += "            vec4 warp[3] = vec4[3](vec4(0.0), vec4(0.0), vec4(0.0));\n";
    code += glsl_biome_mod(p_e, "", r_params);
    code += "            vec4 amp[3] = vec4[3](cl_flush(a0), cl_flush(a1), cl_flush(a2));\n";
    for (size_t i = 0; i < p_e.parts.size(); i++) {
        code += put_carry_glsl(p_spec, p_e.carry + int(i), part_glsl(p_e.parts[i]), "            ");
    }
    return code + "        }\n        }\n"; // glsl_biome_mod's block, then this one
}

ov::str Modulator::glsl_carried(const Cascade::Spec &p_spec, Cascade::ShaderParams &r_params) const {
    // The last level's: what the level above derived (no weights, so no helpers).
    ov::str code = "\n// Climate::Modulator::modulate: each modulated band's BandMod, carried from the level above.\n";
    code += glsl_signature(p_spec, false, r_params);
    code += "    switch (m) {\n";
    for (int m = 0; m <= Cascade::MAX_LEVEL; m++) {
        if (entries[m].empty()) {
            continue;
        }
        code += ov::format("    case %d:\n", m);
        for (const Entry &e : entries[m]) {
            const ov::str o = ov::format("mod[%d]", e.slot);
            code += ov::format("        %s.features = %s;\n", o, uint_glsl(e.features, r_params));
            for (size_t i = 0; i < e.parts.size(); i++) {
                code += ov::format("        %s.%s = %s;\n", o, part_glsl(e.parts[i]), carried_glsl(p_spec, "pre", e.carry + int(i)));
            }
        }
        code += "        break;\n";
    }
    code += "    default: break;\n    }\n    return mask;\n}\n";
    return code;
}

// cascade_mods' first line (Cascade::Modulation::glsl), modulated(m) per level with an entry (a generic table's from
// r_params: its constant notes' levels are 0), out where no band is modulated (with the derive: not on a level deriving
// either), then its BandMods at 0.
ov::str Modulator::glsl_signature(const Cascade::Spec &p_spec, bool p_derive, Cascade::ShaderParams &r_params) const {
    ov::str code = ov::format("uint cascade_mods(int m, %s, out BandMod mod[%d]%s) {\n", p_derive ? "inout Ch pre" : "Ch pre", p_spec.mod_count(), p_derive ? ", WorldPos p" : "");
    code += "    uint mask = 0u;\n    switch (m) {\n";
    for (int m = 0; m <= Cascade::MAX_LEVEL; m++) {
        code += entries[m].empty() ? ov::str() : ov::format("    case %d: mask = %s; break;\n", m, uint_glsl(mask[m], r_params));
    }
    code += p_derive ? "    default: break;\n    }\n    if (mask == 0u && g_carry_n == 0u) {\n        return 0u;\n    }\n" : "    default: break;\n    }\n    if (mask == 0u) {\n        return 0u;\n    }\n";
    code += ov::format("    for (int g = 0; g < %d; g++) {\n        mod[g].features = 0u;\n", p_spec.mod_count());
    code += "        for (int t = 0; t < 3; t++) {\n            mod[g].amp[t] = vec4(0.0);\n            mod[g].warp[t] = vec4(0.0);\n        }\n";
    return code + "        for (int t = 0; t < 4; t++) {\n            mod[g].state[t] = vec4(0.0);\n        }\n    }\n";
}

// modulate()'s loop body for one entry, op for op.
ov::str Modulator::glsl_entry(const Entry &p_e, Cascade::ShaderParams &r_params) const {
    const ov::str o = ov::format("mod[%d]", p_e.slot);
    ov::str code = ov::format("        %s.features = %s;\n", o, uint_glsl(p_e.features, r_params));
    code += glsl_biome_mod(p_e, o + ov::str("."), r_params);
    code += ov::format("            %s.amp[0] = cl_flush(a0);\n            %s.amp[1] = cl_flush(a1);\n            %s.amp[2] = cl_flush(a2);\n        }\n", o, o, o);
    return code;
}

ov::str Modulator::glsl_weighted(const Cascade::Spec &p_spec, bool p_derive, Cascade::ShaderParams &r_params) const {
    ov::str code = GLSL_MULTIFRACTAL; // after the shared helpers, spliced by the level shader (get_climate_helpers_glsl)
    code += "\n// Climate::Modulator::modulate: the field terms, the biome weights, then each modulated band's BandMod.\n";
    code += glsl_signature(p_spec, p_derive, r_params);
    code += glsl_weights(table, p_spec, r_params);
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
            code += glsl_entry(e, r_params);
        }
        code += "        break;\n    }\n";
    }
    code += "    default: break;\n    }\n";
    code += p_derive ? derive_glsl(p_spec, r_params) : ov::str();
    return code + "    return mask;\n}\n";
}

// Climate::Modulator::derive at the end of the level's modulation, from its weights (the channels before the level's
// insertions), where g_carry_n says (j = 1): into the carry of the level below, a run's last (its ground's BandMod parts,
// its samples' terms: the carry_used groups, all of them written).
ov::str Modulator::derive_glsl(const Cascade::Spec &p_spec, Cascade::ShaderParams &r_params) const {
    ov::str code = "    if (g_carry_n == 0u) {\n        return mask;\n    }\n";
    code += "    switch (m) {\n";
    for (int m = 1; m <= Cascade::MAX_LEVEL; m++) {
        if (used[m - 1] == sample.count) {
            continue;
        }
        code += ov::format("    case %d:\n", m);
        for (const Entry &e : entries[m - 1]) {
            if (!e.parts.empty()) {
                code += glsl_derive_entry(e, p_spec, r_params);
            }
        }
        code += "        break;\n";
    }
    code += "    default: break;\n    }\n";
    if (sample.count > 0) {
        code += "    {\n";
        for (size_t b = 0; b < table.biomes.size(); b++) {
            if (sample.weight[b] >= 0) {
                code += put_carry_glsl(p_spec, sample.weight[b], ov::format("w%d", int64_t(b)), "        ");
            }
        }
        if (sample.pick >= 0) {
            code += pick_statements_glsl(table, "pk");
            code += put_carry_glsl(p_spec, sample.pick, "vec4(0.0, 0.0, 0.0, pk)", "        ");
        }
        for (size_t f = 0; f < table.fields.size(); f++) {
            if (sample.field[f] >= 0) {
                code += put_carry_glsl(p_spec, sample.field[f], ov::format("f%d", int64_t(f)), "        ");
            }
        }
        code += "    }\n";
    }
    return code;
}

} // namespace

Cascade::Spec combined_spec(const Table &p_table, const Cascade::Spec &p_ground) {
    Cascade::Spec spec = p_ground.ground_only();
    spec.bands = p_table.notes;
    for (const Field &f : p_table.fields) {
        spec.fields.push_back(f.bands);
    }
    if (p_table.has_state()) {
        spec.fields.emplace_back(); // the multifractal state: band-less, the ground's band overwrites it
    }
    std::shared_ptr<Modulator> mod = std::make_shared<Modulator>(p_table, spec);
    if (mod->needed()) {
        spec.modulation = mod;
    }
    return spec;
}

} // namespace Climate
