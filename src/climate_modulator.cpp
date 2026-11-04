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

// The multifractal weight clamp(mul f + add, 0, 1) of its parent's field f (Multifractal).
Float4 multifractal_weight(const Float4 &p_f, float p_mul, float p_add) {
    const Float4 u = dadd(dscale(p_f, p_mul), Float4{ 0.0f, 0.0f, 0.0f, p_add });
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
vec4 cl_multifractal(vec4 f, float mul, float add) {
    vec4 u = cl_add(cl_scale(f, mul), vec4(0.0, 0.0, 0.0, add));
    if (!(u.w > 0.0)) {
        return vec4(0.0);
    }
    if (u.w >= 1.0) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    return u;
}
)glsl";

// A BandMod member a band reads from the carry: amp[index] or warp[index]; member() numbers them as a generic text's
// data does (amp 0 .. 2, warp 3 .. 5).
enum PartKind {
    AMP,
    WARP,
};
struct Part {
    PartKind kind = AMP;
    int index = 0;
    int member() const { return kind == AMP ? index : 3 + index; }
};

const Float4 &part_of(const BandMod &p_mod, const Part &p_part) {
    return p_part.kind == AMP ? p_mod.amp[p_part.index] : p_mod.warp[p_part.index];
}

ov::str part_glsl(const Part &p_part) {
    return ov::format("%s[%d]", p_part.kind == AMP ? "amp" : "warp", p_part.index);
}

// Carry group p_c of a point's slots.
void put_carry(float r_slots[Cascade::SLOTS], int p_c, const Float4 &p_v) {
    r_slots[Cascade::carry_slot(p_c, Cascade::VALUE)] = p_v.w;
    r_slots[Cascade::carry_slot(p_c, Cascade::DX)] = p_v.x;
    r_slots[Cascade::carry_slot(p_c, Cascade::DY)] = p_v.y;
    r_slots[Cascade::carry_slot(p_c, Cascade::DZ)] = p_v.z;
}

// The same in GLSL, into `Ch pre` (carried_glsl reads it back); p_c an int expression.
ov::str put_carry_glsl(const Cascade::Spec &p_spec, const ov::str &p_c, const ov::str &p_v) {
    const ov::str b = ov::format("%d + %d * (%s)", p_spec.gpu_carry(0), Cascade::FIELD_CHANNELS, p_c);
    static const char *COMP = "wxyz";
    ov::str code;
    for (int c = 0; c < 4; c++) {
        code += ov::format("pre.c[%s + %d] = %s.%c;\n", b, c, p_v, COMP[c]);
    }
    return code;
}

// biome_mod's terms of one biome, op for op, into a0 .. a2 (declared) and p_o's warp[] (after glsl_weights). Its operands as text: a specialised text's params and constant indices, a generic one's run-time reads
// of its record. A term is in the text where `in` says (a specialised text's: its value isn't 0); `tested`, the text
// tests its value where biome_mod does (a generic text's: the values are data).
struct TermText {
    ov::str w, k[3], mf_field, mf_mul, mf_add, warp, field[3];
    bool k_in[3] = {}, mf_in = false, warp_in = false;
    bool tested = false;

    ov::str glsl(const ov::str &p_o) const {
        auto guarded = [&](const ov::str &p_test, const ov::str &p_code) { return tested ? "if (" + p_test + ") {\n" + p_code + "}\n" : p_code; };
        const ov::str mf = ov::format("cl_mul(%s, cl_multifractal(f[int(%s)], %s, %s))", w, mf_field, mf_mul, mf_add);
        ov::str code = ov::format("vec4 wt = %s;\n", !mf_in ? w : (tested ? ov::format("%s >= 0.0 ? %s : %s", mf_field, mf, w) : mf));
        for (int t = 0; t < 3; t++) {
            if (k_in[t]) {
                code += guarded(k[t] + " != 0.0", ov::format("a%d = a%d + wt * %s;\n", t, t, k[t]));
            }
        }
        if (warp_in) {
            ov::str axes;
            for (int ax = 0; ax < 3; ax++) {
                axes += ov::format("%swarp[%d] = cl_add(%swarp[%d], cl_scale(cl_mul(%s, %s), %s));\n", p_o, ax, p_o, ax, w, field[ax], warp);
            }
            code += guarded(warp + " != 0.0", axes);
        }
        return code;
    }
};

class Modulator : public Cascade::Modulation {
public:
    Modulator(const Table &p_table, const Cascade::Spec &p_spec) :
            table(p_table), sample(sample_carry(p_table)) {
        // Biomes play the ground's notes and the sub-mixes' (Field::channel).
        for (const Cascade::Band &band : p_spec.bands) {
            add_entry(p_spec, band, 0, p_spec.note_mod(p_spec.note_rank(band)));
        }
        for (size_t f = 0; f < p_table.fields.size(); f++) {
            for (const Cascade::Band &band : p_table.fields[f].channel ? p_spec.fields[f] : std::vector<Cascade::Band>()) {
                add_entry(p_spec, band, 1 + int(f), p_spec.group_mod(1 + int(f)));
            }
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
        if (p_table.generic) {
            // Its text holds every feature, and its carry at its capacity (the parts are data).
            features[0] = Cascade::MOD_ABS | Cascade::MOD_SQ | Cascade::MOD_WARP;
            for (size_t f = 0; f < p_table.fields.size(); f++) {
                features[1 + f] = p_table.fields[f].channel ? Cascade::MOD_ABS | Cascade::MOD_SQ | Cascade::MOD_WARP : 0u;
            }
            carry = cap_for(carry, p_table.caps.carry, Cascade::MAX_CARRY);
        }
        range.sample = p_table.has_programs();
    }

    // A band some biome modulates, or a sample term: something reads the derive.
    bool needed() const {
        return sample.count > 0 || std::any_of(std::begin(entries), std::end(entries), [](const std::vector<Entry> &p_e) { return !p_e.empty(); });
    }

    uint32_t modulated(int p_m) const override { return p_m >= 0 && p_m <= Cascade::MAX_LEVEL ? mask[p_m] : 0u; }
    Cascade::FieldRange field_range() const override { return range; }
    uint32_t group_features(int p_group) const override { return p_group >= 0 && p_group < Cascade::MAX_GROUPS ? features[p_group] : 0u; }
    int carry_groups() const override { return carry; }
    int copied_carry() const override { return sample.pick; }
    int carry_used(int p_m) const override {
        if (table.generic) {
            return carry; // its layout is data: every last level loads them all
        }
        return p_m >= 0 && p_m <= Cascade::MAX_LEVEL ? used[p_m] : sample.count;
    }

    void modulate(int p_m, bool p_carried, const float p_pre[Cascade::SLOTS], BandMod r_mod[Cascade::MAX_MODS]) const override {
        if (p_m < 0 || p_m > Cascade::MAX_LEVEL || entries[p_m].empty()) {
            return;
        }
        // Above a run's last level: the climate of the point's channels before the level's insertions.
        State st;
        if (!p_carried) {
            Climate::state(table, p_pre, st);
        }
        for (const Entry &e : entries[p_m]) {
            BandMod &o = r_mod[e.slot];
            o = BandMod();
            o.features = e.features;
            if (!p_carried) {
                o = biome_mod(e, st);
                continue;
            }
            for (size_t i = 0; i < e.parts.size(); i++) {
                const Part &pt = e.parts[i];
                Float4 *member = pt.kind == AMP ? o.amp : o.warp;
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
            for (const Entry &e : entries[next]) {
                if (e.parts.empty()) {
                    continue;
                }
                const BandMod o = biome_mod(e, st);
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
        const int64_t dir = table.generic ? plan(r_params) : -1;
        return p_carried ? glsl_carried(p_spec, dir) : glsl_weighted(p_spec, p_derive, dir, r_params);
    }

private:
    // How one biome shapes one band: k = amplitude times the shape's n, |n|, n^2 coefficients (the inserted, centred
    // terms), its multifractal (Multifractal) and its warp.
    struct BiomeBand {
        int biome = 0;
        float k[3] = {};
        Multifractal mf;
        float warp = 0.0f;
        int warp_fields[3] = { -1, -1, -1 };
    };
    // One modulated ground band: the biomes playing it (a term not 0: biome_mod skips the others' terms). parts: the
    // BandMod members some biome makes non-zero, carried from carry group `carry` on (the rest stay 0); slot: its
    // BandMod's (Spec::note_mod, 0 for a level's first note).
    struct Entry {
        uint32_t features = 0;
        std::vector<BiomeBand> biomes;
        std::vector<Part> parts;
        int carry = 0;
        int slot = 0;
    };
    // A generic text's record of a BiomeBand (plan): biome, k[3], multifractal field, mul, add, warp, warp fields[3].
    static constexpr int RECORD = 11;

    // The band's BandMod from the climate p_st of the level above.
    BandMod biome_mod(const Entry &p_e, const State &p_st) const {
        BandMod o;
        o.features = p_e.features;
        const Float4 *f = p_st.fields;
        Float4 a[3] = {};
        for (const BiomeBand &bb : p_e.biomes) {
            const Float4 &w = p_st.w[bb.biome];
            // The weight times the multifractal's.
            const Float4 wt = bb.mf.field >= 0 ? dmul(w, multifractal_weight(f[bb.mf.field], bb.mf.mul, bb.mf.add)) : w;
            for (int t = 0; t < 3; t++) {
                const float k = bb.k[t];
                if (k != 0.0f) {
                    a[t] = Float4{ a[t].x + wt.x * k, a[t].y + wt.y * k, a[t].z + wt.z * k, a[t].w + wt.w * k };
                }
            }
            if (bb.warp != 0.0f) {
                for (int ax = 0; ax < 3; ax++) {
                    o.warp[ax] = dadd(o.warp[ax], dscale(dmul(w, f[bb.warp_fields[ax]]), bb.warp));
                }
            }
        }
        for (int t = 0; t < 3; t++) {
            o.amp[t] = dflush(a[t]);
        }
        return o;
    }

    // Band p_band of group p_group (the ground's, or a sub-mix field's) at BandMod slot p_slot.
    void add_entry(const Cascade::Spec &p_spec, const Cascade::Band &p_band, int p_group, int p_slot) {
        if (std::none_of(table.biomes.begin(), table.biomes.end(), [&](const Biome &p_b) { return find_amp(p_b, p_band, p_group) != nullptr; })) {
            return; // its constant amplitude, unmodulated
        }
        Entry e;
        e.slot = p_slot;
        for (size_t b = 0; b < table.biomes.size(); b++) {
            BiomeBand bb;
            bb.biome = int(b);
            const Amp *a = find_amp(table.biomes[b], p_band, p_group);
            const Shape shape = band_shape(table.biomes[b], p_band, p_group);
            const double amp = biome_amp(table.biomes[b], p_band, p_group);
            for (int t = 0; t < 3; t++) {
                bb.k[t] = float(amp * double(shape.c[1 + t]));
            }
            bb.mf = a ? a->multifractal : Multifractal();
            // Its first warp term on its record, each further one on a record of its own (no amplitude or multifractal).
            const auto set_warp = [](BiomeBand &r_bb, const WarpTerm &p_w) {
                r_bb.warp = p_w.scale;
                for (int ax = 0; ax < 3; ax++) {
                    r_bb.warp_fields[ax] = p_w.field + ax;
                }
            };
            const auto push = [&](const BiomeBand &p_bb) {
                const uint32_t f = (p_bb.k[1] != 0.0f ? Cascade::MOD_ABS : 0u) | (p_bb.k[2] != 0.0f ? Cascade::MOD_SQ : 0u) |
                        (p_bb.warp != 0.0f ? Cascade::MOD_WARP : 0u);
                if (p_bb.k[0] != 0.0f || f != 0u) {
                    e.features |= f;
                    e.biomes.push_back(p_bb);
                }
            };
            if (a && !a->warps.empty()) {
                set_warp(bb, a->warps[0]);
            }
            push(bb);
            for (size_t w = 1; a && w < a->warps.size(); w++) {
                BiomeBand extra;
                extra.biome = int(b);
                set_warp(extra, a->warps[w]);
                push(extra);
            }
        }
        add_parts(e);
        features[p_group] |= e.features;
        const int m = p_spec.insert_level(p_band);
        if (m < 0 || m > Cascade::MAX_LEVEL) {
            return;
        }
        entries[m].push_back(e);
        mask[m] |= 1u << p_group;
        // The fields are carried for the ground's modulated levels.
        range.floor = ov::min(range.floor, m);
        range.ceiling = ov::max(range.ceiling, m);
    }

    // The members biome_mod can make non-zero: an amplitude some biome's k names, the warp.
    static void add_parts(Entry &r_e) {
        auto any = [&](auto p_pred) { return std::any_of(r_e.biomes.begin(), r_e.biomes.end(), p_pred); };
        for (int t = 0; t < 3; t++) {
            if (any([&](const BiomeBand &p_b) { return p_b.k[t] != 0.0f; })) {
                r_e.parts.push_back(Part{ AMP, t });
            }
        }
        if (any([](const BiomeBand &p_b) { return p_b.warp != 0.0f; })) {
            for (int ax = 0; ax < 3; ax++) {
                r_e.parts.push_back(Part{ WARP, ax });
            }
        }
    }

    ov::str glsl_biome_mod(const Entry *p_e, const ov::str &p_target, Cascade::ShaderParams &r_params) const;
    ov::str glsl_derive_entry(const Entry *p_e, const Cascade::Spec &p_spec, Cascade::ShaderParams &r_params) const;
    ov::str glsl_carried(const Cascade::Spec &p_spec, int64_t p_dir) const;
    ov::str glsl_weighted(const Cascade::Spec &p_spec, bool p_derive, int64_t p_dir, Cascade::ShaderParams &r_params) const;
    ov::str glsl_signature(const Cascade::Spec &p_spec, bool p_derive, int64_t p_dir) const;
    ov::str derive_glsl(const Cascade::Spec &p_spec, int64_t p_dir, Cascade::ShaderParams &r_params) const;
    int64_t plan(Cascade::ShaderParams &r_params) const;

    Table table;
    SampleCarry sample;
    int carry = 0;
    std::vector<Entry> entries[Cascade::MAX_LEVEL + 1];
    int used[Cascade::MAX_LEVEL + 1] = {}; // carry_used: the samples' terms and the ground's parts
    uint32_t mask[Cascade::MAX_LEVEL + 1] = {};
    uint32_t features[Cascade::MAX_GROUPS] = {}; // per group
    Cascade::FieldRange range;
};

// A generic text's data (Table::generic), into r_params: a directory of the levels (m at its index + m), each level
// its mask, its entry count and its entries' indices; each entry its slot, features, record count, part count, parts
// (member, carry group) and records (RECORD floats). The directory's index.
int64_t Modulator::plan(Cascade::ShaderParams &r_params) const {
    std::vector<float> levels, records;
    const int64_t base = int64_t(r_params.heap.size());
    int64_t at = base;
    for (int m = 0; m <= Cascade::MAX_LEVEL; m++) {
        at += 2 + int64_t(entries[m].size());
    }
    std::vector<float> d;
    for (int m = 0; m <= Cascade::MAX_LEVEL; m++) {
        d.push_back(float(base + int64_t(levels.size())));
        levels.insert(levels.end(), { float(mask[m]), float(entries[m].size()) });
        for (const Entry &e : entries[m]) {
            levels.push_back(float(at + int64_t(records.size())));
            records.insert(records.end(), { float(e.slot), float(e.features), float(e.biomes.size()), float(e.parts.size()) });
            for (size_t i = 0; i < e.parts.size(); i++) {
                records.insert(records.end(), { float(e.parts[i].member()), float(e.carry + int(i)) });
            }
            for (const BiomeBand &bb : e.biomes) {
                records.insert(records.end(), { float(bb.biome), bb.k[0], bb.k[1], bb.k[2], float(bb.mf.field), bb.mf.mul, bb.mf.add, bb.warp, float(bb.warp_fields[0]), float(bb.warp_fields[1]), float(bb.warp_fields[2]) });
            }
        }
    }
    r_params.put(levels);
    r_params.put(records);
    return r_params.append(d);
}

// biome_mod's loop over the biomes, op for op, as statements declaring a0 .. a2 then adding each biome's terms (into
// p_target's warp[]): entry p_e's biomes in a specialised text; a loop over the records of the entry at
// `int e` in a generic one (p_e null).
ov::str Modulator::glsl_biome_mod(const Entry *p_e, const ov::str &p_target, Cascade::ShaderParams &r_params) const {
    ov::str code = "precise vec4 a0 = vec4(0.0);\nprecise vec4 a1 = vec4(0.0);\nprecise vec4 a2 = vec4(0.0);\n";
    if (p_e == nullptr) {
        using Cascade::ShaderParams;
        TermText t;
        auto at = [](int p_o) { return ShaderParams::heap_at(ov::format("r + %d", p_o)); };
        t.w = "w[int(" + at(0) + ")]";
        for (int i = 0; i < 3; i++) {
            t.k[i] = at(1 + i);
            t.field[i] = "f[int(" + at(8 + i) + ")]";
            t.k_in[i] = true;
        }
        t.mf_field = at(4);
        t.mf_mul = at(5);
        t.mf_add = at(6);
        t.warp = at(7);
        t.mf_in = t.warp_in = t.tested = true;
        code += "int r0 = e + 4 + 2 * int(cl_heap(e + 3));\n";
        code += ov::format("for (int j = 0; j < int(cl_heap(e + 2)); j++) {\nint r = r0 + %d * j;\n", RECORD);
        return code + t.glsl(p_target) + "}\n";
    }
    for (const BiomeBand &bb : p_e->biomes) {
        TermText t;
        t.w = ov::format("w[%d]", bb.biome);
        t.mf_in = bb.mf.field >= 0;
        t.mf_field = ov::itos(bb.mf.field);
        t.mf_mul = t.mf_in ? r_params(bb.mf.mul) : ov::str();
        t.mf_add = t.mf_in ? r_params(bb.mf.add) : ov::str();
        t.warp_in = bb.warp != 0.0f;
        t.warp = t.warp_in ? r_params(bb.warp) : ov::str();
        for (int i = 0; i < 3; i++) {
            t.k_in[i] = bb.k[i] != 0.0f;
            t.k[i] = t.k_in[i] ? r_params(bb.k[i]) : ov::str();
            t.field[i] = ov::format("f[%d]", bb.warp_fields[i]);
        }
        code += "{\n" + t.glsl(p_target) + "}\n";
    }
    return code;
}

// biome_mod for one entry into its carry groups (derive_glsl): p_e's in a specialised text, the entry at `int e` in a
// generic one (p_e null: its parts are data).
ov::str Modulator::glsl_derive_entry(const Entry *p_e, const Cascade::Spec &p_spec, Cascade::ShaderParams &r_params) const {
    ov::str code = "{\nvec4 warp[3] = vec4[3](vec4(0.0), vec4(0.0), vec4(0.0));\n";
    code += glsl_biome_mod(p_e, "", r_params);
    code += "vec4 amp[3] = vec4[3](cl_flush(a0), cl_flush(a1), cl_flush(a2));\n";
    if (p_e == nullptr) {
        code += "for (int q = 0; q < int(cl_heap(e + 3)); q++) {\nint member = int(cl_heap(e + 4 + 2 * q));\n";
        code += "vec4 v = member < 3 ? amp[member] : warp[member - 3];\n";
        return code + put_carry_glsl(p_spec, "int(cl_heap(e + 5 + 2 * q))", "v") + "}\n}\n";
    }
    for (size_t i = 0; i < p_e->parts.size(); i++) {
        code += put_carry_glsl(p_spec, ov::itos(p_e->carry + int(i)), part_glsl(p_e->parts[i]));
    }
    return code + "}\n";
}

namespace {

// A generic loop's skip of an entry whose slot p_spec's view lacks (a field group it doesn't insert: its samples read
// the field carried).
ov::str skip_absent(const Cascade::Spec &p_spec) {
    return ov::format("if (slot >= %d) {\ncontinue;\n}\n", p_spec.mod_count());
}

} // namespace

ov::str Modulator::glsl_carried(const Cascade::Spec &p_spec, int64_t p_dir) const {
    // The last level's: what the level above derived (no weights, so no helpers).
    ov::str code = "\n// Climate::Modulator::modulate: each modulated band's BandMod, carried from the level above.\n";
    code += glsl_signature(p_spec, false, p_dir);
    if (p_dir >= 0) {
        code += "for (int i = 0; i < int(cl_heap(lv + 1)); i++) {\nint e = int(cl_heap(lv + 2 + i));\nint slot = int(cl_heap(e));\n" + skip_absent(p_spec);
        code += "mod[slot].features = uint(cl_heap(e + 1));\nfor (int q = 0; q < int(cl_heap(e + 3)); q++) {\nint member = int(cl_heap(e + 4 + 2 * q));\n";
        code += ov::format("int b = %d + %d * int(cl_heap(e + 5 + 2 * q));\nvec4 v = vec4(pre.c[b + 1], pre.c[b + 2], pre.c[b + 3], pre.c[b]);\n", p_spec.gpu_carry(0), Cascade::FIELD_CHANNELS);
        code += "if (member < 3) {\nmod[slot].amp[member] = v;\n} else {\nmod[slot].warp[member - 3] = v;\n}\n}\n}\n";
        return code + "return mask;\n}\n";
    }
    code += "switch (m) {\n";
    for (int m = 0; m <= Cascade::MAX_LEVEL; m++) {
        if (entries[m].empty()) {
            continue;
        }
        code += ov::format("case %d:\n", m);
        for (const Entry &e : entries[m]) {
            if (e.slot >= p_spec.mod_count()) {
                continue; // a field group the view doesn't insert (its samples read the field carried)
            }
            const ov::str o = ov::format("mod[%d]", e.slot);
            code += ov::format("%s.features = %du;\n", o, int64_t(e.features));
            for (size_t i = 0; i < e.parts.size(); i++) {
                code += ov::format("%s.%s = %s;\n", o, part_glsl(e.parts[i]), carried_glsl(p_spec, "pre", e.carry + int(i)));
            }
        }
        code += "break;\n";
    }
    return code + "default: break;\n}\nreturn mask;\n}\n";
}

// cascade_mods' first lines (Cascade::Modulation::glsl): modulated(m), out where no band is modulated (with the derive:
// not on a level deriving either), then its BandMods at 0. A generic text reads the mask from its level's data
// (`int lv`, p_dir's).
ov::str Modulator::glsl_signature(const Cascade::Spec &p_spec, bool p_derive, int64_t p_dir) const {
    ov::str code = ov::format("uint cascade_mods(int m, %s, out BandMod mod[%d]%s) {\n", p_derive ? "inout Ch pre" : "Ch pre", p_spec.mod_count(), p_derive ? ", WorldPos p" : "");
    if (p_dir >= 0) {
        code += ov::format("int lv = int(cl_param(%d + min(m, %d)));\nuint mask = uint(cl_heap(lv));\n", p_dir, Cascade::MAX_LEVEL);
    } else {
        code += "uint mask = 0u;\nswitch (m) {\n";
        for (int m = 0; m <= Cascade::MAX_LEVEL; m++) {
            code += entries[m].empty() ? ov::str() : ov::format("case %d: mask = %du; break;\n", m, int64_t(mask[m]));
        }
        code += "default: break;\n}\n";
    }
    code += p_derive ? "if (mask == 0u && g_carry_n == 0u) {\nreturn 0u;\n}\n" : "if (mask == 0u) {\nreturn 0u;\n}\n";
    code += ov::format("for (int g = 0; g < %d; g++) {\nmod[g].features = 0u;\n", p_spec.mod_count());
    return code + "for (int t = 0; t < 3; t++) {\nmod[g].amp[t] = vec4(0.0);\nmod[g].warp[t] = vec4(0.0);\n}\n}\n";
}

ov::str Modulator::glsl_weighted(const Cascade::Spec &p_spec, bool p_derive, int64_t p_dir, Cascade::ShaderParams &r_params) const {
    ov::str code = GLSL_MULTIFRACTAL; // after the shared helpers, spliced by the level shader (get_climate_helpers_glsl)
    code += "\n// Climate::Modulator::modulate: the field terms, the biome weights, then each modulated band's BandMod.\n";
    code += glsl_signature(p_spec, p_derive, p_dir);
    code += glsl_weights(table, p_spec, r_params);
    auto entry = [&](const Entry *p_e, const ov::str &p_o) {
        return "{\n" + glsl_biome_mod(p_e, p_o + ".", r_params) + ov::format("%s.amp[0] = cl_flush(a0);\n%s.amp[1] = cl_flush(a1);\n%s.amp[2] = cl_flush(a2);\n}\n", p_o, p_o, p_o);
    };
    if (p_dir >= 0) {
        code += "for (int i = 0; i < int(cl_heap(lv + 1)); i++) {\nint e = int(cl_heap(lv + 2 + i));\nint slot = int(cl_heap(e));\n" + skip_absent(p_spec);
        code += "mod[slot].features = uint(cl_heap(e + 1));\n" + entry(nullptr, "mod[slot]") + "}\n";
    } else {
        code += "switch (m) {\n";
        for (int m = 0; m <= Cascade::MAX_LEVEL; m++) {
            if (entries[m].empty()) {
                continue;
            }
            code += ov::format("case %d: {\n", m);
            for (const Entry &e : entries[m]) {
                if (e.slot >= p_spec.mod_count()) {
                    continue;
                }
                const ov::str o = ov::format("mod[%d]", e.slot);
                code += ov::format("%s.features = %du;\n", o, int64_t(e.features)) + entry(&e, o);
            }
            code += "break;\n}\n";
        }
        code += "default: break;\n}\n";
    }
    code += p_derive ? derive_glsl(p_spec, p_dir, r_params) : ov::str();
    return code + "return mask;\n}\n";
}

// Climate::Modulator::derive at the end of the level's modulation, from its weights (the channels before the level's
// insertions), where g_carry_n says (j = 1): into the carry of the level below, a run's last (its ground's BandMod parts,
// its samples' terms: the carry_used groups, all of them written).
ov::str Modulator::derive_glsl(const Cascade::Spec &p_spec, int64_t p_dir, Cascade::ShaderParams &r_params) const {
    ov::str code = "if (g_carry_n == 0u) {\nreturn mask;\n}\n";
    if (p_dir >= 0) {
        code += ov::format("if (m >= 1) {\nint lv1 = int(cl_param(%d + min(m - 1, %d)));\n", p_dir, Cascade::MAX_LEVEL);
        code += "for (int i = 0; i < int(cl_heap(lv1 + 1)); i++) {\nint e = int(cl_heap(lv1 + 2 + i));\nif (int(cl_heap(e + 3)) == 0) {\ncontinue;\n}\n";
        code += glsl_derive_entry(nullptr, p_spec, r_params) + "}\n}\n";
    } else {
        code += "switch (m) {\n";
        for (int m = 1; m <= Cascade::MAX_LEVEL; m++) {
            if (used[m - 1] == sample.count) {
                continue;
            }
            code += ov::format("case %d:\n", m);
            for (const Entry &e : entries[m - 1]) {
                if (!e.parts.empty()) {
                    code += glsl_derive_entry(&e, p_spec, r_params);
                }
            }
            code += "break;\n";
        }
        code += "default: break;\n}\n";
    }
    if (sample.count > 0) {
        code += "{\n";
        for (size_t b = 0; b < table.biomes.size(); b++) {
            if (sample.weight[b] >= 0) {
                code += put_carry_glsl(p_spec, ov::itos(sample.weight[b]), ov::format("w[%d]", int64_t(b)));
            }
        }
        if (sample.pick >= 0) {
            code += pick_statements_glsl(table, "pk");
            code += put_carry_glsl(p_spec, ov::itos(sample.pick), "vec4(0.0, 0.0, 0.0, pk)");
        }
        for (size_t f = 0; f < table.fields.size(); f++) {
            if (sample.field[f] >= 0) {
                code += put_carry_glsl(p_spec, ov::itos(sample.field[f]), ov::format("f[%d]", int64_t(f)));
            }
        }
        code += "}\n";
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
    spec.generic = p_table.generic;
    spec.level_notes_cap = 0;
    spec.level_notes_cap = p_table.generic ? cap_for(spec.max_level_notes(), p_table.caps.level_notes, Cascade::MAX_LEVEL_NOTES) : 0;
    std::shared_ptr<Modulator> mod = std::make_shared<Modulator>(p_table, spec);
    if (mod->needed()) {
        spec.modulation = mod;
    }
    return spec;
}

} // namespace Climate
