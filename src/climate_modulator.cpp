#include "climate.hpp"

#include <algorithm>

// The climate's Cascade::Modulation (combined_spec): per modulated band (group, level), how each biome shapes it, on
// the CPU (modulate) and in generated GLSL (glsl), op for op.
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

class Modulator : public Cascade::Modulation {
public:
    Modulator(const Table &p_table, const Cascade::Spec &p_spec) :
            table(p_table) {
        state = p_table.has_state() ? p_spec.group_count() - 1 : -1;
        for (int g = 0; g < p_spec.group_count(); g++) {
            const std::vector<Cascade::Band> &bands = p_spec.group_bands(g);
            for (size_t i = 0; i < bands.size(); i++) {
                add_entry(p_spec, g, bands[i], g == 0 ? Shape() : table.fields[size_t(g - 1)].shapes[i]);
            }
        }
    }

    bool any() const {
        return std::any_of(std::begin(mask), std::end(mask), [](uint32_t p_m) { return p_m != 0u; });
    }

    uint32_t modulated(int p_m) const override { return p_m >= 0 && p_m <= Cascade::MAX_LEVEL ? mask[p_m] : 0u; }
    Cascade::FieldRange field_range() const override { return range; }
    int state_group() const override { return state; }
    bool shaped() const override { return shaped_; }
    uint32_t group_features(int p_group) const override { return p_group >= 0 && p_group < Cascade::MAX_GROUPS ? features[p_group] : 0u; }

    void modulate(int p_m, const float p_pre[Cascade::SLOTS], BandMod r_mod[Cascade::MAX_GROUPS]) const override {
        if (p_m < 0 || p_m > Cascade::MAX_LEVEL || entries[p_m].empty()) {
            return;
        }
        Float4 f[MAX_FIELDS];
        apply_terms(table, p_pre, f);
        Float4 w[MAX_BIOMES];
        weights(table, f, w);
        const Float4 c = state >= 0 ? to_float4(p_pre + Cascade::field_slot(state - 1)) : Float4{ 0.0f, 0.0f, 0.0f, 0.0f };
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
            Float4 a[3] = {};
            for (size_t b = 0; b < e.biomes.size(); b++) {
                const BiomeBand &bb = e.biomes[b];
                // The weight times the multifractal's.
                const Float4 wt = bb.gain > 0.0f ? dmul(w[b], multifractal_weight(bb.gain, c)) : w[b];
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
        }
    }

    String glsl(const Cascade::Spec &p_spec) const override;

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
    // One modulated band: per biome, or constant (a field band's own shape, no biome naming it).
    struct Entry {
        int group = 0;
        uint32_t features = 0;
        bool constant = false;
        float k[3] = {};
        std::vector<BiomeBand> biomes;
    };

    static uint32_t shape_features(const float p_k[3]) {
        return (p_k[1] != 0.0f ? Cascade::MOD_ABS : 0u) | (p_k[2] != 0.0f ? Cascade::MOD_SQ : 0u);
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
            range.floor = MIN(range.floor, m);
            range.ceiling = MAX(range.ceiling, m);
        }
    }

    String glsl_plain(const Cascade::Spec &p_spec) const;
    String glsl_entry(const Entry &p_e) const;

    Table table;
    int state = -1;
    bool shaped_ = false;
    std::vector<Entry> entries[Cascade::MAX_LEVEL + 1];
    uint32_t mask[Cascade::MAX_LEVEL + 1] = {};
    uint32_t features[Cascade::MAX_GROUPS] = {};
    Cascade::FieldRange range;
};

String mask_switch(const uint32_t p_mask[Cascade::MAX_LEVEL + 1]) {
    String code = "    uint mask = 0u;\n    switch (m) {\n";
    for (int m = 0; m <= Cascade::MAX_LEVEL; m++) {
        if (p_mask[m] != 0u) {
            code += vformat("    case %d: mask = %du; break;\n", m, int64_t(p_mask[m]));
        }
    }
    return code + "    default: break;\n    }\n    if (mask == 0u) {\n        return 0u;\n    }\n";
}

// The amplitudes alone (no band shaped anywhere): the text the climate had before shaped bands.
String Modulator::glsl_plain(const Cascade::Spec &p_spec) const {
    using ExactNoise::glsl_float;
    String code = splices_helpers(table, p_spec) ? String() : glsl_helpers();
    code += "\n// Climate::Modulator::amplitudes: the field terms, the biome weights, then each modulated band's amplitude.\n";
    code += vformat("uint cascade_amplitudes(int m, Ch pre, out vec4 amp[%d]) {\n", p_spec.group_count());
    code += vformat("    for (int g = 0; g < %d; g++) {\n        amp[g] = vec4(0.0);\n    }\n", p_spec.group_count());
    code += mask_switch(mask);
    code += glsl_weights(table, p_spec);
    code += "    switch (m) {\n";
    for (int m = 0; m <= Cascade::MAX_LEVEL; m++) {
        if (entries[m].empty()) {
            continue;
        }
        code += vformat("    case %d: {\n", m);
        for (const Entry &e : entries[m]) {
            code += "        { precise vec4 a = vec4(0.0);";
            for (size_t b = 0; b < e.biomes.size(); b++) {
                if (e.biomes[b].k[0] != 0.0f) {
                    code += vformat(" a = a + w%d * %s;", int64_t(b), glsl_float(e.biomes[b].k[0]));
                }
            }
            code += vformat(" amp[%d] = cl_flush(a); }\n", e.group);
        }
        code += "        break;\n    }\n";
    }
    code += "    default: break;\n    }\n    return mask;\n}\n";
    return code;
}

// modulate()'s loop body for one entry, op for op.
String Modulator::glsl_entry(const Entry &p_e) const {
    using ExactNoise::glsl_float;
    const String o = vformat("mod[%d]", p_e.group);
    String code = vformat("        %s.features = %du;\n", o, int64_t(p_e.features));
    if (p_e.constant) {
        for (int t = 0; t < 3; t++) {
            code += vformat("        %s.amp[%d] = vec4(0.0, 0.0, 0.0, %s);\n", o, t, glsl_float(p_e.k[t]));
        }
        return code;
    }
    code += "        {\n            precise vec4 a0 = vec4(0.0);\n            precise vec4 a1 = vec4(0.0);\n            precise vec4 a2 = vec4(0.0);\n";
    for (size_t b = 0; b < p_e.biomes.size(); b++) {
        const BiomeBand &bb = p_e.biomes[b];
        const String wb = vformat("w%d", int64_t(b));
        const String wt = bb.gain > 0.0f ? vformat("cl_mul(%s, cl_multifractal(%s, state_c))", wb, glsl_float(bb.gain)) : wb;
        code += vformat("            {\n                vec4 wt = %s;\n", wt);
        for (int t = 0; t < 3; t++) {
            if (bb.k[t] != 0.0f) {
                code += vformat("                a%d = a%d + wt * %s;\n", t, t, glsl_float(bb.k[t]));
            }
        }
        if (bb.gain > 0.0f) {
            code += vformat("                %s.state[0] = cl_add(%s.state[0], cl_sub(%s, cl_scale(wt, %s)));\n", o, o, wb, glsl_float(bb.shape.c[0]));
            for (int t = 1; t < 4; t++) {
                if (bb.shape.c[t] != 0.0f) {
                    code += vformat("                %s.state[%d] = cl_sub(%s.state[%d], cl_scale(wt, %s));\n", o, t, o, t, glsl_float(bb.shape.c[t]));
                }
            }
        }
        if (bb.warp != 0.0f) {
            for (int ax = 0; ax < 3; ax++) {
                code += vformat("                %s.warp[%d] = cl_add(%s.warp[%d], cl_scale(cl_mul(%s, f%d), %s));\n", o, ax, o, ax, wb, bb.warp_fields[ax], glsl_float(bb.warp));
            }
        }
        code += "            }\n";
    }
    code += vformat("            %s.amp[0] = cl_flush(a0);\n            %s.amp[1] = cl_flush(a1);\n            %s.amp[2] = cl_flush(a2);\n        }\n", o, o, o);
    return code;
}

String Modulator::glsl(const Cascade::Spec &p_spec) const {
    if (!shaped_) {
        return glsl_plain(p_spec);
    }
    using ExactNoise::glsl_float;
    String code = GLSL_MULTIFRACTAL; // after the shared helpers, spliced by the level shader (splices_helpers)
    code += "\n// Climate::Modulator::modulate: the field terms, the biome weights, then each modulated band's BandMod.\n";
    code += vformat("uint cascade_mods(int m, Ch pre, out BandMod mod[%d]) {\n", p_spec.group_count());
    code += vformat("    for (int g = 0; g < %d; g++) {\n        mod[g].features = 0u;\n", p_spec.group_count());
    code += "        for (int t = 0; t < 3; t++) {\n            mod[g].amp[t] = vec4(0.0);\n            mod[g].warp[t] = vec4(0.0);\n        }\n";
    code += "        for (int t = 0; t < 4; t++) {\n            mod[g].state[t] = vec4(0.0);\n        }\n    }\n";
    code += mask_switch(mask);
    code += glsl_weights(table, p_spec);
    // The state's channels (the multifractal's complement).
    if (state >= 0) {
        const int sb = p_spec.gpu_base(state);
        code += vformat("    vec4 state_c = vec4(pre.c[%d], pre.c[%d], pre.c[%d], pre.c[%d]);\n", sb + 1, sb + 2, sb + 3, sb);
    }
    code += "    switch (m) {\n";
    for (int m = 0; m <= Cascade::MAX_LEVEL; m++) {
        if (entries[m].empty()) {
            continue;
        }
        code += vformat("    case %d: {\n", m);
        for (const Entry &e : entries[m]) {
            code += glsl_entry(e);
        }
        code += "        break;\n    }\n";
    }
    code += "    default: break;\n    }\n    return mask;\n}\n";
    return code;
}

} // namespace

Cascade::Spec combined_spec(const Table &p_table, const Cascade::Spec &p_ground) {
    Cascade::Spec spec = p_ground;
    for (Cascade::Band &b : spec.bands) {
        for (const std::pair<int, int> &s : p_table.ground_stretch) {
            b.stretch = s.first == b.k ? s.second : b.stretch;
        }
    }
    spec.fields.clear();
    spec.modulation.reset();
    spec.cell_groups = 0;
    for (size_t i = 0; i < p_table.fields.size(); i++) {
        spec.fields.push_back(p_table.fields[i].bands);
        spec.cell_groups |= p_table.fields[i].cell ? 1u << (1 + i) : 0u;
    }
    if (p_table.has_state()) {
        spec.fields.emplace_back(); // the multifractal state: band-less, the ground's band overwrites it
    }
    std::shared_ptr<Modulator> mod = std::make_shared<Modulator>(p_table, spec);
    if (mod->any()) {
        ERR_FAIL_COND_V_MSG(spec.set != Cascade::ChannelSet::LINEAR4, spec, "Climate: amplitude multipliers need a LINEAR4 ground; the climate is carried unmodulated.");
        spec.modulation = mod;
    }
    return spec;
}

} // namespace Climate
