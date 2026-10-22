#include "climate.hpp"

#include <algorithm>

// The climate's Cascade::Modulation (combined_spec): per modulated band (group, level), each biome's amplitude, on the
// CPU (amplitudes) and in generated GLSL (glsl), op for op.
namespace Climate {

using Cascade::dflush;

namespace {

class Modulator : public Cascade::Modulation {
public:
    Modulator(const Table &p_table, const Cascade::Spec &p_spec) :
            table(p_table) {
        for (int g = 0; g < p_spec.group_count(); g++) {
            for (const Cascade::Band &b : p_spec.group_bands(g)) {
                add_entry(p_spec, g, b);
            }
        }
    }

    bool any() const {
        return std::any_of(std::begin(mask), std::end(mask), [](uint32_t p_m) { return p_m != 0u; });
    }

    uint32_t modulated(int p_m) const override { return p_m >= 0 && p_m <= Cascade::MAX_LEVEL ? mask[p_m] : 0u; }
    Cascade::FieldRange field_range() const override { return range; }

    void amplitudes(int p_m, const float p_pre[Cascade::SLOTS], Float4 r_amp[Cascade::MAX_GROUPS]) const override {
        if (p_m < 0 || p_m > Cascade::MAX_LEVEL || entries[p_m].empty()) {
            return;
        }
        State st;
        Climate::state(table, p_pre, st);
        const Float4 *w = st.w;
        for (const Entry &e : entries[p_m]) {
            Float4 a{ 0.0f, 0.0f, 0.0f, 0.0f };
            for (size_t b = 0; b < e.amps.size(); b++) {
                const float k = e.amps[b];
                if (k != 0.0f) {
                    a = Float4{ a.x + w[b].x * k, a.y + w[b].y * k, a.z + w[b].z * k, a.w + w[b].w * k };
                }
            }
            r_amp[e.group] = dflush(a);
        }
    }

    String glsl(const Cascade::Spec &p_spec) const override;

private:
    // One modulated band: each biome's amplitude (biome_amp).
    struct Entry {
        int group = 0;
        std::vector<float> amps;
    };

    void add_entry(const Cascade::Spec &p_spec, int p_group, const Cascade::Band &p_band) {
        const int field = p_group - 1; // the ground is GROUND
        const bool named = std::any_of(table.biomes.begin(), table.biomes.end(), [&](const Biome &p_b) { return find_amp(p_b, field, p_band.k) != nullptr; });
        if (!named) {
            return; // its constant amplitude, unmodulated
        }
        Entry e;
        e.group = p_group;
        for (const Biome &b : table.biomes) {
            e.amps.push_back(biome_amp(b, field, p_band));
        }
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

    Table table;
    std::vector<Entry> entries[Cascade::MAX_LEVEL + 1];
    uint32_t mask[Cascade::MAX_LEVEL + 1] = {};
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

// amplitudes(), op for op.
String Modulator::glsl(const Cascade::Spec &p_spec) const {
    using ExactNoise::glsl_float;
    String code = glsl_helpers();
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
            for (size_t b = 0; b < e.amps.size(); b++) {
                if (e.amps[b] != 0.0f) {
                    code += vformat(" a = a + w%d * %s;", int64_t(b), glsl_float(e.amps[b]));
                }
            }
            code += vformat(" amp[%d] = cl_flush(a); }\n", e.group);
        }
        code += "        break;\n    }\n";
    }
    code += "    default: break;\n    }\n    return mask;\n}\n";
    return code;
}

} // namespace

Cascade::Spec combined_spec(const Table &p_table, const Cascade::Spec &p_ground) {
    Cascade::Spec spec = p_ground.ground_only();
    for (const Field &f : p_table.fields) {
        spec.fields.push_back(f.bands);
    }
    std::shared_ptr<Modulator> mod = std::make_shared<Modulator>(p_table, spec);
    if (mod->any()) {
        spec.modulation = mod; // a LINEAR4 ground: validate refuses multipliers on another
    }
    return spec;
}

} // namespace Climate
