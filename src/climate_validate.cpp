#include "climate.hpp"

#include <algorithm>
#include <cmath>

// A climate table's validation (validate): what parse() refuses, and what a table built by hand must satisfy.
namespace Climate {

namespace {

// Coefficients, bounds and scales: |x| within [2^-40, 2^40], so their products with field values stay normal.
bool moderate(double p_v) {
    const double a = std::fabs(p_v);
    return std::isfinite(p_v) && a >= std::ldexp(1.0, -40) && a <= std::ldexp(1.0, 40);
}

const Cascade::Band *find_band(const std::vector<Cascade::Band> &p_bands, int p_k) {
    for (const Cascade::Band &b : p_bands) {
        if (b.k == p_k) {
            return &b;
        }
    }
    return nullptr;
}

template <typename T>
bool name_taken_before(const std::vector<T> &p_list, size_t p_i) {
    for (size_t j = 0; j < p_i; j++) {
        if (p_list[j].name == p_list[p_i].name) {
            return true;
        }
    }
    return false;
}

String validate_field(const Table &p_table, int p_i, const Ground &p_ground) {
    const Field &f = p_table.fields[size_t(p_i)];
    if (!f.name.is_valid_ascii_identifier() || f.name == "ground") {
        return vformat("climate field %d: the name must be an identifier (letters, digits, _), and 'ground' is reserved.", p_i);
    }
    const String where = "climate field '" + f.name + String("': ");
    if (name_taken_before(p_table.fields, size_t(p_i))) {
        return where + String("the name is used twice.");
    }
    for (const Cascade::Band &b : f.bands) {
        if (b.k < ExactNoise::BAND_SHIFT || b.k > p_ground.k_max) {
            return where + vformat("octave k = %d is outside the ground's range [%d, %d].", b.k, ExactNoise::BAND_SHIFT, p_ground.k_max);
        }
        if (find_band(f.bands, b.k) != &b) {
            return where + vformat("octave k = %d is listed twice.", b.k);
        }
        if (!ExactNoise::in_exact_range(b.amp)) {
            return where + vformat("octave k = %d: amplitude %s outside [2^-16, 2^40].", b.k, b.amp);
        }
    }
    for (const Term &t : f.terms) {
        if (t.source != GROUND && (t.source < 0 || t.source >= p_i)) {
            return where + String("a term reads the ground or an earlier field only.");
        }
        if (bands_of(p_table, p_ground, t.source) == nullptr) {
            return where + String("a term reads the ground, and there is none.");
        }
        if (!moderate(t.coef)) {
            return where + vformat("term coefficient %s outside [2^-40, 2^40].", t.coef);
        }
    }
    return String();
}

String validate_amp(const Table &p_table, const Ground &p_ground, const Amp &p_amp) {
    const std::vector<Cascade::Band> *bands = bands_of(p_table, p_ground, p_amp.field);
    const Cascade::Band *band = bands ? find_band(*bands, p_amp.k) : nullptr;
    if (band == nullptr) {
        return vformat("an amplitude multiplier names octave k = %d of a field that doesn't have it.", p_amp.k);
    }
    if (!std::isfinite(p_amp.mult) || p_amp.mult < 0.0f) {
        return vformat("octave k = %d: the multiplier must be >= 0.", p_amp.k);
    }
    // The inserted amplitude: in the exact range, a ground one under the ground's cap.
    const double amp = double(p_amp.mult) * double(band->amp);
    const bool ok = ExactNoise::in_exact_range(float(amp)) && (p_amp.field != GROUND || amp <= p_ground.amp_cap);
    return p_amp.mult == 0.0f || ok ? String() : vformat("octave k = %d: the modulated amplitude %s is out of range.", p_amp.k, amp);
}

String validate_biome(const Table &p_table, size_t p_i, const Ground &p_ground) {
    const Biome &b = p_table.biomes[p_i];
    if (b.name.is_empty()) {
        return vformat("biome %d: a name is required.", int64_t(p_i));
    }
    const String where = "biome '" + b.name + String("': ");
    if (name_taken_before(p_table.biomes, p_i)) {
        return where + String("the name is used twice.");
    }
    const bool last = p_i + 1 == p_table.biomes.size();
    if (last != b.edges.empty()) {
        return where + String(last ? "the last biome is the default: no region." : "only the last biome (the default) has no region.");
    }
    for (const Edge &e : b.edges) {
        if (e.field < 0 || e.field >= int(p_table.fields.size())) {
            return where + String("an edge reads a field that doesn't exist.");
        }
        if ((e.anchor != 0.0f && !moderate(e.anchor)) || !moderate(e.scale)) {
            return where + vformat("edge on '%s': bound or fade out of range.", p_table.fields[size_t(e.field)].name);
        }
    }
    for (size_t j = 0; j < b.amps.size(); j++) {
        for (size_t i = 0; i < j; i++) {
            if (b.amps[i].field == b.amps[j].field && b.amps[i].k == b.amps[j].k) {
                return where + vformat("octave k = %d of one field has two multipliers.", b.amps[j].k);
            }
        }
        const String err = validate_amp(p_table, p_ground, b.amps[j]);
        if (!err.is_empty()) {
            return where + err;
        }
    }
    return String();
}

} // namespace

String validate(const Table &p_table, const Ground &p_ground) {
    if (p_table.fields.size() > size_t(MAX_FIELDS)) {
        return vformat("climate: %d fields, at most %d.", int64_t(p_table.fields.size()), MAX_FIELDS);
    }
    if (p_table.biomes.empty() || p_table.biomes.size() > size_t(MAX_BIOMES)) {
        return vformat("climate: %d biomes, 1 to %d.", int64_t(p_table.biomes.size()), MAX_BIOMES);
    }
    for (int i = 0; i < int(p_table.fields.size()); i++) {
        const String err = validate_field(p_table, i, p_ground);
        if (!err.is_empty()) {
            return err;
        }
    }
    for (size_t i = 0; i < p_table.biomes.size(); i++) {
        const String err = validate_biome(p_table, i, p_ground);
        if (!err.is_empty()) {
            return err;
        }
    }
    // The one check of the ground's channel set (Cascade's checked_top only guards a malformed spec): a modulated band
    // inserts value and gradient (LINEAR4).
    const bool linear4 = p_ground.spec != nullptr && p_ground.spec->set == Cascade::ChannelSet::LINEAR4;
    const bool multiplied = std::any_of(p_table.biomes.begin(), p_table.biomes.end(), [](const Biome &p_b) { return !p_b.amps.empty(); });
    if (multiplied && p_ground.spec != nullptr && !linear4) {
        return "climate: amplitude multipliers need the LINEAR4 ground (channel_set).";
    }
    // A field's multiplier only reaches the mesh through the ground's modulation (the fields are carried where a ground
    // band is modulated, FieldRange): without one it would be ignored.
    const auto multiplies = [&p_table](bool p_ground_band) {
        for (const Biome &b : p_table.biomes) {
            for (const Amp &a : b.amps) {
                if ((a.field == GROUND) == p_ground_band) {
                    return true;
                }
            }
        }
        return false;
    };
    if (multiplies(false) && !multiplies(true)) {
        return "climate: a field's amplitude multiplier needs a ground multiplier too (the fields only reach the mesh through it).";
    }
    return String();
}

const std::vector<Cascade::Band> *bands_of(const Table &p_table, const Ground &p_ground, int p_source) {
    if (p_source == GROUND) {
        return p_ground.spec ? &p_ground.spec->bands : nullptr;
    }
    return p_source >= 0 && p_source < int(p_table.fields.size()) ? &p_table.fields[size_t(p_source)].bands : nullptr;
}

} // namespace Climate
