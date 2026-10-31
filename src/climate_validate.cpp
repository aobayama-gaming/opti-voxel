#include "climate.hpp"
#include "ov/format.hpp"
#include "ov/str.hpp"

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

ov::str validate_field(const Table &p_table, int p_i, const Ground &p_ground) {
    const Field &f = p_table.fields[size_t(p_i)];
    if (!ov::is_valid_ascii_identifier(f.name) || f.name == "ground") {
        return ov::format("climate field %d: the name must be an identifier (letters, digits, _), and 'ground' is reserved.", p_i);
    }
    const ov::str where = "climate field '" + f.name + ov::str("': ");
    if (name_taken_before(p_table.fields, size_t(p_i))) {
        return where + ov::str("the name is used twice.");
    }
    for (const Cascade::Band &b : f.bands) {
        if (b.k < ExactNoise::BAND_SHIFT || b.k > p_ground.k_max) {
            return where + ov::format("octave k = %d is outside the ground's range [%d, %d].", b.k, ExactNoise::BAND_SHIFT, p_ground.k_max);
        }
        if (find_band(f.bands, b.k) != &b) {
            return where + ov::format("octave k = %d is listed twice.", b.k);
        }
        if (!ExactNoise::in_exact_range(b.amp)) {
            return where + ov::format("octave k = %d: amplitude %s outside [2^-16, 2^40].", b.k, b.amp);
        }
    }
    for (const Term &t : f.terms) {
        if (t.source != GROUND && (t.source < 0 || t.source >= p_i)) {
            return where + ov::str("a term reads the ground or an earlier field only.");
        }
        if (bands_of(p_table, p_ground, t.source) == nullptr) {
            return where + ov::str("a term reads the ground, and there is none.");
        }
        if (!moderate(t.coef)) {
            return where + ov::format("term coefficient %s outside [2^-40, 2^40].", t.coef);
        }
    }
    return ov::str();
}

ov::str validate_amp(const Table &p_table, const Ground &p_ground, const Biome &p_biome, const Amp &p_amp) {
    const Cascade::Band *band = p_ground.spec ? find_band(p_ground.spec->bands, p_amp.k) : nullptr;
    if (band == nullptr) {
        return ov::format("an amplitude multiplier names octave k = %d, which the ground doesn't have.", p_amp.k);
    }
    if (!std::isfinite(p_amp.mult) || p_amp.mult < 0.0f) {
        return ov::format("octave k = %d: the multiplier must be >= 0.", p_amp.k);
    }
    if ((p_amp.gain != 0.0f && !(moderate(p_amp.gain) && p_amp.gain > 0.0f)) ||
            (p_amp.warp != 0.0f && !(moderate(p_amp.warp) && std::fabs(p_amp.warp) <= Cascade::WARP_LIMIT))) {
        return ov::format("octave k = %d: multifractal gain or warp out of range.", p_amp.k);
    }
    for (const int f : p_amp.warp_fields) {
        if ((p_amp.warp != 0.0f && f < 0) || f >= int(p_table.fields.size())) {
            return ov::format("octave k = %d: a warp reads 3 fields that exist (warp_fields).", p_amp.k);
        }
    }
    for (const float c : p_amp.shape.c) {
        if (p_amp.has_shape && c != 0.0f && !moderate(c)) {
            return ov::format("shape coefficient %s outside [2^-40, 2^40].", c);
        }
    }
    // The inserted amplitude: the multiplier's, each shape term's in the exact range, the shaped band under the ground's cap.
    const double amp = double(p_amp.mult) * double(band->amp);
    const Shape shape = band_shape(p_biome, p_amp.k);
    bool ok = ExactNoise::in_exact_range(float(amp)) && double(p_amp.mult) <= max_ground_mult(p_ground, p_biome, p_amp.k);
    for (int t = 1; t < 4 && ok; t++) {
        ok = shape.c[t] == 0.0f || ExactNoise::in_exact_range(float(amp * std::fabs(double(shape.c[t]))));
    }
    return p_amp.mult == 0.0f || ok ? ov::str() : ov::format("octave k = %d: the modulated amplitude %s (shaped) is out of range.", p_amp.k, amp);
}

ov::str validate_biome(const Table &p_table, size_t p_i, const Ground &p_ground) {
    const Biome &b = p_table.biomes[p_i];
    if (b.name.empty()) {
        return ov::format("biome %d: a name is required.", int64_t(p_i));
    }
    const ov::str where = "biome '" + b.name + ov::str("': ");
    if (name_taken_before(p_table.biomes, p_i)) {
        return where + ov::str("the name is used twice.");
    }
    const bool last = p_i + 1 == p_table.biomes.size();
    if (last != b.edges.empty()) {
        return where + ov::str(last ? "the last biome is the default: no edges." : "only the last biome (the default) has no edges.");
    }
    for (const Edge &e : b.edges) {
        if (e.field < 0 || e.field >= int(p_table.fields.size())) {
            return where + ov::str("an edge reads a field that doesn't exist.");
        }
        if ((e.anchor != 0.0f && !moderate(e.anchor)) || !moderate(e.scale)) {
            return where + ov::format("edge on '%s': bound or fade out of range.", p_table.fields[size_t(e.field)].name);
        }
    }
    // The curve's offset R - h: under the ground's amplitude cap (what the skirts and the cascade's own bands are held
    // to), in the exact range.
    const double bound = b.curve.empty() ? 0.0 : double(curve_bound(b.curve));
    if (bound > p_ground.amp_cap || (bound > 0.0 && !ExactNoise::in_exact_range(float(bound)))) {
        return where + ov::format("its curve moves the ground by up to %s voxels, over the ground's cap of %s.", bound, p_ground.amp_cap);
    }
    for (size_t j = 0; j < b.amps.size(); j++) {
        for (size_t i = 0; i < j; i++) {
            if (b.amps[i].k == b.amps[j].k) {
                return where + ov::format("octave k = %d has two multipliers.", b.amps[j].k);
            }
        }
        const ov::str err = validate_amp(p_table, p_ground, b, b.amps[j]);
        if (!err.empty()) {
            return where + err;
        }
    }
    return ov::str();
}

} // namespace

ov::str validate(const Table &p_table, const Ground &p_ground) {
    if (p_table.fields.size() > size_t(MAX_FIELDS)) {
        return ov::format("climate: %d fields, at most %d.", int64_t(p_table.fields.size()), MAX_FIELDS);
    }
    if (p_table.biomes.empty() || p_table.biomes.size() > size_t(MAX_BIOMES)) {
        return ov::format("climate: %d biomes, 1 to %d.", int64_t(p_table.biomes.size()), MAX_BIOMES);
    }
    for (int i = 0; i < int(p_table.fields.size()); i++) {
        const ov::str err = validate_field(p_table, i, p_ground);
        if (!err.empty()) {
            return err;
        }
    }
    for (size_t i = 0; i < p_table.biomes.size(); i++) {
        const ov::str err = validate_biome(p_table, i, p_ground);
        if (!err.empty()) {
            return err;
        }
    }
    // The one check of the ground's channel set (Cascade's checked_top only guards a malformed spec): a modulated band
    // inserts value and gradient (LINEAR4), and perlin3_cross has no stretch.
    const bool multiplied = std::any_of(p_table.biomes.begin(), p_table.biomes.end(), [](const Biome &p_b) { return !p_b.amps.empty(); });
    if ((multiplied || p_table.has_stretch()) && p_ground.spec != nullptr && p_ground.spec->set != Cascade::ChannelSet::LINEAR4) {
        return "climate: amplitude multipliers and stretched octaves need the LINEAR4 ground (channel_set).";
    }
    if (p_table.has_state() && p_table.fields.size() + 1 > size_t(Cascade::MAX_FIELD_GROUPS)) {
        return ov::format("climate: a multifractal carries its state as a field group: at most %d fields.", Cascade::MAX_FIELD_GROUPS - 1);
    }
    // What a level derives for the one below (the samples' terms, then one level's band parts) fits the carry.
    if (p_ground.spec != nullptr) {
        const int carry = combined_spec(p_table, *p_ground.spec).carry_groups();
        if (carry > Cascade::MAX_CARRY) {
            return ov::format("climate: a level derives %d carry groups for the one below, more than the %d the cascade carries (fewer modulated bands "
                              "per level, curved biomes, or shaped / warped multipliers).",
                    carry, Cascade::MAX_CARRY);
        }
    }
    return ov::str();
}

const std::vector<Cascade::Band> *bands_of(const Table &p_table, const Ground &p_ground, int p_source) {
    if (p_source == GROUND) {
        return p_ground.spec ? &p_ground.spec->bands : nullptr;
    }
    return p_source >= 0 && p_source < int(p_table.fields.size()) ? &p_table.fields[size_t(p_source)].bands : nullptr;
}

double max_ground_mult(const Ground &p_ground, const Biome &p_biome, int p_k) {
    const Cascade::Band *band = p_ground.spec ? find_band(p_ground.spec->bands, p_k) : nullptr;
    if (band == nullptr) {
        return 0.0;
    }
    const double unit = double(band->amp) * double(shape_bound(band_shape(p_biome, p_k)));
    return unit > 0.0 ? double(p_ground.amp_cap) / unit : INFINITY;
}

} // namespace Climate
