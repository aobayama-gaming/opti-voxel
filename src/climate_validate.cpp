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

// The first band of p_bands like p_like: its octave (p_identity: and its seed and stretch, a note's identity).
const Cascade::Band *find_band(const std::vector<Cascade::Band> &p_bands, const Cascade::Band &p_like, bool p_identity) {
    for (const Cascade::Band &b : p_bands) {
        if (b.k == p_like.k && (!p_identity || (b.seed == p_like.seed && b.stretch == p_like.stretch))) {
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

// A field's bands or the ground's notes: each octave in the cascade's range, once (p_notes: each note, several may share
// an octave), its amplitude in the exact range.
ov::str validate_bands(const std::vector<Cascade::Band> &p_bands, const Ground &p_ground, const ov::str &p_where, bool p_notes = false) {
    for (const Cascade::Band &b : p_bands) {
        if (b.k < ExactNoise::BAND_SHIFT || b.k > p_ground.k_max) {
            return p_where + ov::format("octave k = %d is outside the cascade's range [%d, %d].", b.k, ExactNoise::BAND_SHIFT, p_ground.k_max);
        }
        if (find_band(p_bands, b, p_notes) != &b) {
            return p_where + ov::format("octave k = %d is listed twice.", b.k);
        }
        if (!ExactNoise::in_exact_range(b.amp)) {
            return p_where + ov::format("octave k = %d: amplitude %s outside [2^-16, 2^40].", b.k, b.amp);
        }
    }
    return ov::str();
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
    const ov::str err = validate_bands(f.bands, p_ground, where);
    if (!err.empty()) {
        return err;
    }
    for (const Term &t : f.terms) {
        if (t.source != GROUND && (t.source < 0 || t.source >= p_i)) {
            return where + ov::str("a term reads the ground or an earlier field only.");
        }
        if (bands_of(p_table, t.source) == nullptr) {
            return where + ov::str("a term reads the ground, and there is none.");
        }
        if (!moderate(t.coef)) {
            return where + ov::format("term coefficient %s outside [2^-40, 2^40].", t.coef);
        }
    }
    return ov::str();
}

ov::str validate_amp(const Table &p_table, const Ground &p_ground, const Amp &p_amp) {
    Cascade::Band like;
    like.k = p_amp.k;
    like.seed = p_amp.seed;
    like.stretch = p_amp.stretch;
    if (find_band(p_table.notes, like, true) == nullptr) {
        return ov::format("an amplitude names octave k = %d, which no note has.", p_amp.k);
    }
    if (!std::isfinite(p_amp.amp) || p_amp.amp < 0.0f) {
        return ov::format("octave k = %d: the amplitude must be >= 0.", p_amp.k);
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
    if (p_amp.has_shape) {
        for (const float c : p_amp.shape.c) {
            if (c != 0.0f && !moderate(c)) {
                return ov::format("shape coefficient %s outside [2^-40, 2^40].", c);
            }
        }
    }
    // The inserted amplitude, each shape term's in the exact range, the shaped band under the cap.
    const double amp = double(p_amp.amp);
    const Shape shape = p_amp.has_shape ? p_amp.shape : Shape();
    bool ok = ExactNoise::in_exact_range(p_amp.amp) && amp * double(shape_bound(shape)) <= double(p_ground.amp_cap);
    for (int t = 1; t < 4 && ok; t++) {
        ok = shape.c[t] == 0.0f || ExactNoise::in_exact_range(float(amp * std::fabs(double(shape.c[t]))));
    }
    return p_amp.amp == 0.0f || ok ? ov::str() : ov::format("octave k = %d: the modulated amplitude %s (shaped) is out of range, or over the cap %s.", p_amp.k, amp, p_ground.amp_cap);
}

// A biome's amplitudes: one per note, each validate_amp's.
ov::str validate_amps(const Table &p_table, const Ground &p_ground, const Biome &p_biome, const ov::str &p_where) {
    for (size_t j = 0; j < p_biome.amps.size(); j++) {
        for (size_t i = 0; i < j; i++) {
            const Amp &a = p_biome.amps[i], &b = p_biome.amps[j];
            if (a.k == b.k && a.seed == b.seed && a.stretch == b.stretch) {
                return p_where + ov::format("the note of octave k = %d has two amplitudes.", b.k);
            }
        }
        const ov::str err = validate_amp(p_table, p_ground, p_biome.amps[j]);
        if (!err.empty()) {
            return p_where + err;
        }
    }
    return ov::str();
}

// The ground's largest |h|: each note's largest amplitude in any biome, shaped (max_amp).
double ground_reach(const Table &p_table) {
    double h = 0.0;
    for (const Cascade::Band &band : p_table.notes) {
        h += double(max_amp(p_table, band));
    }
    return h;
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
    if (last != b.at.empty()) {
        return where + ov::str(last ? "the last biome is the default: no 'at'." : "only the last biome (the default) has no 'at'.");
    }
    for (size_t i = 0; i < b.at.size(); i++) {
        const Axis &a = b.at[i];
        if (a.field < 0 || a.field >= int(p_table.fields.size())) {
            return where + ov::str("'at' names a field that doesn't exist.");
        }
        if (i > 0 && b.at[i - 1].field >= a.field) {
            return where + ov::format("'at' names '%s' twice, or out of field order.", p_table.fields[size_t(a.field)].name);
        }
        if ((a.centre != 0.0f && !moderate(a.centre)) || !moderate(a.inv_span) || !(a.inv_span > 0.0f)) {
            return where + ov::format("'at' on '%s': centre or span out of range.", p_table.fields[size_t(a.field)].name);
        }
    }
    if (b.curve.empty()) {
        return validate_amps(p_table, p_ground, b, where);
    }
    // The curve's offset R - h: under the ground's amplitude cap (what the skirts and the cascade's own bands are held
    // to), in the exact range.
    const double bound = double(curve_bound(b.curve));
    if (bound > p_ground.amp_cap || (bound > 0.0 && !ExactNoise::in_exact_range(float(bound)))) {
        return where + ov::format("its curve moves the ground by up to %s voxels, over the ground's cap of %s.", bound, p_ground.amp_cap);
    }
    // Its slope scales the ground's gradient (1 + w C') and its step at the seams (ground_step): a cliff in the curve
    // would outgrow the skirts and the distance estimate.
    const double slope = double(curve_slope_bound(b.curve));
    if (slope > MAX_CURVE_SLOPE) {
        return where + ov::format("its curve's offset R - h has a slope up to %s, over %s: spread its points.", slope, MAX_CURVE_SLOPE);
    }
    // Points all past the heights the ground reaches make the curve a constant offset (a wrong unit or sign).
    const double reach = ground_reach(p_table);
    if (reach > 0.0 && (double(b.curve.x.front()) > reach || double(b.curve.x.back()) < -reach)) {
        return where + ov::format("its curve's points are outside the ground's heights (|h| <= %s voxels): it would be a constant.", reach);
    }
    return validate_amps(p_table, p_ground, b, where);
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
    const ov::str notes = validate_bands(p_table.notes, p_ground, "climate: the ground's notes: ", true);
    if (!notes.empty()) {
        return notes;
    }
    for (size_t i = 0; i < p_table.biomes.size(); i++) {
        const ov::str err = validate_biome(p_table, i, p_ground);
        if (!err.empty()) {
            return err;
        }
    }
    if (p_table.has_state() && p_table.fields.size() + 1 > size_t(Cascade::MAX_FIELD_GROUPS)) {
        return ov::format("climate: a multifractal carries its state as a field group: at most %d fields.", Cascade::MAX_FIELD_GROUPS - 1);
    }
    // What a level derives for the one below (the samples' terms, then one level's band parts) fits the carry.
    const int carry = combined_spec(p_table, Cascade::Spec()).carry_groups();
    if (carry > Cascade::MAX_CARRY) {
        return ov::format("climate: a level derives %d carry groups for the one below, more than the %d the cascade carries (fewer modulated notes "
                          "per level, curved biomes, or shaped / warped notes).",
                carry, Cascade::MAX_CARRY);
    }
    return ov::str();
}

const std::vector<Cascade::Band> *bands_of(const Table &p_table, int p_source) {
    if (p_source == GROUND) {
        return &p_table.notes;
    }
    return p_source >= 0 && p_source < int(p_table.fields.size()) ? &p_table.fields[size_t(p_source)].bands : nullptr;
}

} // namespace Climate
