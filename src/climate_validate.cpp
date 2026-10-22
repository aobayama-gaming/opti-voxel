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

// A shape's coefficients: each 0 or within [2^-40, 2^40]; a cell noise field's bands (p_cell) are plain.
String validate_shape(const Shape &p_shape, bool p_cell) {
    for (const float c : p_shape.c) {
        if (c != 0.0f && !moderate(c)) {
            return vformat("shape coefficient %s outside [2^-40, 2^40].", c);
        }
    }
    return p_cell && !p_shape.plain() ? String("a cell noise field's bands take no shape.") : String();
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
    for (const Shape &sh : f.shapes) {
        const String err = validate_shape(sh, f.cell);
        if (!err.is_empty()) {
            return where + err;
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

String validate_amp(const Table &p_table, const Ground &p_ground, const Biome &p_biome, const Amp &p_amp) {
    const std::vector<Cascade::Band> *bands = bands_of(p_table, p_ground, p_amp.field);
    const Cascade::Band *band = bands ? find_band(*bands, p_amp.k) : nullptr;
    if (band == nullptr) {
        return vformat("an amplitude multiplier names octave k = %d of a field that doesn't have it.", p_amp.k);
    }
    if (!std::isfinite(p_amp.mult) || p_amp.mult < 0.0f) {
        return vformat("octave k = %d: the multiplier must be >= 0.", p_amp.k);
    }
    if ((p_amp.gain != 0.0f || p_amp.warp != 0.0f) && p_amp.field != GROUND) {
        return vformat("octave k = %d: multifractal and warp shape ground bands only.", p_amp.k);
    }
    if ((p_amp.gain != 0.0f && !(moderate(p_amp.gain) && p_amp.gain > 0.0f)) ||
            (p_amp.warp != 0.0f && !(moderate(p_amp.warp) && std::fabs(p_amp.warp) <= Cascade::WARP_LIMIT))) {
        return vformat("octave k = %d: multifractal gain or warp out of range.", p_amp.k);
    }
    for (const int f : p_amp.warp_fields) {
        if ((p_amp.warp != 0.0f && f < 0) || f >= int(p_table.fields.size())) {
            return vformat("octave k = %d: a warp reads 3 fields that exist (warp_fields).", p_amp.k);
        }
    }
    if (p_amp.has_shape) {
        const String err = validate_shape(p_amp.shape, p_amp.field != GROUND && p_table.fields[size_t(p_amp.field)].cell);
        if (!err.is_empty()) {
            return err;
        }
    }
    // The inserted amplitude: the multiplier's, each shape term's in the exact range, the shaped band under the ground's cap.
    const double amp = double(p_amp.mult) * double(band->amp);
    const Shape shape = band_shape(p_table, &p_biome, p_amp.field, p_amp.k);
    bool ok = ExactNoise::in_exact_range(float(amp)) && (p_amp.field != GROUND || double(p_amp.mult) <= max_ground_mult(p_table, p_ground, p_biome, p_amp.k));
    for (int t = 1; t < 4 && ok; t++) {
        ok = shape.c[t] == 0.0f || ExactNoise::in_exact_range(float(amp * std::fabs(double(shape.c[t]))));
    }
    return p_amp.mult == 0.0f || ok ? String() : vformat("octave k = %d: the modulated amplitude %s (shaped) is out of range.", p_amp.k, amp);
}

// max |value| of group p_source (GROUND or a field): its bands' largest amplitudes times their shapes' bounds, plus its
// terms' |coef| times their sources' bounds.
double value_bound(const Table &p_table, const Ground &p_ground, int p_source) {
    const std::vector<Cascade::Band> *bands = bands_of(p_table, p_ground, p_source);
    double b = 0.0;
    if (bands != nullptr) {
        for (const Cascade::Band &band : *bands) {
            b += double(max_amp(p_table, p_source, band));
        }
    }
    if (p_source != GROUND) {
        for (const Term &t : p_table.fields[size_t(p_source)].terms) {
            b += std::fabs(double(t.coef)) * value_bound(p_table, p_ground, t.source);
        }
    }
    return b;
}

String validate_height(const Table &p_table, const Ground &p_ground, const HeightTerm &p_h) {
    if (p_h.field < 0 || p_h.field >= int(p_table.fields.size())) {
        return "climate: a height term reads a field that doesn't exist.";
    }
    if (!moderate(p_h.coef)) {
        return vformat("climate: height term coefficient %s outside [2^-40, 2^40] voxels.", p_h.coef);
    }
    const String &name = p_table.fields[size_t(p_h.field)].name;
    const double range = value_bound(p_table, p_ground, p_h.field);
    // A curve flat over the field's whole range (its points all past the values the field reaches) is a constant.
    if (p_h.has_curve && (double(p_h.curve.x.front()) > range || double(p_h.curve.x.back()) < -range)) {
        return vformat("climate: the height term on '%s' has its curve's points outside the field's values (|%s| <= %s): it would be a constant.", name,
                name, range);
    }
    // The term's largest value: in the exact range, and under the ground's amplitude cap (what the skirts and the
    // cascade's own bands are held to).
    const double bound = std::fabs(double(p_h.coef)) * (p_h.has_curve ? double(curve_bound(p_h.curve)) : range);
    if (bound > p_ground.amp_cap || (bound > 0.0 && !ExactNoise::in_exact_range(float(bound)))) {
        return vformat("climate: the height term on '%s' reaches %s voxels, over the ground's cap of %s.", name, bound, p_ground.amp_cap);
    }
    return String();
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
    if (b.terrace.step != 0.0f && !(moderate(b.terrace.step) && b.terrace.step > 0.0f)) {
        return where + String("the terrace step must be > 0.");
    }
    for (size_t j = 0; j < b.amps.size(); j++) {
        for (size_t i = 0; i < j; i++) {
            if (b.amps[i].field == b.amps[j].field && b.amps[i].k == b.amps[j].k) {
                return where + vformat("octave k = %d of one field has two multipliers.", b.amps[j].k);
            }
        }
        const String err = validate_amp(p_table, p_ground, b, b.amps[j]);
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
    // inserts value and gradient (LINEAR4), and perlin3_cross has no stretch.
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
    if (p_table.has_stretch() && p_ground.spec != nullptr && !linear4) {
        return "climate: stretched bands need a LINEAR4 ground.";
    }
    if (p_table.has_state() && p_table.fields.size() + 1 > size_t(Cascade::MAX_FIELD_GROUPS)) {
        return vformat("climate: a multifractal carries its state as a field group: at most %d fields.", Cascade::MAX_FIELD_GROUPS - 1);
    }
    for (const HeightTerm &h : p_table.heights) {
        const String err = validate_height(p_table, p_ground, h);
        if (!err.is_empty()) {
            return err;
        }
    }
    return String();
}

const std::vector<Cascade::Band> *bands_of(const Table &p_table, const Ground &p_ground, int p_source) {
    if (p_source == GROUND) {
        return p_ground.spec ? &p_ground.spec->bands : nullptr;
    }
    return p_source >= 0 && p_source < int(p_table.fields.size()) ? &p_table.fields[size_t(p_source)].bands : nullptr;
}

double max_ground_mult(const Table &p_table, const Ground &p_ground, const Biome &p_biome, int p_k) {
    const Cascade::Band *band = p_ground.spec ? find_band(p_ground.spec->bands, p_k) : nullptr;
    if (band == nullptr) {
        return 0.0;
    }
    const double unit = double(band->amp) * double(shape_bound(band_shape(p_table, &p_biome, GROUND, p_k)));
    return unit > 0.0 ? double(p_ground.amp_cap) / unit : INFINITY;
}

} // namespace Climate
