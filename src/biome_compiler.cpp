#include "biome_compiler.hpp"

#include "exact_noise.hpp"

#include "ov/format.hpp"
#include "ov/math.hpp"

#include <algorithm>
#include <cmath>

namespace Climate {

std::vector<int> note_octaves(const NoteNode &p_note, const Ground &p_ground) {
    std::vector<int> ks;
    for (int k = p_note.hi; k >= p_note.lo; k--) {
        if (!p_note.range || (k >= ExactNoise::BAND_SHIFT && k <= p_ground.k_max)) {
            ks.push_back(k);
        }
    }
    return ks;
}

float note_amp(const NoteNode &p_note, int p_k, const Ground &p_ground) {
    double a = p_note.steepness > 0.0 ? p_note.steepness * std::ldexp(1.0, p_k) : p_note.amp * std::pow(p_note.persistence, p_note.hi - p_k);
    a = ov::min(a, p_note.max_amp);
    return float(ov::min(a, p_ground.note_cap));
}

namespace {

// Whether two amplitude entries play a note the same way (a constant note: every biome's alike and plain).
bool plain(const Amp &p_a) {
    return !p_a.has_shape && p_a.gain == 0.0f && p_a.warp == 0.0f;
}

// One compiled note: its node's seed, its band, per biome its entry (-1: the biome doesn't mix it), whether every biome
// plays it alike (a constant band) and its largest amplitude.
struct Note {
    uint32_t seed = 0;
    Cascade::Band band;
    std::vector<int> entry;
    bool constant = false;
    float top = 0.0f;
};

// Every biome's rows into r_notes (one per identity) and each biome's entries r_amps[b]: empty, or the first error.
ov::str gather(const std::vector<BiomeGraph> &p_graphs, const Ground &p_ground, const Table &p_table, std::vector<Note> &r_notes, std::vector<std::vector<Amp>> &r_amps) {
    const size_t n = p_table.biomes.size();
    for (size_t b = 0; b < n; b++) {
        const ov::str where = "biome '" + p_table.biomes[b].name + ov::str("': ");
        for (const GraphRow &row : p_graphs[b].rows) {
            for (const int k : note_octaves(row.note, p_ground)) {
                const int stretch = ov::max(0, ov::min(row.note.stretch, k - ExactNoise::BAND_SHIFT));
                auto same = [&](const Note &p_c) { return p_c.band.k == k && p_c.seed == row.note.seed && p_c.band.stretch == stretch; };
                auto it = std::find_if(r_notes.begin(), r_notes.end(), same);
                if (it == r_notes.end()) {
                    Note note{ row.note.seed, ExactNoise::make_octave(row.note.seed, uint32_t(k), k, 0.0f), std::vector<int>(n, -1) };
                    note.band.stretch = stretch;
                    r_notes.push_back(note);
                    it = r_notes.end() - 1;
                }
                if (it->entry[b] >= 0) {
                    return where + ov::format("the note of octave k = %d is mixed twice.", k);
                }
                Amp a;
                a.k = k;
                a.seed = it->band.seed;
                a.stretch = stretch;
                a.amp = float(double(row.x) * double(note_amp(row.note, k, p_ground)));
                if (!(a.amp >= 0.0f)) {
                    return where + ov::format("octave k = %d: the amplitude must be >= 0 (its note's, times the row's volume).", k);
                }
                a.has_shape = row.has_shape;
                a.shape = row.shape;
                a.gain = row.gain;
                a.warp = row.warp;
                std::copy(std::begin(row.warp_fields), std::end(row.warp_fields), std::begin(a.warp_fields));
                it->entry[b] = int(r_amps[b].size());
                r_amps[b].push_back(a);
            }
        }
    }
    return ov::str();
}

// Each note's fold (constant: every biome plays it plain at one amplitude) and largest amplitude; an octave is modulated
// or constant as a whole (the cascade's levels are: Cascade::MAX_LEVEL_NOTES), at most MAX_LEVEL_NOTES notes and one
// multifractal note (the state group holds one octave's) on it. Empty, or the first error.
ov::str fold(const std::vector<std::vector<Amp>> &p_amps, std::vector<Note> &r_notes) {
    for (Note &note : r_notes) {
        const Amp *first = note.entry[0] >= 0 ? &p_amps[0][size_t(note.entry[0])] : nullptr;
        note.constant = first && plain(*first);
        for (size_t b = 0; b < note.entry.size(); b++) {
            const Amp *a = note.entry[b] >= 0 ? &p_amps[b][size_t(note.entry[b])] : nullptr;
            note.constant = note.constant && a && plain(*a) && ExactNoise::float_bits(a->amp) == ExactNoise::float_bits(first->amp);
            note.top = a ? ov::max(note.top, a->amp) : note.top;
        }
    }
    for (Note &note : r_notes) {
        int count = 0, multifractal = 0;
        bool modulated = false;
        for (const Note &o : r_notes) {
            if (o.band.k == note.band.k && o.top > 0.0f) {
                count++;
                modulated = modulated || !o.constant;
                for (size_t b = 0; b < o.entry.size(); b++) {
                    if (o.entry[b] >= 0 && p_amps[b][size_t(o.entry[b])].gain > 0.0f) {
                        multifractal++;
                        break;
                    }
                }
            }
        }
        if (count > Cascade::MAX_LEVEL_NOTES || multifractal > 1) {
            return ov::format("octave k = %d has %d notes, %d of them multifractal: at most %d, one multifractal (its state).", note.band.k, count, multifractal,
                    Cascade::MAX_LEVEL_NOTES);
        }
        note.constant = note.constant && !modulated;
    }
    return ov::str();
}

// What a sample program's register can be, given |H| <= the ground's reach and each field within its bound: its value
// and its deviation v - H (a curve of H stays tight: R(H) - H = C(H)), each an interval, and the bounds of |dv/dH|,
// |d(v - H)/dH| and |dv/df|.
struct Range {
    double lo = 0.0, hi = 0.0;
    double dlo = 0.0, dhi = 0.0;
    double s_h = 0.0, ds_h = 0.0;
    double s_f[MAX_FIELDS] = {};
    double peak() const { return ov::max(std::fabs(lo), std::fabs(hi)); }
};

// Each field's bound: its bands' amplitudes, plus its terms' |coef| times their source's (the ground: p_reach).
std::vector<double> field_bounds(const Table &p_table, double p_reach) {
    std::vector<double> f(p_table.fields.size(), 0.0);
    for (size_t i = 0; i < p_table.fields.size(); i++) {
        for (const Cascade::Band &b : p_table.fields[i].bands) {
            f[i] += std::fabs(double(b.amp));
        }
        for (const Term &t : p_table.fields[i].terms) {
            f[i] += std::fabs(double(t.coef)) * (t.source == GROUND ? p_reach : f[size_t(t.source)]);
        }
    }
    return f;
}

// r_p's bounds (SampleProgram::bound, slope, field_slope) for a ground within p_reach and fields within p_fields.
void bound_program(SampleProgram &r_p, double p_reach, const std::vector<double> &p_fields) {
    std::vector<Range> r(r_p.code.size());
    for (size_t i = 0; i < r_p.code.size(); i++) {
        const SampleInstr &in = r_p.code[i];
        const Range a = in.a >= 0 ? r[size_t(in.a)] : Range(), b = in.b >= 0 ? r[size_t(in.b)] : Range();
        Range &o = r[i];
        auto deviation = [&]() { // v - H, nothing known of their link
            o.dlo = o.lo - p_reach;
            o.dhi = o.hi + p_reach;
            o.ds_h = o.s_h + 1.0;
        };
        auto fields = [&](double p_ka, double p_kb) {
            for (int f = 0; f < MAX_FIELDS; f++) {
                o.s_f[f] = p_ka * a.s_f[f] + p_kb * b.s_f[f];
            }
        };
        switch (in.op) {
            case SampleOp::GROUND:
                o.lo = -p_reach;
                o.hi = p_reach;
                o.s_h = 1.0;
                break;
            case SampleOp::CONST:
                o.lo = o.hi = double(in.value);
                deviation();
                break;
            case SampleOp::FIELD:
                o.hi = in.index >= 0 && in.index < int(p_fields.size()) ? p_fields[size_t(in.index)] : 0.0;
                o.lo = -o.hi;
                o.s_f[in.index] = 1.0;
                deviation();
                break;
            case SampleOp::ADD:
            case SampleOp::SUB: {
                const double sign = in.op == SampleOp::ADD ? 1.0 : -1.0;
                o.lo = a.lo + (sign > 0.0 ? b.lo : -b.hi);
                o.hi = a.hi + (sign > 0.0 ? b.hi : -b.lo);
                o.dlo = a.dlo + (sign > 0.0 ? b.lo : -b.hi); // (a - H) +- b
                o.dhi = a.dhi + (sign > 0.0 ? b.hi : -b.lo);
                o.s_h = a.s_h + b.s_h;
                o.ds_h = a.ds_h + b.s_h;
                fields(1.0, 1.0);
                break;
            }
            case SampleOp::MUL: {
                const double p[4] = { a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
                o.lo = *std::min_element(p, p + 4);
                o.hi = *std::max_element(p, p + 4);
                o.s_h = a.peak() * b.s_h + b.peak() * a.s_h;
                fields(b.peak(), a.peak());
                deviation();
                break;
            }
            default: { // MIN, MAX, SMIN, SMAX: v - H is the same op of a - H and b - H; a smooth one dips by k / 4 more
                const bool lower = in.op == SampleOp::MIN || in.op == SampleOp::SMIN;
                const double dip = in.op == SampleOp::SMIN || in.op == SampleOp::SMAX ? double(in.value) / 4.0 : 0.0;
                o.lo = (lower ? ov::min(a.lo, b.lo) : ov::max(a.lo, b.lo)) - (lower ? dip : 0.0);
                o.hi = (lower ? ov::min(a.hi, b.hi) : ov::max(a.hi, b.hi)) + (lower ? 0.0 : dip);
                o.dlo = (lower ? ov::min(a.dlo, b.dlo) : ov::max(a.dlo, b.dlo)) - (lower ? dip : 0.0);
                o.dhi = (lower ? ov::min(a.dhi, b.dhi) : ov::max(a.dhi, b.dhi)) + (lower ? 0.0 : dip);
                o.s_h = ov::max(a.s_h, b.s_h);
                o.ds_h = ov::max(a.ds_h, b.ds_h);
                for (int f = 0; f < MAX_FIELDS; f++) {
                    o.s_f[f] = ov::max(a.s_f[f], b.s_f[f]);
                }
                break;
            }
            case SampleOp::CURVE: {
                const Curve &c = r_p.curves[size_t(in.index)];
                const double bound = double(curve_bound(c)), slope = double(curve_slope_bound(c));
                o.lo = a.lo - bound;
                o.hi = a.hi + bound;
                o.dlo = a.dlo - bound;
                o.dhi = a.dhi + bound;
                o.s_h = a.s_h * (1.0 + slope);
                o.ds_h = a.ds_h + slope * a.s_h;
                fields(1.0 + slope, 0.0);
                break;
            }
        }
    }
    const Range &out = r.back();
    r_p.bound = ov::max(std::fabs(out.dlo), std::fabs(out.dhi));
    r_p.slope = out.ds_h;
    std::copy(std::begin(out.s_f), std::end(out.s_f), std::begin(r_p.field_slope));
}

} // namespace

ov::str compile(const std::vector<BiomeGraph> &p_graphs, const Ground &p_ground, Table &r_table) {
    const size_t n = r_table.biomes.size();
    if (p_graphs.size() != n) {
        return "climate: one graph per biome.";
    }
    std::vector<Note> notes;
    std::vector<std::vector<Amp>> amps(n);
    ov::str err = gather(p_graphs, p_ground, r_table, notes, amps);
    err = err.empty() ? fold(amps, notes) : err;
    if (!err.empty()) {
        return err;
    }
    r_table.notes.clear();
    for (Biome &bi : r_table.biomes) {
        bi.amps.clear();
    }
    for (Note &note : notes) {
        if (!(note.top > 0.0f)) {
            continue; // silent everywhere
        }
        note.band.amp = note.top;
        r_table.notes.push_back(note.band);
        for (size_t b = 0; b < n && !note.constant; b++) {
            Amp a;
            a.k = note.band.k;
            a.seed = note.band.seed;
            a.stretch = note.band.stretch;
            r_table.biomes[b].amps.push_back(note.entry[b] >= 0 ? amps[b][size_t(note.entry[b])] : a);
        }
    }
    std::stable_sort(r_table.notes.begin(), r_table.notes.end(), [](const Cascade::Band &p_a, const Cascade::Band &p_b) { return p_a.k > p_b.k; });
    const double reach = ground_reach(r_table);
    const std::vector<double> fields = field_bounds(r_table, reach);
    for (Biome &bi : r_table.biomes) {
        if (!bi.program.empty()) {
            bound_program(bi.program, reach, fields);
        }
    }
    return ov::str();
}

int template_top(double p_voxel_m) {
    int top = ExactNoise::BAND_SHIFT;
    while (std::ldexp(TEMPLATE_STEEPNESS * p_voxel_m, top) < TEMPLATE_MAX_AMP_M * (1.0 - 1e-9) && top < ExactNoise::MAX_K) {
        top++;
    }
    return top;
}

ov::Value template_graph(double p_voxel_m) {
    const int top = template_top(p_voxel_m);
    ov::Value note = ov::Value::object();
    note["id"] = "ground";
    note["kind"] = "note";
    note["from_m"] = std::ldexp(p_voxel_m, ExactNoise::BAND_SHIFT);
    note["to_m"] = std::ldexp(p_voxel_m, top);
    note["seed"] = int64_t(TEMPLATE_SEED);
    note["steepness"] = TEMPLATE_STEEPNESS;
    note["max_amp_m"] = TEMPLATE_MAX_AMP_M;
    ov::Value out = ov::Value::object();
    out["id"] = "out";
    out["kind"] = "output";
    out["height"] = "ground";
    ov::Value nodes = ov::Value::array();
    nodes.push_back(note);
    nodes.push_back(out);
    ov::Value graph = ov::Value::object();
    graph["nodes"] = nodes;
    return graph;
}

ov::Value template_biome(double p_voxel_m, const ov::str &p_name) {
    ov::Value b = ov::Value::object();
    b["name"] = p_name;
    b["graph"] = template_graph(p_voxel_m);
    return b;
}

} // namespace Climate
