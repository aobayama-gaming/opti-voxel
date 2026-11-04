#include "biome_compiler.hpp"

#include "exact_noise.hpp"

#include "ov/format.hpp"
#include "ov/math.hpp"

#include <algorithm>
#include <cmath>

namespace Climate {

int note_stretch(const NoteNode &p_note, int p_k) {
    return ov::max(0, ov::min(p_note.stretch, p_k - ExactNoise::BAND_SHIFT));
}

bool note_plays(const NoteNode &p_note, const Ground &p_ground) {
    return p_note.k >= ExactNoise::BAND_SHIFT && p_note.k <= p_ground.k_max;
}

float note_amp(const NoteNode &p_note, const Ground &p_ground) {
    return float(ov::min(p_note.amp_m / p_ground.voxel_size, p_ground.note_cap));
}

namespace {

// Whether two amplitude entries play a note the same way (a constant note: every biome's alike and plain).
bool plain(const Amp &p_a) {
    return !p_a.has_shape && p_a.multifractal.field < 0 && p_a.warps.empty();
}

// One compiled note: its node's seed, its band, its cascade group (0 the ground, else 1 + its channel field's index), per
// biome its entry (-1: the biome doesn't mix it), whether every biome plays it alike (a constant band) and its largest
// amplitude.
struct Note {
    uint32_t seed = 0;
    Cascade::Band band;
    int group = 0;
    std::vector<int> entry;
    bool constant = false;
    float top = 0.0f;
};

// Each biome's sub-mixes onto channel fields (Field::channel, after the table's own, at least p_ground.caps.channels of
// them in a generic table): the first whose notes take none of the sub-mix's octaves with another note (a field group
// inserts one band per level) and none of the biome's other sub-mixes is on, else a new one. r_field[b][c - 1]: biome
// b's sub-mix c's field. An error past MAX_FIELDS fields (before validate: the programs' bounds index by field).
ov::str channels(const std::vector<BiomeGraph> &p_graphs, const Ground &p_ground, Table &r_table, std::vector<std::vector<int>> &r_field) {
    const size_t own = r_table.fields.size();
    std::vector<std::vector<NoteId>> taken;
    for (size_t b = 0; b < p_graphs.size(); b++) {
        int count = 0;
        for (const GraphRow &row : p_graphs[b].rows) {
            count = ov::max(count, row.channel);
        }
        for (int c = 1; c <= count; c++) {
            std::vector<NoteId> ids;
            for (const GraphRow &row : p_graphs[b].rows) {
                if (row.channel == c && note_plays(row.note, p_ground)) {
                    ids.push_back(NoteId{ row.note.k, row.note.seed, note_stretch(row.note, row.note.k), 0 });
                }
            }
            auto fits = [&](const std::vector<NoteId> &p_taken) {
                return std::none_of(ids.begin(), ids.end(), [&](const NoteId &p_i) {
                    return std::any_of(p_taken.begin(), p_taken.end(), [&](const NoteId &p_t) { return p_t.k == p_i.k && !(p_t == p_i); });
                });
            };
            size_t f = 0;
            while (f < taken.size() && (!fits(taken[f]) || std::find(r_field[b].begin(), r_field[b].end(), int(own + f)) != r_field[b].end())) {
                f++;
            }
            if (f == taken.size()) {
                taken.emplace_back();
            }
            taken[f].insert(taken[f].end(), ids.begin(), ids.end());
            r_field[b].push_back(int(own + f));
        }
    }
    const size_t count = ov::max(taken.size(), r_table.generic ? size_t(ov::max(0, p_ground.caps.channels)) : size_t(0));
    if (own + count > size_t(MAX_FIELDS)) {
        const int64_t warps = std::count_if(r_table.fields.begin(), r_table.fields.end(), [](const Field &p_f) { return p_f.warp; });
        return ov::format("%d fields, %d warp and multifractal fields (3 per warp octave, 1 per parent note) and %d sub-mix fields, at most %d in all.", int64_t(own) - warps, warps, int64_t(count),
                MAX_FIELDS);
    }
    for (size_t f = 0; f < count; f++) {
        Field field;
        field.name = ov::format("~submix%d", int64_t(f + 1));
        field.channel = true;
        r_table.fields.push_back(field);
    }
    return ov::str();
}

// Every biome's rows into r_notes (one per identity) and each biome's entries r_amps[b]: empty, or the first error.
ov::str gather(const std::vector<BiomeGraph> &p_graphs, const Ground &p_ground, const Table &p_table, const std::vector<std::vector<int>> &p_field, std::vector<Note> &r_notes,
        std::vector<std::vector<Amp>> &r_amps) {
    const size_t n = p_table.biomes.size();
    for (size_t b = 0; b < n; b++) {
        const ov::str where = "biome '" + p_table.biomes[b].name + ov::str("': ");
        for (const GraphRow &row : p_graphs[b].rows) {
            const int group = row.channel == 0 ? 0 : 1 + p_field[b][size_t(row.channel - 1)];
            if (!note_plays(row.note, p_ground)) {
                continue;
            }
            const int k = row.note.k;
            const int stretch = note_stretch(row.note, k);
            auto same = [&](const Note &p_c) { return p_c.band.k == k && p_c.seed == row.note.seed && p_c.band.stretch == stretch && p_c.group == group; };
            auto it = std::find_if(r_notes.begin(), r_notes.end(), same);
            if (it == r_notes.end()) {
                Note note{ row.note.seed, ExactNoise::make_octave(row.note.seed, uint32_t(k), k, 0.0f), group, std::vector<int>(n, -1) };
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
            a.group = group;
            a.amp = float(double(row.x) * double(note_amp(row.note, p_ground)));
            if (!(a.amp >= 0.0f)) {
                return where + ov::format("octave k = %d: the amplitude must be >= 0 (its note's, times the row's volume).", k);
            }
            a.has_shape = row.has_shape;
            a.shape = row.shape;
            a.multifractal = row.multifractal;
            a.warps = row.warps;
            it->entry[b] = int(r_amps[b].size());
            r_amps[b].push_back(a);
        }
    }
    return ov::str();
}

// Each note's fold (constant: every biome plays it plain at one amplitude) and largest amplitude; an octave of the
// ground is modulated or constant as a whole (the cascade's levels are: Cascade::MAX_LEVEL_NOTES), at most
// MAX_LEVEL_NOTES notes on it; a sub-mix's, one note (a field group). Empty, or the first error.
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
        int count = 0;
        bool modulated = false;
        for (const Note &o : r_notes) {
            if (o.band.k == note.band.k && o.group == note.group && o.top > 0.0f) {
                count++;
                modulated = modulated || !o.constant;
            }
        }
        if (note.group > 0 && count > 1) {
            return ov::format("octave k = %d of a sub-mix has %d notes: a sub-mix plays one note per octave.", note.band.k, count);
        }
        if (count > Cascade::MAX_LEVEL_NOTES) {
            return ov::format("octave k = %d has %d notes: at most %d.", note.band.k, count, Cascade::MAX_LEVEL_NOTES);
        }
        note.constant = note.constant && !modulated;
    }
    return ov::str();
}

// What a sample program's register v can be, given |H| <= the ground's reach and each field within its bound: affine
// forms in z = UP - H (the sample's height over the ground, unbounded) and H, v <= each upper form and >= each lower one
// (an empty side: unbounded). H stays exact, so a curve of H stays tight (R(H) - H = C(H)), and z too, so a cave's tent
// min(depth - top, floor - depth) bounds smin(h, up - C) - h by its floor. Plus the bounds of |dv/dH|, |d(v - H)/dH|
// and |dv/df|.
struct Form {
    double z = 0.0, h = 0.0, c = 0.0;
    Form operator+(const Form &p_o) const { return Form{ z + p_o.z, h + p_o.h, c + p_o.c }; }
    Form operator*(double p_k) const { return Form{ z * p_k, h * p_k, c * p_k }; }
};
struct Range {
    std::vector<Form> up, lo;
    double s_h = 0.0, ds_h = 0.0;
    double s_f[MAX_FIELDS] = {};
};

// A side's forms at most: the rest are dropped (fewer forms bound less tightly, never wrongly).
constexpr size_t MAX_FORMS = 8;

// The largest min of p_forms over z and |H| <= p_reach (+inf: unbounded): with each form's H at its worst, a min of
// lines in z peaks on a flat one or where a rising one crosses a falling one.
double sup_of(const std::vector<Form> &p_forms, double p_reach) {
    double s = INFINITY;
    for (const Form &a : p_forms) {
        const double ca = a.c + std::fabs(a.h) * p_reach;
        s = a.z == 0.0 ? ov::min(s, ca) : s;
        for (const Form &b : p_forms) {
            if (a.z > 0.0 && b.z < 0.0) {
                s = ov::min(s, (a.z * (b.c + std::fabs(b.h) * p_reach) - b.z * ca) / (a.z - b.z));
            }
        }
    }
    return s;
}

std::vector<Form> negated(std::vector<Form> p_forms) {
    for (Form &f : p_forms) {
        f = f * -1.0;
    }
    return p_forms;
}

double inf_of(const std::vector<Form> &p_forms, double p_reach) {
    return -sup_of(negated(p_forms), p_reach);
}

double peak(const Range &p_r, double p_reach) {
    return ov::max(std::fabs(inf_of(p_r.lo, p_reach)), std::fabs(sup_of(p_r.up, p_reach)));
}

// p_forms each plus p_by.
std::vector<Form> shifted(std::vector<Form> p_forms, const Form &p_by) {
    for (Form &f : p_forms) {
        f = f + p_by;
    }
    return p_forms;
}

// p_a's forms plus p_k times p_b's, pair by pair.
std::vector<Form> sums(const std::vector<Form> &p_a, const std::vector<Form> &p_b, double p_k) {
    std::vector<Form> r;
    for (const Form &a : p_a) {
        for (const Form &b : p_b) {
            if (r.size() < MAX_FORMS) {
                r.push_back(a + b * p_k);
            }
        }
    }
    return r;
}

// Lower forms of min(a, b) from a's and b's: min(la, lb) for a pair along the same z (H at its worst where they
// differ in it), then the smaller of their lowest values.
std::vector<Form> min_lower(const std::vector<Form> &p_a, const std::vector<Form> &p_b, double p_reach) {
    std::vector<Form> r;
    for (const Form &a : p_a) {
        for (const Form &b : p_b) {
            if (a.z == b.z && r.size() < MAX_FORMS) {
                r.push_back(a.h == b.h ? Form{ a.z, a.h, ov::min(a.c, b.c) } : Form{ a.z, 0.0, ov::min(a.c - std::fabs(a.h) * p_reach, b.c - std::fabs(b.h) * p_reach) });
            }
        }
    }
    const double low = ov::min(inf_of(p_a, p_reach), inf_of(p_b, p_reach));
    if (std::isfinite(low)) {
        r.push_back(Form{ 0.0, 0.0, low });
    }
    return r;
}

// Whether p_r is a constant (a CONST register), *r_k its value.
bool constant(const Range &p_r, double *r_k) {
    const bool k = p_r.up.size() == 1 && p_r.lo.size() == 1 && p_r.up[0].z == 0.0 && p_r.up[0].h == 0.0 && p_r.lo[0].z == 0.0 && p_r.lo[0].h == 0.0 &&
            p_r.up[0].c == p_r.lo[0].c;
    *r_k = k ? p_r.up[0].c : 0.0;
    return k;
}

// p times a slope s (0 where s is: an unbounded value times a constant).
double times(double p_p, double p_s) {
    return p_s == 0.0 ? 0.0 : p_p * p_s;
}

// Each field's bound: its bands' amplitudes (a sub-mix's, each biome's times its shape's bound: max_amp), plus its
// terms' |coef| times their source's (the ground: p_reach).
std::vector<double> field_bounds(const Table &p_table, double p_reach) {
    std::vector<double> f(p_table.fields.size(), 0.0);
    for (size_t i = 0; i < p_table.fields.size(); i++) {
        for (const Cascade::Band &b : p_table.fields[i].bands) {
            f[i] += band_bound(p_table, int(i), b);
        }
        for (const Term &t : p_table.fields[i].terms) {
            f[i] += std::fabs(double(t.coef)) * (t.source == GROUND ? p_reach : f[size_t(t.source)]);
        }
    }
    return f;
}

// r_o = p_k x + p_c.
void affine(Range &r_o, const Range &p_x, double p_k, double p_c) {
    r_o.up = shifted(sums({ Form() }, p_k >= 0.0 ? p_x.up : p_x.lo, p_k), Form{ 0.0, 0.0, p_c });
    r_o.lo = shifted(sums({ Form() }, p_k >= 0.0 ? p_x.lo : p_x.up, p_k), Form{ 0.0, 0.0, p_c });
    r_o.s_h = std::fabs(p_k) * p_x.s_h;
    for (int f = 0; f < MAX_FIELDS; f++) {
        r_o.s_f[f] = std::fabs(p_k) * p_x.s_f[f];
    }
}

// r_o within +-p_hi (unbounded if p_hi isn't finite).
void within(Range &r_o, double p_hi) {
    if (std::isfinite(p_hi)) {
        r_o.up = { Form{ 0.0, 0.0, p_hi } };
        r_o.lo = { Form{ 0.0, 0.0, -p_hi } };
    }
}

// MIN, MAX, SMIN, SMAX of a and b into r_o: the bound of the larger / smaller side, the pairs' for the other; a smooth
// one dips by k / 4 more.
void min_max(const SampleInstr &p_in, const Range &a, const Range &b, double p_reach, Range &r_o) {
    const bool lower = p_in.op == SampleOp::MIN || p_in.op == SampleOp::SMIN;
    const double dip = p_in.op == SampleOp::SMIN || p_in.op == SampleOp::SMAX ? double(p_in.value) / 4.0 : 0.0;
    std::vector<Form> &free = lower ? r_o.up : r_o.lo;
    free = lower ? a.up : a.lo;
    for (const Form &f : lower ? b.up : b.lo) {
        if (free.size() < MAX_FORMS) {
            free.push_back(f);
        }
    }
    r_o.lo = lower ? min_lower(a.lo, b.lo, p_reach) : r_o.lo;
    r_o.up = lower ? r_o.up : negated(min_lower(negated(a.up), negated(b.up), p_reach));
    (lower ? r_o.lo : r_o.up) = shifted(lower ? r_o.lo : r_o.up, Form{ 0.0, 0.0, lower ? -dip : dip });
    r_o.s_h = ov::max(a.s_h, b.s_h);
    r_o.ds_h = ov::max(a.ds_h, b.ds_h);
    for (int f = 0; f < MAX_FIELDS; f++) {
        r_o.s_f[f] = ov::max(a.s_f[f], b.s_f[f]);
    }
}

// SHAPE of a into r_o: |phi(a)| <= |c0| + (|c1| + |c2|) |a| + |c3| a^2, |phi'(a)| <= |c1| + |c2| + 2 |c3| |a|; an
// affine one exactly.
void shape_of(const Shape &p_s, const Range &a, double p_reach, Range &r_o) {
    const float *c = p_s.c;
    if (c[2] == 0.0f && c[3] == 0.0f) {
        affine(r_o, a, double(c[1]), double(c[0]));
    } else {
        const double p = peak(a, p_reach);
        const double k = std::fabs(double(c[1])) + std::fabs(double(c[2])) + times(p, 2.0 * std::fabs(double(c[3])));
        within(r_o, std::fabs(double(c[0])) + times(p, std::fabs(double(c[1])) + std::fabs(double(c[2]))) + times(p * p, std::fabs(double(c[3]))));
        r_o.s_h = times(k, a.s_h);
        for (int f = 0; f < MAX_FIELDS; f++) {
            r_o.s_f[f] = times(k, a.s_f[f]);
        }
    }
    r_o.ds_h = r_o.s_h + 1.0;
}

// r_p's bounds (SampleProgram::bound, slope, field_slope; bound +inf: unbounded) for a ground within p_reach and fields
// within p_fields.
void bound_program(SampleProgram &r_p, double p_reach, const std::vector<double> &p_fields) {
    std::vector<Range> r(r_p.code.size());
    for (size_t i = 0; i < r_p.code.size(); i++) {
        const SampleInstr &in = r_p.code[i];
        static const Range NONE;
        const Range &a = in.a >= 0 ? r[size_t(in.a)] : NONE, &b = in.b >= 0 ? r[size_t(in.b)] : NONE;
        Range &o = r[i];
        auto fields = [&](double p_ka, double p_kb) {
            for (int f = 0; f < MAX_FIELDS; f++) {
                o.s_f[f] = times(p_ka, a.s_f[f]) + times(p_kb, b.s_f[f]);
            }
        };
        double k = 0.0;
        switch (in.op) {
            case SampleOp::GROUND:
                o.up = o.lo = { Form{ 0.0, 1.0, 0.0 } };
                o.s_h = 1.0;
                break;
            case SampleOp::UP: // z + H
                o.up = o.lo = { Form{ 1.0, 1.0, 0.0 } };
                o.ds_h = 1.0;
                break;
            case SampleOp::CONST:
                o.up = o.lo = { Form{ 0.0, 0.0, double(in.value) } };
                o.ds_h = 1.0;
                break;
            case SampleOp::FIELD:
                within(o, in.index >= 0 && in.index < int(p_fields.size()) ? p_fields[size_t(in.index)] : 0.0);
                o.s_f[in.index] = 1.0;
                o.ds_h = 1.0;
                break;
            case SampleOp::ADD:
            case SampleOp::SUB: {
                const double sign = in.op == SampleOp::ADD ? 1.0 : -1.0;
                o.up = sums(a.up, sign > 0.0 ? b.up : b.lo, sign);
                o.lo = sums(a.lo, sign > 0.0 ? b.lo : b.up, sign);
                o.s_h = a.s_h + b.s_h;
                o.ds_h = a.ds_h + b.s_h;
                fields(1.0, 1.0);
                break;
            }
            case SampleOp::MUL: {
                const bool by_b = constant(b, &k);
                if (by_b || constant(a, &k)) {
                    affine(o, by_b ? a : b, k, 0.0);
                } else {
                    const double pa = peak(a, p_reach), pb = peak(b, p_reach);
                    within(o, pa * pb);
                    o.s_h = times(pa, b.s_h) + times(pb, a.s_h);
                    fields(pb, pa);
                }
                o.ds_h = o.s_h + 1.0;
                break;
            }
            case SampleOp::MIN:
            case SampleOp::MAX:
            case SampleOp::SMIN:
            case SampleOp::SMAX:
                min_max(in, a, b, p_reach, o);
                break;
            case SampleOp::SHAPE:
                shape_of(r_p.shapes[size_t(in.index)], a, p_reach, o);
                break;
            case SampleOp::CURVE: {
                const Curve &c = r_p.curves[size_t(in.index)];
                const double bound = double(curve_bound(c)), slope = double(curve_slope_bound(c));
                o.up = shifted(a.up, Form{ 0.0, 0.0, bound });
                o.lo = shifted(a.lo, Form{ 0.0, 0.0, -bound });
                o.s_h = a.s_h * (1.0 + slope);
                o.ds_h = a.ds_h + slope * a.s_h;
                fields(1.0 + slope, 0.0);
                break;
            }
        }
    }
    const Range &out = r.back();
    const std::vector<Form> d_up = shifted(out.up, Form{ 0.0, -1.0, 0.0 }), d_lo = shifted(out.lo, Form{ 0.0, -1.0, 0.0 }); // D = v - H
    r_p.bound = ov::max(std::fabs(inf_of(d_lo, p_reach)), std::fabs(sup_of(d_up, p_reach)));
    r_p.slope = out.ds_h;
    std::copy(std::begin(out.s_f), std::end(out.s_f), std::begin(r_p.field_slope));
}

} // namespace

ov::str compile(const std::vector<BiomeGraph> &p_graphs, const Ground &p_ground, Table &r_table) {
    const size_t n = r_table.biomes.size();
    if (p_graphs.size() != n) {
        return "climate: one graph per biome.";
    }
    r_table.generic = !p_ground.specialize;
    r_table.caps = p_ground.caps;
    std::vector<std::vector<int>> field(n);
    ov::str err = channels(p_graphs, p_ground, r_table, field);
    std::vector<Note> notes;
    std::vector<std::vector<Amp>> amps(n);
    err = err.empty() ? gather(p_graphs, p_ground, r_table, field, notes, amps) : err;
    err = err.empty() ? fold(amps, notes) : err;
    if (!err.empty()) {
        return err;
    }
    r_table.notes.clear();
    for (Biome &bi : r_table.biomes) {
        bi.amps.clear();
    }
    for (Note &note : notes) {
        if (!(note.top > 0.0f) && !r_table.generic) {
            continue; // silent everywhere (a generic table keeps it, at amplitude 0: a volume edit keeps the structure)
        }
        note.band.amp = note.top;
        (note.group == 0 ? r_table.notes : r_table.fields[size_t(note.group - 1)].bands).push_back(note.band);
        for (size_t b = 0; b < n && !note.constant; b++) {
            Amp a;
            a.k = note.band.k;
            a.seed = note.band.seed;
            a.stretch = note.band.stretch;
            a.group = note.group;
            r_table.biomes[b].amps.push_back(note.entry[b] >= 0 ? amps[b][size_t(note.entry[b])] : a);
        }
    }
    auto coarsest_first = [](const Cascade::Band &p_a, const Cascade::Band &p_b) { return p_a.k > p_b.k; };
    std::stable_sort(r_table.notes.begin(), r_table.notes.end(), coarsest_first);
    for (Field &f : r_table.fields) {
        if (f.channel) {
            std::stable_sort(f.bands.begin(), f.bands.end(), coarsest_first);
        }
    }
    // The programs' sub-mix reads (FIELD -c) onto their fields.
    for (size_t b = 0; b < n; b++) {
        for (SampleInstr &in : r_table.biomes[b].program.code) {
            if (in.op == SampleOp::FIELD && in.index < 0) {
                in.index = field[b][size_t(-in.index - 1)];
            }
        }
    }
    const double reach = ground_reach(r_table);
    const std::vector<double> fields = field_bounds(r_table, reach);
    for (Biome &bi : r_table.biomes) {
        if (!bi.program.empty()) {
            bound_program(bi.program, reach, fields);
        }
    }
    return ov::str();
}

namespace {

double template_amp_m(double p_voxel_m, int p_k) {
    return ov::min(TEMPLATE_STEEPNESS * std::ldexp(1.0, p_k), TEMPLATE_MAX_AMP_M / p_voxel_m) * p_voxel_m;
}

} // namespace

int template_top(double p_voxel_m, const Ground *p_ground) {
    int top = ExactNoise::BAND_SHIFT;
    double reach = 0.0; // the sum of the notes' amplitudes up to top (LOD0 voxels), as p_ground caps them
    auto capped = [&](int p_k) {
        NoteNode n;
        n.amp_m = template_amp_m(p_voxel_m, p_k);
        return double(note_amp(n, *p_ground));
    };
    reach = p_ground ? capped(top) : 0.0;
    while (std::ldexp(TEMPLATE_STEEPNESS * p_voxel_m, top) < TEMPLATE_MAX_AMP_M * (1.0 - 1e-9) && top < ExactNoise::MAX_K) {
        if (p_ground && reach + capped(top + 1) > double(p_ground->amp_cap)) {
            break;
        }
        reach += p_ground ? capped(top + 1) : 0.0;
        top++;
    }
    return top;
}

ov::Value template_graph(double p_voxel_m, const Ground *p_ground) {
    ov::Value nodes = ov::Value::array();
    ov::Value rows = ov::Value::array();
    for (int k = template_top(p_voxel_m, p_ground); k >= ExactNoise::BAND_SHIFT; k--) {
        ov::Value note = ov::Value::object();
        note["id"] = ov::format("ground_%d", int64_t(k));
        note["kind"] = "note";
        note["wavelength_m"] = std::ldexp(p_voxel_m, k);
        note["seed"] = int64_t(TEMPLATE_SEED);
        note["amp_m"] = template_amp_m(p_voxel_m, k);
        nodes.push_back(note);
        ov::Value row = ov::Value::object();
        row["in"] = note["id"];
        rows.push_back(row);
    }
    ov::Value mix = ov::Value::object();
    mix["id"] = "ground";
    mix["kind"] = "mix";
    mix["rows"] = rows;
    nodes.push_back(mix);
    ov::Value out = ov::Value::object();
    out["id"] = "out";
    out["kind"] = "output";
    out["height"] = "ground";
    nodes.push_back(out);
    ov::Value graph = ov::Value::object();
    graph["nodes"] = nodes;
    return graph;
}

ov::Value template_biome(double p_voxel_m, const ov::str &p_name, const Ground *p_ground) {
    ov::Value b = ov::Value::object();
    b["name"] = p_name;
    b["graph"] = template_graph(p_voxel_m, p_ground);
    return b;
}

} // namespace Climate
