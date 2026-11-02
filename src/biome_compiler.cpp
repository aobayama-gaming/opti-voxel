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

// One compiled note: its node's seed, its band and, per biome, its entry (-1: the biome doesn't mix it).
struct Note {
    uint32_t seed = 0;
    Cascade::Band band;
    std::vector<int> entry;
};

} // namespace

ov::str compile(const std::vector<BiomeGraph> &p_graphs, const Ground &p_ground, Table &r_table) {
    const size_t n = r_table.biomes.size();
    if (p_graphs.size() != n) {
        return "climate: one graph per biome.";
    }
    std::vector<Note> notes;
    std::vector<std::vector<Amp>> amps(n);
    for (size_t b = 0; b < n; b++) {
        const ov::str where = "biome '" + r_table.biomes[b].name + ov::str("': ");
        for (const GraphRow &row : p_graphs[b].rows) {
            for (const int k : note_octaves(row.note, p_ground)) {
                const int stretch = ov::max(0, ov::min(row.note.stretch, k - ExactNoise::BAND_SHIFT));
                Note *note = nullptr;
                for (Note &c : notes) {
                    if (c.band.k == k && c.seed == row.note.seed && c.band.stretch == stretch) {
                        note = &c;
                    } else if (c.band.k == k) {
                        return where + ov::format("two notes on octave k = %d (another seed or stretch): the cascade holds one per octave.", k);
                    }
                }
                if (note == nullptr) {
                    notes.push_back(Note{ row.note.seed, ExactNoise::make_octave(row.note.seed, uint32_t(k), k, 0.0f), std::vector<int>(n, -1) });
                    note = &notes.back();
                    note->band.stretch = stretch;
                }
                if (note->entry[b] >= 0) {
                    return where + ov::format("the note of octave k = %d is mixed twice.", k);
                }
                Amp a;
                a.k = k;
                a.amp = float(double(row.x) * double(note_amp(row.note, k, p_ground)));
                if (!(a.amp >= 0.0f)) {
                    return where + ov::format("octave k = %d: the amplitude must be >= 0 (its note's, times the row's volume).", k);
                }
                a.has_shape = row.has_shape;
                a.shape = row.shape;
                a.gain = row.gain;
                a.warp = row.warp;
                std::copy(std::begin(row.warp_fields), std::end(row.warp_fields), std::begin(a.warp_fields));
                note->entry[b] = int(amps[b].size());
                amps[b].push_back(a);
            }
        }
    }
    r_table.notes.clear();
    for (Biome &bi : r_table.biomes) {
        bi.amps.clear();
    }
    for (Note &note : notes) {
        const Amp *first = note.entry[0] >= 0 ? &amps[0][size_t(note.entry[0])] : nullptr;
        bool constant = first && plain(*first);
        float top = 0.0f;
        for (size_t b = 0; b < n; b++) {
            const Amp *a = note.entry[b] >= 0 ? &amps[b][size_t(note.entry[b])] : nullptr;
            constant = constant && a && plain(*a) && ExactNoise::float_bits(a->amp) == ExactNoise::float_bits(first->amp);
            top = a ? ov::max(top, a->amp) : top;
        }
        if (!(top > 0.0f)) {
            continue; // silent everywhere
        }
        note.band.amp = top;
        r_table.notes.push_back(note.band);
        for (size_t b = 0; b < n && !constant; b++) {
            Amp a;
            a.k = note.band.k;
            r_table.biomes[b].amps.push_back(note.entry[b] >= 0 ? amps[b][size_t(note.entry[b])] : a);
        }
    }
    std::stable_sort(r_table.notes.begin(), r_table.notes.end(), [](const Cascade::Band &p_a, const Cascade::Band &p_b) { return p_a.k > p_b.k; });
    return ov::str();
}

ov::Value template_graph(double p_voxel_m) {
    int top = ExactNoise::BAND_SHIFT;
    while (std::ldexp(TEMPLATE_STEEPNESS * p_voxel_m, top) < TEMPLATE_MAX_AMP_M * (1.0 - 1e-9) && top < ExactNoise::MAX_K) {
        top++;
    }
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
