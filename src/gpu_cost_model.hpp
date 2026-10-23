#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

// The GPU time of a mesher batch, estimated per chunk and per LOD plus per word of its instance list, and corrected from
// the batches' timestamps: a batch is estimated as the sum of its chunks' LOD costs plus its list words times the cost of
// a word; a measured batch scales each of them (the LODs it holds and the words) by measured / estimated, in proportion to
// its share of the estimate. A LOD never measured takes the nearest measured LOD's cost (DEFAULT_US before any measure);
// its first measure is what the measured time leaves once the words are paid, within MAX_SCALE of the borrowed cost when
// words are in the batch (a word cost far off would otherwise hand the LOD the whole error).
//
// Before its list is built, a chunk only knows a bound of its words (the streamed records of the keys it reads:
// SDFBase::admit): it is expected to list list_share[lod] of it (its LOD's streamed list words over their bounds: a fine
// chunk meets a sliver of its part tiles, a coarse one most of them) plus words_per_chunk[lod] (the scene's own records
// and the edits', all of a list without streamed sources). A LOD's first list replaces their seeds (the bound, nothing).
struct GPUCostModel {
    static constexpr double DEFAULT_US = 60.0; // a forest batch at 5 cm: ~7 ms for 128 chunks
    // The sdf pass per list word: the 1M forest's batches fit 5,842 + 1.004 x words us (docs/forest-1m-profile.md).
    static constexpr double WORD_US = 1.0;
    static constexpr double RATE = 0.25; // weight of one batch in a measured cost
    static constexpr double MAX_SCALE = 8.0; // one batch moves a cost by at most this factor
    static constexpr double LIST_SHARE_MIN = 1.0 / 1024.0, LIST_SHARE_MAX = 4.0; // a list's words over its bound
    // A word's cost stays within this factor of WORD_US: batches with the same words per chunk can't tell it from the LODs'.
    static constexpr double WORD_RANGE = 64.0;

    void resize(int p_lods) {
        us_per_chunk.assign(std::max(p_lods, 0), -1.0);
        list_share.assign(std::max(p_lods, 0), -1.0);
        words_per_chunk.assign(std::max(p_lods, 0), -1.0);
    }
    int lods() const { return (int)us_per_chunk.size(); }
    bool measured(int p_lod) const { return p_lod >= 0 && p_lod < lods() && us_per_chunk[p_lod] >= 0.0; }

    // p_per_lod[p_lod] if seen (>= 0), else the nearest LOD's seen value (the finer first at equal distance), else p_none.
    static double nearest(const std::vector<double> &p_per_lod, int p_lod, double p_none) {
        const int n = (int)p_per_lod.size();
        for (int d = 0; d < n; d++) {
            if (p_lod - d >= 0 && p_lod - d < n && p_per_lod[p_lod - d] >= 0.0) {
                return p_per_lod[p_lod - d];
            }
            if (d > 0 && p_lod + d >= 0 && p_lod + d < n && p_per_lod[p_lod + d] >= 0.0) {
                return p_per_lod[p_lod + d];
            }
        }
        return p_none;
    }
    double estimate(int p_lod) const { return nearest(us_per_chunk, p_lod, DEFAULT_US); }
    // A chunk not batched yet: its LOD's cost plus the words its bound p_bound (SDFBase::admit) is expected to list (a LOD
    // with no list yet borrows the nearest one's share and rest; the whole bound when no LOD has a list).
    double estimate_chunk(int p_lod, uint64_t p_bound) const {
        return estimate(p_lod) + (double(p_bound) * nearest(list_share, p_lod, 1.0) + nearest(words_per_chunk, p_lod, 0.0)) * us_per_word;
    }

    // p_counts[lod] = the batch's chunks at that LOD; p_words: its list's words.
    double estimate(const std::vector<uint32_t> &p_counts, uint64_t p_words = 0) const {
        double e = double(p_words) * us_per_word;
        for (int lod = 0; lod < (int)p_counts.size(); lod++) {
            e += p_counts[lod] * estimate(lod);
        }
        return e;
    }

    void observe(const std::vector<uint32_t> &p_counts, double p_measured_us, uint64_t p_words = 0) {
        const double words = double(p_words) * us_per_word;
        const double e = estimate(p_counts, p_words);
        if (e <= 0.0 || p_measured_us <= 0.0) {
            return;
        }
        // The LODs taking their first measure (borrowed costs) and the measured ones.
        double fresh = 0.0;
        for (int lod = 0; lod < (int)p_counts.size() && lod < lods(); lod++) {
            fresh += measured(lod) ? 0.0 : p_counts[lod] * estimate(lod);
        }
        if (fresh > 0.0) {
            // Only the LODs taking their first measure move: they replace their borrowed costs with what the words and
            // the measured LODs leave of the measure (at least their share of it); with words in the batch, within
            // MAX_SCALE (the words' cost may be far off; without, a first LOD may well cost 10x its neighbour's).
            const double leftover = std::max(p_measured_us - (e - fresh), p_measured_us * fresh / e) / fresh;
            const double first = p_words > 0 ? std::clamp(leftover, 1.0 / MAX_SCALE, MAX_SCALE) : leftover;
            std::vector<double> next = us_per_chunk; // borrowed from the LODs measured before this batch
            for (int lod = 0; lod < (int)p_counts.size() && lod < lods(); lod++) {
                if (p_counts[lod] > 0 && !measured(lod)) {
                    next[lod] = estimate(lod) * first;
                }
            }
            us_per_chunk = std::move(next);
            return;
        }
        // Every LOD measured: each cost, and the words', moves a damped, bounded step by its share of the estimate.
        const double scale = std::clamp(p_measured_us / e, 1.0 / MAX_SCALE, MAX_SCALE);
        for (int lod = 0; lod < (int)p_counts.size() && lod < lods(); lod++) {
            if (p_counts[lod] > 0) {
                const double cost = us_per_chunk[lod];
                us_per_chunk[lod] = cost * (1.0 + RATE * (p_counts[lod] * cost / e) * (scale - 1.0));
            }
        }
        us_per_word = std::clamp(us_per_word * (1.0 + RATE * (words / e) * (scale - 1.0)), WORD_US / WORD_RANGE, WORD_US * WORD_RANGE);
    }
    // A batch's lists at LOD p_lod: p_streamed streamed words for p_chunks chunks whose bounds summed to p_bound, and
    // p_other words of the scene's and the edits' records.
    void observe_list(int p_lod, uint64_t p_bound, uint64_t p_streamed, uint64_t p_other, uint32_t p_chunks) {
        if (p_chunks == 0 || p_lod < 0 || p_lod >= lods()) {
            return;
        }
        const auto follow = [](double &r_value, double p_seen) { r_value = r_value < 0.0 ? p_seen : r_value + RATE * (p_seen - r_value); };
        follow(words_per_chunk[p_lod], double(p_other) / double(p_chunks));
        if (p_bound > 0) {
            follow(list_share[p_lod], std::clamp(double(p_streamed) / double(p_bound), LIST_SHARE_MIN, LIST_SHARE_MAX));
        }
    }

    std::vector<double> us_per_chunk; // < 0: never measured
    double us_per_word = WORD_US;
    std::vector<double> list_share; // per LOD: a bound's words that reach the list (< 0: no list yet, the bound)
    std::vector<double> words_per_chunk; // per LOD: a chunk's words of the scene's and the edits' records (< 0: none yet, 0)
};
