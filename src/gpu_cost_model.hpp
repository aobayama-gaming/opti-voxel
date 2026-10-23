#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

// The GPU time of a mesher batch, estimated per chunk and per LOD plus per word of its instance list, and corrected from
// the batches' timestamps: a batch is estimated as the sum of its chunks' LOD costs plus its list words times the cost of
// a word; a measured batch scales each of them (the LODs it holds and the words) by measured / estimated, in proportion to
// its share of the estimate. A LOD never measured takes the nearest measured LOD's cost (DEFAULT_US before any measure);
// its first measure is what the measured time leaves once the words are paid.
//
// Before its list is built, a chunk only knows a bound of its words (the streamed records of the keys it reads:
// SDFBase::admit): it is expected to list list_share of it (the batches' list words over their bounds) plus
// words_per_chunk (what the batches list past that: the scene's own records, all of a list without streamed sources).
struct GPUCostModel {
    static constexpr double DEFAULT_US = 60.0; // a forest batch at 5 cm: ~7 ms for 128 chunks
    // The sdf pass per list word: the 1M forest's batches fit 5,842 + 1.004 x words us (docs/forest-1m-profile.md).
    static constexpr double WORD_US = 1.0;
    static constexpr double RATE = 0.25; // weight of one batch in a measured cost
    static constexpr double MAX_SCALE = 8.0; // one batch moves a cost by at most this factor
    static constexpr double LIST_SHARE_MIN = 1.0 / 1024.0, LIST_SHARE_MAX = 4.0; // a list's words over its bound

    void resize(int p_lods) { us_per_chunk.assign(std::max(p_lods, 0), -1.0); }
    int lods() const { return (int)us_per_chunk.size(); }
    bool measured(int p_lod) const { return p_lod >= 0 && p_lod < lods() && us_per_chunk[p_lod] >= 0.0; }

    double estimate(int p_lod) const {
        if (measured(p_lod)) {
            return us_per_chunk[p_lod];
        }
        for (int d = 1; d < lods(); d++) {
            if (measured(p_lod - d)) {
                return us_per_chunk[p_lod - d];
            }
            if (measured(p_lod + d)) {
                return us_per_chunk[p_lod + d];
            }
        }
        return DEFAULT_US;
    }
    // A chunk not batched yet: its LOD's cost plus the words its bound p_bound (SDFBase::admit) is expected to list.
    double estimate_chunk(int p_lod, uint64_t p_bound) const { return estimate(p_lod) + (double(p_bound) * list_share + words_per_chunk) * us_per_word; }

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
        // A LOD's first measure replaces the borrowed cost with what the words leave of the measure (at least the chunks'
        // share of it); a measured cost, and the words', only move a damped, bounded step.
        const double chunks = e - words;
        const double first = chunks > 0.0 ? std::max(p_measured_us - words, p_measured_us * chunks / e) / chunks : 1.0;
        const double scale = std::clamp(p_measured_us / e, 1.0 / MAX_SCALE, MAX_SCALE);
        std::vector<double> next = us_per_chunk;
        for (int lod = 0; lod < (int)p_counts.size() && lod < lods(); lod++) {
            if (p_counts[lod] == 0) {
                continue;
            }
            const double cost = estimate(lod);
            const double share = p_counts[lod] * cost / e;
            next[lod] = measured(lod) ? cost * (1.0 + RATE * share * (scale - 1.0)) : cost * first;
        }
        us_per_chunk = std::move(next);
        us_per_word *= 1.0 + RATE * (words / e) * (scale - 1.0);
    }
    // A batch's list: p_words for p_chunks chunks whose bounds summed to p_bound.
    void observe_list(uint64_t p_bound, uint64_t p_words, uint32_t p_chunks) {
        if (p_chunks == 0) {
            return;
        }
        const double rest = std::max(0.0, double(p_words) - double(p_bound) * list_share) / double(p_chunks);
        words_per_chunk += RATE * (rest - words_per_chunk);
        if (p_bound > 0) {
            const double share = std::clamp(double(p_words) / double(p_bound), LIST_SHARE_MIN, LIST_SHARE_MAX);
            list_share += RATE * (share - list_share);
        }
    }

    std::vector<double> us_per_chunk; // < 0: never measured
    double us_per_word = WORD_US;
    double list_share = 1.0; // a bound's words that reach the list (1: the bound, until a list says otherwise)
    double words_per_chunk = 0.0; // a chunk's words past its bound's share
};
