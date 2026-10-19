#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

// The GPU time of a mesher batch, estimated per chunk and per LOD and corrected from the batches'
// timestamps: a batch is estimated as the sum of its chunks' LOD costs; a measured batch scales the
// costs of the LODs it holds by measured / estimated, each in proportion to its share of the estimate.
// A LOD never measured takes the nearest measured LOD's cost (DEFAULT_US before any measure).
struct GPUCostModel {
    static constexpr double DEFAULT_US = 60.0; // a forest batch at 5 cm: ~7 ms for 128 chunks
    static constexpr double RATE = 0.25; // weight of one batch in a measured LOD's cost
    static constexpr double MAX_SCALE = 8.0; // one batch moves a cost by at most this factor

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

    // p_counts[lod] = the batch's chunks at that LOD.
    double estimate(const std::vector<uint32_t> &p_counts) const {
        double e = 0.0;
        for (int lod = 0; lod < (int)p_counts.size(); lod++) {
            e += p_counts[lod] * estimate(lod);
        }
        return e;
    }

    void observe(const std::vector<uint32_t> &p_counts, double p_measured_us) {
        const double e = estimate(p_counts);
        if (e <= 0.0 || p_measured_us <= 0.0) {
            return;
        }
        // A LOD's first measure replaces the borrowed cost; a measured cost only moves a damped, bounded step.
        const double first = p_measured_us / e;
        const double scale = std::clamp(first, 1.0 / MAX_SCALE, MAX_SCALE);
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
    }

    std::vector<double> us_per_chunk; // < 0: never measured
};
