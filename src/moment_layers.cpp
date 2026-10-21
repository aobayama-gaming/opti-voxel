// The far field of the archetype instances (moment_layers.hpp).
#include "moment_layers.hpp"

#include <godot_cpp/core/error_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <tuple>

namespace Archetype {

namespace {

struct CellKey {
    int64_t x, y, z;
    uint32_t material;
    bool operator<(const CellKey &p_o) const { return std::tie(x, y, z, material) < std::tie(p_o.x, p_o.y, p_o.z, p_o.material); }
};

// The cell index of an absolute coordinate p_voxel + p_offset on cells of p_cell voxels (floor), and the position in
// it in cells (in [0, 1)).
int64_t cell_of(int64_t p_voxel, double p_offset, int64_t p_cell, double &r_frac) {
    const int64_t base = WorldCoord::floor_div(p_voxel, p_cell);
    const double u = (double(p_voxel - base * p_cell) + p_offset) / double(p_cell);
    const double fl = std::floor(u);
    r_frac = u - fl;
    return base + int64_t(fl);
}

// The world-axis half extents of a blob's support (an ellipsoid of semi-axes s along the rotation's rows).
void support_extent(const MomentBlob &p_b, double r_e[3]) {
    for (int j = 0; j < 3; j++) {
        double e2 = 0.0;
        for (int i = 0; i < 3; i++) {
            const double v = double(p_b.rotation[3 * i + j]) * p_b.s[i];
            e2 += v * v;
        }
        r_e[j] = std::sqrt(e2);
    }
}

// The centre of cell p_key (cells of p_cell voxels, an even count): the cell's moments' anchor.
VoxelPos cell_centre(const CellKey &p_key, int64_t p_cell) {
    return VoxelPos(p_key.x * p_cell + p_cell / 2, p_key.y * p_cell + p_cell / 2, p_key.z * p_cell + p_cell / 2);
}

// Each blob's peak_bound: the summed amplitudes of the blobs whose support box meets its own (any material: the
// occupancy is their sum). Blobs whose support reaches at most NEAR_CELLS cells from their anchor (nearly all: a cell's
// parts plus the filter) sit in bins of 2 NEAR_CELLS cells, so two such blobs that meet are in neighbouring bins; the
// few wider ones (a long part dropped at a fine LOD) are tested against every blob.
void bound_peaks(std::vector<MomentBlob> &r_blobs, size_t p_first, int64_t p_cell) {
    constexpr int64_t NEAR_CELLS = 2;
    const int64_t bin = 2 * NEAR_CELLS * p_cell;
    auto bin_of = [bin](const VoxelPos &p_v) {
        return std::make_tuple(WorldCoord::floor_div(p_v.x, bin), WorldCoord::floor_div(p_v.y, bin), WorldCoord::floor_div(p_v.z, bin));
    };
    const size_t n = r_blobs.size() - p_first;
    std::vector<std::array<double, 3>> extent(n);
    std::map<std::tuple<int64_t, int64_t, int64_t>, std::vector<size_t>> bins;
    std::vector<size_t> wide;
    std::vector<bool> is_wide(n, false);
    for (size_t i = 0; i < n; i++) {
        support_extent(r_blobs[p_first + i], extent[i].data());
        if (std::max(extent[i][0], std::max(extent[i][1], extent[i][2])) <= double(NEAR_CELLS * p_cell)) {
            bins[bin_of(r_blobs[p_first + i].anchor)].push_back(i);
        } else {
            wide.push_back(i);
            is_wide[i] = true;
        }
    }
    auto meets = [&](size_t p_i, size_t p_j) {
        for (int a = 0; a < 3; a++) {
            if (std::abs(double(r_blobs[p_first + p_i].anchor[a] - r_blobs[p_first + p_j].anchor[a])) > extent[p_i][a] + extent[p_j][a]) {
                return false;
            }
        }
        return true;
    };
    for (size_t i = 0; i < n; i++) {
        double sum = 0.0;
        if (is_wide[i]) {
            for (size_t j = 0; j < n; j++) {
                sum += meets(i, j) ? double(r_blobs[p_first + j].amplitude) : 0.0;
            }
        } else {
            const auto [bx, by, bz] = bin_of(r_blobs[p_first + i].anchor);
            for (int k = 0; k < 27; k++) {
                const auto it = bins.find({ bx + k % 3 - 1, by + k / 3 % 3 - 1, bz + k / 9 - 1 });
                for (size_t m = 0; it != bins.end() && m < it->second.size(); m++) {
                    sum += meets(i, it->second[m]) ? double(r_blobs[p_first + it->second[m]].amplitude) : 0.0;
                }
            }
            for (const size_t j : wide) {
                sum += meets(i, j) ? double(r_blobs[p_first + j].amplitude) : 0.0;
            }
        }
        r_blobs[p_first + i].peak_bound = sum;
    }
}

} // namespace

double MomentBlob::occupancy(const double p_offset[3]) const {
    double q = 0.0;
    for (int i = 0; i < 3; i++) {
        const double l = double(rotation[3 * i]) * p_offset[0] + double(rotation[3 * i + 1]) * p_offset[1] + double(rotation[3 * i + 2]) * p_offset[2];
        q += l * l * double(inv_s2[i]);
    }
    const double t = std::max(1.0 - q, 0.0);
    return double(amplitude) * t * t;
}

MomentBlob make_blob(const Moments &p_cell, const VoxelPos &p_anchor, int p_lod, uint32_t p_material, double p_filter_voxels) {
    MomentBlob b;
    b.lod = p_lod;
    b.material = p_material;
    b.mass = p_cell.m0;
    b.anchor = p_anchor + VoxelPos(std::llround(p_cell.c[0]), std::llround(p_cell.c[1]), std::llround(p_cell.c[2]));
    double lambda[3], axes[9];
    p_cell.principal(lambda, axes);
    // Right-handed (a proper rotation, like every other record's).
    const double det = axes[0] * (axes[4] * axes[8] - axes[5] * axes[7]) - axes[1] * (axes[3] * axes[8] - axes[5] * axes[6]) +
            axes[2] * (axes[3] * axes[7] - axes[4] * axes[6]);
    for (int i = 0; i < 9; i++) {
        b.rotation[i] = float(i >= 6 && det < 0.0 ? -axes[i] : axes[i]);
    }
    for (int i = 0; i < 3; i++) {
        b.s[i] = 3.0 * std::sqrt(lambda[i] + p_filter_voxels * p_filter_voxels);
        b.inv_s2[i] = float(1.0 / (b.s[i] * b.s[i]));
    }
    b.amplitude = float(p_cell.m0 * 105.0 / (32.0 * M_PI * b.s[0] * b.s[1] * b.s[2]));
    return b;
}

void build_moment_blobs(const std::vector<MomentItem> &p_items, const MomentLayerParams &p_params, std::vector<MomentBlob> &r_out) {
    r_out.clear();
    // A blob under this many times min_occupancy on its own is a sliver of a part's tent.
    constexpr double SLIVER = 1.0 / 16.0;
    ERR_FAIL_COND_MSG(p_params.cell_shift < 1 || p_params.lod_max + p_params.cell_shift > 60, "build_moment_blobs: cells of 2^(lod + cell_shift) voxels must be within [2, 2^60].");
    for (int lod = p_params.lod_min; lod <= p_params.lod_max; lod++) {
        const int64_t cell = int64_t(1) << (lod + p_params.cell_shift);
        std::map<CellKey, Moments> cells;
        for (const MomentItem &item : p_items) {
            if (item.drop_lod > lod || item.moments.empty()) {
                continue;
            }
            // Tent weights over the 2 x 2 x 2 cells around the centroid (cell centres at (i + 1/2) cell).
            double f[3];
            int64_t i0[3];
            for (int a = 0; a < 3; a++) {
                i0[a] = cell_of(item.anchor[a], item.moments.c[a] - 0.5 * double(cell), cell, f[a]);
            }
            for (int k = 0; k < 8; k++) {
                const int d[3] = { k & 1, (k >> 1) & 1, k >> 2 };
                const double w = (d[0] ? f[0] : 1.0 - f[0]) * (d[1] ? f[1] : 1.0 - f[1]) * (d[2] ? f[2] : 1.0 - f[2]);
                if (!(w > 0.0)) {
                    continue;
                }
                const CellKey key{ i0[0] + d[0], i0[1] + d[1], i0[2] + d[2], item.material };
                const VoxelPos centre = cell_centre(key, cell);
                const double offset[3] = { double(item.anchor.x - centre.x), double(item.anchor.y - centre.y), double(item.anchor.z - centre.z) };
                cells[key].add(item.moments.scaled(w), offset);
            }
        }
        const size_t first = r_out.size();
        for (const std::pair<const CellKey, Moments> &c : cells) {
            r_out.push_back(make_blob(c.second, cell_centre(c.first, cell), lod, c.first.material, p_params.filter * std::ldexp(1.0, lod)));
        }
        bound_peaks(r_out, first, cell);
        // Pruned: too faint to matter even with its neighbours, or a sliver of a part (a tent corner) that only spreads
        // its neighbours' surface.
        r_out.erase(std::remove_if(r_out.begin() + first, r_out.end(),
                            [&](const MomentBlob &p_b) {
                                return p_b.peak_bound < p_params.min_occupancy || double(p_b.amplitude) < SLIVER * p_params.min_occupancy;
                            }),
                r_out.end());
    }
}

} // namespace Archetype
