// The far field of the archetype instances (moment_layers.hpp).
#include "moment_layers.hpp"

#include <godot_cpp/core/error_macros.hpp>

#include <algorithm>
#include <cmath>

namespace Archetype {

namespace {

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

bool boxes_meet(const VoxelPos &p_a, const double *p_ea, const VoxelPos &p_b, const double *p_eb) {
    for (int a = 0; a < 3; a++) {
        if (std::abs(double(p_a[a] - p_b[a])) > p_ea[a] + p_eb[a]) {
            return false;
        }
    }
    return true;
}

// The amplitude's factor across a sheet (moment_layers.hpp): lambda ascending, p_amplitude the mass-keeping one.
double sheet_gain(const double p_lambda[3], double p_filter_voxels, double p_sheet_ratio, double p_max_gain, double p_amplitude) {
    if (p_sheet_ratio <= 0.0 || !(p_lambda[0] > 0.0) || !(p_amplitude > 0.0)) {
        return 1.0;
    }
    const double f2 = p_filter_voxels * p_filter_voxels;
    const double sheetness = std::sqrt((p_lambda[1] + f2) / (p_lambda[0] + f2)); // as the LOD sees it: a lone flat part is none
    const double lost = std::sqrt((p_lambda[0] + f2) / p_lambda[0]);
    const double gain = std::clamp(sheetness / p_sheet_ratio, 1.0, std::max(std::min(lost, p_max_gain), 1.0));
    return std::max(std::min(gain, 1.0 / p_amplitude), 1.0);
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

MomentBlob make_blob(const Moments &p_cell, const VoxelPos &p_anchor, int p_lod, uint32_t p_material, double p_filter_voxels, double p_sheet_ratio, double p_max_gain) {
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
    const double kept_mass = p_cell.m0 * 105.0 / (32.0 * M_PI * b.s[0] * b.s[1] * b.s[2]);
    b.amplitude = float(kept_mass * sheet_gain(lambda, p_filter_voxels, p_sheet_ratio, p_max_gain, kept_mass));
    return b;
}

MomentJoins moment_joins(const std::vector<MomentItem> &p_items, const MomentLayerParams &p_params) {
    std::map<uint32_t, std::vector<double>> dropped; // material -> mass dropped at each LOD (the last entry: never)
    for (const MomentItem &item : p_items) {
        if (item.moments.empty()) {
            continue;
        }
        std::vector<double> &d = dropped[item.material];
        d.resize(p_params.lod_max + 2, 0.0);
        d[MIN(MAX(item.drop_lod, 0), p_params.lod_max + 1)] += item.moments.m0;
    }
    MomentJoins joins;
    for (const std::pair<const uint32_t, std::vector<double>> &d : dropped) {
        double total = 0.0, sum = 0.0;
        for (const double m : d.second) {
            total += m;
        }
        int lod = p_params.lod_max + 1;
        for (int l = 0; l <= p_params.lod_max; l++) {
            sum += d.second[l];
            if (sum >= p_params.min_mass_share * total) {
                lod = l;
                break;
            }
        }
        joins[d.first] = lod;
    }
    return joins;
}

bool MomentLayers::on_layer(const MomentItem &p_item, int p_lod) const {
    const auto join = joins.find(p_item.material);
    return !p_item.moments.empty() && p_item.drop_lod <= p_lod && join != joins.end() && join->second <= p_lod;
}

int MomentLayers::tent(const MomentItem &p_item, const Layer &p_layer, CellKey r_keys[8], double r_weights[8]) const {
    // Cell centres at (i + 1/2) cell: the 2 x 2 x 2 around the centroid.
    double f[3];
    int64_t i0[3];
    for (int a = 0; a < 3; a++) {
        i0[a] = cell_of(p_item.anchor[a], p_item.moments.c[a] - 0.5 * double(p_layer.cell), p_layer.cell, f[a]);
    }
    int n = 0;
    for (int k = 0; k < 8; k++) {
        const int d[3] = { k & 1, (k >> 1) & 1, k >> 2 };
        const double w = (d[0] ? f[0] : 1.0 - f[0]) * (d[1] ? f[1] : 1.0 - f[1]) * (d[2] ? f[2] : 1.0 - f[2]);
        if (w > 0.0) {
            r_keys[n] = CellKey{ i0[0] + d[0], i0[1] + d[1], i0[2] + d[2], p_item.material };
            r_weights[n++] = w;
        }
    }
    return n;
}

void MomentLayers::resum_block(const Layer &p_layer, const CellKey &p_key, uint32_t p_block, Cell &r_cell) const {
    const int64_t c = p_layer.cell;
    const VoxelPos centre(p_key.x * c + c / 2, p_key.y * c + c / 2, p_key.z * c + c / 2);
    Moments sum;
    auto it = std::lower_bound(r_cell.members.begin(), r_cell.members.end(), p_block << BLOCK_SHIFT);
    for (; it != r_cell.members.end() && (*it >> BLOCK_SHIFT) == p_block; ++it) {
        const MomentItem &item = items[*it];
        CellKey keys[8];
        double weights[8];
        const int n = tent(item, p_layer, keys, weights);
        for (int k = 0; k < n; k++) {
            if (keys[k] == p_key) {
                const double offset[3] = { double(item.anchor.x - centre.x), double(item.anchor.y - centre.y), double(item.anchor.z - centre.z) };
                sum.add(item.moments.scaled(weights[k]), offset);
            }
        }
    }
    if (sum.empty()) {
        r_cell.blocks.erase(p_block);
    } else {
        r_cell.blocks[p_block] = sum;
    }
}

void MomentLayers::resum(const Layer &p_layer, const CellKey &p_key, Cell &r_cell) const {
    const int64_t c = p_layer.cell;
    const VoxelPos centre(p_key.x * c + c / 2, p_key.y * c + c / 2, p_key.z * c + c / 2);
    Moments sum;
    for (const std::pair<const uint32_t, Moments> &b : r_cell.blocks) {
        sum.add(b.second);
    }
    // The cover takes over one doubling per LOD from the material's first layer: no seam steps more than a mass-kept one
    // (a first layer next to the parts still drawn whole, the rock scatter's worst).
    const auto join = joins.find(p_key.material);
    const double max_gain = join == joins.end() ? 1.0 : std::ldexp(1.0, p_layer.lod - join->second);
    r_cell.blob = make_blob(sum, centre, p_layer.lod, p_key.material, params.filter * std::ldexp(1.0, p_layer.lod), params.sheet_ratio, max_gain);
    r_cell.blob.cell[0] = p_key.x;
    r_cell.blob.cell[1] = p_key.y;
    r_cell.blob.cell[2] = p_key.z;
    support_extent(r_cell.blob, r_cell.extent);
}

bool MomentLayers::bin_range(const Layer &p_layer, const VoxelPos &p_anchor, const double p_extent[3], Bin &r_lo, Bin &r_hi) const {
    // The box's integer hull: a point where two boxes meet floors into both hulls, so into a bin both are filed in.
    const int64_t bin = BIN_CELLS * p_layer.cell;
    int64_t lo[3], hi[3];
    double bins = 1.0;
    for (int a = 0; a < 3; a++) {
        const double e = std::ceil(p_extent[a]);
        if (!(e < double(bin) * WIDE_BINS)) {
            return false;
        }
        lo[a] = WorldCoord::floor_div(p_anchor[a] - int64_t(e), bin);
        hi[a] = WorldCoord::floor_div(p_anchor[a] + int64_t(e), bin);
        bins *= double(hi[a] - lo[a] + 1);
    }
    r_lo = Bin(lo[0], lo[1], lo[2]);
    r_hi = Bin(hi[0], hi[1], hi[2]);
    return bins <= WIDE_BINS;
}

void MomentLayers::place(Layer &r_layer, const CellKey &p_key, Cell &r_cell) {
    Bin lo, hi;
    if (!bin_range(r_layer, r_cell.blob.anchor, r_cell.extent, lo, hi)) {
        r_layer.wide.insert(p_key);
    } else {
        for (int64_t x = std::get<0>(lo); x <= std::get<0>(hi); x++) {
            for (int64_t y = std::get<1>(lo); y <= std::get<1>(hi); y++) {
                for (int64_t z = std::get<2>(lo); z <= std::get<2>(hi); z++) {
                    r_layer.bins[Bin(x, y, z)].push_back(p_key);
                }
            }
        }
    }
    r_cell.placed = true;
}

void MomentLayers::unplace(Layer &r_layer, const CellKey &p_key, Cell &r_cell) {
    r_cell.placed = false;
    Bin lo, hi;
    if (!bin_range(r_layer, r_cell.blob.anchor, r_cell.extent, lo, hi)) {
        r_layer.wide.erase(p_key);
        return;
    }
    for (int64_t x = std::get<0>(lo); x <= std::get<0>(hi); x++) {
        for (int64_t y = std::get<1>(lo); y <= std::get<1>(hi); y++) {
            for (int64_t z = std::get<2>(lo); z <= std::get<2>(hi); z++) {
                const auto it = r_layer.bins.find(Bin(x, y, z));
                if (it == r_layer.bins.end()) {
                    continue;
                }
                it->second.erase(std::remove(it->second.begin(), it->second.end(), p_key), it->second.end());
                if (it->second.empty()) {
                    r_layer.bins.erase(it);
                }
            }
        }
    }
}

void MomentLayers::candidates(const Layer &p_layer, const VoxelPos &p_anchor, const double p_extent[3], std::vector<CellKey> &r_out) const {
    r_out.clear();
    Bin lo, hi;
    if (!bin_range(p_layer, p_anchor, p_extent, lo, hi)) {
        for (const std::pair<const CellKey, Cell> &c : p_layer.cells) {
            r_out.push_back(c.first); // a wide blob may meet any
        }
        return;
    }
    for (int64_t x = std::get<0>(lo); x <= std::get<0>(hi); x++) {
        for (int64_t y = std::get<1>(lo); y <= std::get<1>(hi); y++) {
            for (int64_t z = std::get<2>(lo); z <= std::get<2>(hi); z++) {
                const auto it = p_layer.bins.find(Bin(x, y, z));
                if (it != p_layer.bins.end()) {
                    r_out.insert(r_out.end(), it->second.begin(), it->second.end());
                }
            }
        }
    }
    r_out.insert(r_out.end(), p_layer.wide.begin(), p_layer.wide.end());
    std::sort(r_out.begin(), r_out.end());
    r_out.erase(std::unique(r_out.begin(), r_out.end()), r_out.end());
}

void MomentLayers::judge(const Layer &p_layer, const CellKey &p_key, Cell &r_cell) const {
    thread_local std::vector<CellKey> near;
    candidates(p_layer, r_cell.blob.anchor, r_cell.extent, near);
    box_tests += near.size();
    double sum = 0.0; // in cell order: the same bits whichever way the bins were filled
    for (const CellKey &k : near) {
        const Cell &o = k == p_key ? r_cell : p_layer.cells.at(k);
        sum += boxes_meet(r_cell.blob.anchor, r_cell.extent, o.blob.anchor, o.extent) ? double(o.blob.amplitude) : 0.0;
    }
    r_cell.blob.peak_bound = sum;
    r_cell.kept = sum >= params.min_occupancy && double(r_cell.blob.amplitude) >= SLIVER * params.min_occupancy;
}

void MomentLayers::build_layer(Layer &r_layer) {
    for (uint32_t id = 0; id < items.size(); id++) {
        if (!on_layer(items[id], r_layer.lod)) {
            continue;
        }
        CellKey keys[8];
        double weights[8];
        const int n = tent(items[id], r_layer, keys, weights);
        for (int k = 0; k < n; k++) {
            r_layer.cells[keys[k]].members.push_back(id);
        }
    }
    for (std::pair<const CellKey, Cell> &c : r_layer.cells) {
        for (size_t m = 0; m < c.second.members.size(); m++) {
            const uint32_t block = c.second.members[m] >> BLOCK_SHIFT;
            if (m == 0 || block != (c.second.members[m - 1] >> BLOCK_SHIFT)) {
                resum_block(r_layer, c.first, block, c.second);
            }
        }
        resum(r_layer, c.first, c.second);
        place(r_layer, c.first, c.second);
    }
    for (std::pair<const CellKey, Cell> &c : r_layer.cells) {
        judge(r_layer, c.first, c.second);
    }
}

void MomentLayers::build(const std::vector<MomentItem> &p_items, const MomentLayerParams &p_params) {
    ERR_FAIL_COND_MSG(p_params.cell_shift < 1 || p_params.lod_max + p_params.cell_shift > 60, "MomentLayers: cells of 2^(lod + cell_shift) voxels must be within [2, 2^60].");
    items = p_items;
    params = p_params;
    joins = moment_joins(items, params);
    layers.assign(MAX(params.lod_max - params.lod_min + 1, 0), Layer());
    for (size_t i = 0; i < layers.size(); i++) {
        layers[i].lod = params.lod_min + int(i);
        layers[i].cell = int64_t(1) << (layers[i].lod + params.cell_shift);
        build_layer(layers[i]);
    }
}

void MomentLayers::update(uint32_t p_id, const MomentItem &p_item, std::vector<MomentBlobChange> &r_changes) {
    if (items.size() <= p_id) {
        items.resize(p_id + 1);
    }
    const MomentItem before = items[p_id];
    items[p_id] = p_item;
    for (Layer &layer : layers) {
        update_layer(layer, p_id, before, r_changes);
    }
}

void MomentLayers::update_layer(Layer &layer, uint32_t p_id, const MomentItem &p_before, std::vector<MomentBlobChange> &r_changes) {
    const MomentItem &before = p_before;
    const MomentItem &p_item = items[p_id];
    {
        // The cells it leaves and joins.
        std::vector<CellKey> dirty;
        CellKey keys[8];
        double weights[8];
        if (on_layer(before, layer.lod)) {
            const int n = tent(before, layer, keys, weights);
            for (int k = 0; k < n; k++) {
                std::vector<uint32_t> &m = layer.cells[keys[k]].members;
                m.erase(std::remove(m.begin(), m.end(), p_id), m.end());
                dirty.push_back(keys[k]);
            }
        }
        if (on_layer(p_item, layer.lod)) {
            const int n = tent(p_item, layer, keys, weights);
            for (int k = 0; k < n; k++) {
                std::vector<uint32_t> &m = layer.cells[keys[k]].members;
                m.insert(std::lower_bound(m.begin(), m.end(), p_id), p_id);
                dirty.push_back(keys[k]);
            }
        }
        if (dirty.empty()) {
            return;
        }
        std::sort(dirty.begin(), dirty.end());
        dirty.erase(std::unique(dirty.begin(), dirty.end()), dirty.end());
        // Re-sum them; every blob whose box meets one of theirs, before or after, is judged again.
        std::vector<CellKey> affected = dirty, near;
        // The blobs whose box meets a box (their bound reads its amplitude).
        auto meeting = [&](const VoxelPos &p_anchor, const double *p_extent) {
            candidates(layer, p_anchor, p_extent, near);
            for (const CellKey &n : near) {
                const auto it = layer.cells.find(n);
                if (it != layer.cells.end() && boxes_meet(p_anchor, p_extent, it->second.blob.anchor, it->second.extent)) {
                    affected.push_back(n);
                }
            }
        };
        std::map<CellKey, std::pair<bool, MomentBlob>> was; // kept and blob before, for the changes
        for (const CellKey &k : dirty) {
            Cell &cell = layer.cells[k];
            const bool existed = cell.placed;
            if (existed) {
                was[k] = { cell.kept, cell.blob };
                meeting(cell.blob.anchor, cell.extent);
                unplace(layer, k, cell);
            }
            if (cell.members.empty()) {
                if (existed && cell.kept) {
                    MomentBlobChange gone;
                    gone.blob = cell.blob;
                    r_changes.push_back(gone);
                }
                layer.cells.erase(k);
                continue;
            }
            resum_block(layer, k, p_id >> BLOCK_SHIFT, cell);
            resum(layer, k, cell);
            place(layer, k, cell);
        }
        for (const CellKey &k : dirty) {
            const auto it = layer.cells.find(k);
            if (it != layer.cells.end()) {
                meeting(it->second.blob.anchor, it->second.extent);
            }
        }
        std::sort(affected.begin(), affected.end());
        affected.erase(std::unique(affected.begin(), affected.end()), affected.end());
        for (const CellKey &k : affected) {
            const auto it = layer.cells.find(k);
            if (it == layer.cells.end()) {
                continue;
            }
            Cell &cell = it->second;
            const auto w = was.find(k);
            const bool kept_before = w != was.end() ? w->second.first : cell.kept;
            judge(layer, k, cell);
            const bool reblobbed = w != was.end();
            if (cell.kept != kept_before || (cell.kept && reblobbed)) {
                MomentBlobChange change;
                change.blob = cell.blob;
                change.kept = cell.kept;
                r_changes.push_back(change);
            }
        }
    }
}

void MomentLayers::kept(std::vector<MomentBlob> &r_out) const {
    r_out.clear();
    for (const Layer &layer : layers) {
        for (const std::pair<const CellKey, Cell> &c : layer.cells) {
            if (c.second.kept) {
                r_out.push_back(c.second.blob);
            }
        }
    }
}

void MomentLayers::rebuild(std::vector<MomentBlob> &r_out) const {
    MomentLayers fresh;
    fresh.items = items;
    fresh.params = params;
    fresh.joins = joins;
    fresh.layers.assign(layers.size(), Layer());
    for (size_t i = 0; i < layers.size(); i++) {
        fresh.layers[i].lod = layers[i].lod;
        fresh.layers[i].cell = layers[i].cell;
        fresh.build_layer(fresh.layers[i]);
    }
    fresh.kept(r_out);
}

void build_moment_blobs(const std::vector<MomentItem> &p_items, const MomentLayerParams &p_params, std::vector<MomentBlob> &r_out) {
    MomentLayers layers;
    layers.build(p_items, p_params);
    layers.kept(r_out);
}

} // namespace Archetype
