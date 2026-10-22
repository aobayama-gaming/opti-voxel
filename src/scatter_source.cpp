// A procedural source as a field of keys (scatter_source.hpp).
#include "scatter_source.hpp"

#include "tree_archetype.hpp"

#include <godot_cpp/core/error_macros.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>

namespace Archetype {

namespace {

// A record with the largest smooth radius and locality margin of a set (its influence_margin is theirs at most), over
// p_box (none: only the margin counts), listed below p_max_lod: a group's box test passes wherever a member's can.
TableRecord bounds_record(float p_k, float p_margin_c, const VoxelBox &p_box = VoxelBox(), int p_max_lod = TableRecord::NEVER_DROPPED) {
    TableRecord g;
    if (p_box.valid) {
        g.anchor = p_box.lo;
        for (int a = 0; a < 3; a++) {
            g.box_hi[a] = int32_t(MIN<int64_t>(p_box.hi[a] - p_box.lo[a], INT32_MAX));
        }
    }
    g.k = p_k;
    g.margin_c = p_margin_c;
    g.max_lod = uint8_t(p_max_lod);
    return g;
}

// Appends the records of p_records listed at p_lod (all of them: p_lod < 0) as one group of r_out (none if no record is).
void append_group(const TableRecord *p_records, uint32_t p_count, int p_lod, SourceContent &r_out) {
    const uint32_t first = uint32_t(r_out.records.size());
    VoxelBox box;
    float k = 0.0f, margin_c = 0.0f;
    int max_lod = 0;
    for (uint32_t i = 0; i < p_count; i++) {
        const TableRecord &rec = p_records[i];
        if (p_lod >= 0 && !rec.listed_at(p_lod)) {
            continue;
        }
        box = box.merged(record_box(rec));
        k = MAX(k, rec.k);
        margin_c = MAX(margin_c, rec.margin_c);
        max_lod = MAX(max_lod, int(rec.max_lod));
        r_out.records.push_back(rec);
    }
    if (r_out.records.size() > first) {
        r_out.groups.push_back(SourceContent::Group{ first, uint32_t(r_out.records.size()) - first, bounds_record(k, margin_c, box, max_lod) });
    }
}

std::atomic<bool> profile_reported{ false };

} // namespace

void SourceContent::meeting(const std::vector<Group> &p_groups, RecordView p_records, uint32_t p_first, const VoxelPos &p_lo, const VoxelPos &p_hi, int p_lod,
        std::vector<uint32_t> &r_out) {
    for (const Group &g : p_groups) {
        if (!g.bounds.listed_at(p_lod) || !influence_meets(g.bounds, p_lod, p_lo, p_hi)) {
            continue;
        }
        for (uint32_t i = p_first + g.first; i < p_first + g.first + g.count; i++) {
            const TableRecord &rec = p_records[i];
            if (rec.listed_at(p_lod) && influence_meets(rec, p_lod, p_lo, p_hi)) {
                r_out.push_back(i);
            }
        }
    }
}

ScatterSource::ScatterSource(const Setup &p_setup) :
        set(p_setup) {
    ERR_FAIL_COND_MSG(!set.library || !set.scatter.valid() || set.max_extent <= 0 || set.chunk_voxels <= 0,
            "ScatterSource: no library, valid scatter, largest extent or chunk side.");
    prof = SourceProfile(*set.library, set.scatter, set.finish, set.layers, set.lod_count);
    has_blobs = set.blob_type >= 0 && set.finish.band && !prof.joins.empty();
    first_layer = set.lod_count;
    for (const std::pair<const uint32_t, MomentJoin> &j : prof.joins) {
        first_layer = MIN(first_layer, j.second.lod);
    }
    if (has_blobs) {
        MomentBlob unit;
        std::fill(unit.inv_s2, unit.inv_s2 + 3, 1.0f);
        unit.rotation[0] = unit.rotation[4] = unit.rotation[8] = 1.0f;
        blob_margin_c = blob_record(*set.library, set.blob_type, unit, set.finish.skirt_depth).margin_c;
    }
    // The pyramid: each LOD's side the previous one's or twice it, so a tile covers 1 or 8 tiles of the LOD under it.
    int64_t side = PART_TILE_CELLS * set.scatter.side;
    for (int l = 0; l < set.lod_count; l++) {
        while (side < 2 * (set.chunk_voxels << l)) {
            side *= 2;
        }
        part_sides.push_back(side);
    }
    // A base lies within a voxel of the ground (cell_site), a blob's cell within a cell of the records' reach; the largest
    // over the LOD and the finer ones (a culled node hides its subtree).
    int64_t r = 0;
    for (int l = 0; l < set.lod_count; l++) {
        if (l < prof.lod_end) {
            r = MAX(r, part_grow(l) + 1);
        }
        if (has_layer(l)) {
            r = MAX(r, int64_t(std::ceil(prof.reach)) + layer_cell(l) + blob_grow(l) + 1);
        }
        ground_reaches.push_back(r);
    }
}

int64_t ScatterSource::blob_tile_cells(int p_layer) const {
    const int64_t c = layer_cell(p_layer);
    const int64_t side = MAX(PART_TILE_CELLS * set.scatter.side, 2 * (set.chunk_voxels << p_layer));
    return MAX(BLOB_TILE_CELLS, (side + c - 1) / c);
}

VoxelBox ScatterSource::tile_box(const SourceKey &p_key) const {
    const int64_t side = p_key.kind == SourceKey::PARTS ? part_tile_side(p_key.lod) : blob_tile_cells(p_key.lod) * layer_cell(p_key.lod);
    return VoxelBox{ p_key.tile * side, p_key.tile * side + VoxelPos(side - 1, side - 1, side - 1), true };
}

int64_t ScatterSource::blob_reach(int p_layer) const {
    // A cell's members' centroids lie within a cell of its centre (their tents), so its blob's anchor does, and the
    // blob's covariance is at most a member's own plus the spread of their centroids (2 cells); its support's semi-axes
    // are 3 sigma with the layer's filter (make_blob).
    const double c = double(layer_cell(p_layer));
    const double own = part_layer(p_layer) ? prof.part_variance : prof.lost[p_layer].variance;
    const double filter = set.layers.filter * std::ldexp(1.0, p_layer);
    return int64_t(std::ceil(c + 3.0 * std::sqrt(own + 4.0 * c * c + filter * filter)));
}

int64_t ScatterSource::part_grow(int p_lod) const {
    return int64_t(std::ceil(prof.reach)) + influence_margin(bounds_record(prof.k_max, prof.margin_max), p_lod);
}

int64_t ScatterSource::blob_grow(int p_layer) const {
    return blob_reach(p_layer) + influence_margin(bounds_record(0.0f, blob_margin_c), p_layer);
}

int64_t ScatterSource::ground_reach(int p_lod) const {
    return ground_reaches.empty() ? 0 : ground_reaches[MIN(MAX(p_lod, 0), int(ground_reaches.size()) - 1)];
}

void ScatterSource::append_tiles(int p_kind, int p_lod, int64_t p_side, const VoxelBox &p_box, int64_t p_zone_grow, std::vector<SourceKey> &r_out) const {
    const VoxelBox box = p_box.clipped(set.scatter.zone.grown(p_zone_grow));
    if (!box.valid) {
        return;
    }
    VoxelPos lo, hi;
    for (int a = 0; a < 3; a++) {
        lo[a] = WorldCoord::floor_div(box.lo[a], p_side);
        hi[a] = WorldCoord::floor_div(box.hi[a], p_side);
    }
    for (int64_t z = lo.z; z <= hi.z; z++) {
        for (int64_t y = lo.y; y <= hi.y; y++) {
            for (int64_t x = lo.x; x <= hi.x; x++) {
                r_out.push_back(SourceKey{ set.id, p_kind, p_lod, VoxelPos(x, y, z) });
            }
        }
    }
}

void ScatterSource::keys(const VoxelPos &p_lo, const VoxelPos &p_hi, int p_lod, std::vector<SourceKey> &r_out) const {
    r_out.clear();
    if (part_sides.empty()) {
        return; // the constructor refused the setup
    }
    const VoxelBox box{ p_lo, p_hi, true };
    // Every tile whose content can pass a box test in the box: a part tile's records lie within its box grown by the
    // reach, and its bases in the zone; a blob tile's within blob_reach, its cells within a cell of the records' reach.
    if (p_lod < prof.lod_end) {
        append_tiles(SourceKey::PARTS, p_lod, part_tile_side(p_lod), box.grown(part_grow(p_lod)), 0, r_out);
    }
    if (has_layer(p_lod)) {
        const int64_t c = layer_cell(p_lod);
        append_tiles(SourceKey::BLOBS, p_lod, blob_tile_cells(p_lod) * c, box.grown(blob_grow(p_lod)), int64_t(std::ceil(prof.reach)) + c, r_out);
    }
}

void ScatterSource::chunk_keys(const ChunkId &p_chunk, const ChunkParam &p_param, std::vector<SourceKey> &r_out) const {
    VoxelPos lo, hi;
    chunk_sample_box(p_chunk, p_param, lo, hi);
    keys(lo, hi, ChunkMath::get_lod(p_chunk), r_out);
}

void ScatterSource::inputs(const SourceKey &p_key, std::vector<SourceKey> &r_out) const {
    r_out.clear();
    if (part_sides.empty()) {
        return;
    }
    if (p_key.kind == SourceKey::PARTS) {
        if (p_key.lod > 0) {
            // The L - 1 tiles inside this one (the sides are powers of two apart).
            const int64_t n = part_tile_side(p_key.lod) / part_tile_side(p_key.lod - 1);
            for (int64_t z = 0; z < n; z++) {
                for (int64_t y = 0; y < n; y++) {
                    for (int64_t x = 0; x < n; x++) {
                        r_out.push_back(SourceKey{ set.id, SourceKey::PARTS, p_key.lod - 1, p_key.tile * n + VoxelPos(x, y, z) });
                    }
                }
            }
        }
        return;
    }
    if (part_layer(p_key.lod)) {
        // The LOD-0 part tiles whose records can have their centroid in the tile's cells or MARGIN_CELLS past them.
        const VoxelBox region = tile_box(p_key).grown(MARGIN_CELLS * layer_cell(p_key.lod));
        append_tiles(SourceKey::PARTS, 0, part_tile_side(0), region.grown(int64_t(std::ceil(prof.reach))), 0, r_out);
    }
}

const SourceContent &ScatterSource::input(const SourceKey &p_key, const Inputs &p_inputs, size_t p_i, SourceContent &r_local) const {
    if (p_i < p_inputs.size() && p_inputs[p_i]) {
        return *p_inputs[p_i];
    }
    content(p_key, Inputs(), r_local);
    return r_local;
}

void ScatterSource::content(const SourceKey &p_key, const Inputs &p_inputs, SourceContent &r_out) const {
    r_out = SourceContent();
    if (part_sides.empty() || p_key.source != set.id) {
        return;
    }
    if (p_key.kind == SourceKey::PARTS && p_key.lod < prof.lod_end) {
        part_tile(p_key, p_inputs, r_out);
    } else if (p_key.kind == SourceKey::BLOBS && has_layer(p_key.lod)) {
        blob_tile(p_key, p_inputs, r_out);
    }
}

void ScatterSource::part_tile(const SourceKey &p_key, const Inputs &p_inputs, SourceContent &r_out) const {
    if (p_key.lod > 0) {
        std::vector<SourceKey> under;
        inputs(p_key, under);
        SourceContent local;
        for (size_t i = 0; i < under.size(); i++) {
            const SourceContent &c = input(under[i], p_inputs, i, local);
            for (const SourceContent::Group &g : c.groups) {
                append_group(c.records.data() + g.first, g.count, p_key.lod, r_out);
            }
            r_out.out_of_profile += c.out_of_profile;
        }
        return;
    }
    std::vector<CellSite> sites;
    set.scatter.sites(tile_box(p_key), sites);
    const DefLibrary &lib = *set.library;
    std::vector<TableRecord> grown;
    for (const CellSite &cs : sites) {
        grown.clear();
        set.scatter.grow(cs.site, grown);
        // The table's limits (SDFTable::admit): a site with a record it would refuse is left out whole.
        if (std::any_of(grown.begin(), grown.end(), [&](const TableRecord &p_r) { return !record_fit_error(lib[p_r.def], p_r, set.max_extent).is_empty(); })) {
            continue;
        }
        bool out = false;
        for (TableRecord &rec : grown) {
            finish_table_record(lib, set.finish, rec);
            // keys() and blob_reach() trust the profile, read from samples: a record past it may be missing from chunks.
            out = out || !prof.covers(cs.site.base, rec);
        }
        append_group(grown.data(), uint32_t(grown.size()), -1, r_out);
        if (out) {
            r_out.out_of_profile++;
            if (!profile_reported.exchange(true)) {
                ERR_PRINT("ScatterSource: a site has a record past the profile (reach, last LOD or margin): the chunks it reaches may miss it.");
            }
        }
    }
}

void ScatterSource::near_items(const SourceKey &p_key, const Inputs &p_inputs, std::vector<MomentItem> &r_items) const {
    const VoxelBox region = tile_box(p_key).grown(MARGIN_CELLS * layer_cell(p_key.lod));
    std::vector<SourceKey> tiles;
    inputs(p_key, tiles);
    SourceContent local;
    for (size_t t = 0; t < tiles.size(); t++) {
        const SourceContent &parts = input(tiles[t], p_inputs, t, local);
        for (const SourceContent::Group &g : parts.groups) {
            // A record's centroid lies in its box: a site whose records' box misses the region has none in it.
            if (!record_box(g.bounds).intersects(region)) {
                continue;
            }
            for (uint32_t i = g.first; i < g.first + g.count; i++) {
                const MomentItem item = record_item(*set.library, parts.records[i]);
                if (item.moments.empty()) {
                    continue;
                }
                bool inside = true;
                for (int a = 0; a < 3; a++) {
                    const double at = double(item.anchor[a]) + item.moments.c[a];
                    inside = inside && at >= double(region.lo[a]) && at <= double(region.hi[a]) + 1.0;
                }
                if (inside) {
                    r_items.push_back(item);
                }
            }
        }
    }
}

void ScatterSource::far_items(const SourceKey &p_key, std::vector<MomentItem> &r_items) const {
    // One proxy per surface cell of g voxels: the sites of its patch of ground ((g / side)^2 cells' worth, one site per
    // source cell on flat ground, times the share of the cell inside the zone), each what the mean site has lost at the
    // layer turned to the up there. A point mass at its site: the hashed sites, two per cell side, already spread over the
    // cells as the sites do (spreading each over its patch as well made the blobs 1.6-2 times too wide).
    const int64_t c = layer_cell(p_key.lod);
    const int64_t g = MAX<int64_t>(1, c / PROXY_PER_CELL);
    const CellScatter &s = set.scatter;
    // A proxy outside the zone weighs nothing: the walk stops a cell past it.
    const VoxelBox region = tile_box(p_key).grown(MARGIN_CELLS * c + int64_t(std::ceil(prof.reach))).clipped(s.zone.grown(g));
    std::vector<CellSite> sites;
    if (region.valid) {
        surface_sites(region.lo, region.hi, g, s.salt ^ PROXY_SALT, s.ground, sites, false); // a proxy's base needn't be on the LOD-0 ground
    }
    const double side = double(s.side);
    const SourceProfile::Lost &lost = prof.lost[p_key.lod];
    for (const CellSite &cs : sites) {
        const VoxelPos &base = cs.site.base;
        double inside = 1.0;
        for (int a = 0; a < 3; a++) {
            const int64_t lo = cs.cell[a] * g;
            inside *= double(MAX<int64_t>(0, MIN(lo + g - 1, s.zone.hi[a]) - MAX(lo, s.zone.lo[a]) + 1)) / double(g);
        }
        if (!(inside > 0.0) || (s.accept && !s.accept(base))) {
            continue;
        }
        const double weight = (double(g) / side) * (double(g) / side) * inside;
        float rows[9];
        up_rotation(s.up(base), rows);
        for (const std::pair<uint32_t, Moments> &m : lost.moments.parts) {
            MomentItem item;
            item.anchor = base;
            item.material = m.first;
            item.drop_lod = 0; // its moments are what is lost
            item.moments = m.second.rotated(rows).scaled(weight);
            const auto area = lost.area.find(m.first);
            if (area != lost.area.end()) {
                rotate_symmetric(area->second.data(), rows, item.area);
            }
            for (double &v : item.area) {
                v *= weight;
            }
            item.area_given = true;
            r_items.push_back(item);
        }
    }
}

void ScatterSource::blobs(const SourceKey &p_key, const Inputs &p_inputs, std::vector<MomentBlob> &r_out) const {
    r_out.clear();
    if (part_sides.empty() || p_key.kind != SourceKey::BLOBS || !has_layer(p_key.lod)) {
        return;
    }
    std::vector<MomentItem> items;
    if (part_layer(p_key.lod)) {
        near_items(p_key, p_inputs, items);
    } else {
        far_items(p_key, items);
    }
    // An item tents into the cells within a cell of its centroid, which lies within two reaches of its anchor (a record's
    // anchor and centroid are both in its box, within the reach of its base): none near the tile's own cells, no blob
    // (most tiles of a column: their region reaches the ground, their cells don't).
    const VoxelBox near = tile_box(p_key).grown(layer_cell(p_key.lod) + 2 * int64_t(std::ceil(prof.reach)));
    if (std::none_of(items.begin(), items.end(), [&near](const MomentItem &p_i) { return near.contains(p_i.anchor); })) {
        return;
    }
    MomentLayerParams params = set.layers;
    params.lod_min = params.lod_max = p_key.lod;
    const int64_t n = blob_tile_cells(p_key.lod);
    params.judged = VoxelBox{ p_key.tile * n, p_key.tile * n + VoxelPos(n - 1, n - 1, n - 1), true }; // its own cells
    MomentLayers layers;
    layers.build(items, params, &prof.joins);
    layers.kept(r_out);
}

void ScatterSource::blob_tile(const SourceKey &p_key, const Inputs &p_inputs, SourceContent &r_out) const {
    std::vector<MomentBlob> kept;
    blobs(p_key, p_inputs, kept);
    const DefLibrary &lib = *set.library;
    for (const MomentBlob &b : kept) {
        const TableRecord rec = blob_record(lib, set.blob_type, b, set.finish.skirt_depth);
        if (!record_fit_error(lib[rec.def], rec, set.max_extent).is_empty()) {
            continue; // refused like the table's (SDFTable::blob_fits)
        }
        r_out.groups.push_back(SourceContent::Group{ uint32_t(r_out.records.size()), 1, rec });
        r_out.records.push_back(rec);
    }
}

} // namespace Archetype
