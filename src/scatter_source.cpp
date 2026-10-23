// A procedural source as a field of keys (scatter_source.hpp).
#include "scatter_source.hpp"

#include "tree_archetype.hpp"

#include <godot_cpp/core/error_macros.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>

namespace Archetype {

namespace {

// p_fn(tile index) for every tile of side p_side meeting p_box, z y x; none for an invalid box.
template <typename F>
void for_each_tile(const VoxelBox &p_box, int64_t p_side, F p_fn) {
    if (!p_box.valid) {
        return;
    }
    VoxelPos lo, hi;
    for (int a = 0; a < 3; a++) {
        lo[a] = WorldCoord::floor_div(p_box.lo[a], p_side);
        hi[a] = WorldCoord::floor_div(p_box.hi[a], p_side);
    }
    for (int64_t z = lo.z; z <= hi.z; z++) {
        for (int64_t y = lo.y; y <= hi.y; y++) {
            for (int64_t x = lo.x; x <= hi.x; x++) {
                p_fn(VoxelPos(x, y, z));
            }
        }
    }
}

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
    // Each LOD's side the previous one's or twice it, so a tile covers 1 or 8 tiles of the LOD under it; a LOD with the
    // previous one's side reads its keys.
    int64_t side = PART_TILE_CELLS * set.scatter.side;
    for (int l = 0; l < set.lod_count; l++) {
        while (side < 2 * (set.chunk_voxels << l)) {
            side *= 2;
        }
        part_key_lods.push_back(l > 0 && side == part_sides.back() ? part_key_lods.back() : l);
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
    const bool parts_grid = p_key.kind == SourceKey::PARTS || p_key.kind == SourceKey::SITES;
    const int64_t side = parts_grid ? part_tile_side(p_key.lod) : blob_tile_cells(p_key.lod) * layer_cell(p_key.lod);
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
    for_each_tile(p_box.clipped(set.scatter.zone.grown(p_zone_grow)), p_side,
            [&](const VoxelPos &p_tile) { r_out.push_back(SourceKey{ set.id, p_kind, p_lod, p_tile }); });
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
        append_tiles(SourceKey::PARTS, part_key_lod(p_lod), part_tile_side(p_lod), box.grown(part_grow(p_lod)), 0, r_out);
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
        append_part_sites(p_key.tile, p_key.lod, r_out);
    } else if (p_key.kind == SourceKey::BLOBS && part_layer(p_key.lod)) {
        // The LOD-0 part tiles whose records can have their centroid in the tile's cells or MARGIN_CELLS past them.
        const VoxelBox region = tile_box(p_key).grown(MARGIN_CELLS * layer_cell(p_key.lod));
        append_tiles(SourceKey::PARTS, 0, part_tile_side(0), region.grown(int64_t(std::ceil(prof.reach))), 0, r_out);
    } else if (p_key.kind == SourceKey::BLOBS && has_layer(p_key.lod)) {
        // A far layer: the proxy tiles its region meets (each walked once, shared by the blob tiles around it).
        proxy_tiles(proxy_region(p_key), p_key.lod, r_out);
    }
}

void ScatterSource::proxy_tiles(const VoxelBox &p_region, int p_layer, std::vector<SourceKey> &r_out) const {
    for_each_tile(p_region, blob_tile_cells(p_layer) * layer_cell(p_layer),
            [&](const VoxelPos &p_tile) { r_out.push_back(SourceKey{ set.id, SourceKey::PROXIES, p_layer, p_tile }); });
}

void ScatterSource::append_part_sites(const VoxelPos &p_tile, int p_lod, std::vector<SourceKey> &r_out) const {
    const VoxelBox box = tile_box(SourceKey{ set.id, SourceKey::PARTS, p_lod, p_tile }).clipped(set.scatter.zone);
    if (!box.valid) {
        return; // no base in it: an empty walk
    }
    if (p_lod <= 0) {
        r_out.push_back(SourceKey{ set.id, SourceKey::SITES, 0, p_tile });
        return;
    }
    // The L - 1 tiles inside this one (1 or 8) that meet the zone.
    for_each_tile(box, part_tile_side(p_lod - 1), [&](const VoxelPos &p_sub) { append_part_sites(p_sub, p_lod - 1, r_out); });
}

const SourceContent &ScatterSource::input(const SourceKey &p_key, const Inputs &p_inputs, size_t p_i, SourceContent &r_local) const {
    if (p_i < p_inputs.size() && p_inputs[p_i]) {
        return *p_inputs[p_i];
    }
    content(p_key, Inputs(), r_local);
    return r_local;
}

template <typename F>
void ScatterSource::for_each_input(const std::vector<SourceKey> &p_keys, const Inputs &p_inputs, F p_fn) const {
    SourceContent local;
    for (size_t i = 0; i < p_keys.size(); i++) {
        p_fn(input(p_keys[i], p_inputs, i, local));
    }
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
    } else if (p_key.walk()) {
        // A site tile is on the LOD-0 part grid, a proxy tile on a far layer's blob grid (inputs()): any other walks a
        // box no key reads (a part layer's proxies: a full walk for nothing).
        const bool walked = p_key.kind == SourceKey::SITES ? p_key.lod == 0 : has_layer(p_key.lod) && !part_layer(p_key.lod);
        ERR_FAIL_COND_MSG(!walked, vformat("ScatterSource: no %s tile at LOD %d.", SourceKey::kind_name(p_key.kind), p_key.lod));
        walk_tile(p_key, r_out);
    }
}

void ScatterSource::walk_tile(const SourceKey &p_key, SourceContent &r_out) const {
    const CellScatter &s = set.scatter;
    if (p_key.kind == SourceKey::SITES) {
        s.sites(tile_box(p_key), r_out.sites);
        return;
    }
    // From the regions' level: the sites a tile keeps are those a region walk keeps over it (surface_sites).
    const VoxelBox box = tile_box(p_key).clipped(proxy_zone(p_key.lod));
    if (box.valid) {
        surface_sites(box.lo, box.hi, proxy_side(p_key.lod), s.salt ^ PROXY_SALT, s.ground, r_out.sites, false, // a proxy's base needn't be on the LOD-0 ground
                proxy_level(p_key.lod));
    }
}

int64_t ScatterSource::proxy_grow(int p_layer) const {
    return MARGIN_CELLS * layer_cell(p_layer) + int64_t(std::ceil(prof.reach));
}

VoxelBox ScatterSource::proxy_region(const SourceKey &p_key) const {
    return tile_box(p_key).grown(proxy_grow(p_key.lod)).clipped(proxy_zone(p_key.lod));
}

int ScatterSource::proxy_level(int p_layer) const {
    // Every tile's region before the zone clips it has the same extent: the level of any of them.
    return walk_level(tile_box(SourceKey{ set.id, SourceKey::BLOBS, p_layer, VoxelPos() }).grown(proxy_grow(p_layer)), proxy_side(p_layer));
}

void ScatterSource::part_tile(const SourceKey &p_key, const Inputs &p_inputs, SourceContent &r_out) const {
    std::vector<SourceKey> in;
    inputs(p_key, in);
    std::vector<TableRecord> grown;
    for_each_input(in, p_inputs, [&](const SourceContent &p_sites) {
        for (const CellSite &cs : p_sites.sites) {
            grow_site(cs, p_key.lod > 0 ? p_key.lod : -1, grown, r_out);
        }
    });
}

void ScatterSource::grow_site(const CellSite &p_site, int p_lod, std::vector<TableRecord> &r_grown, SourceContent &r_out) const {
    const DefLibrary &lib = *set.library;
    r_grown.clear();
    set.scatter.grow(p_site.site, r_grown);
    // The table's limits (SDFTable::admit): a site with a record it would refuse is left out whole.
    if (std::any_of(r_grown.begin(), r_grown.end(), [&](const TableRecord &p_r) { return !record_fit_error(lib[p_r.def], p_r, set.max_extent).is_empty(); })) {
        return;
    }
    bool out = false;
    for (TableRecord &rec : r_grown) {
        finish_table_record(lib, set.finish, rec);
        // keys() and blob_reach() trust the profile, read from samples: a record past it may be missing from chunks.
        out = out || !prof.covers(p_site.site.base, rec);
    }
    append_group(r_grown.data(), uint32_t(r_grown.size()), p_lod, r_out);
    if (out) {
        r_out.out_of_profile++;
        if (!profile_reported.exchange(true)) {
            ERR_PRINT("ScatterSource: a site has a record past the profile (reach, last LOD or margin): the chunks it reaches may miss it.");
        }
    }
}

void ScatterSource::near_items(const SourceKey &p_key, const Inputs &p_inputs, std::vector<MomentItem> &r_items) const {
    const VoxelBox region = tile_box(p_key).grown(MARGIN_CELLS * layer_cell(p_key.lod));
    std::vector<SourceKey> tiles;
    inputs(p_key, tiles);
    for_each_input(tiles, p_inputs, [&](const SourceContent &parts) {
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
    });
}

void ScatterSource::far_items(const SourceKey &p_key, const Inputs &p_inputs, std::vector<MomentItem> &r_items) const {
    // One proxy per surface cell of g voxels: the sites of its patch of ground ((g / side)^2 cells' worth, one site per
    // source cell on flat ground, times the share of the cell inside the zone), each what the mean site has lost at the
    // layer turned to the up there. A point mass at its site: the hashed sites, two per cell side, already spread over the
    // cells as the sites do (spreading each over its patch as well made the blobs 1.6-2 times too wide).
    const int64_t g = proxy_side(p_key.lod);
    const CellScatter &s = set.scatter;
    // The proxy tiles' sites in the region, in cell order: the tiles walk from the regions' level (proxy_level), so this is
    // the region's own walk from it (surface_sites), and the layers sum the items in the same order. With a tile missing
    // (the CPU's point evaluation computes a key alone) that one walk is cheaper than the tiles' (each a whole tile, where
    // the region meets a sliver of most of them).
    const VoxelBox region = proxy_region(p_key);
    std::vector<SourceKey> tiles;
    proxy_tiles(region, p_key.lod, tiles);
    std::vector<CellSite> sites;
    const bool resident = p_inputs.size() == tiles.size() && std::none_of(p_inputs.begin(), p_inputs.end(), [](const SourceContent *p_in) { return !p_in; });
    if (resident) {
        for_each_input(tiles, p_inputs, [&](const SourceContent &p_walk) {
            std::copy_if(p_walk.sites.begin(), p_walk.sites.end(), std::back_inserter(sites), [&region](const CellSite &p_s) { return region.contains(p_s.site.base); });
        });
        std::sort(sites.begin(), sites.end(), [](const CellSite &p_a, const CellSite &p_b) { return cell_order(p_a.cell, p_b.cell); });
    } else if (region.valid) {
        surface_sites(region.lo, region.hi, g, s.salt ^ PROXY_SALT, s.ground, sites, false, proxy_level(p_key.lod)); // in cell order
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
        far_items(p_key, p_inputs, items);
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
