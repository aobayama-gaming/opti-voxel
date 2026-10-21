// A procedural forest as a field of keys (forest_field.hpp).
#include "forest_field.hpp"

#include "parallel_for.hpp"

#include <godot_cpp/core/error_macros.hpp>

#include <algorithm>
#include <cmath>

namespace Archetype {

namespace {

double largest_variance(const Moments &p_m) {
    double lambda[3];
    p_m.principal(lambda);
    return lambda[2];
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

// Appends the records of p_records listed at p_lod as one group of r_out (none if no record is).
void append_group(const TableRecord *p_records, uint32_t p_count, int p_lod, FieldContent &r_out) {
    const uint32_t first = uint32_t(r_out.records.size());
    VoxelBox box;
    float k = 0.0f, margin_c = 0.0f;
    int max_lod = 0;
    for (uint32_t i = 0; i < p_count; i++) {
        const TableRecord &rec = p_records[i];
        if (!rec.listed_at(p_lod)) {
            continue;
        }
        box = box.merged(record_box(rec));
        k = MAX(k, rec.k);
        margin_c = MAX(margin_c, rec.margin_c);
        max_lod = MAX(max_lod, int(rec.max_lod));
        r_out.records.push_back(rec);
    }
    if (r_out.records.size() > first) {
        r_out.groups.push_back(FieldContent::Group{ first, uint32_t(r_out.records.size()) - first, bounds_record(k, margin_c, box, max_lod) });
    }
}

} // namespace

ForestProfile::ForestProfile(const DefLibrary &p_library, const TreeSamples &p_samples, const RecordFinish &p_finish, const MomentLayerParams &p_layers,
        int p_lod_count) {
    const size_t n = p_samples.size();
    ERR_FAIL_COND_MSG(n == 0, "ForestProfile: no sample tree.");
    std::vector<std::vector<TableRecord>> parts(n);
    std::vector<std::vector<MomentItem>> items(n);
    parallel_for(uint32_t(n), "ForestProfile", [&](uint32_t p_i) {
        parts[p_i] = p_samples.parts[p_i];
        for (TableRecord &rec : parts[p_i]) {
            finish_table_record(p_library, p_finish, rec);
            MomentItem item = record_item(p_library, rec);
            prepare_moment_item(item);
            items[p_i].push_back(item);
        }
    });
    // In sample order: the same bits however the pool ran.
    const double share = 1.0 / double(n);
    lost.assign(p_lod_count, Lost());
    std::vector<MomentItem> all;
    for (size_t i = 0; i < n; i++) {
        for (const TableRecord &rec : parts[i]) {
            // A part's shape lies in its box, so within the box's farthest corner of the base (the origin) under any
            // turn of the tree: a turned tree's boxes stay within that distance on every axis.
            for (int corner = 0; corner < 8; corner++) {
                double d2 = 0.0;
                for (int a = 0; a < 3; a++) {
                    const double v = double(rec.anchor[a] + ((corner >> a) & 1 ? rec.box_hi[a] : rec.box_lo[a]));
                    d2 += v * v;
                }
                reach = MAX(reach, std::sqrt(d2));
            }
            lod_end = MAX(lod_end, rec.max_lod == TableRecord::NEVER_DROPPED ? p_lod_count : int(rec.max_lod));
            k_max = MAX(k_max, rec.k);
            margin_max = MAX(margin_max, rec.margin_c);
        }
        for (const MomentItem &item : items[i]) {
            if (item.moments.empty()) {
                continue;
            }
            part_variance = MAX(part_variance, largest_variance(item.moments));
            const double offset[3] = { double(item.anchor.x), double(item.anchor.y), double(item.anchor.z) }; // from the base
            for (int l = 0; l < p_lod_count; l++) {
                const double w = share * item.lost_share(l);
                if (!(w > 0.0)) {
                    continue;
                }
                Lost &at = lost[l];
                at.moments.add(item.material, item.moments.scaled(w), offset);
                std::array<double, 6> &area = at.area[item.material];
                for (int k = 0; k < 6; k++) {
                    area[k] += w * item.area[k];
                }
            }
            all.push_back(item);
        }
    }
    // Lengths grow with a tree's scale, and the samples' largest is short of the forest's.
    reach *= REACH_SAFETY;
    k_max = float(double(k_max) * REACH_SAFETY);
    lod_end = MIN(lod_end, p_lod_count);
    MomentLayerParams params = p_layers;
    params.lod_min = 0;
    params.lod_max = p_lod_count - 1;
    joins = moment_joins(all, params);
    for (Lost &at : lost) {
        for (const std::pair<uint32_t, Moments> &m : at.moments.parts) {
            at.variance = MAX(at.variance, largest_variance(m.second));
        }
    }
}

void FieldContent::meeting(const std::vector<Group> &p_groups, RecordView p_records, uint32_t p_first, const VoxelPos &p_lo, const VoxelPos &p_hi, int p_lod,
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

ForestField::ForestField(const Setup &p_setup) :
        set(p_setup) {
    ERR_FAIL_COND_MSG(!set.library || set.generator.side <= 0 || set.max_extent <= 0, "ForestField: no library, forest cell side or largest extent.");
    const TreeSamples samples(set.types, set.voxel);
    prof = ForestProfile(*set.library, samples, set.finish, set.layers, set.lod_count);
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
    // A base lies within a voxel of the ground (cell_site), a blob's cell within a cell of the trees' reach; the largest
    // over the LOD and the finer ones (a culled node hides its subtree).
    int64_t r = 0;
    for (int l = 0; l < set.lod_count; l++) {
        if (l < prof.lod_end) {
            r = MAX(r, tree_grow(l) + 1);
        }
        if (has_layer(l)) {
            r = MAX(r, int64_t(std::ceil(prof.reach)) + layer_cell(l) + blob_grow(l) + 1);
        }
        ground_reaches.push_back(r);
    }
}

int64_t ForestField::blob_tile_cells(int p_layer) const {
    const int64_t c = layer_cell(p_layer);
    return MAX(BLOB_TILE_CELLS, (tree_tile_side() + c - 1) / c);
}

int64_t ForestField::blob_reach(int p_layer) const {
    // A cell's members' centroids lie within a cell of its centre (their tents), so its blob's anchor does, and the
    // blob's covariance is at most a member's own plus the spread of their centroids (2 cells); its support's semi-axes
    // are 3 sigma with the layer's filter (make_blob).
    const double c = double(layer_cell(p_layer));
    const double own = part_layer(p_layer) ? prof.part_variance : prof.lost[p_layer].variance;
    const double filter = set.layers.filter * std::ldexp(1.0, p_layer);
    return int64_t(std::ceil(c + 3.0 * std::sqrt(own + 4.0 * c * c + filter * filter)));
}

void ForestField::append_tiles(int p_kind, int p_lod, int64_t p_side, const VoxelBox &p_box, int64_t p_zone_grow, std::vector<FieldKey> &r_out) const {
    const VoxelBox box = p_box.clipped(set.generator.zone.grown(p_zone_grow));
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
                r_out.push_back(FieldKey{ p_kind, p_lod, VoxelPos(x, y, z) });
            }
        }
    }
}

VoxelBox ForestField::tile_box(const FieldKey &p_key) const {
    const int64_t side = p_key.kind == FieldKey::TREES ? tree_tile_side() : blob_tile_cells(p_key.lod) * layer_cell(p_key.lod);
    return VoxelBox{ p_key.tile * side, p_key.tile * side + VoxelPos(side - 1, side - 1, side - 1), true };
}

void ForestField::keys(const VoxelPos &p_lo, const VoxelPos &p_hi, int p_lod, std::vector<FieldKey> &r_out) const {
    r_out.clear();
    if (!set.generator.zone.valid || set.generator.side <= 0) {
        return;
    }
    const VoxelBox box{ p_lo, p_hi, true };
    // Every tile whose content can pass a box test in the box: a tree tile's records lie within its box grown by the
    // reach, and its bases in the zone; a blob tile's within blob_reach, its cells within a cell of the trees' reach.
    if (p_lod < prof.lod_end) {
        append_tiles(FieldKey::TREES, p_lod, tree_tile_side(), box.grown(tree_grow(p_lod)), 0, r_out);
    }
    if (has_layer(p_lod)) {
        const int64_t c = layer_cell(p_lod);
        append_tiles(FieldKey::BLOBS, p_lod, blob_tile_cells(p_lod) * c, box.grown(blob_grow(p_lod)), int64_t(std::ceil(prof.reach)) + c, r_out);
    }
}

int64_t ForestField::tree_grow(int p_lod) const {
    return int64_t(std::ceil(prof.reach)) + influence_margin(bounds_record(prof.k_max, prof.margin_max), p_lod);
}

int64_t ForestField::blob_grow(int p_layer) const {
    return blob_reach(p_layer) + influence_margin(bounds_record(0.0f, blob_margin_c), p_layer);
}

int64_t ForestField::ground_reach(int p_lod) const {
    return ground_reaches.empty() ? 0 : ground_reaches[MIN(MAX(p_lod, 0), int(ground_reaches.size()) - 1)];
}

void ForestField::chunk_keys(const ChunkId &p_chunk, const ChunkParam &p_param, std::vector<FieldKey> &r_out) const {
    VoxelPos lo, hi;
    chunk_sample_box(p_chunk, p_param, lo, hi);
    keys(lo, hi, ChunkMath::get_lod(p_chunk), r_out);
}

void ForestField::inputs(const FieldKey &p_key, std::vector<FieldKey> &r_out) const {
    r_out.clear();
    if (p_key.kind == FieldKey::TREES) {
        if (p_key.lod > 0) {
            r_out.push_back(FieldKey{ FieldKey::TREES, 0, p_key.tile });
        }
        return;
    }
    if (part_layer(p_key.lod)) {
        // The tree tiles whose parts can have their centroid in the tile's cells or MARGIN_CELLS past them.
        const VoxelBox region = tile_box(p_key).grown(MARGIN_CELLS * layer_cell(p_key.lod));
        append_tiles(FieldKey::TREES, 0, tree_tile_side(), region.grown(int64_t(std::ceil(prof.reach))), 0, r_out);
    }
}

void ForestField::content(const FieldKey &p_key, FieldContent &r_out, const Lookup &p_inputs) const {
    r_out = FieldContent();
    if (!set.library) {
        return; // the constructor refused the setup
    }
    if (p_key.kind == FieldKey::TREES) {
        tree_tile(p_key, r_out, p_inputs);
    } else if (p_key.kind == FieldKey::BLOBS && has_layer(p_key.lod)) {
        blob_tile(p_key, r_out, p_inputs);
    }
}

void ForestField::tree_tile(const FieldKey &p_key, FieldContent &r_out, const Lookup &p_inputs) const {
    if (p_key.lod > 0) {
        const FieldKey whole{ FieldKey::TREES, 0, p_key.tile };
        FieldContent local;
        const FieldContent *all = p_inputs ? p_inputs(whole) : nullptr;
        if (!all) {
            tree_tile(whole, local, nullptr);
            all = &local;
        }
        for (const FieldContent::Group &g : all->groups) {
            append_group(all->records.data() + g.first, g.count, p_key.lod, r_out);
        }
        r_out.out_of_profile = all->out_of_profile;
        return;
    }
    std::vector<CellSite> sites; // ForestGenerator::trees, from the kept blocks (a tile computed again only grows)
    const VoxelBox box = tile_box(p_key).clipped(set.generator.zone);
    if (box.valid) {
        block_sites(box, set.generator.side, true, sites);
    }
    const DefLibrary &lib = *set.library;
    std::vector<TableRecord> parts;
    for (const CellSite &cs : sites) {
        const TreeSite &site = cs.site;
        if (set.keep_out && set.keep_out(site.base)) {
            continue;
        }
        parts.clear();
        grow_tree(tree_species()[site.species], site.base, set.up(site.base), site.seed, set.voxel, set.types, parts);
        // The table's limits (SDFTable::admit): a tree with a part it would refuse is left out whole.
        if (std::any_of(parts.begin(), parts.end(), [&](const TableRecord &p_r) { return !record_fit_error(lib[p_r.def], p_r, set.max_extent).is_empty(); })) {
            continue;
        }
        const VoxelBox reach = VoxelBox{ site.base, site.base, true }.grown(int64_t(std::floor(prof.reach)));
        bool out = false;
        for (TableRecord &rec : parts) {
            finish_table_record(lib, set.finish, rec);
            const VoxelBox b = record_box(rec);
            // keys() and blob_reach() trust the profile, read from samples: a part past it may be missing from chunks.
            out = out || !reach.contains(b.lo) || !reach.contains(b.hi) || int(rec.max_lod) > prof.lod_end || rec.k > prof.k_max ||
                    rec.margin_c > prof.margin_max;
        }
        append_group(parts.data(), uint32_t(parts.size()), 0, r_out);
        if (out) {
            r_out.out_of_profile++;
            if (!profile_reported.exchange(true)) {
                ERR_PRINT("ForestField: a tree has a part past the profile (reach, last LOD or margin): the chunks it reaches may miss it.");
            }
        }
    }
}

void ForestField::near_items(const FieldKey &p_key, const Lookup &p_inputs, std::vector<MomentItem> &r_items) const {
    const VoxelBox region = tile_box(p_key).grown(MARGIN_CELLS * layer_cell(p_key.lod));
    std::vector<FieldKey> tiles;
    inputs(p_key, tiles);
    FieldContent local;
    for (const FieldKey &t : tiles) {
        const FieldContent *trees = p_inputs ? p_inputs(t) : nullptr;
        if (!trees) {
            local = FieldContent();
            tree_tile(t, local, nullptr);
            trees = &local;
        }
        for (const FieldContent::Group &g : trees->groups) {
            // A part's centroid lies in its box: a tree whose box misses the region has none in it.
            if (!record_box(g.bounds).intersects(region)) {
                continue;
            }
            for (uint32_t i = g.first; i < g.first + g.count; i++) {
                const MomentItem item = record_item(*set.library, trees->records[i]);
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

void ForestField::far_items(const FieldKey &p_key, std::vector<MomentItem> &r_items) const {
    // One proxy per surface cell of g voxels: the trees of its patch of ground ((g / side)^2 cells' worth, one tree per
    // forest cell on flat ground, times the share of the cell inside the zone), each what the mean tree has lost at the
    // layer turned to the up there. A point mass at its site: the hashed sites, two per cell side, already spread over the
    // cells as the trees do (spreading each over its patch as well made the blobs 1.6-2 times too wide).
    const int64_t c = layer_cell(p_key.lod);
    const int64_t g = MAX<int64_t>(1, c / PROXY_PER_CELL);
    // A proxy outside the zone weighs nothing: the walk stops a cell past it.
    const VoxelBox region = tile_box(p_key).grown(MARGIN_CELLS * c + int64_t(std::ceil(prof.reach))).clipped(set.generator.zone.grown(g));
    std::vector<CellSite> sites;
    if (region.valid) {
        block_sites(region, g, false, sites); // a proxy's base needn't be on the LOD-0 ground
    }
    const double side = double(set.generator.side);
    const VoxelBox &zone = set.generator.zone;
    const ForestProfile::Lost &lost = prof.lost[p_key.lod];
    for (const CellSite &cs : sites) {
        const VoxelPos &base = cs.site.base;
        double inside = 1.0;
        for (int a = 0; a < 3; a++) {
            const int64_t lo = cs.cell[a] * g;
            inside *= double(MAX<int64_t>(0, MIN(lo + g - 1, zone.hi[a]) - MAX(lo, zone.lo[a]) + 1)) / double(g);
        }
        if (!(inside > 0.0) || (set.keep_out && set.keep_out(base))) {
            continue;
        }
        const double weight = (double(g) / side) * (double(g) / side) * inside;
        float rows[9];
        up_rotation(set.up(base), rows);
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

void ForestField::block_sites(const VoxelBox &p_box, int64_t p_side, bool p_snap, std::vector<CellSite> &r_out) const {
    const int64_t block = SITE_BLOCK_CELLS * p_side;
    VoxelPos lo, hi;
    for (int a = 0; a < 3; a++) {
        lo[a] = WorldCoord::floor_div(p_box.lo[a], block);
        hi[a] = WorldCoord::floor_div(p_box.hi[a], block);
    }
    const size_t first = r_out.size();
    for (int64_t z = lo.z; z <= hi.z; z++) {
        for (int64_t y = lo.y; y <= hi.y; y++) {
            for (int64_t x = lo.x; x <= hi.x; x++) {
                const SiteBlock key(p_side, p_snap, x, y, z);
                std::shared_ptr<const std::vector<CellSite>> sites;
                {
                    const std::lock_guard<std::mutex> lock(site_mutex);
                    const auto it = site_blocks.find(key);
                    if (it != site_blocks.end()) {
                        it->second.used = ++site_clock;
                        sites = it->second.sites;
                    }
                }
                if (!sites) { // walked outside the lock: two tiles may walk a block at once, the same sites
                    auto walked = std::make_shared<std::vector<CellSite>>();
                    const VoxelPos b = VoxelPos(x, y, z) * block;
                    surface_sites(b, b + VoxelPos(block - 1, block - 1, block - 1), p_side, set.ground, *walked, p_snap);
                    sites = walked;
                    const std::lock_guard<std::mutex> lock(site_mutex);
                    site_blocks[key] = Sites{ sites, ++site_clock };
                    if (site_blocks.size() > SITE_BLOCKS_KEPT) {
                        std::vector<std::pair<uint64_t, SiteBlock>> old;
                        for (const std::pair<const SiteBlock, Sites> &kv : site_blocks) {
                            old.push_back({ kv.second.used, kv.first });
                        }
                        std::sort(old.begin(), old.end());
                        for (size_t i = 0; i < old.size() / 2; i++) {
                            site_blocks.erase(old[i].second);
                        }
                    }
                }
                for (const CellSite &s : *sites) {
                    if (p_box.contains(s.site.base)) {
                        r_out.push_back(s);
                    }
                }
            }
        }
    }
    std::sort(r_out.begin() + first, r_out.end(), [](const CellSite &p_a, const CellSite &p_b) { return cell_order(p_a.cell, p_b.cell); });
}

void ForestField::blobs(const FieldKey &p_key, std::vector<MomentBlob> &r_out, const Lookup &p_inputs) const {
    r_out.clear();
    if (!has_layer(p_key.lod)) {
        return;
    }
    std::vector<MomentItem> items;
    if (part_layer(p_key.lod)) {
        near_items(p_key, p_inputs, items);
    } else {
        far_items(p_key, items);
    }
    // An item tents into the cells within a cell of its centroid, which lies within two reaches of its anchor (a part's
    // anchor and centroid are both in its box, within the reach of its base): none near the tile's own cells, no blob
    // (most tiles of a column: their region reaches the ground, their cells don't).
    const VoxelBox near = tile_box(p_key).grown(layer_cell(p_key.lod) + 2 * int64_t(std::ceil(prof.reach)));
    if (std::none_of(items.begin(), items.end(), [&near](const MomentItem &p_i) { return near.contains(p_i.anchor); })) {
        return;
    }
    MomentLayerParams params = set.layers;
    params.lod_min = params.lod_max = p_key.lod;
    const int64_t own = blob_tile_cells(p_key.lod);
    params.judged = VoxelBox{ p_key.tile * own, p_key.tile * own + VoxelPos(own - 1, own - 1, own - 1), true }; // its own cells
    MomentLayers layers;
    layers.build(items, params, &prof.joins);
    layers.kept(r_out);
    const int64_t n = blob_tile_cells(p_key.lod);
    r_out.erase(std::remove_if(r_out.begin(), r_out.end(),
                        [&](const MomentBlob &p_b) {
                            for (int a = 0; a < 3; a++) {
                                if (WorldCoord::floor_div(p_b.cell[a], n) != p_key.tile[a]) {
                                    return true;
                                }
                            }
                            return false;
                        }),
            r_out.end());
}

void ForestField::blob_tile(const FieldKey &p_key, FieldContent &r_out, const Lookup &p_inputs) const {
    std::vector<MomentBlob> kept;
    blobs(p_key, kept, p_inputs);
    const DefLibrary &lib = *set.library;
    for (const MomentBlob &b : kept) {
        const TableRecord rec = blob_record(lib, set.blob_type, b, set.finish.skirt_depth);
        if (!record_fit_error(lib[rec.def], rec, set.max_extent).is_empty()) {
            continue; // refused like the table's (SDFTable::blob_fits)
        }
        FieldContent::Group group;
        group.first = uint32_t(r_out.records.size());
        group.count = 1;
        group.bounds = rec;
        r_out.records.push_back(rec);
        r_out.groups.push_back(group);
    }
}

} // namespace Archetype
