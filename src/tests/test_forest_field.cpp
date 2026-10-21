#include "test_common.hpp"

#include "forest_field.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>

using namespace Archetype;

namespace {

// A 5 cm forest of about 900 trees on gentle hills far from the origin, its field built once.
struct FieldFixture {
    static constexpr double VOXEL = 0.05;
    std::shared_ptr<ForestField> field;
    VoxelPos origin = VoxelPos(int64_t(1) << 33, 1700, -(int64_t(1) << 20));

    FieldFixture() {
        auto lib = std::make_shared<DefLibrary>();
        ForestField::Setup s;
        s.types = add_tree_types(*lib, VOXEL);
        s.blob_type = lib->add(make_moment_blob_type());
        s.library = lib;
        s.voxel = VOXEL;
        s.finish = RecordFinish{ 4.0f, true, ErodeRule() };
        s.layers.min_occupancy = MOMENT_MATERIAL_ISO; // as SDFTable::add_moment_blobs
        s.lod_count = 15;
        s.max_extent = int64_t(1) << 29; // SDFTable::max_instance_extent() at 15 LODs and up
        s.generator.side = 100;
        s.generator.zone = VoxelBox{ origin - VoxelPos(2250, 400, 2250), origin + VoxelPos(2249, 400, 2249), true };
        const VoxelPos o = origin;
        s.ground = [o](const VoxelPos &p_v) {
            const double tau = 6.283185307179586, amp = 60.0, wave = 2400.0;
            const double x = double(p_v.x - o.x), z = double(p_v.z - o.z);
            const double w = double(p_v.y - o.y) - amp * std::sin(tau * x / wave) * std::cos(tau * z / wave);
            return ExactNoise::Float4{ float(-amp * tau / wave * std::cos(tau * x / wave) * std::cos(tau * z / wave)), 1.0f,
                float(amp * tau / wave * std::sin(tau * x / wave) * std::sin(tau * z / wave)), float(w) };
        };
        s.up = [](const VoxelPos &) { return UP_Y; };
        field = std::make_shared<ForestField>(s);
    }
    static const FieldFixture &get() {
        static const FieldFixture fixture;
        return fixture;
    }
    const ForestField &f() const { return *field; }
    VoxelBox zone() const { return f().setup().generator.zone; }

    // Every key of a kind (layer) the zone's box has at p_lod, with its content (the tree tiles from the cache: a blob
    // tile reads them).
    std::vector<std::pair<FieldKey, FieldContent>> tiles(int p_kind, int p_lod) const {
        std::vector<FieldKey> keys;
        f().keys(zone().lo, zone().hi, p_lod, keys);
        std::vector<std::pair<FieldKey, FieldContent>> out;
        for (const FieldKey &k : keys) {
            if (k.kind == p_kind) {
                out.push_back({ k, FieldContent() });
                if (k.kind == FieldKey::TREES) {
                    out.back().second = *tree_tile(k);
                } else {
                    f().content(k, out.back().second, [this](const FieldKey &p_k) { return tree_tile(p_k); });
                }
            }
        }
        return out;
    }

    // Any key's content (its tree tiles from the cache).
    FieldContent content(const FieldKey &p_key) const {
        if (p_key.kind == FieldKey::TREES) {
            return *tree_tile(p_key);
        }
        FieldContent c;
        f().content(p_key, c, [this](const FieldKey &p_k) { return tree_tile(p_k); });
        return c;
    }

    // The items of every part of the zone's trees, in key order.
    std::vector<MomentItem> part_items() const {
        std::vector<MomentItem> items;
        for (const std::pair<FieldKey, FieldContent> &t : tiles(FieldKey::TREES, 0)) {
            for (const TableRecord &r : t.second.records) {
                items.push_back(record_item(*f().setup().library, r));
            }
        }
        return items;
    }

    // Layer p_layer of one MomentLayers build over p_items, with the field's joins: the reference.
    std::vector<MomentBlob> whole_build(const std::vector<MomentItem> &p_items, int p_layer) const {
        MomentLayerParams params = f().setup().layers;
        params.lod_min = params.lod_max = p_layer;
        MomentLayers layers;
        layers.build(p_items, params, &f().profile().joins);
        std::vector<MomentBlob> out;
        layers.kept(out);
        return out;
    }

    // The kept blobs of every blob tile of layer p_layer over the zone.
    std::vector<MomentBlob> layer_blobs(int p_layer) const {
        std::vector<FieldKey> keys;
        f().keys(zone().lo, zone().hi, p_layer, keys);
        std::vector<MomentBlob> out, one;
        for (const FieldKey &k : keys) {
            if (k.kind == FieldKey::BLOBS) {
                f().blobs(k, one, [this](const FieldKey &p_k) { return tree_tile(p_k); });
                out.insert(out.end(), one.begin(), one.end());
            }
        }
        return out;
    }

private:
    mutable std::map<std::tuple<int, int64_t, int64_t, int64_t>, FieldContent> trees;
    const FieldContent *tree_tile(const FieldKey &p_key) const {
        const auto at = std::make_tuple(p_key.lod, p_key.tile.x, p_key.tile.y, p_key.tile.z);
        auto it = trees.find(at);
        if (it == trees.end()) {
            it = trees.emplace(at, FieldContent()).first;
            f().content(p_key, it->second, [this](const FieldKey &p_k) { return tree_tile(p_k); }); // map nodes stay put
        }
        return &it->second;
    }
};

bool same_record(const TableRecord &p_a, const TableRecord &p_b) {
    return p_a.def == p_b.def && p_a.anchor == p_b.anchor && std::equal(p_a.params, p_a.params + 8, p_b.params) &&
            std::equal(p_a.rotation, p_a.rotation + 9, p_b.rotation) && p_a.max_lod == p_b.max_lod && p_a.min_lod == p_b.min_lod &&
            p_a.material == p_b.material && p_a.erode == p_b.erode && p_a.erode_from == p_b.erode_from;
}

using Cell = std::tuple<int64_t, int64_t, int64_t, uint32_t>;
Cell cell_of(const MomentBlob &p_b) {
    return Cell(p_b.cell[0], p_b.cell[1], p_b.cell[2], p_b.material);
}

} // namespace

TEST_CASE("Forest field: the profile reads the trees' reach, LODs and joins") {
    const ForestProfile &p = FieldFixture::get().f().profile();
    MESSAGE("reach ", p.reach, " voxels, lod_end ", p.lod_end, ", k ", p.k_max, ", margin ", p.margin_max, ", part sigma ", std::sqrt(p.part_variance),
            ", joins ", int64_t(p.joins.size()));
    for (const std::pair<const uint32_t, MomentJoin> &j : p.joins) {
        MESSAGE("material ", j.first, " joins at LOD ", j.second.lod, ", cover from ", j.second.cover_from);
    }
    CHECK(p.reach > 100.0);
    CHECK(p.lod_end > 2);
    CHECK(p.lod_end < 12);
    CHECK(p.joins.size() == 2); // bark and leaves
    CHECK(p.lost[p.lod_end].moments.parts.size() == 2); // every part lost there
}

// The tree tiles partition the forest: each tree once, its records those a whole placement gives.
TEST_CASE("Forest field: the tree tiles hold the zone's trees, each once") {
    const FieldFixture &fx = FieldFixture::get();
    std::vector<CellSite> sites;
    fx.f().setup().generator.trees(fx.zone(), fx.f().setup().ground, sites);
    std::map<std::tuple<int64_t, int64_t, int64_t>, std::vector<TableRecord>> whole;
    for (const CellSite &cs : sites) {
        std::vector<TableRecord> &parts = whole[std::make_tuple(cs.site.base.x, cs.site.base.y, cs.site.base.z)];
        grow_tree(tree_species()[cs.site.species], cs.site.base, UP_Y, cs.site.seed, FieldFixture::VOXEL, fx.f().setup().types, parts);
        for (TableRecord &r : parts) {
            finish_table_record(*fx.f().setup().library, fx.f().setup().finish, r);
        }
    }
    int64_t trees = 0, matched = 0, out_of_profile = 0;
    for (const std::pair<FieldKey, FieldContent> &t : fx.tiles(FieldKey::TREES, 0)) {
        out_of_profile += t.second.out_of_profile;
        for (const FieldContent::Group &g : t.second.groups) {
            trees++;
            const TableRecord &first = t.second.records[g.first];
            for (const std::pair<const std::tuple<int64_t, int64_t, int64_t>, std::vector<TableRecord>> &w : whole) {
                if (w.second.size() == g.count && same_record(w.second[0], first)) {
                    bool all = true;
                    for (uint32_t i = 0; i < g.count; i++) {
                        all = all && same_record(w.second[i], t.second.records[g.first + i]);
                    }
                    matched += all;
                    break;
                }
            }
        }
    }
    MESSAGE(trees, " trees in the tiles, ", int64_t(whole.size()), " placed whole, ", matched, " identical");
    CHECK(whole.size() > 500);
    CHECK(trees == int64_t(whole.size()));
    CHECK(matched == trees);
    CHECK(out_of_profile == 0);
    // A coarser LOD's tiles hold the parts listed there: a few per tree past the twigs' drops.
    for (int lod = 1; lod < fx.f().profile().lod_end; lod++) {
        int64_t listed = 0, held = 0;
        for (const std::pair<FieldKey, FieldContent> &t : fx.tiles(FieldKey::TREES, 0)) {
            listed += std::count_if(t.second.records.begin(), t.second.records.end(), [lod](const TableRecord &p_r) { return p_r.listed_at(lod); });
        }
        for (const std::pair<FieldKey, FieldContent> &t : fx.tiles(FieldKey::TREES, lod)) {
            held += int64_t(t.second.records.size());
        }
        MESSAGE("LOD ", lod, ": ", double(held) / double(trees), " parts per tree");
        CHECK(held == listed);
    }
}
// A part layer's blob tile is what a whole build of the zone's trees gives for its cells: its items around it are enough.
TEST_CASE("Forest field: the part layers' blob tiles are a whole build's blobs") {
    const FieldFixture &fx = FieldFixture::get();
    const ForestField &f = fx.f();
    const std::vector<MomentItem> items = fx.part_items();
    int layers_checked = 0;
    for (int layer = 0; layer < f.setup().lod_count; layer++) {
        if (!f.has_layer(layer) || !f.part_layer(layer)) {
            continue;
        }
        layers_checked++;
        std::map<Cell, MomentBlob> whole;
        for (const MomentBlob &b : fx.whole_build(items, layer)) {
            whole[cell_of(b)] = b;
        }
        int64_t got = 0, missing = 0, differ = 0;
        double worst = 0.0;
        for (const MomentBlob &b : fx.layer_blobs(layer)) {
            got++;
            const auto it = whole.find(cell_of(b));
            missing += it == whole.end();
            if (it == whole.end()) {
                continue;
            }
            const MomentBlob &w = it->second;
            for (int i = 0; i < 3; i++) {
                worst = MAX(worst, std::fabs(double(w.inv_s2[i]) / double(b.inv_s2[i]) - 1.0));
            }
            worst = MAX(worst, std::fabs(double(w.amplitude) / double(b.amplitude) - 1.0));
            differ += w.anchor != b.anchor || std::fabs(w.mass / b.mass - 1.0) > 1e-9;
        }
        MESSAGE("layer ", layer, ": ", got, " blobs from the tiles, ", int64_t(whole.size()), " from a whole build, ", missing, " not in it, ", differ,
                " off (anchor, mass), worst param ratio - 1: ", worst);
        CHECK(got == int64_t(whole.size()));
        CHECK(missing == 0);
        CHECK(differ == 0);
        CHECK(worst < 1e-5); // the sums in another order, rounded to float
    }
    CHECK(layers_checked >= 2);
}

// A proxy layer's tiles stand for the trees with proxies: the mass its kept blobs hold is the trees' within a few
// percent, and its canopy reads about the same cover.
TEST_CASE("Forest field: the proxy layers weigh what the trees do") {
    const FieldFixture &fx = FieldFixture::get();
    const ForestField &f = fx.f();
    const std::vector<MomentItem> items = fx.part_items();
    int layers_checked = 0;
    for (int layer = 0; layer < f.setup().lod_count; layer++) {
        // Layers with a few cells across the zone (a coarser one is all edge).
        if (!f.has_layer(layer) || f.part_layer(layer) || 4 * f.layer_cell(layer) > fx.zone().hi.x - fx.zone().lo.x) {
            continue;
        }
        layers_checked++;
        double truth[2] = {}, proxies[2] = {}, truth_amp = 0.0, proxy_amp = 0.0;
        const std::vector<MomentBlob> whole = fx.whole_build(items, layer);
        for (const MomentBlob &b : whole) {
            truth[b.material == Material::LEAVES] += b.mass;
            truth_amp += double(b.amplitude) * b.mass;
        }
        const std::vector<MomentBlob> tiles = fx.layer_blobs(layer);
        for (const MomentBlob &b : tiles) {
            proxies[b.material == Material::LEAVES] += b.mass;
            proxy_amp += double(b.amplitude) * b.mass;
        }
        // The mass-weighted mean peak: what the canopy reads where the mass is.
        truth_amp /= truth[0] + truth[1];
        proxy_amp /= proxies[0] + proxies[1];
        MESSAGE("layer ", layer, ": mass proxies / trees bark ", proxies[0] / truth[0], " leaves ", proxies[1] / truth[1], "; ", int64_t(tiles.size()),
                " blobs, ", int64_t(whole.size()), " from the trees; mean peak ", proxy_amp, " vs ", truth_amp);
        // The cells both keep: mass, support volume and peak, mean ratio per material.
        std::map<Cell, MomentBlob> by_cell;
        for (const MomentBlob &b : whole) {
            by_cell[cell_of(b)] = b;
        }
        double ratio[2][3] = {};
        int64_t pairs[2] = {};
        for (const MomentBlob &b : tiles) {
            const auto it = by_cell.find(cell_of(b));
            if (it == by_cell.end()) {
                continue;
            }
            const int m = b.material == Material::LEAVES;
            pairs[m]++;
            ratio[m][0] += b.mass / it->second.mass;
            ratio[m][1] += (b.s[0] * b.s[1] * b.s[2]) / (it->second.s[0] * it->second.s[1] * it->second.s[2]);
            ratio[m][2] += double(b.amplitude) / double(it->second.amplitude);
        }
        for (int m = 0; m < 2; m++) {
            MESSAGE("  ", doctest::String(m ? "leaves" : "bark"), ": ", pairs[m], " cells in both, mean ratio mass ", ratio[m][0] / double(MAX(pairs[m], int64_t(1))),
                    " support volume ", ratio[m][1] / double(MAX(pairs[m], int64_t(1))), " peak ", ratio[m][2] / double(MAX(pairs[m], int64_t(1))));
        }
        CHECK(std::fabs(proxies[0] / truth[0] - 1.0) < 0.1);
        CHECK(std::fabs(proxies[1] / truth[1] - 1.0) < 0.1);
        CHECK(std::fabs(proxy_amp / truth_amp - 1.0) < 0.25);
    }
    CHECK(layers_checked >= 2);
}

// A key a box needs is any key with a record whose box test passes in it, at every LOD.
TEST_CASE("Forest field: a box's keys hold every record that can reach it") {
    const FieldFixture &fx = FieldFixture::get();
    const ForestField &f = fx.f();
    for (int lod = 0; lod < f.profile().lod_end + 2; lod++) {
        // Every tile of both kinds over the zone grown by the trees' reach and two blob cells, enumerated here and not by
        // keys(): a tile keys() wrongly leaves out is still a candidate.
        std::vector<std::pair<FieldKey, FieldContent>> all;
        const VoxelBox around = fx.zone().grown(int64_t(f.profile().reach) + 2 * f.layer_cell(lod));
        for (int kind = FieldKey::TREES; kind <= FieldKey::BLOBS; kind++) {
            const int64_t side = kind == FieldKey::TREES ? f.tree_tile_side() : f.blob_tile_cells(lod) * f.layer_cell(lod);
            for (int64_t z = WorldCoord::floor_div(around.lo.z, side); z <= WorldCoord::floor_div(around.hi.z, side); z++) {
                for (int64_t y = WorldCoord::floor_div(around.lo.y, side); y <= WorldCoord::floor_div(around.hi.y, side); y++) {
                    for (int64_t x = WorldCoord::floor_div(around.lo.x, side); x <= WorldCoord::floor_div(around.hi.x, side); x++) {
                        const FieldKey k{ kind, lod, VoxelPos(x, y, z) };
                        all.push_back({ k, fx.content(k) });
                    }
                }
            }
        }
        int64_t needed = 0, missed = 0;
        for (int i = 0; i < 24; i++) {
            const ExactNoise::UVec3 h = ExactNoise::hash3(uint32_t(i), uint32_t(lod), 0xB0Bu);
            const int64_t span = int64_t(28) << lod;
            const VoxelPos lo = fx.zone().lo + VoxelPos(int64_t(h.x % 3400u) - 200, int64_t(h.y % 1000u) - 200, int64_t(h.z % 3400u) - 200);
            const VoxelPos hi = lo + VoxelPos(span, span, span);
            std::vector<FieldKey> keys;
            f.keys(lo, hi, lod, keys);
            for (const std::pair<FieldKey, FieldContent> &t : all) {
                std::vector<uint32_t> hit;
                t.second.meeting(lo, hi, lod, hit);
                if (!hit.empty()) {
                    needed++;
                    missed += !std::binary_search(keys.begin(), keys.end(), t.first);
                }
            }
        }
        MESSAGE("LOD ", lod, ": ", needed, " keys reached a box, ", missed, " missed by keys()");
        CHECK(missed == 0);
    }
}
