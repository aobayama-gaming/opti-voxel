#include "test_common.hpp"

#include "scatter_source.hpp"
#include "tree_archetype.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>

using namespace Archetype;

namespace {

// A 5 cm forest of about 2,000 trees on gentle hills far from the origin, its source built once.
struct SourceFixture {
    static constexpr double VOXEL = 0.05;
    std::shared_ptr<ScatterSource> source;
    VoxelPos origin = VoxelPos(int64_t(1) << 33, 1700, -(int64_t(1) << 20));

    SourceFixture() {
        auto lib = std::make_shared<DefLibrary>();
        const TreeTypes types = add_tree_types(*lib, VOXEL);
        ScatterSource::Setup s;
        s.id = 3;
        s.blob_type = lib->add(make_moment_blob_type());
        s.library = lib;
        s.finish = RecordFinish{ 4.0f, true, ErodeRule() };
        s.layers.min_occupancy = MOMENT_MATERIAL_ISO; // as SDFTable::moment_params
        s.lod_count = 15;
        s.max_extent = int64_t(1) << 29; // SDFTable::max_instance_extent() at 15 LODs and up
        CellScatter &c = s.scatter;
        c.salt = TREE_SITE_SALT;
        c.side = 100;
        c.zone = VoxelBox{ origin - VoxelPos(2250, 400, 2250), origin + VoxelPos(2249, 400, 2249), true };
        const VoxelPos o = origin;
        c.ground = [o](const VoxelPos &p_v, int) {
            const double tau = 6.283185307179586, amp = 60.0, wave = 2400.0;
            const double x = double(p_v.x - o.x), z = double(p_v.z - o.z);
            const double w = double(p_v.y - o.y) - amp * std::sin(tau * x / wave) * std::cos(tau * z / wave);
            return ExactNoise::Float4{ float(-amp * tau / wave * std::cos(tau * x / wave) * std::cos(tau * z / wave)), 1.0f,
                float(amp * tau / wave * std::sin(tau * x / wave) * std::sin(tau * z / wave)), float(w) };
        };
        c.up = [](const VoxelPos &) { return UP_Y; };
        c.generate = [types](const Site &p_site, const UpAxis &p_up, std::vector<TableRecord> &r_out) { // SDFTable::forest_source's
            grow_tree(tree_species()[tree_species_of(p_site.variant)], p_site.base, p_up, p_site.seed, VOXEL, types, r_out);
        };
        source = std::make_shared<ScatterSource>(s);
    }
    static const SourceFixture &get() {
        static const SourceFixture fixture;
        return fixture;
    }
    const ScatterSource &f() const { return *source; }
    VoxelBox zone() const { return f().setup().scatter.zone; }

    // Any key's content, its inputs from the cache (as the stream hands them).
    const SourceContent &content(const SourceKey &p_key) const {
        auto it = cache.find(p_key);
        if (it == cache.end()) {
            std::vector<SourceKey> in;
            f().inputs(p_key, in);
            ScatterSource::Inputs inputs;
            for (const SourceKey &k : in) {
                inputs.push_back(&content(k)); // map nodes stay put
            }
            SourceContent c;
            f().content(p_key, inputs, c);
            it = cache.emplace(p_key, std::move(c)).first;
        }
        return it->second;
    }

    // Every key of a kind the zone's box has at p_lod.
    std::vector<SourceKey> zone_keys(int p_kind, int p_lod) const {
        std::vector<SourceKey> keys, out;
        f().keys(zone().lo, zone().hi, p_lod, keys);
        std::copy_if(keys.begin(), keys.end(), std::back_inserter(out), [p_kind](const SourceKey &p_k) { return p_k.kind == p_kind; });
        return out;
    }

    // The items of every record of the zone's sites, in key order.
    std::vector<MomentItem> part_items() const {
        std::vector<MomentItem> items;
        for (const SourceKey &k : zone_keys(SourceKey::PARTS, 0)) {
            for (const TableRecord &r : content(k).records) {
                items.push_back(record_item(*f().setup().library, r));
            }
        }
        return items;
    }

    // Layer p_layer of one MomentLayers build over p_items, with the source's joins: the reference.
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
        std::vector<MomentBlob> out, one;
        for (const SourceKey &k : zone_keys(SourceKey::BLOBS, p_layer)) {
            std::vector<SourceKey> in;
            f().inputs(k, in);
            ScatterSource::Inputs inputs;
            for (const SourceKey &i : in) {
                inputs.push_back(&content(i));
            }
            f().blobs(k, inputs, one);
            out.insert(out.end(), one.begin(), one.end());
        }
        return out;
    }

private:
    mutable std::map<SourceKey, SourceContent> cache;
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

// The LOD-0 part tiles partition the source: each site once, its records those a whole placement gives; a coarser LOD's
// tiles (the pyramid) hold exactly the records listed there.
TEST_CASE("Scatter source: the part tiles hold the zone's sites, each once, and the pyramid what each LOD lists") {
    const SourceFixture &fx = SourceFixture::get();
    const ScatterSource &f = fx.f();
    std::vector<CellSite> sites;
    f.setup().scatter.sites(fx.zone(), sites);
    std::vector<std::vector<TableRecord>> whole(sites.size());
    for (size_t i = 0; i < sites.size(); i++) {
        f.setup().scatter.grow(sites[i].site, whole[i]);
        for (TableRecord &r : whole[i]) {
            finish_table_record(*f.setup().library, f.setup().finish, r);
        }
    }
    int64_t groups = 0, matched = 0, out_of_profile = 0, records = 0;
    for (const SourceKey &k : fx.zone_keys(SourceKey::PARTS, 0)) {
        CHECK(k.source == 3);
        const SourceContent &c = fx.content(k);
        out_of_profile += c.out_of_profile;
        records += int64_t(c.records.size());
        for (const SourceContent::Group &g : c.groups) {
            groups++;
            const TableRecord &first = c.records[g.first];
            for (const std::vector<TableRecord> &w : whole) {
                if (w.size() == g.count && same_record(w[0], first)) {
                    bool all = true;
                    for (uint32_t i = 0; i < g.count; i++) {
                        all = all && same_record(w[i], c.records[g.first + i]);
                    }
                    matched += all;
                    break;
                }
            }
        }
    }
    MESSAGE(groups, " sites in the tiles, ", int64_t(whole.size()), " placed whole, ", matched, " identical, ", records, " records");
    CHECK(whole.size() > 1000);
    CHECK(groups == int64_t(whole.size()));
    CHECK(matched == groups);
    CHECK(out_of_profile == 0);
    for (int lod = 1; lod < f.profile().lod_end; lod++) {
        CAPTURE(lod);
        const int64_t ratio = f.part_tile_side(lod) / f.part_tile_side(lod - 1);
        CHECK((ratio == 1 || ratio == 2));
        CHECK(f.part_tile_side(lod) % f.part_tile_side(lod - 1) == 0);
        CHECK(f.part_tile_side(lod) >= 2 * (f.setup().chunk_voxels << lod));
        int64_t listed = 0, held = 0;
        for (const std::vector<TableRecord> &w : whole) {
            listed += std::count_if(w.begin(), w.end(), [lod](const TableRecord &p_r) { return p_r.listed_at(lod); });
        }
        for (const SourceKey &k : fx.zone_keys(SourceKey::PARTS, lod)) {
            std::vector<SourceKey> in;
            f.inputs(k, in);
            CHECK(int64_t(in.size()) == ratio * ratio * ratio);
            const SourceContent &c = fx.content(k);
            held += int64_t(c.records.size());
            CHECK(std::all_of(c.records.begin(), c.records.end(), [lod](const TableRecord &p_r) { return p_r.listed_at(lod); }));
            // Computed without its inputs: the same records.
            if (lod == 1 && !c.records.empty()) {
                SourceContent alone;
                f.content(k, ScatterSource::Inputs(), alone);
                CHECK(alone.records.size() == c.records.size());
                for (size_t i = 0; i < MIN(alone.records.size(), c.records.size()); i++) {
                    CHECK(same_record(alone.records[i], c.records[i]));
                }
            }
        }
        MESSAGE("LOD ", lod, ": tile ", f.part_tile_side(lod), " voxels, ", double(held) / double(groups), " records per site");
        CHECK(held == listed);
    }
}

// A part layer's blob tile is what a whole build of the zone's records gives for its cells: the items within
// MARGIN_CELLS of it are enough.
TEST_CASE("Scatter source: the part layers' blob tiles are a whole build's blobs") {
    const SourceFixture &fx = SourceFixture::get();
    const ScatterSource &f = fx.f();
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
        MESSAGE("layer ", layer, ": ", got, " blobs from the tiles (", f.blob_tile_cells(layer), " cells a side), ", int64_t(whole.size()),
                " from a whole build, ", missing, " not in it, ", differ, " off (anchor, mass), worst param ratio - 1: ", worst);
        CHECK(got == int64_t(whole.size()));
        CHECK(missing == 0);
        CHECK(differ == 0);
        CHECK(worst < 1e-5); // the sums in another order, rounded to float
    }
    CHECK(layers_checked >= 2);
}

// A proxy layer's tiles stand for the sites with proxies: the mass its kept blobs hold is the records' within a few
// percent, and its canopy reads about the same cover.
TEST_CASE("Scatter source: the proxy layers weigh what the records do") {
    const SourceFixture &fx = SourceFixture::get();
    const ScatterSource &f = fx.f();
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
        MESSAGE("layer ", layer, ": mass proxies / records bark ", proxies[0] / truth[0], " leaves ", proxies[1] / truth[1], "; ", int64_t(tiles.size()),
                " blobs, ", int64_t(whole.size()), " from the records; mean peak ", proxy_amp, " vs ", truth_amp);
        CHECK(std::fabs(proxies[0] / truth[0] - 1.0) < 0.1);
        CHECK(std::fabs(proxies[1] / truth[1] - 1.0) < 0.1);
        CHECK(std::fabs(proxy_amp / truth_amp - 1.0) < 0.25);
    }
    CHECK(layers_checked >= 2);
}

// A key a box needs is any key with a record whose box test passes in it, at every LOD.
TEST_CASE("Scatter source: a box's keys hold every record that can reach it") {
    const SourceFixture &fx = SourceFixture::get();
    const ScatterSource &f = fx.f();
    for (int lod = 0; lod < f.profile().lod_end + 2; lod++) {
        CAPTURE(lod);
        // Every tile of both kinds over the zone grown by the records' reach and two blob cells, enumerated here and not by
        // keys(): a tile keys() wrongly leaves out is still a candidate.
        std::vector<SourceKey> all;
        const VoxelBox around = fx.zone().grown(int64_t(f.profile().reach) + 2 * f.layer_cell(lod));
        for (int kind = SourceKey::PARTS; kind <= SourceKey::BLOBS; kind++) {
            const int64_t side = kind == SourceKey::PARTS ? f.part_tile_side(lod) : f.blob_tile_cells(lod) * f.layer_cell(lod);
            for (int64_t z = WorldCoord::floor_div(around.lo.z, side); z <= WorldCoord::floor_div(around.hi.z, side); z++) {
                for (int64_t y = WorldCoord::floor_div(around.lo.y, side); y <= WorldCoord::floor_div(around.hi.y, side); y++) {
                    for (int64_t x = WorldCoord::floor_div(around.lo.x, side); x <= WorldCoord::floor_div(around.hi.x, side); x++) {
                        all.push_back(SourceKey{ 3, kind, lod, VoxelPos(x, y, z) });
                    }
                }
            }
        }
        int64_t needed = 0, missed = 0, asked = 0;
        for (int i = 0; i < 24; i++) {
            const ExactNoise::UVec3 h = ExactNoise::hash3(uint32_t(i), uint32_t(lod), 0xB0Bu);
            const int64_t span = f.setup().chunk_voxels << lod;
            const VoxelPos lo = fx.zone().lo + VoxelPos(int64_t(h.x % 4700u) - 200, int64_t(h.y % 1000u) - 200, int64_t(h.z % 4700u) - 200);
            const VoxelPos hi = lo + VoxelPos(span, span, span);
            std::vector<SourceKey> keys;
            f.keys(lo, hi, lod, keys);
            CHECK(std::is_sorted(keys.begin(), keys.end()));
            asked += int64_t(keys.size());
            for (const SourceKey &k : all) {
                std::vector<uint32_t> hit;
                fx.content(k).meeting(lo, hi, lod, hit);
                if (!hit.empty()) {
                    needed++;
                    missed += !std::binary_search(keys.begin(), keys.end(), k);
                }
            }
        }
        MESSAGE("LOD ", lod, ": ", asked, " keys asked, ", needed, " reached a box, ", missed, " missed by keys()");
        CHECK(missed == 0);
    }
}
