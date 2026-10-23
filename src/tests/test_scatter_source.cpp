#include "test_stream_common.hpp"

#include "instance_stream.hpp"
#include "scatter_source.hpp"
#include "tree_archetype.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <tuple>

using namespace Archetype;
using StreamTest::Mirror;
using StreamTest::same_bits;

namespace {

// Any key's content, its inputs from the cache (as the stream hands them).
class KeyCache {
public:
    explicit KeyCache(const ScatterSource *p_source = nullptr) :
            source(p_source) {}
    const SourceContent &content(const SourceKey &p_key) {
        auto it = cache.find(p_key);
        if (it == cache.end()) {
            SourceContent c;
            source->content(p_key, inputs(p_key), c);
            it = cache.emplace(p_key, std::move(c)).first;
        }
        return it->second;
    }
    // The contents of the key's inputs, in inputs() order.
    ScatterSource::Inputs inputs(const SourceKey &p_key) {
        std::vector<SourceKey> in;
        source->inputs(p_key, in);
        ScatterSource::Inputs out;
        for (const SourceKey &k : in) {
            out.push_back(&content(k)); // map nodes stay put
        }
        return out;
    }

private:
    const ScatterSource *source;
    std::map<SourceKey, SourceContent> cache;
};

// A 5 cm forest of about 2,000 trees on gentle hills far from the origin, its source built once.
struct SourceFixture {
    static constexpr double VOXEL = 0.05;
    std::shared_ptr<ScatterSource> source;
    VoxelPos origin = VoxelPos(int64_t(1) << 33, 1700, -(int64_t(1) << 20));

    // p_species >= 0: every site of that species (one class, its own profile), the mixed forest otherwise.
    explicit SourceFixture(int64_t p_part_cell_sides = ScatterSource::PART_CELL_SIDES, int p_species = -1) {
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
        s.part_cell_sides = p_part_cell_sides;
        CellScatter &c = s.scatter;
        set_forest_scatter(c, VOXEL, *lib, s.finish, types, s.max_extent); // SDFTable::forest_source's
        if (p_species >= 0) {
            const std::function<void(const Site &, const UpAxis &, int, std::vector<TableRecord> &)> mixed = c.generate;
            c.generate = [mixed, p_species](const Site &p_site, const UpAxis &p_up, int p_lod, std::vector<TableRecord> &r_out) {
                Site one = p_site;
                one.variant = uint32_t(p_species); // tree_species_of(p_species) == p_species
                mixed(one, p_up, p_lod, r_out);
            };
            c.classes = 1;
            c.profile_key.clear();
        }
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
        source = std::make_shared<ScatterSource>(s);
        cache = KeyCache(source.get());
    }
    static const SourceFixture &get() {
        static const SourceFixture fixture;
        return fixture;
    }
    // The streamed sources' setting: proxies from the source cell on.
    static const SourceFixture &streamed() {
        static const SourceFixture fixture(ScatterSource::STREAM_PART_CELL_SIDES);
        return fixture;
    }
    // The streamed setting on a one-species forest.
    static const SourceFixture &one_species(int p_species) {
        static const SourceFixture fixtures[3] = { SourceFixture(ScatterSource::STREAM_PART_CELL_SIDES, 0), SourceFixture(ScatterSource::STREAM_PART_CELL_SIDES, 1),
            SourceFixture(ScatterSource::STREAM_PART_CELL_SIDES, 2) };
        return fixtures[p_species];
    }
    const ScatterSource &f() const { return *source; }
    VoxelBox zone() const { return f().setup().scatter.zone; }

    const SourceContent &content(const SourceKey &p_key) const { return cache.content(p_key); }

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
            f().blobs(k, cache.inputs(k), one);
            out.insert(out.end(), one.begin(), one.end());
        }
        return out;
    }

private:
    mutable KeyCache cache;
};

// The tiles of p_side voxels meeting p_box, in cell order.
std::vector<VoxelPos> tiles_over(const VoxelBox &p_box, int64_t p_side) {
    std::vector<VoxelPos> out;
    for (int64_t z = WorldCoord::floor_div(p_box.lo.z, p_side); z <= WorldCoord::floor_div(p_box.hi.z, p_side); z++) {
        for (int64_t y = WorldCoord::floor_div(p_box.lo.y, p_side); y <= WorldCoord::floor_div(p_box.hi.y, p_side); y++) {
            for (int64_t x = WorldCoord::floor_div(p_box.lo.x, p_side); x <= WorldCoord::floor_div(p_box.hi.x, p_side); x++) {
                out.push_back(VoxelPos(x, y, z));
            }
        }
    }
    return out;
}

// The LOD-0 tiles inside part tile p_tile of LOD p_lod in the record pyramid's order: its 1 or 8 L - 1 tiles, z y x,
// recursively (the zone left out).
void pyramid_tiles(const ScatterSource &p_f, int p_lod, const VoxelPos &p_tile, std::vector<VoxelPos> &r_out) {
    if (p_lod == 0) {
        r_out.push_back(p_tile);
        return;
    }
    const int64_t n = p_f.part_tile_side(p_lod) / p_f.part_tile_side(p_lod - 1);
    for (int64_t z = 0; z < n; z++) {
        for (int64_t y = 0; y < n; y++) {
            for (int64_t x = 0; x < n; x++) {
                pyramid_tiles(p_f, p_lod - 1, p_tile * n + VoxelPos(x, y, z), r_out);
            }
        }
    }
}

using Cell = std::tuple<int64_t, int64_t, int64_t, uint32_t>;
Cell cell_of(const MomentBlob &p_b) {
    return Cell(p_b.cell[0], p_b.cell[1], p_b.cell[2], p_b.material);
}

// The same sites, cell by cell, in the same order.
bool same_sites(const std::vector<CellSite> &p_a, const std::vector<CellSite> &p_b) {
    bool equal = p_a.size() == p_b.size();
    for (size_t i = 0; equal && i < p_a.size(); i++) {
        const CellSite &a = p_a[i], &b = p_b[i];
        equal = a.cell == b.cell && a.site.base == b.site.base && a.site.seed == b.site.seed && a.site.variant == b.site.variant;
    }
    return equal;
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
        f.setup().scatter.grow(sites[i].site, -1, whole[i]);
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
                if (w.size() == g.count && same_bits(w[0], first)) {
                    bool all = true;
                    for (uint32_t i = 0; i < g.count; i++) {
                        all = all && same_bits(w[i], c.records[g.first + i]);
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
        // A LOD with the LOD-0 side reads the LOD-0 keys (all records: the chunks filter), a coarser one has its own.
        const int key_lod = f.part_key_lod(lod);
        CHECK(f.part_tile_side(key_lod) == f.part_tile_side(lod));
        CHECK(key_lod == (f.part_tile_side(lod) == f.part_tile_side(0) ? 0 : lod));
        int64_t listed = 0, held = 0, inputs = 0, cube = 0, walked = 0;
        for (const std::vector<TableRecord> &w : whole) {
            listed += std::count_if(w.begin(), w.end(), [lod](const TableRecord &p_r) { return p_r.listed_at(lod); });
        }
        for (const SourceKey &k : fx.zone_keys(SourceKey::PARTS, lod)) {
            CHECK(k.lod == key_lod);
            std::vector<SourceKey> in;
            f.inputs(k, in);
            // The LOD-0 site tiles of its cube meeting the zone, in the pyramid's order: a LOD-0 part tile reads its own, a
            // coarser one none, its one walk giving the union of theirs (a walk is a pure function of the cell), in cell order.
            std::vector<VoxelPos> all;
            pyramid_tiles(f, k.lod, k.tile, all);
            std::vector<SourceKey> site_tiles;
            std::vector<CellSite> from_tiles;
            for (const VoxelPos &t : all) {
                const SourceKey site{ 3, SourceKey::SITES, 0, t };
                if (f.tile_box(site).intersects(fx.zone())) {
                    site_tiles.push_back(site);
                    const std::vector<CellSite> &w = fx.content(site).sites;
                    from_tiles.insert(from_tiles.end(), w.begin(), w.end());
                }
            }
            if (k.lod == 0) {
                CHECK(in == site_tiles);
            } else {
                CHECK(in.empty());
                std::sort(from_tiles.begin(), from_tiles.end(), [](const CellSite &p_a, const CellSite &p_b) { return cell_order(p_a.cell, p_b.cell); });
                std::vector<CellSite> own;
                f.setup().scatter.sites(f.tile_box(k), own);
                CHECK(same_sites(own, from_tiles));
                walked += int64_t(own.size());
            }
            inputs += int64_t(in.size());
            cube += int64_t(all.size());
            const SourceContent &c = fx.content(k);
            held += std::count_if(c.records.begin(), c.records.end(), [lod](const TableRecord &p_r) { return p_r.listed_at(lod); });
            if (key_lod == lod) {
                CHECK(std::all_of(c.records.begin(), c.records.end(), [lod](const TableRecord &p_r) { return p_r.listed_at(lod); }));
            }
        }
        MESSAGE("LOD ", lod, ": tile ", f.part_tile_side(lod), " voxels (key LOD ", key_lod, "), ", double(held) / double(groups), " records per site, ", inputs,
                " site tiles read of ", cube, ", ", walked, " sites walked by the coarse tiles");
        CHECK(held == listed);
        CHECK((key_lod == 0 || (inputs == 0 && walked == groups)));
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
// percent, and its canopy reads about the same cover. p_gate false: reported only.
static void check_proxy_layers(const SourceFixture &fx, bool p_gate = true) {
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
        // The layers the streamed sources switch to proxies (a cell of 1-3 source cells: few sites per proxy) are looser: on
        // this forest, with the per-species moments, bark mass 0.97 and 1.15, peak 0.59 and 0.97 at layers 4 and 5 (0.62 and
        // 0.78 with one mean tree). A switched layer losing bark only (layer 4: thin trunks, mean peak 0.015 of 0.025, far
        // under the material iso 0.125, so neither side draws there) gets 0.45.
        const bool switched = f.layer_cell(layer) < ScatterSource::PART_CELL_SIDES * f.setup().scatter.side;
        const double mass_tol = switched ? 0.2 : 0.1, peak_tol = switched ? (truth[1] == 0.0 ? 0.45 : 0.4) : 0.25;
        if (!p_gate) {
            continue;
        }
        for (int m = 0; m < 2; m++) {
            CAPTURE(m);
            if (truth[m] == 0.0) {
                CHECK(proxies[m] == 0.0); // nothing of the material lost at the layer
            } else {
                CHECK(std::fabs(proxies[m] / truth[m] - 1.0) < mass_tol);
            }
        }
        CHECK(std::fabs(proxy_amp / truth_amp - 1.0) < peak_tol);
    }
    CHECK(layers_checked >= 2);
}

// On the whole build's setting, and on the streamed sources' (proxies from the source cell on: the layers that switch).
TEST_CASE("Scatter source: the proxy layers weigh what the records do") {
    for (const SourceFixture *fixture : { &SourceFixture::get(), &SourceFixture::streamed() }) {
        check_proxy_layers(*fixture);
    }
}

// Each species alone (its sites forced to it, one class): what the per-species moments stand for, reported.
TEST_CASE("Scatter source: the proxy layers of a one-species forest (reported)") {
    for (int species = 0; species < int(tree_species().size()); species++) {
        MESSAGE("species ", species, " alone:");
        check_proxy_layers(SourceFixture::one_species(species), false);
    }
}

// The forest's profile is per species and shared by its tree set: a source with the same key and another zone takes the
// same object, another voxel size another; per species, the lost mass weighted by the species' share of the sites is the
// mean site's (a one-class profile of hashed variants, other samples: within sampling noise).
TEST_CASE("Scatter source: one profile per tree set, per-species far moments") {
    const SourceFixture &fx = SourceFixture::get();
    const ScatterSource &f = fx.f();
    CHECK(f.profile().lost.size() == tree_species().size());
    ScatterSource::Setup other = f.setup();
    other.scatter.zone = VoxelBox{ fx.origin + VoxelPos(9000, -400, 0), fx.origin + VoxelPos(12000, 400, 3000), true };
    other.scatter.side = 140; // another density
    const ScatterSource b(other);
    CHECK(b.profile_shared());
    CHECK(&b.profile() == &f.profile());

    auto lib = std::make_shared<DefLibrary>();
    const TreeTypes types = add_tree_types(*lib, 0.1);
    ScatterSource::Setup coarse = other;
    coarse.blob_type = lib->add(make_moment_blob_type()); // the fixture's library order
    coarse.library = lib;
    set_forest_scatter(coarse.scatter, 0.1, *lib, coarse.finish, types, coarse.max_extent);
    const ScatterSource c(coarse);
    CHECK(&c.profile() != &f.profile());

    // The mixed mean: one class, hashed variants (the profile before classes).
    ScatterSource::Setup mixed = f.setup();
    mixed.scatter.classes = 1;
    mixed.scatter.profile_key.clear();
    const ScatterSource m(mixed);
    CHECK_FALSE(m.profile_shared());
    CHECK(m.profile().lost.size() == 1);
    const double n = double(tree_species().size());
    int checked = 0;
    for (int l = 0; l < f.setup().lod_count; l++) {
        double per_species = 0.0, mean = 0.0;
        for (uint32_t s = 0; s < tree_species().size(); s++) {
            for (const std::pair<uint32_t, Moments> &p : f.profile().lost_of(s, l).moments.parts) {
                per_species += p.second.m0 / n;
            }
        }
        for (const std::pair<uint32_t, Moments> &p : m.profile().lost_of(0, l).moments.parts) {
            mean += p.second.m0;
        }
        if (mean > 0.0) {
            checked++;
            MESSAGE("LOD ", l, ": lost mass per site, species-weighted ", per_species, " vs the mixed samples' ", mean);
            CHECK(std::fabs(per_species / mean - 1.0) < 0.15);
        }
    }
    CHECK(checked > 3);
}

TEST_CASE("Scatter source: the streamed sources' proxies start at the source cell") {
    const ScatterSource &whole = SourceFixture::get().f(), &streamed = SourceFixture::streamed().f();
    int switched = 0;
    for (int layer = 0; layer < whole.setup().lod_count; layer++) {
        CHECK((!streamed.part_layer(layer) || whole.part_layer(layer)));
        CHECK(streamed.part_layer(layer) == (layer < streamed.profile().lod_end && streamed.layer_cell(layer) < streamed.setup().scatter.side));
        switched += whole.has_layer(layer) && whole.part_layer(layer) && !streamed.part_layer(layer);
    }
    CHECK(switched >= 1);
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
            for (const VoxelPos &t : tiles_over(around, side)) {
                all.push_back(SourceKey{ 3, kind, kind == SourceKey::PARTS ? f.part_key_lod(lod) : lod, t });
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

namespace {

// Two sources on the forest's hills, in a smaller zone: rocks (id 0, spheres of 6-14 voxels, no blobs) and the 5 cm forest
// (id 1, with its moment layers), as the stream holds them.
struct SceneFixture {
    std::vector<std::shared_ptr<const ScatterSource>> sources;
    std::vector<std::unique_ptr<KeyCache>> caches; // per source
    ChunkParam param;
    VoxelPos origin = SourceFixture::get().origin;

    SceneFixture() {
        auto lib = std::make_shared<DefLibrary>();
        add_tree_types(*lib, SourceFixture::VOXEL);
        const int sphere = lib->add(make_sphere_type());
        const int blob = lib->add(make_moment_blob_type());
        const SourceFixture &forest = SourceFixture::get();
        ScatterSource::Setup s = forest.f().setup();
        s.library = lib;
        s.scatter.zone = VoxelBox{ origin - VoxelPos(1200, 400, 1200), origin + VoxelPos(1199, 400, 1199), true };
        s.part_cell_sides = ScatterSource::STREAM_PART_CELL_SIDES;
        ScatterSource::Setup rocks = s;
        rocks.id = 0;
        rocks.blob_type = -1;
        rocks.scatter.salt = 0x40C5u;
        rocks.scatter.side = 40;
        rocks.scatter.generate = [sphere](const Site &p_site, const UpAxis &, int, std::vector<TableRecord> &r_out) {
            TableRecord r;
            r.def = uint8_t(sphere);
            r.anchor = p_site.base;
            r.params[0] = 6.0f + float(p_site.seed % 9u);
            r_out.push_back(r);
        };
        s.id = 1;
        s.blob_type = blob; // the forest's generate kept: its tree types have the same indices in this library
        for (const ScatterSource::Setup &setup : { rocks, s }) {
            sources.push_back(std::make_shared<ScatterSource>(setup));
            caches.push_back(std::make_unique<KeyCache>(sources.back().get()));
        }
    }
};

// The record pyramid: a LOD-0 tile grows its sites (cell order, every record, a site the table would refuse left out); a
// coarser one holds the groups of the LOD-0 tiles inside it, site by site in cell order (one walk of its box gives them in
// that order), each filtered to the records listed at L. The oracle of the part tiles.
class Pyramid {
public:
    explicit Pyramid(const ScatterSource &p_source) :
            f(p_source) {}

    const SourceContent &tile(int p_lod, const VoxelPos &p_tile) {
        const std::tuple<int, int64_t, int64_t, int64_t> at(p_lod, p_tile.x, p_tile.y, p_tile.z);
        const auto it = memo.find(at);
        if (it != memo.end()) {
            return it->second.content;
        }
        Grown c;
        if (p_lod == 0) {
            grow(p_tile, c);
        } else {
            // The LOD-0 tiles' groups, merged in cell order.
            std::vector<std::pair<VoxelPos, std::pair<const Grown *, size_t>>> sites;
            std::vector<VoxelPos> under;
            pyramid_tiles(f, p_lod, p_tile, under);
            for (const VoxelPos &t : under) {
                tile(0, t);
                const Grown &g = memo.find(std::tuple<int, int64_t, int64_t, int64_t>(0, t.x, t.y, t.z))->second;
                for (size_t i = 0; i < g.cells.size(); i++) {
                    sites.emplace_back(g.cells[i], std::make_pair(&g, i));
                }
                c.content.out_of_profile += g.content.out_of_profile;
            }
            std::sort(sites.begin(), sites.end(), [](const auto &p_a, const auto &p_b) { return cell_order(p_a.first, p_b.first); });
            for (const auto &s : sites) {
                const SourceContent::Group &g = s.second.first->content.groups[s.second.second];
                append(s.second.first->content.records.data() + g.first, g.count, p_lod, c.content);
            }
        }
        return memo.emplace(at, std::move(c)).first->second.content;
    }

private:
    struct Grown {
        SourceContent content;
        std::vector<VoxelPos> cells; // a LOD-0 tile's: per group, its site's cell
    };
    const ScatterSource &f;
    std::map<std::tuple<int, int64_t, int64_t, int64_t>, Grown> memo;

    void grow(const VoxelPos &p_tile, Grown &r_out) const {
        const ScatterSource::Setup &set = f.setup();
        std::vector<CellSite> sites;
        set.scatter.sites(f.tile_box(SourceKey{ set.id, SourceKey::PARTS, 0, p_tile }), sites);
        for (const CellSite &cs : sites) {
            std::vector<TableRecord> grown;
            set.scatter.grow(cs.site, -1, grown); // the full growth: the oracle filters it per LOD
            if (std::any_of(grown.begin(), grown.end(), [&](const TableRecord &p_r) { return !record_fit_error((*set.library)[p_r.def], p_r, set.max_extent).is_empty(); })) {
                continue;
            }
            bool out = false;
            for (TableRecord &rec : grown) {
                finish_table_record(*set.library, set.finish, rec);
                out = out || !f.profile().covers(cs.site.base, rec);
            }
            if (append(grown.data(), uint32_t(grown.size()), -1, r_out.content)) {
                r_out.cells.push_back(cs.cell);
            }
            r_out.content.out_of_profile += out;
        }
    }
    // The records listed at p_lod (all: p_lod < 0) as one group, its bounds their union box, largest k, margin and LOD;
    // whether one was appended.
    static bool append(const TableRecord *p_records, uint32_t p_count, int p_lod, SourceContent &r_out) {
        const uint32_t first = uint32_t(r_out.records.size());
        VoxelBox box;
        TableRecord bounds;
        int max_lod = 0;
        for (uint32_t i = 0; i < p_count; i++) {
            const TableRecord &rec = p_records[i];
            if (p_lod >= 0 && !rec.listed_at(p_lod)) {
                continue;
            }
            box = box.merged(record_box(rec));
            bounds.k = MAX(bounds.k, rec.k);
            bounds.margin_c = MAX(bounds.margin_c, rec.margin_c);
            max_lod = MAX(max_lod, int(rec.max_lod));
            r_out.records.push_back(rec);
        }
        if (r_out.records.size() == first) {
            return false;
        }
        bounds.anchor = box.lo;
        for (int a = 0; a < 3; a++) {
            bounds.box_hi[a] = int32_t(MIN<int64_t>(box.hi[a] - box.lo[a], INT32_MAX));
        }
        bounds.max_lod = uint8_t(max_lod);
        r_out.groups.push_back(SourceContent::Group{ first, uint32_t(r_out.records.size()) - first, bounds });
        return true;
    }
};

bool same_content(const SourceContent &p_a, const SourceContent &p_b) {
    if (p_a.records.size() != p_b.records.size() || p_a.groups.size() != p_b.groups.size() || p_a.out_of_profile != p_b.out_of_profile) {
        return false;
    }
    for (size_t i = 0; i < p_a.records.size(); i++) {
        if (!same_bits(p_a.records[i], p_b.records[i])) {
            return false;
        }
    }
    for (size_t i = 0; i < p_a.groups.size(); i++) {
        const SourceContent::Group &a = p_a.groups[i], &b = p_b.groups[i];
        if (a.first != b.first || a.count != b.count || !same_bits(a.bounds, b.bounds)) {
            return false;
        }
    }
    return true;
}

} // namespace

// The part tiles (the LOD-0 ones grown from their site tiles, the coarser from their own walk) are the record pyramid's,
// bit for bit, at every LOD (the LODs that no longer have keys of their own included), for both sources.
TEST_CASE("Scatter source: the part tiles grown from the site tiles are the record pyramid's") {
    const SceneFixture fx;
    for (const std::shared_ptr<const ScatterSource> &s : fx.sources) {
        Pyramid pyramid(*s);
        KeyCache &cache = *fx.caches[s->setup().id];
        for (int lod = 0; lod < s->profile().lod_end; lod++) {
            CAPTURE(s->setup().id);
            CAPTURE(lod);
            int64_t tiles = 0, same = 0, records = 0;
            for (const VoxelPos &t : tiles_over(s->setup().scatter.zone, s->part_tile_side(lod))) {
                const SourceKey key{ s->setup().id, SourceKey::PARTS, lod, t };
                const SourceContent &got = cache.content(key);
                tiles++;
                same += same_content(got, pyramid.tile(lod, t));
                records += int64_t(got.records.size());
            }
            MESSAGE("source ", s->setup().id, " LOD ", lod, ": ", tiles, " tiles, ", records, " records, ", same, " identical");
            CHECK(same == tiles);
            CHECK((lod > 0 || records > 0));
        }
    }
}

// Through the stream (inline executor), a grid of chunks at LODs 0-7 lists the record pyramid's records in the same order:
// the LODs sharing the LOD-0 side read the LOD-0 keys, the coarser ones the tiles grown from the site tiles.
TEST_CASE("Scatter source: the chunks' lists through the stream are the record pyramid's") {
    const SceneFixture fx;
    InstanceStream::Settings settings;
    settings.slot_capacity = InstanceStream::SLOTS_MAX;
    InstanceStream stream(fx.sources, fx.param, settings, std::make_unique<StreamExecutor>());
    std::vector<Pyramid> pyramids;
    for (const std::shared_ptr<const ScatterSource> &s : fx.sources) {
        pyramids.emplace_back(*s);
    }
    Mirror mirror;
    std::set<SourceKey> proxy_tiles; // every proxy tile the chunks' blob tiles read
    for (int lod = 0; lod <= 7; lod++) {
        CAPTURE(lod);
        int64_t chunks = 0, right = 0, records = 0;
        for (int64_t i = 0; i < 25; i++) {
            const ChunkId chunk = ChunkMath::chunk_at_voxel(fx.origin + VoxelPos(-1100 + 550 * (i % 5), 0, -1100 + 550 * (i / 5)), lod, fx.param);
            for (const std::shared_ptr<const ScatterSource> &s : fx.sources) {
                std::vector<SourceKey> keys, in;
                s->chunk_keys(chunk, fx.param, keys);
                for (const SourceKey &k : keys) {
                    s->inputs(k, in);
                    std::copy_if(in.begin(), in.end(), std::inserter(proxy_tiles, proxy_tiles.end()), [](const SourceKey &p_k) { return p_k.kind == SourceKey::PROXIES; });
                }
            }
            InstanceStream::Admit answer = InstanceStream::LATER;
            std::vector<std::pair<ChunkId, uint64_t>> back;
            for (int frame = 0; frame < 8 && answer != InstanceStream::READY; frame++) {
                answer = stream.admit(chunk, 0, 0, 0);
                stream.poll();
                stream.take_ready(0, back);
            }
            CHECK(answer == InstanceStream::READY);
            mirror.update(stream);
            std::vector<uint32_t> ids;
            stream.chunk_list(chunk, ids);
            // The reference: per source, the pyramid's tiles over the zone at the chunk's LOD, then its blob tiles.
            VoxelPos lo, hi;
            chunk_sample_box(chunk, fx.param, lo, hi);
            std::vector<TableRecord> want;
            std::vector<uint32_t> hit;
            for (const std::shared_ptr<const ScatterSource> &s : fx.sources) {
                std::vector<const SourceContent *> contents;
                if (lod < s->profile().lod_end) {
                    for (const VoxelPos &t : tiles_over(s->setup().scatter.zone, s->part_tile_side(lod))) {
                        contents.push_back(&pyramids[s->setup().id].tile(lod, t));
                    }
                }
                std::vector<SourceKey> keys;
                s->keys(lo, hi, lod, keys);
                for (const SourceKey &k : keys) {
                    if (k.kind == SourceKey::BLOBS) {
                        contents.push_back(&fx.caches[k.source]->content(k));
                    }
                }
                for (const SourceContent *c : contents) {
                    hit.clear();
                    c->meeting(lo, hi, lod, hit);
                    for (const uint32_t h : hit) {
                        want.push_back(c->records[h]);
                    }
                }
            }
            bool same = ids.size() == want.size();
            for (size_t j = 0; same && j < ids.size(); j++) {
                const uint32_t at = ids[j] - stream.slot_base();
                same = ids[j] >= stream.slot_base() && at < mirror.records.size() && same_bits(mirror.records[at], want[j]);
            }
            chunks++;
            right += same;
            records += int64_t(want.size());
        }
        MESSAGE("LOD ", lod, ": ", chunks, " chunks, ", records, " records listed, ", right, " lists identical");
        CHECK(right == chunks);
    }
    CHECK(int64_t(stream.get_stats()["lists_missing"]) == 0);
    // A proxy tile shared by several blob tiles is walked once (resident while they wait, and after).
    const Dictionary by = stream.get_stats()["computed_by_key"];
    int64_t walked = 0;
    for (int layer = 0; layer < ChunkMath::MAX_LOD_NUMBERS; layer++) {
        const String name = vformat("proxies %d", layer);
        walked += by.has(name) ? int64_t(Dictionary(by[name])["count"]) : 0;
    }
    MESSAGE(int64_t(proxy_tiles.size()), " proxy tiles read, ", walked, " walks");
    CHECK(proxy_tiles.size() > 4);
    CHECK(walked == int64_t(proxy_tiles.size()));
}

// A site tile is the walk of its box; a far layer's blob tile reads the proxy tiles around it, whose sites in its region
// are the walk of the region itself (the one walk a blob tile did before), and is what it computes alone.
TEST_CASE("Scatter source: the site tiles are the walks their tiles would do") {
    const SourceFixture &fx = SourceFixture::get();
    const ScatterSource &f = fx.f();
    const CellScatter &sc = f.setup().scatter;
    int64_t sites = 0, tiles = 0, same = 0;
    for (const VoxelPos &t : tiles_over(fx.zone(), f.part_tile_side(0))) {
        const SourceKey key{ 3, SourceKey::SITES, 0, t };
        std::vector<CellSite> walk;
        const VoxelBox box = f.tile_box(key).clipped(sc.zone);
        if (box.valid) {
            surface_sites(box.lo, box.hi, sc.side, sc.salt, sc.ground, walk, true, walk_level(sc.zone, sc.side)); // from the zone's level
        }
        const SourceContent &c = fx.content(key);
        tiles++;
        same += c.records.empty() && c.groups.empty() && same_sites(c.sites, walk);
        sites += int64_t(walk.size());
    }
    MESSAGE(tiles, " LOD-0 site tiles, ", sites, " sites, ", same, " identical");
    CHECK(same == tiles);
    CHECK(sites > 1000);

    const ScatterSource &streamed = SourceFixture::streamed().f();
    int layers = 0;
    for (int layer = 0; layer < streamed.setup().lod_count; layer++) {
        if (!streamed.has_layer(layer) || streamed.part_layer(layer)) {
            continue;
        }
        layers++;
        const int64_t g = streamed.proxy_side(layer);
        std::vector<SourceKey> keys, in;
        streamed.keys(fx.zone().lo, fx.zone().hi, layer, keys);
        std::map<SourceKey, SourceContent> walks;
        int64_t blobs = 0, alike = 0, checked = 0, reads = 0, regions_same = 0, proxies = 0;
        for (const SourceKey &k : keys) {
            if (k.kind != SourceKey::BLOBS) {
                continue;
            }
            streamed.inputs(k, in);
            ScatterSource::Inputs raw;
            std::vector<CellSite> gathered;
            const VoxelBox region = streamed.proxy_region(k);
            for (const SourceKey &p : in) {
                CHECK((p.kind == SourceKey::PROXIES && p.lod == layer));
                auto it = walks.find(p);
                if (it == walks.end()) {
                    it = walks.emplace(p, SourceContent()).first;
                    streamed.content(p, {}, it->second);
                    proxies += int64_t(it->second.sites.size());
                }
                raw.push_back(&it->second);
                std::copy_if(it->second.sites.begin(), it->second.sites.end(), std::back_inserter(gathered),
                        [&region](const CellSite &p_s) { return region.contains(p_s.site.base); });
            }
            reads += int64_t(in.size());
            // The one-region walk from the same level (an unclipped region's: the old walk, where the zone doesn't clip it).
            std::vector<CellSite> walk;
            if (region.valid) {
                surface_sites(region.lo, region.hi, g, sc.salt ^ ScatterSource::PROXY_SALT, sc.ground, walk, false, streamed.proxy_level(layer));
            }
            std::sort(gathered.begin(), gathered.end(), [](const CellSite &p_a, const CellSite &p_b) { return cell_order(p_a.cell, p_b.cell); });
            regions_same += same_sites(gathered, walk);
            // Alone or with a tile missing (the CPU's point path): one walk of the region, from the same level.
            SourceContent with, alone, partial;
            streamed.content(k, raw, with);
            streamed.content(k, {}, alone);
            ScatterSource::Inputs missing = raw;
            if (!missing.empty()) {
                missing[0] = nullptr;
            }
            streamed.content(k, missing, partial);
            checked++;
            alike += same_content(with, alone) && same_content(with, partial);
            blobs += int64_t(with.records.size());
        }
        MESSAGE("layer ", layer, ": ", checked, " far blob tiles, ", blobs, " blobs, ", int64_t(walks.size()), " proxy tiles (", proxies, " proxies) read ",
                reads, " times, ", regions_same, " regions' walks identical, ", alike, " the same without their proxy tiles (all, or one)");
        CHECK(regions_same == checked);
        CHECK(alike == checked);
        CHECK(reads > int64_t(walks.size())); // shared
    }
    CHECK(layers >= 2);
}
