#include "test_common.hpp"

#include "cell_sites.hpp"
#include "debug_log.hpp"
#include "sdf_cascade.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iterator>

using namespace Archetype;

namespace {

constexpr uint32_t SALT = 0x5A17u;

// The sites of every cell meeting [p_lo, p_hi], cell by cell: what surface_sites' walk must find.
std::vector<Site> brute_sites(const VoxelPos &p_lo, const VoxelPos &p_hi, int64_t p_side, const FieldFn &p_field) {
    std::vector<Site> out;
    const VoxelBox box{ p_lo, p_hi, true };
    for (int64_t z = WorldCoord::floor_div(p_lo.z, p_side); z <= WorldCoord::floor_div(p_hi.z, p_side); z++) {
        for (int64_t y = WorldCoord::floor_div(p_lo.y, p_side); y <= WorldCoord::floor_div(p_hi.y, p_side); y++) {
            for (int64_t x = WorldCoord::floor_div(p_lo.x, p_side); x <= WorldCoord::floor_div(p_hi.x, p_side); x++) {
                Site site;
                if (cell_site(VoxelPos(x, y, z), p_side, SALT, p_field, site) && box.contains(site.base)) {
                    out.push_back(site);
                }
            }
        }
    }
    return out;
}

bool same_sites(const std::vector<CellSite> &p_a, const std::vector<Site> &p_b) {
    if (p_a.size() != p_b.size()) {
        return false;
    }
    for (size_t i = 0; i < p_a.size(); i++) {
        const Site &s = p_a[i].site;
        if (!(s.base == p_b[i].base) || s.seed != p_b[i].seed || s.variant != p_b[i].variant) {
            return false;
        }
    }
    return true;
}

// surface_sites' sites without their cells.
std::vector<Site> walk(const VoxelPos &p_lo, const VoxelPos &p_hi, int64_t p_side, const FieldFn &p_field) {
    std::vector<CellSite> cells;
    surface_sites(p_lo, p_hi, p_side, SALT, p_field, cells);
    std::vector<Site> out;
    for (const CellSite &cs : cells) {
        out.push_back(cs.site);
    }
    return out;
}

double closest_pair(const std::vector<Site> &p_sites) {
    double best = 1e30;
    for (size_t i = 0; i < p_sites.size(); i++) {
        for (size_t j = i + 1; j < p_sites.size(); j++) {
            const VoxelPos d = p_sites[i].base - p_sites[j].base;
            best = MIN(best, std::sqrt(double(d.x) * d.x + double(d.y) * d.y + double(d.z) * d.z));
        }
    }
    return best;
}

// A plane through p_origin with unit normal (p_nx, p_ny, p_nz), as a field in LOD0 voxels (the same at every LOD).
FieldFn plane_field(const VoxelPos &p_origin, double p_nx, double p_ny, double p_nz) {
    return [=](const VoxelPos &p_v, int) {
        const double w = double(p_v.x - p_origin.x) * p_nx + double(p_v.y - p_origin.y) * p_ny + double(p_v.z - p_origin.z) * p_nz;
        return ExactNoise::Float4{ float(p_nx), float(p_ny), float(p_nz), float(w) };
    };
}

} // namespace

TEST_CASE("Cell sites: on flat ground, one site per column of cells, on the surface") {
    const int64_t side = 100;
    const VoxelPos origin(-3000, 6017, 1200);
    const FieldFn flat = plane_field(origin, 0.0, 1.0, 0.0);
    const std::vector<Site> sites = walk(origin - VoxelPos(2000, 2000, 2000), origin + VoxelPos(2000, 2000, 2000), side, flat);
    const int64_t columns = (4000 / side) * (4000 / side);
    MESSAGE("flat: ", sites.size(), " sites for ", columns, " columns of cells");
    CHECK(int64_t(sites.size()) >= columns * 9 / 10);
    CHECK(int64_t(sites.size()) <= columns + 4 * (4000 / side) + 1);
    std::vector<VoxelPos> cells;
    for (const Site &s : sites) {
        CHECK(std::abs(s.base.y - origin.y) <= 1);
        cells.push_back(VoxelPos(WorldCoord::floor_div(s.base.x, side), 0, WorldCoord::floor_div(s.base.z, side)));
    }
    std::sort(cells.begin(), cells.end(), [](const VoxelPos &p_a, const VoxelPos &p_b) { return p_a.x != p_b.x ? p_a.x < p_b.x : p_a.z < p_b.z; });
    CHECK(std::adjacent_find(cells.begin(), cells.end()) == cells.end()); // one per column
    // The middle part: on flat ground two sites are 2 SITE_INNER cells apart across x and z.
    CHECK(closest_pair(sites) >= 2.0 * SITE_INNER * double(side) - 2.0);
}

TEST_CASE("Cell sites: the octree walk finds every cell's site (brute force), on hills and a 60 degree slope") {
    const int64_t side = 80;
    const VoxelPos origin(int64_t(1) << 40, -517, -(int64_t(1) << 33));
    const double tau = 6.283185307179586;
    const FieldFn hills = [origin, tau](const VoxelPos &p_v, int) {
        const double amp = 60.0, wave = 1500.0;
        const double x = double(p_v.x - origin.x);
        const double w = double(p_v.y - origin.y) - amp * std::sin(tau * x / wave);
        return ExactNoise::Float4{ float(-amp * tau / wave * std::cos(tau * x / wave)), 1.0f, 0.0f, float(w) };
    };
    const FieldFn slope = plane_field(origin, std::sin(1.0471975511965976), std::cos(1.0471975511965976), 0.0);
    const VoxelPos lo = origin - VoxelPos(1200, 1300, 1200), hi = origin + VoxelPos(1200, 1300, 1200);
    for (const FieldFn *f : { &hills, &slope }) {
        std::vector<CellSite> walked;
        surface_sites(lo, hi, side, SALT, *f, walked);
        const std::vector<Site> brute = brute_sites(lo, hi, side, *f);
        MESSAGE("walk ", walked.size(), " sites, brute force ", brute.size());
        CHECK(walked.size() > 100);
        CHECK(same_sites(walked, brute));
        for (const CellSite &s : walked) {
            CHECK(std::fabs((*f)(s.site.base, 0).w) <= 1.0f);
        }
        // The middle part keeps cells side by side along the ground 2 SITE_INNER cells apart; across the face of the
        // normal's axis two sites can meet, rarely (their jitters along the ground are independent).
        int64_t close = 0;
        for (size_t i = 0; i < brute.size(); i++) {
            for (size_t j = 0; j < brute.size(); j++) {
                const VoxelPos d = brute[i].base - brute[j].base;
                if (i != j && std::sqrt(double(d.x) * d.x + double(d.y) * d.y + double(d.z) * d.z) < 2.0 * SITE_INNER * double(side)) {
                    close++;
                    break;
                }
            }
        }
        MESSAGE("closest pair ", closest_pair(brute) / double(side), " cells; ", close, " of ", brute.size(), " sites closer than ", 2.0 * SITE_INNER,
                " cells to another");
        CHECK(close * 100 <= 3 * int64_t(brute.size()));
    }
}

// The walk keeps a node by its centre's coarse distance; the cascade's field is a first-order height-field estimate,
// and the coarse one leaves the fine octaves out. On the real ground at 5 cm the walk must still find every cell's site
// (SITE_SLACK), at a few evaluations per site.
TEST_CASE("Cell sites: on the cascade ground, the octree walk finds every cell's site (brute force)") {
    ChunkParam param;
    param.voxel_size = 0.05f;
    SDFCascade sdf;
    sdf.configure(param, 18);
    const int64_t side = 100;
    std::atomic<int64_t> coarse{ 0 }, fine{ 0 };
    const FieldFn ground = [&](const VoxelPos &p_v, int p_lod) {
        (p_lod > 0 ? coarse : fine)++;
        return sdf.ground_field(p_v, p_lod);
    };
    int64_t total = 0;
    for (const int64_t x : { int64_t(-1000), int64_t(250000), int64_t(-3700000) }) {
        const int64_t z = x / 3 + 40000;
        const VoxelPos centre(x, sdf.ground_at(x, z).y, z);
        const VoxelPos reach(1000, 4000, 1000);
        coarse = 0;
        fine = 0;
        std::vector<CellSite> walked;
        surface_sites(centre - reach, centre + reach, side, SALT, ground, walked);
        const int64_t walk_evals = coarse + fine;
        const std::vector<Site> brute = brute_sites(centre - reach, centre + reach, side, ground);
        MESSAGE("cascade at x ", x, ": walk ", walked.size(), " sites, brute force ", brute.size(), "; ", double(walk_evals) / double(MAX<size_t>(walked.size(), 1)),
                " evaluations per site");
        CHECK(walked.size() > 100);
        CHECK(same_sites(walked, brute));
        CHECK(walk_evals <= 12 * int64_t(walked.size()));
        total += int64_t(walked.size());
        // Unsnapped (far proxies): no LOD-0 evaluation, every snapped site's cell kept, the sites within a cell of the ground.
        fine = 0;
        std::vector<CellSite> unsnapped;
        surface_sites(centre - reach, centre + reach, side, SALT, ground, unsnapped, false);
        CHECK(fine == 0);
        size_t j = 0, matched = 0;
        double off = 0.0;
        for (const CellSite &s : walked) {
            while (j < unsnapped.size() && cell_order(unsnapped[j].cell, s.cell)) {
                j++;
            }
            if (j < unsnapped.size() && unsnapped[j].cell == s.cell) {
                matched++;
                off = MAX(off, std::fabs(double(sdf.ground_field(unsnapped[j].site.base, 0).w)));
            }
        }
        CHECK(matched == walked.size());
        CHECK(off <= double(side));
        CHECK(unsnapped.size() * 100 <= walked.size() * 105);
    }
    CHECK(total > 300);
    // An empty box (a node of 8 cells far above the ground): one evaluation.
    const int64_t tile = 8 * side;
    const VoxelPos air(0, WorldCoord::floor_div(sdf.ground_at(0, 0).y, tile) + 20, 0);
    coarse = 0;
    fine = 0;
    std::vector<CellSite> none;
    surface_sites(air * tile, air * tile + VoxelPos(tile - 1, tile - 1, tile - 1), side, SALT, ground, none);
    CHECK(none.empty());
    CHECK(coarse + fine == 1);
}

TEST_CASE("Cell sites: a cell holds the same site whatever the query box, another salt another site") {
    const int64_t side = 100;
    const VoxelPos origin(500, 300, -700);
    const FieldFn slope = plane_field(origin, 0.6, 0.8, 0.0);
    const std::vector<Site> a = walk(origin - VoxelPos(1500, 1500, 1500), origin + VoxelPos(1500, 1500, 1500), side, slope);
    const std::vector<Site> b = walk(origin - VoxelPos(300, 900, 2500), origin + VoxelPos(2600, 1100, 400), side, slope);
    int64_t shared = 0;
    for (const Site &s : a) {
        for (const Site &t : b) {
            if (s.base == t.base) {
                CHECK(s.seed == t.seed);
                CHECK(s.variant == t.variant);
                shared++;
            }
        }
    }
    MESSAGE("query boxes: ", a.size(), " and ", b.size(), " sites, ", shared, " in both");
    CHECK(shared > 50);
    std::vector<CellSite> other;
    surface_sites(origin - VoxelPos(1500, 1500, 1500), origin + VoxelPos(1500, 1500, 1500), side, SALT + 1, slope, other);
    int64_t same = 0;
    for (size_t i = 0; i < MIN(a.size(), other.size()); i++) {
        same += a[i].base == other[i].site.base ? 1 : 0;
    }
    CHECK(same * 10 < int64_t(a.size()));
}

// What a LOD-0 site tile's walk costs on the cascade at 5 cm (the 1M forest's cells: 100 voxels, tiles of 8 cells): the
// tiles of a LOD-7 part tile's footprint (16^3 site tiles, most of them air), by the walk's outcome. An air tile with no
// ground within its own node's plane slack costs the one evaluation at its centre; the ones next to the ground descend.
TEST_CASE("Cell sites: an air tile's walk costs one evaluation, a ground tile's a few per cell") {
    ChunkParam param;
    param.voxel_size = 0.05f;
    SDFCascade sdf;
    sdf.configure(param, 18);
    const int64_t side = 100, tile = 8 * side, n = 16;
    std::atomic<int64_t> evals{ 0 };
    const FieldFn ground = [&](const VoxelPos &p_v, int p_lod) {
        evals++;
        return sdf.ground_field(p_v, p_lod);
    };
    const int64_t x0 = 250000, z0 = 40000;
    const int64_t ty = WorldCoord::floor_div(sdf.ground_at(x0, z0).y, tile) - n / 2;
    // [0]: tiles with sites, [1]: empty tiles that descended (more than one evaluation), [2]: empty at one evaluation.
    int64_t tiles[3] = {}, cost[3] = {}, sites = 0;
    const uint64_t t0 = now_usec();
    for (int64_t z = 0; z < n; z++) {
        for (int64_t y = 0; y < n; y++) {
            for (int64_t x = 0; x < n; x++) {
                const VoxelPos lo = VoxelPos(WorldCoord::floor_div(x0, tile) + x, ty + y, WorldCoord::floor_div(z0, tile) + z) * tile;
                std::vector<CellSite> found;
                evals = 0;
                surface_sites(lo, lo + VoxelPos(tile - 1, tile - 1, tile - 1), side, SALT, ground, found);
                const int kind = !found.empty() ? 0 : (evals > 1 ? 1 : 2);
                tiles[kind]++;
                cost[kind] += evals;
                sites += int64_t(found.size());
            }
        }
    }
    const double usec = double(now_usec() - t0);
    const int64_t all = cost[0] + cost[1] + cost[2];
    MESSAGE("16^3 site tiles: ", tiles[0], " with ", sites, " sites at ", double(cost[0]) / double(MAX<int64_t>(tiles[0], 1)), " evaluations each, ",
            tiles[1], " empty that descend at ", double(cost[1]) / double(MAX<int64_t>(tiles[1], 1)), ", ", tiles[2], " air at 1; ", all,
            " evaluations, ", usec / double(MAX<int64_t>(all, 1)), " us each");
    CHECK(tiles[2] > tiles[0] + tiles[1]); // most of a part tile's footprint is air
    CHECK(cost[2] == tiles[2]);
    CHECK(cost[0] > 10 * tiles[0]);
}

// Tiles walked from a box's level keep, over each tile, the sites the box's own walk keeps: on the cascade ground, where
// the field is far from its planes (the prune's slack is a heuristic there, so only the shared start makes it exact). The
// far proxies' tiles rest on it (ScatterSource::proxy_level).
TEST_CASE("Cell sites: tiles walked from a box's level keep the box walk's sites, on the cascade ground") {
    ChunkParam param;
    param.voxel_size = 0.05f;
    SDFCascade sdf;
    sdf.configure(param, 18);
    const FieldFn ground = [&](const VoxelPos &p_v, int p_lod) { return sdf.ground_field(p_v, p_lod); };
    const int64_t side = 50, tile = 16 * side; // a far layer's proxies: cells of half a layer cell, tiles of 16 of them
    int64_t sites = 0;
    for (const int64_t x0 : { int64_t(-3700000), int64_t(250000), int64_t(1200000) }) {
        const int64_t z0 = x0 / 3 + 40000;
        const VoxelPos t0(WorldCoord::floor_div(x0, tile) - 1, WorldCoord::floor_div(sdf.ground_at(x0, z0).y, tile) - 1, WorldCoord::floor_div(z0, tile) - 1);
        // A region off the tile grid (a blob tile grown by its margin), its walk, and its tiles' walks from its level.
        const VoxelBox region{ t0 * tile + VoxelPos(tile / 3, tile / 5, tile / 7), (t0 + VoxelPos(3, 3, 3)) * tile - VoxelPos(tile / 4, tile / 6, tile / 9), true };
        const int level = walk_level(region, side);
        std::vector<CellSite> whole, tiles;
        surface_sites(region.lo, region.hi, side, SALT, ground, whole, false, level);
        for (int64_t z = 0; z < 3; z++) {
            for (int64_t y = 0; y < 3; y++) {
                for (int64_t x = 0; x < 3; x++) {
                    const VoxelPos lo = (t0 + VoxelPos(x, y, z)) * tile;
                    std::vector<CellSite> one;
                    surface_sites(lo, lo + VoxelPos(tile - 1, tile - 1, tile - 1), side, SALT, ground, one, false, level);
                    std::copy_if(one.begin(), one.end(), std::back_inserter(tiles), [&region](const CellSite &p_s) { return region.contains(p_s.site.base); });
                }
            }
        }
        std::sort(tiles.begin(), tiles.end(), [](const CellSite &p_a, const CellSite &p_b) { return cell_order(p_a.cell, p_b.cell); });
        std::vector<Site> want;
        for (const CellSite &cs : whole) {
            want.push_back(cs.site);
        }
        CAPTURE(x0);
        CHECK(same_sites(tiles, want));
        sites += int64_t(whole.size());
    }
    CHECK(sites > 100);
}

// Many boxes walked in lockstep (surface_sites_many) find each box's own sites, whatever walks share the rounds: site
// tiles (snapped) and proxy tiles (unsnapped, another cell side and salt) side by side, an air box, a walk with no cell.
// A round reads once what several walks ask (the tiles' shared ancestors), and the rounds don't grow with the boxes.
TEST_CASE("Cell sites: boxes walked in lockstep keep each box's sites, in rounds shared by all of them") {
    ChunkParam param;
    param.voxel_size = 0.05f;
    SDFCascade sdf;
    sdf.configure(param, 18);
    const FieldFn ground = [&](const VoxelPos &p_v, int p_lod) { return sdf.ground_field(p_v, p_lod); };
    int calls = 0;
    const FieldBatchFn counted = [&](const VoxelPos *p_points, const int *p_lods, int p_count, ExactNoise::Float4 *r_out) {
        calls++;
        batch_of(ground)(p_points, p_lods, p_count, r_out);
    };
    const int64_t x0 = 250000, z0 = x0 / 3 + 40000;
    const int64_t side = 100, tile = 8 * side;
    const VoxelBox zone{ VoxelPos(x0 - 64 * tile, -(int64_t(1) << 20), z0 - 64 * tile), VoxelPos(x0 + 64 * tile, int64_t(1) << 20, z0 + 64 * tile), true };
    const int level = walk_level(zone, side);
    const VoxelPos t0(WorldCoord::floor_div(x0, tile) - 2, WorldCoord::floor_div(sdf.ground_at(x0, z0).y, tile) - 1, WorldCoord::floor_div(z0, tile) - 2);
    std::vector<SiteWalk> walks;
    std::vector<std::vector<CellSite>> got(64), want(64);
    for (int64_t z = 0; z < 4; z++) {
        for (int64_t y = 0; y < 3; y++) {
            for (int64_t x = 0; x < 4; x++) {
                const VoxelPos lo = (t0 + VoxelPos(x, y, z)) * tile;
                const VoxelBox box{ lo, lo + VoxelPos(tile - 1, tile - 1, tile - 1), true };
                const bool proxies = (x + y + z) % 3 == 0;
                walks.push_back(SiteWalk{ box, proxies ? side / 2 : side, proxies ? SALT ^ 0x9999u : SALT, !proxies, level, &got[walks.size()] });
            }
        }
    }
    const VoxelPos sky = (t0 + VoxelPos(0, 40, 0)) * tile;
    walks.push_back(SiteWalk{ VoxelBox{ sky, sky + VoxelPos(tile - 1, tile - 1, tile - 1), true }, side, SALT, true, level, &got[walks.size()] });
    walks.push_back(SiteWalk{ walks[0].box, 0, SALT, true, level, &got[walks.size()] });
    WalkStats stats;
    surface_sites_many(walks.data(), walks.size(), counted, &stats);
    int64_t sites = 0;
    for (size_t w = 0; w < walks.size(); w++) {
        const SiteWalk &s = walks[w];
        surface_sites(s.box.lo, s.box.hi, s.side, s.salt, ground, want[w], s.snap, s.min_level);
        std::vector<Site> expected;
        for (const CellSite &cs : want[w]) {
            expected.push_back(cs.site);
        }
        CAPTURE(w);
        CHECK(same_sites(got[w], expected));
        sites += int64_t(got[w].size());
    }
    CHECK(sites > 50);
    CHECK(got[walks.size() - 2].empty());
    CHECK(got[walks.size() - 1].empty());
    // One call per round: the levels from the zone's down to the cells, then the snap's reads.
    CHECK(calls == stats.rounds);
    CHECK(stats.rounds <= level + 1 + SNAP_STEPS + 1);
    CHECK(stats.unique < stats.readings);
    MESSAGE(walks.size(), " walks: ", stats.rounds, " rounds, ", stats.readings, " readings asked, ", stats.unique, " read");
}
