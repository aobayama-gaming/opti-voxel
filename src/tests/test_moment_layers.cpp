#include "test_common.hpp"

#include "archetype_table.hpp"
#include "moment_layers.hpp"

#include <map>

using namespace Archetype;

namespace {

// A slab of spheres of radius p_r on a jittered square grid of spacing p_d (voxels), centred on p_centre, p_n x p_n,
// dropped at LOD p_drop: a canopy.
std::vector<MomentItem> sphere_slab(const VoxelPos &p_centre, int p_n, double p_d, double p_r, int p_drop, uint32_t p_material) {
    std::vector<MomentItem> items;
    for (int i = 0; i < p_n * p_n; i++) {
        const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ uint32_t(i), 0x51ABu, p_material });
        MomentItem it;
        it.anchor = p_centre +
                VoxelPos(std::llround((i % p_n - p_n / 2 + 0.6 * (ExactNoise::unit_of(h.x) - 0.5)) * p_d), std::llround(0.3 * p_r * (ExactNoise::unit_of(h.z) - 0.5)),
                        std::llround((i / p_n - p_n / 2 + 0.6 * (ExactNoise::unit_of(h.y) - 0.5)) * p_d));
        it.moments = sphere_moments(p_r);
        it.material = p_material;
        it.drop_lod = p_drop;
        items.push_back(it);
    }
    return items;
}

double occupancy_at(const std::vector<MomentBlob> &p_blobs, int p_lod, const VoxelPos &p_at) {
    double sum = 0.0;
    for (const MomentBlob &b : p_blobs) {
        if (b.lod != p_lod) {
            continue;
        }
        const double off[3] = { double(p_at.x - b.anchor.x), double(p_at.y - b.anchor.y), double(p_at.z - b.anchor.z) };
        sum += b.occupancy(off);
    }
    return sum;
}

} // namespace

TEST_CASE("Moment layers: each layer holds the mass dropped at or before its LOD, per material") {
    std::vector<MomentItem> items = sphere_slab(VoxelPos(-(int64_t(1) << 40), 500, 77), 12, 20.0, 6.0, 3, 1);
    const std::vector<MomentItem> more = sphere_slab(VoxelPos(-(int64_t(1) << 40) + 30, 520, 90), 8, 25.0, 9.0, 5, 2);
    items.insert(items.end(), more.begin(), more.end());
    MomentLayerParams params;
    params.lod_min = 2;
    params.lod_max = 7;
    params.min_occupancy = 0.0; // keep every blob: the mass is exact
    std::vector<MomentBlob> blobs;
    build_moment_blobs(items, params, blobs);
    for (int lod = params.lod_min; lod <= params.lod_max; lod++) {
        for (const uint32_t material : { 1u, 2u }) {
            double want = 0.0, got = 0.0;
            for (const MomentItem &it : items) {
                want += it.material == material && it.drop_lod <= lod ? it.moments.m0 : 0.0;
            }
            for (const MomentBlob &b : blobs) {
                got += b.lod == lod && b.material == material ? b.mass : 0.0;
            }
            CAPTURE(lod);
            CAPTURE(material);
            CHECK(got == doctest::Approx(want).epsilon(1e-9));
        }
    }
    // Every rotation is proper and orthonormal (rows within 1e-5, like a runtime edit's), every amplitude finite.
    for (const MomentBlob &b : blobs) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double d = 0.0;
                for (int c = 0; c < 3; c++) {
                    d += double(b.rotation[3 * i + c]) * double(b.rotation[3 * j + c]);
                }
                CHECK(std::abs(d - (i == j ? 1.0 : 0.0)) < 1e-5);
            }
        }
        const float *r = b.rotation;
        const double det = double(r[0]) * (double(r[4]) * r[8] - double(r[5]) * r[7]) - double(r[1]) * (double(r[3]) * r[8] - double(r[5]) * r[6]) +
                double(r[2]) * (double(r[3]) * r[7] - double(r[4]) * r[6]);
        CHECK(det > 0.0);
        CHECK(std::isfinite(b.amplitude));
    }
}

TEST_CASE("Moment layers: a uniform canopy reads a near-constant occupancy, its column integral is its mass") {
    // Spheres of radius 6 every 16 voxels: per unit area 4/3 pi 6^3 / 16^2 = 3.53 voxels of solid. From LOD 4 the cells
    // (64 voxels) hold 4 x 4 spheres and a LOD cell is the spacing (below, the spheres are still about resolved); the slab
    // lies across a cell boundary at both LODs (y = 128).
    const VoxelPos centre(1000, 128, -2000);
    const std::vector<MomentItem> items = sphere_slab(centre, 32, 16.0, 6.0, 0, 1);
    for (const int lod : { 4, 5 }) {
        MomentLayerParams params;
        params.cell_shift = 2; // cells of 4 LOD cells: the slab spans many of them at both LODs
        params.keep_cover = false; // the mass kept: the cover is the next tests'
        params.lod_min = lod;
        params.lod_max = lod;
        std::vector<MomentBlob> blobs;
        build_moment_blobs(items, params, blobs);
        // Along x through the middle of the slab, inside it (away from its edges): the occupancy at the slab's height, and
        // the column integral over y (should be the solid per unit area).
        double lo = 1e30, hi = 0.0, column_sum = 0.0;
        int columns = 0;
        for (int x = -64; x <= 64; x += 3) {
            const VoxelPos at = centre + VoxelPos(x, 0, 5);
            const double phi = occupancy_at(blobs, lod, at);
            lo = std::min(lo, phi);
            hi = std::max(hi, phi);
            double column = 0.0;
            for (int y = -80; y <= 80; y++) {
                column += occupancy_at(blobs, lod, at + VoxelPos(0, y, 0));
            }
            column_sum += column;
            columns++;
        }
        const double per_area = 4.0 / 3.0 * M_PI * 216.0 / 256.0;
        MESSAGE("LOD ", lod, ": ", blobs.size(), " blobs, occupancy in the slab ", lo, " .. ", hi, ", column integral ", column_sum / columns,
                " (solid per unit area ", per_area, ")");
        CHECK(column_sum / columns == doctest::Approx(per_area).epsilon(0.05));
        CHECK(hi - lo < 0.2 * hi); // no holes along the cell edges
        // The slab is thin: its blobs reach far less up and down than across (their spread is the parts' positions').
        int thin = 0;
        for (const MomentBlob &blob : blobs) {
            double e[3] = {};
            for (int j = 0; j < 3; j++) {
                for (int i = 0; i < 3; i++) {
                    e[j] += double(blob.rotation[3 * i + j]) * double(blob.rotation[3 * i + j]) * blob.s[i] * blob.s[i];
                }
            }
            thin += e[1] < 0.5 * e[0] && e[1] < 0.5 * e[2] ? 1 : 0;
        }
        CHECK(thin > int(blobs.size()) / 2);
    }
}

TEST_CASE("Moment layers: an absolute grid, the same bits for the same items, and faint blobs pruned") {
    const std::vector<MomentItem> items = sphere_slab(VoxelPos(5, 40, -9), 10, 18.0, 5.0, 2, 3);
    MomentLayerParams params;
    params.lod_min = 2;
    params.lod_max = 6;
    std::vector<MomentBlob> a, b;
    build_moment_blobs(items, params, a);
    build_moment_blobs(items, params, b);
    // Moved by a multiple of every layer's cell: the same blobs, moved.
    const int64_t shift = int64_t(1) << 40;
    std::vector<MomentItem> moved = items;
    for (MomentItem &it : moved) {
        it.anchor += VoxelPos(shift, -shift, 3 * shift);
    }
    std::vector<MomentBlob> c;
    build_moment_blobs(moved, params, c);
    CHECK(a.size() == b.size());
    CHECK(a.size() == c.size());
    for (size_t i = 0; i < MIN(a.size(), c.size()); i++) {
        CHECK(a[i].anchor == b[i].anchor);
        CHECK(ExactNoise::float_bits(a[i].amplitude) == ExactNoise::float_bits(b[i].amplitude));
        CHECK(a[i].anchor + VoxelPos(shift, -shift, 3 * shift) == c[i].anchor);
        CHECK(ExactNoise::float_bits(a[i].amplitude) == ExactNoise::float_bits(c[i].amplitude));
        CHECK(ExactNoise::float_bits(a[i].inv_s2[0]) == ExactNoise::float_bits(c[i].inv_s2[0]));
        CHECK(ExactNoise::float_bits(a[i].rotation[4]) == ExactNoise::float_bits(c[i].rotation[4]));
    }
    // A lone small part (a 4-voxel pebble) far from the slab: drawn by its blobs at LOD 2, where it is about a cell, and
    // pruned once the LOD's filter spreads its cover fainter than min_occupancy (from LOD 5).
    std::vector<MomentItem> lone = items;
    MomentItem pebble;
    pebble.anchor = VoxelPos(100000, 40, 100000);
    pebble.moments = sphere_moments(2.0);
    pebble.material = 3;
    pebble.drop_lod = 2;
    lone.push_back(pebble);
    std::vector<MomentBlob> d;
    build_moment_blobs(lone, params, d);
    int far_fine = 0, far_coarse = 0;
    for (const MomentBlob &blob : d) {
        (blob.lod < 5 ? far_fine : far_coarse) += blob.anchor.x > 50000 ? 1 : 0;
        CHECK(blob.peak_bound >= params.min_occupancy);
    }
    CHECK(far_fine > 0);
    CHECK(far_coarse == 0);
    // The slab itself survives the pruning at every layer.
    for (int lod = params.lod_min; lod <= params.lod_max; lod++) {
        int slab = 0;
        for (const MomentBlob &blob : d) {
            slab += blob.lod == lod && blob.anchor.x < 50000 ? 1 : 0;
        }
        CAPTURE(lod);
        CHECK(slab > 0);
    }
}

TEST_CASE("Moment layers: the pruning's cost per item stays flat as a forest of tall trunks grows at a fixed density") {
    // 5 cm voxels: 20 m trunks 5 m apart under 6 m crowns. A trunk's blob reaches tens of fine cells up and down; the
    // pruning once tested such blobs against every blob of the layer (a 40,000-tree forest never finished loading).
    auto field = [](int p_n) {
        std::vector<MomentItem> items;
        for (int i = 0; i < p_n * p_n; i++) {
            MomentItem trunk;
            trunk.anchor = VoxelPos(int64_t(i % p_n) * 100, 200, int64_t(i / p_n) * 100);
            trunk.moments = cylinder_moments(200.0, 8.0);
            trunk.material = 1;
            trunk.drop_lod = 1;
            MomentItem crown = trunk;
            crown.anchor.y = 450;
            crown.moments = sphere_moments(60.0);
            crown.material = 2;
            crown.drop_lod = 3;
            items.push_back(trunk);
            items.push_back(crown);
        }
        return items;
    };
    MomentLayerParams params;
    params.lod_min = 1;
    params.lod_max = 8;
    double per_item[2];
    for (int k = 0; k < 2; k++) {
        const std::vector<MomentItem> items = field(16 << k);
        MomentLayers layers;
        layers.build(items, params);
        per_item[k] = double(layers.pruning_tests()) / double(items.size());
        MESSAGE(items.size(), " items: ", per_item[k], " box tests per item");
    }
    CHECK(per_item[1] < 1.5 * per_item[0]);
}

TEST_CASE("Moment layers: from far away a canopy keeps its cover, a lone clump fades, whatever the up axis") {
    // Spheres of radius 48 every 128 voxels across 8,192 (64 x 64): a canopy, four LOD-8 cells wide. Mass alone spreads
    // it over the LOD's filter, thinner every LOD (it vanished from far away); the cover keeps its occupancy. p_swap lays
    // the slab in the y z plane (x and y swapped): no axis is up.
    auto canopy_phi = [](int p_lod, bool p_cover, bool p_swap) {
        std::vector<MomentItem> items = sphere_slab(VoxelPos(0, 3000, 0), 64, 128.0, 48.0, 3, 1);
        if (p_swap) {
            for (MomentItem &it : items) {
                std::swap(it.anchor.x, it.anchor.y);
            }
        }
        MomentLayerParams params;
        params.lod_min = p_lod;
        params.lod_max = p_lod;
        params.keep_cover = p_cover;
        std::vector<MomentBlob> blobs;
        build_moment_blobs(items, params, blobs);
        double sum = 0.0;
        int n = 0;
        for (int u = -1600; u <= 1600; u += 160) {
            for (int w = -1600; w <= 1600; w += 160) {
                const VoxelPos at = p_swap ? VoxelPos(3000, u, w) : VoxelPos(u, 3000, w);
                sum += occupancy_at(blobs, p_lod, at);
                n++;
            }
        }
        return sum / n;
    };
    // The spheres' share of the area: pi 48^2 / 128^2.
    const double cover = M_PI * 48.0 * 48.0 / (128.0 * 128.0);
    for (int lod = 3; lod <= 9; lod++) {
        const double mass = canopy_phi(lod, false, false), kept = canopy_phi(lod, true, false), swapped = canopy_phi(lod, true, true);
        MESSAGE("LOD ", lod, ": occupancy in the canopy ", kept, " keeping the cover (", cover, "), ", mass, " keeping the mass, ", swapped,
                " laid on its side");
        CAPTURE(lod);
        CHECK(kept == doctest::Approx(cover).epsilon(0.25)); // its cover at every LOD, over MOMENT_ISO
        CHECK(kept > MOMENT_ISO);
        CHECK(kept >= mass);
        CHECK(swapped == doctest::Approx(kept).epsilon(1e-4));
    }
    CHECK(canopy_phi(8, false, false) < 0.5 * canopy_phi(8, true, false)); // keeping the mass, it fades under the surface
    // A lone clump (four 25-voxel crowns): drawn while about resolved, under MOMENT_ISO once the filter outgrows it.
    std::vector<MomentItem> clump;
    for (int i = 0; i < 4; i++) {
        MomentItem it;
        it.anchor = VoxelPos(5000 + (i & 1) * 30, 900 + (i >> 1) * 30, 5000 + ((i + 1) & 2) * 15);
        it.moments = sphere_moments(25.0);
        it.material = 2;
        it.drop_lod = 0;
        clump.push_back(it);
    }
    MomentLayerParams params;
    params.lod_min = 2;
    params.lod_max = 8;
    params.min_occupancy = 0.0;
    std::vector<MomentBlob> blobs;
    build_moment_blobs(clump, params, blobs);
    double peak[9] = {};
    for (const MomentBlob &b : blobs) {
        peak[b.lod] = MAX(peak[b.lod], double(b.amplitude));
    }
    MESSAGE("lone clump: peak ", peak[2], " at LOD 2, ", peak[8], " at LOD 8");
    CHECK(peak[2] > MOMENT_ISO);
    CHECK(peak[8] < MOMENT_ISO);
}

TEST_CASE("Moment layers: an eroding part hands the share of its mass its erosion took to the layers, all of it at its drop") {
    MomentItem part;
    part.anchor = VoxelPos(0, 0, 0);
    part.moments = sphere_moments(40.0);
    part.material = 1;
    part.drop_lod = 6;
    part.erode = 1.0f;
    part.erode_from = 16.0f; // onset LOD 4: eroded by 2^L - 16 voxels
    prepare_moment_item(part);
    CHECK(part.semi[0] == doctest::Approx(40.0)); // a sphere's semi-axes are its radius
    CHECK(part.area[0] == doctest::Approx(M_PI * 1600.0)); // its silhouette along x
    CHECK(part.first_lost_lod() == 5);
    CHECK(part.lost_share(4) == 0.0);
    const double e = 32.0 - 16.0;
    CHECK(part.lost_share(5) == doctest::Approx(1.0 - std::pow((40.0 - e) / 40.0, 3)));
    CHECK(part.lost_share(6) == 1.0);
    // The layers hold that share of its mass (the material joins at once: it is all there is).
    MomentLayerParams params;
    params.lod_min = 3;
    params.lod_max = 7;
    params.min_occupancy = 0.0;
    std::vector<MomentBlob> blobs;
    build_moment_blobs({ part }, params, blobs);
    for (int lod = params.lod_min; lod <= params.lod_max; lod++) {
        double mass = 0.0;
        for (const MomentBlob &b : blobs) {
            mass += b.lod == lod ? b.mass : 0.0;
        }
        CAPTURE(lod);
        CHECK(mass == doctest::Approx(part.moments.m0 * part.lost_share(lod)).epsilon(1e-9));
    }
}

// The pruning reads the cells within PRUNE_CELLS of a blob's own: a tile judged alone, from the items whose centroid lies
// within PRUNE_CELLS + 1 cells of it (the others emptied, so the ids and their blocks stay), keeps a whole build's blobs.
TEST_CASE("Moment layers: a judged tile keeps what a whole build keeps inside it, bit for bit") {
    const std::vector<MomentItem> items = sphere_slab(VoxelPos(5, 40, -9), 24, 18.0, 5.0, 2, 3);
    MomentLayerParams params;
    params.lod_min = params.lod_max = 3;
    std::vector<MomentBlob> whole;
    build_moment_blobs(items, params, whole);
    const int64_t cell = int64_t(1) << (3 + params.cell_shift);
    const VoxelBox tile{ VoxelPos(-1, 0, -1), VoxelPos(0, 0, 0), true }; // cells
    const int64_t reach = (MomentLayers::PRUNE_CELLS + 1) * cell;
    std::vector<MomentItem> near = items;
    for (MomentItem &it : near) {
        bool inside = true;
        for (int a = 0; a < 3; a++) {
            const double at = double(it.anchor[a]) + it.moments.c[a];
            inside = inside && at >= double(tile.lo[a] * cell - reach) && at < double((tile.hi[a] + 1) * cell + reach);
        }
        if (!inside) {
            it.moments = Moments();
        }
    }
    params.judged = tile;
    MomentLayers layers;
    MomentLayers reference;
    MomentLayerParams unjudged = params;
    unjudged.judged = VoxelBox();
    reference.build(items, unjudged);
    layers.build(near, params, &reference.get_joins());
    std::vector<MomentBlob> got;
    layers.kept(got);
    int64_t expected = 0, same = 0;
    for (const MomentBlob &w : whole) {
        if (!tile.contains(VoxelPos(w.cell[0], w.cell[1], w.cell[2]))) {
            continue;
        }
        expected++;
        for (const MomentBlob &g : got) {
            same += g.cell[0] == w.cell[0] && g.cell[1] == w.cell[1] && g.cell[2] == w.cell[2] && g.anchor == w.anchor &&
                    ExactNoise::float_bits(g.amplitude) == ExactNoise::float_bits(w.amplitude) &&
                    ExactNoise::float_bits(g.inv_s2[0]) == ExactNoise::float_bits(w.inv_s2[0]);
        }
    }
    MESSAGE("tile: ", got.size(), " blobs, whole build ", expected, " inside it, ", same, " identical");
    CHECK(expected > 0);
    CHECK(int64_t(got.size()) == expected);
    CHECK(same == expected);
}
