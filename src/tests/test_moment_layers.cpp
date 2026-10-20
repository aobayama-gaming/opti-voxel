#include "test_common.hpp"

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
    // pruned once the LOD's filter spreads it fainter than min_occupancy (from LOD 4).
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
        (blob.lod < 4 ? far_fine : far_coarse) += blob.anchor.x > 50000 ? 1 : 0;
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
