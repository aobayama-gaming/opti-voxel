#include "test_common.hpp"

#include "sdf_archetype.hpp"
#include "sdf_cascade.hpp"

#include <cmath>

namespace {

struct Lcg {
    uint64_t state;
    double next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return double(state >> 11) * (1.0 / 9007199254740992.0);
    }
};

// The nearest point of Λ_lod (multiples of 2^lod voxels).
int64_t snap(double p_voxel, int p_lod) {
    const double unit = std::ldexp(1.0, p_lod);
    return int64_t(std::llround(p_voxel / unit)) << p_lod;
}

float eval(const SDFBase &p_sdf, const VoxelPos &p_voxel, int p_lod) {
    WorldPos p;
    p.voxel = p_voxel;
    return p_sdf.evaluate(p, p_lod);
}

// The LOD-p_lod node holding voxel p_voxel (LOD0 chunks span 2 * half_chunk_voxels voxels, odd ids).
ChunkId node_at(const VoxelPos &p_voxel, int p_lod, const ChunkParam &p_param) {
    const int64_t chunk = 2 * p_param.half_chunk_voxels();
    ChunkId id(2 * WorldCoord::floor_div(p_voxel.x, chunk) + 1, 2 * WorldCoord::floor_div(p_voxel.y, chunk) + 1, 2 * WorldCoord::floor_div(p_voxel.z, chunk) + 1);
    return ChunkMath::get_parent_from_child_until(id, p_lod);
}

struct Claim {
    int64_t corners = 0; // corners whose ball was tested
    int64_t near_instance = 0; // of them, with the union records' bound
    int64_t points = 0;
    int64_t violations = 0; // points in a ball where a field was not solid
    double worst = 0.0; // the most positive value found there (metres)
};

// The discoverer's claim at LOD p_lod: a corner below the surface with value v < 0 gives a ball of radius
// (-v - deviation) / lipschitz - one cell diagonal of p_lod + 1, at most MAX_OCCLUDER_SPANS node spans (the bound
// of the node holding it), where the field of every LOD from p_lod - 2 to p_lod + 1 is solid at the lattice
// points. Corners at random depths (up to 3 node spans) under random columns of a p_window_m square around
// p_centre.
Claim check_claim(const SDFBase &p_sdf, const ChunkParam &p_param, int p_lod_count, int p_lod, double p_centre_x_m, double p_centre_z_m, double p_window_m, uint32_t p_seed) {
    Claim claim;
    const double vs = p_param.voxel_size;
    const double span_m = double(p_param.chunk_voxel_dimension()) * std::ldexp(vs, p_lod);
    const double cell_diag = std::sqrt(3.0) * std::ldexp(vs, p_lod + 1);
    Lcg rng{ p_seed * 0x9E3779B97F4A7C15ULL + uint64_t(p_lod) };
    for (int column = 0; column < 200; column++) {
        const int64_t x = snap((p_centre_x_m + (rng.next() - 0.5) * p_window_m) / vs, p_lod);
        const int64_t z = snap((p_centre_z_m + (rng.next() - 0.5) * p_window_m) / vs, p_lod);
        // The surface under the column: a few first-order steps from the datum.
        double y = SDFBase::DATUM_M / vs;
        for (int step = 0; step < 4; step++) {
            y -= double(eval(p_sdf, VoxelPos(x, snap(y, p_lod), z), p_lod)) / vs;
        }
        const VoxelPos c(x, snap(y - rng.next() * 3.0 * span_m / vs, p_lod), z);
        const float v = eval(p_sdf, c, p_lod);
        const SDFBase::OccluderBound bound = p_sdf.occluder_bound(node_at(c, p_lod, p_param), p_lod);
        if (!(v < 0.0f) || !(bound.lipschitz > 0.0f)) {
            continue;
        }
        const double radius = MIN((-double(v) - bound.deviation_m) / bound.lipschitz - cell_diag, SDFBase::MAX_OCCLUDER_SPANS * span_m);
        if (radius <= 0.0) {
            continue;
        }
        claim.corners++;
        claim.near_instance += bound.lipschitz > SDFArchetype::GROUND_LIPSCHITZ ? 1 : 0;
        for (int l = MAX(p_lod - 2, 0); l <= MIN(p_lod + 1, p_lod_count - 1); l++) {
            for (int s = 0; s < 8; s++) {
                // A uniform point of the ball.
                double d[3];
                double n2;
                do {
                    for (double &a : d) {
                        a = rng.next() * 2.0 - 1.0;
                    }
                    n2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                } while (n2 > 1.0 || n2 < 1e-6);
                const double k = radius / vs;
                const VoxelPos q(snap(double(c.x) + d[0] * k, l), snap(double(c.y) + d[1] * k, l), snap(double(c.z) + d[2] * k, l));
                const float w = eval(p_sdf, q, l);
                claim.points++;
                if (w > 0.0f) {
                    claim.violations++;
                    claim.worst = MAX(claim.worst, double(w));
                }
            }
        }
    }
    return claim;
}

} // namespace

// The archetype demo at 5 cm: a forest (union records, L from the crowns' bumps and the branches' taper) on the
// fBm ground, and the bare ground. Corners near the cave mouth (a remover) are exempt.
TEST_CASE("SDFArchetype: occluder balls are solid at every LOD they claim") {
    for (const int trees : { 60, 0 }) {
        CAPTURE(trees);
        ChunkParam param;
        param.voxel_size = 0.05f;
        SDFArchetype sdf;
        sdf.set_forest(trees, 40.0);
        param.skirt_depth = MAX(param.skirt_depth, sdf.min_skirt_depth(param, 15));
        sdf.configure(param, 15);
        int64_t corners = 0;
        int64_t near_instance = 0;
        // Around the demo objects (rocks, trunks, the cave, the forest), then bare ground 3 km away.
        for (const double centre : { -100.0, 3000.0 }) {
            for (int lod = 0; lod <= 10; lod++) {
                CAPTURE(centre);
                CAPTURE(lod);
                const Claim c = check_claim(sdf, param, 15, lod, centre + 50.0, centre - 50.0, 40.0 * std::ldexp(1.0, lod / 2), 0xB0u + uint32_t(trees));
                CAPTURE(c.points);
                CAPTURE(c.worst);
                CHECK(c.violations == 0);
                corners += c.corners;
                near_instance += c.near_instance;
            }
        }
        MESSAGE("trees ", trees, ": ", corners, " corners tested, ", near_instance, " near instances (lipschitz ", sdf.get_snapshot()->union_lipschitz, ")");
        CHECK(corners - near_instance > 300); // the ground's bound
        CHECK(near_instance > 50); // the union records' (the demo's rocks and trunks, the trees)
    }
}

TEST_CASE("SDFCascade: occluder balls are solid at every LOD they claim") {
    ChunkParam param;
    param.voxel_size = 0.5f;
    SDFCascade sdf;
    sdf.configure(param, 15);
    int64_t corners = 0;
    for (int lod = 0; lod <= 12; lod++) {
        CAPTURE(lod);
        const Claim c = check_claim(sdf, param, 15, lod, 0.0, 0.0, 200.0 * std::ldexp(1.0, lod), 0xCA5u);
        CAPTURE(c.points);
        CAPTURE(c.worst);
        CHECK(c.violations == 0);
        corners += c.corners;
    }
    MESSAGE(corners, " corners tested");
    CHECK(corners > 500);
}

// The demo's cave mouth subtracts: it is a remover, and nodes near it give no occluder.
TEST_CASE("SDFArchetype: removers exempt their neighbourhood from occluders") {
    ChunkParam param;
    param.voxel_size = 0.05f;
    SDFArchetype sdf;
    param.skirt_depth = MAX(param.skirt_depth, sdf.min_skirt_depth(param, 15));
    sdf.configure(param, 15);
    const auto snap = sdf.get_snapshot();
    CHECK(snap->removers.size() == 1);
    if (snap->removers.empty()) {
        return;
    }
    const VoxelPos at = snap->records[snap->removers[0]].anchor;
    CHECK(sdf.occluder_bound(node_at(at, 0, param), 0).lipschitz == 0.0f);
    CHECK(sdf.occluder_bound(node_at(at, 6, param), 6).lipschitz == 0.0f);
    CHECK(sdf.occluder_bound(node_at(at + VoxelPos(100000, 0, 0), 0, param), 0).lipschitz == SDFArchetype::GROUND_LIPSCHITZ);
    CHECK(snap->union_lipschitz >= 1.0f);
}
