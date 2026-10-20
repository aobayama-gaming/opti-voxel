#include "test_common.hpp"

#include "chunk_palette.hpp"
#include "moment_layers.hpp"
#include "sdf_cascade.hpp"

#include <string>

using namespace Archetype;

namespace {

// A dense 5 cm forest (crowns touching) with or without its moment blobs.
struct Forest {
    ChunkParam param;
    SDFCascade sdf;
    explicit Forest(bool p_moments) {
        param.voxel_size = 0.05f;
        sdf.set_forest(400, 100.0);
        sdf.set_moment_blobs(p_moments);
        param.skirt_depth = MAX(param.skirt_depth, sdf.min_skirt_depth(param, 18));
        sdf.configure(param, 18);
    }
    // The highest solid lattice point of LOD p_lod in column (x, z), scanning down from p_top (LOD0 voxels).
    int64_t top_solid(int64_t p_x, int64_t p_z, int64_t p_top, int p_lod) const {
        const int64_t unit = int64_t(1) << p_lod;
        WorldPos p;
        p.voxel = VoxelPos(WorldCoord::floor_div(p_x, unit) * unit, WorldCoord::floor_div(p_top, unit) * unit, WorldCoord::floor_div(p_z, unit) * unit);
        while (sdf.evaluate(p, p_lod) > 0.0f && p.voxel.y > p_top - 4000) {
            p.voxel.y -= unit;
        }
        return p.voxel.y;
    }
};

} // namespace

TEST_CASE("Moment fold: the blob type's program is the blob's occupancy, in the record's frame") {
    const TableDefPtr def = make_moment_blob_type();
    Moments m = box_moments(30.0, 4.0, 18.0);
    std::vector<std::array<float, 9>> rots;
    rotations(rots);
    m = m.rotated(rots.back().data());
    const MomentBlob blob = make_blob(m, VoxelPos(int64_t(1) << 50, 77, -9000), 6, 1, 0.3 * 64.0);
    TableRecord rec;
    rec.anchor = blob.anchor;
    std::copy(blob.rotation, blob.rotation + 9, rec.rotation);
    std::copy(blob.inv_s2, blob.inv_s2 + 3, rec.params);
    rec.params[3] = blob.amplitude;
    const DefRecord dr = rec.def_record();
    std::vector<Float4> regs;
    double worst = 0.0;
    int inside = 0;
    for (int i = 0; i < 2000; i++) {
        const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ uint32_t(i), 0xF01Du, 0u });
        const double off[3] = { (ExactNoise::unit_of(h.x) - 0.5) * 2.2 * blob.s[2], (ExactNoise::unit_of(h.y) - 0.5) * 2.2 * blob.s[2],
            (ExactNoise::unit_of(h.z) - 0.5) * 2.2 * blob.s[2] };
        run_registers(def->program, WorldCoord::offset_pos(blob.anchor, off), 6, regs, &dr);
        const double got = regs[def->program.output].w, want = blob.occupancy(off);
        worst = MAX(worst, std::abs(got - want));
        inside += want > 0.0 ? 1 : 0;
    }
    MESSAGE("blob program vs occupancy: worst difference ", worst, " (amplitude ", blob.amplitude, "), ", inside, " of 2000 points inside");
    CHECK(worst < 1e-4 * double(blob.amplitude));
    CHECK(inside > 100);
}

TEST_CASE("Moment fold: the far-field surface lifts the ground, draws the canopy at MOMENT_ISO and takes the blobs' material") {
    const int lod = 5;
    const float h = ExactNoise::exact_pow2(lod);
    const Float4 air{ 0.0f, h, 0.0f, 100.0f * h }; // far above the ground
    const Float4 ground{ 0.0f, h, 0.0f, 0.25f * h }; // a quarter cell above it
    auto phi = [](float p_v) { return Float4{ 0.0f, -0.01f, 0.0f, p_v }; }; // falling with height
    uint32_t material = 0;
    // In the air, the surface is where phi reaches MOMENT_ISO.
    CHECK(moment_surface(air, phi(MOMENT_ISO), lod, 7, &material).w == 0.0f);
    CHECK(moment_surface(air, phi(0.5f * MOMENT_ISO), lod, 7, &material).w > 0.0f);
    CHECK(moment_surface(air, phi(2.0f * MOMENT_ISO), lod, 7, &material).w < 0.0f);
    CHECK(material == 7);
    // Flat phi far from any surface: at most 64 cells, never solid.
    material = 0;
    const Float4 flat = moment_surface(air, Float4{ 0.0f, 0.0f, 0.0f, 0.01f }, lod, 7, &material);
    CHECK(flat.w > 0.0f);
    CHECK(flat.w <= 64.0f * h * 1.0001f); // 64 cells where phi is flat
    CHECK(material == 0); // under MOMENT_MATERIAL_ISO: the ground's
    // Near the ground, under MOMENT_ISO, the lift: phi lowers the field by phi cells (the surface rises), and from
    // MOMENT_MATERIAL_ISO on the sample takes the blob's material.
    material = 0;
    const float low = 0.5f * (MOMENT_MATERIAL_ISO + MOMENT_ISO);
    const Float4 lifted = moment_surface(ground, phi(low), lod, 3, &material);
    CHECK(lifted.w == doctest::Approx(ground.w - low * h));
    CHECK(material == 3);
    // No blob material (an inheriting one): the register's stays.
    material = 5;
    moment_surface(ground, phi(low), lod, ChunkPalette::INHERIT_INDEX, &material);
    CHECK(material == 5);
}

TEST_CASE("Moment fold: a dense forest seen from far keeps a canopy, listed at the blobs' LOD only") {
    const Forest with(true), without(false);
    const int64_t cx = std::llround(SDFTable::SCENE_X / 0.05), cz = std::llround(SDFTable::SCENE_Z / 0.05);
    const std::shared_ptr<const InstanceSnapshot> snap = with.sdf.get_snapshot();
    int64_t blobs = 0;
    int64_t per_lod[32] = {};
    for (uint32_t i = 0; i < snap->records.size(); i++) {
        if (snap->records[i].op == Combine::ACCUM) {
            blobs++;
            per_lod[snap->records[i].min_lod]++;
        }
    }
    std::string layers;
    for (int l = 0; l < 18; l++) {
        layers += " " + std::to_string(per_lod[l]);
    }
    MESSAGE("forest: ", snap->records.size(), " records, ", blobs, " moment blobs; per LOD:", layers);
    CHECK(blobs > 0);
    for (const int lod : { 6, 7, 8 }) {
        int canopy = 0, bare = 0, columns = 0;
        for (int i = 0; i < 24; i++) {
            const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ uint32_t(i), uint32_t(lod), 0xCA70u });
            const int64_t x = cx + int64_t(h.x % 1200u) - 600, z = cz + int64_t(h.y % 1200u) - 600; // the middle 60 m
            const int64_t ground = with.sdf.ground_y(x, z);
            const int64_t top = ground + 1200; // 60 m up
            canopy += with.top_solid(x, z, top, lod) > ground + 100 ? 1 : 0; // solid 5 m above the ground
            bare += without.top_solid(x, z, top, lod) > ground + 100 ? 1 : 0;
            columns++;
        }
        MESSAGE("LOD ", lod, ": ", canopy, " of ", columns, " columns with a canopy (moments), ", bare, " without");
        CHECK(canopy > columns / 2);
        if (lod >= 7) {
            CHECK(bare == 0); // every crown dropped: nothing left without the blobs (at LOD 6 some are still drawn)
        }
        // The chunks over the forest list blobs of their own LOD only.
        std::vector<uint32_t> list;
        snap->chunk_list(ChunkMath::chunk_at_voxel(VoxelPos(cx, with.sdf.ground_y(cx, cz), cz), lod, with.param), list);
        int listed = 0;
        for (const uint32_t id : list) {
            if (snap->records[id].op == Combine::ACCUM) {
                CHECK(int(snap->records[id].min_lod) == lod);
                listed++;
            }
        }
        CHECK(listed > 0);
    }
}
