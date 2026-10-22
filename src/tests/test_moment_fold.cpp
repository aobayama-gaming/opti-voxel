#include "test_common.hpp"

#include "chunk_palette.hpp"
#include "moment_layers.hpp"
#include "sdf_cascade.hpp"

#include <chrono>
#include <cstring>
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
        sdf.set_stream_sources(false); // the whole build's blobs, the table's records
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

// A 5 cm forest of 120 trees, with or without its moment blobs, and its trees' record ids.
void small_forest(SDFCascade &r_sdf, bool p_moments, std::vector<uint32_t> &r_parts) {
    ChunkParam param;
    param.voxel_size = 0.05f;
    r_sdf.set_forest(120, 60.0);
    r_sdf.set_stream_sources(false); // its records are edited by id
    r_sdf.set_moment_blobs(p_moments);
    param.skirt_depth = MAX(param.skirt_depth, r_sdf.min_skirt_depth(param, 18));
    r_sdf.configure(param, 18);
    const std::shared_ptr<const InstanceSnapshot> snap = r_sdf.get_snapshot();
    for (uint32_t i = 0; i < snap->records.size(); i++) {
        if (snap->records[i].op != Combine::ACCUM) {
            r_parts.push_back(i);
        }
    }
}

// Edit p_step of a scripted sequence: a tree part removed, one moved by up to 10 m, or a 1-3 m sphere added near the
// forest; its time (microseconds).
int64_t edit_step(SDFCascade &p_sdf, const std::vector<uint32_t> &p_parts, uint32_t p_step) {
    const int64_t cx = std::llround(SDFTable::SCENE_X / 0.05), cz = std::llround(SDFTable::SCENE_Z / 0.05);
    const auto t0 = std::chrono::steady_clock::now();
    const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ p_step, 0xED17u, 0x5u });
    const uint32_t id = p_parts[h.x % p_parts.size()];
    if (p_step % 3 == 0) {
        p_sdf.remove_instance(int(id));
    } else if (p_step % 3 == 1) {
        const TableRecord rec = p_sdf.get_snapshot()->records[id];
        p_sdf.move_instance(int(id), rec.anchor + VoxelPos(int64_t(h.y % 401u) - 200, 0, int64_t(h.z % 401u) - 200), nullptr);
    } else {
        const int64_t x = cx + int64_t(h.y % 1000u) - 500, z = cz + int64_t(h.z % 1000u) - 500;
        const float params[8] = { 20.0f + float(h.x % 40u), 0, 0, 0, 0, 0, 0, 0 }; // type 0: a sphere
        CHECK(p_sdf.add_instance(0, VoxelPos(x, p_sdf.ground_y(x, z) + 200, z), nullptr, params, int(Combine::SMOOTH_UNION), 4.0f) >= 0);
    }
    return int64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
}

} // namespace

TEST_CASE("Moment edits: after adds, moves and removes the table holds the blobs a whole rebuild makes") {
    SDFCascade sdf;
    std::vector<uint32_t> parts;
    small_forest(sdf, true, parts);
    int64_t compared = 0;
    int64_t edit_us = 0;
    for (uint32_t step = 0; step < 40; step++) {
        edit_us += edit_step(sdf, parts, step);
        if (step % 10 != 9) {
            continue;
        }
        std::vector<MomentBlob> reference;
        sdf.moment_blobs_reference(reference);
        const std::shared_ptr<const InstanceSnapshot> snap = sdf.get_snapshot();
        std::vector<const TableRecord *> held;
        for (uint32_t i = 0; i < snap->records.size(); i++) {
            if (snap->records[i].op == Combine::ACCUM && !snap->records[i].removed) {
                held.push_back(&snap->records[i]);
            }
        }
        CAPTURE(step);
        CHECK(held.size() == reference.size());
        // Every reference blob is held, bit for bit (anchor, frame, params, material, LOD).
        int64_t missing = 0;
        for (const MomentBlob &b : reference) {
            bool found = false;
            for (const TableRecord *r : held) {
                found = found || (r->anchor == b.anchor && int(r->min_lod) == b.lod && r->material == b.material && std::memcmp(r->rotation, b.rotation, sizeof(b.rotation)) == 0 && std::memcmp(r->params, b.inv_s2, sizeof(b.inv_s2)) == 0 && ExactNoise::float_bits(r->params[3]) == ExactNoise::float_bits(b.amplitude));
            }
            missing += found ? 0 : 1;
        }
        CHECK(missing == 0);
        compared += int64_t(reference.size());
    }
    // The same edits without the blobs: what they add to an edit.
    SDFCascade bare;
    std::vector<uint32_t> bare_parts;
    small_forest(bare, false, bare_parts);
    int64_t bare_us = 0;
    for (uint32_t step = 0; step < 40; step++) {
        bare_us += edit_step(bare, bare_parts, step);
    }
    MESSAGE("moment edits: ", compared, " blobs compared with a whole rebuild over 40 edits; ", edit_us / 40, " us per edit (", bare_us / 40,
            " without the blobs)");
    CHECK(compared > 0);
}

TEST_CASE("Moment fold: the blob type's program is the blob's occupancy, in the record's frame") {
    const TableDefPtr def = make_moment_blob_type();
    Moments m = box_moments(30.0, 4.0, 18.0);
    std::vector<std::array<float, 9>> rots;
    rotations(rots);
    m = m.rotated(rots.back().data());
    MomentItem item;
    item.moments = m;
    prepare_moment_item(item);
    const MomentBlob blob = make_blob(m, item.area, VoxelPos(int64_t(1) << 50, 77, -9000), 6, 1, 0.3 * 64.0, 4.0);
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
    // Phi over the iso at a near-flat maximum (32 cells first-order): inside, at most MOMENT_DEPTH_CELLS deep (the first-order distance read
    // 10 cells there, next to samples on the surface: a LOD step past the skirt in the 5 cm forest).
    const Float4 crest = moment_surface(air, Float4{ 0.0f, 0.0f, 0.0f, 1.5f * MOMENT_ISO }, lod, 7, &material);
    CHECK(crest.w < 0.0f);
    CHECK(crest.w == -MOMENT_DEPTH_CELLS * h);
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

TEST_CASE("Moment fold: the forest never thins out from its parts to its blobs") {
    const Forest with(true), without(false);
    const int64_t cx = std::llround(SDFTable::SCENE_X / 0.05), cz = std::llround(SDFTable::SCENE_Z / 0.05);
    // The same 48 columns of the middle 60 m at every LOD: the share with a canopy (solid 5 m above the ground) and
    // its mean height. The blobs taking only the parts gone whole left a trough (26 of 48 at LOD 5, 34 at LOD 2): the
    // cover never falls under nine tenths of LOD 2's.
    int reference = 0;
    for (int lod = 2; lod <= 9; lod++) {
        int canopy = 0, bare = 0;
        double height = 0.0;
        const int columns = 48;
        for (int i = 0; i < columns; i++) {
            const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ uint32_t(i), 0xC0E5u, 0u });
            const int64_t x = cx + int64_t(h.x % 1200u) - 600, z = cz + int64_t(h.y % 1200u) - 600;
            const int64_t ground = with.sdf.ground_y(x, z);
            const int64_t top = with.top_solid(x, z, ground + 1200, lod);
            if (top > ground + 100) {
                canopy++;
                height += double(top - ground) * 0.05;
            }
            bare += without.top_solid(x, z, ground + 1200, lod) > ground + 100 ? 1 : 0;
        }
        MESSAGE("LOD ", lod, ": canopy ", canopy, " / ", columns, " columns (", bare, " without blobs), mean height ",
                canopy > 0 ? height / canopy : 0.0, " m");
        reference = lod == 2 ? canopy : reference;
        CAPTURE(lod);
        CHECK(canopy * 10 >= reference * 9);
    }
}
