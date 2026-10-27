#include "test_common.hpp"

#include "cascade_glsl.hpp"
#include "godot_conv.hpp"
#include "gpu_frame.hpp"
#include "sdf_cascade.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstring>
#include <set>
#include <string>
#include <thread>

using namespace Archetype;
using ExactNoise::float_bits;

namespace {

// A configured cascade terrain with the --instances scatter (rocks and pillars around SDFTable::SCENE_X / Z), placed
// whole in the table: these tests are the index's, they count its records and lists and edit them by id.
// The scatter's records alone (p_moments false): the tests count records and lists; the moment blobs have their own.
struct ScatteredCascade {
    ChunkParam param;
    SDFCascade sdf;
    explicit ScatteredCascade(int p_count, double p_area = 200.0, int p_lods = 15, bool p_moments = false) {
        sdf.set_scatter(p_count, p_area);
        sdf.set_stream_sources(false);
        sdf.set_moment_blobs(p_moments);
        param.skirt_depth = MAX(param.skirt_depth, sdf.min_skirt_depth(param, p_lods));
        sdf.configure(param, p_lods);
    }
};

// sample_sdf at a LOD-p_lod lattice point with a given list.
Float4 sample_with(const SDFCascade &p_sdf, const InstanceSnapshot &p_snap, const std::vector<uint32_t> &p_list, const VoxelPos &p_voxel, int p_lod) {
    float ch[Cascade::CHANNELS];
    p_sdf.channels_at(p_voxel, p_lod, ch);
    WorldPos p;
    p.voxel = p_voxel;
    return p_sdf.sample_sdf(p_snap, p_list.data(), uint32_t(p_list.size()), ch, p, p_lod);
}

// The bare cascade at a LOD-p_lod lattice point (metres).
float bare_sdf(const SDFCascade &p_sdf, const VoxelPos &p_voxel, int p_lod) {
    float ch[Cascade::CHANNELS];
    p_sdf.channels_at(p_voxel, p_lod, ch);
    return p_sdf.sdf(ch, p_voxel, p_lod).w;
}

using ChunkMath::chunk_at_voxel;

// Admits p_chunks to the stream (a test's own owner) until every one is READY: their keys are then resident, and the
// CPU's point folds read them (else every point computes its keys and drops them). True when all got there.
bool make_resident(SDFCascade &r_sdf, const std::vector<ChunkId> &p_chunks) {
    constexpr uint64_t OWNER = 1;
    uint64_t words = 0;
    std::vector<ChunkId> retry; // LATER: asked again
    for (const ChunkId &c : p_chunks) {
        if (r_sdf.admit(c, OWNER, 0, 0, words) == SDFBase::Admit::LATER) {
            retry.push_back(c);
        }
    }
    std::vector<std::pair<ChunkId, uint64_t>> back;
    for (int i = 0; i < 100000 && (r_sdf.parked_count(OWNER) > 0 || !retry.empty()); i++) {
        r_sdf.poll();
        r_sdf.take_ready(OWNER, back);
        for (const std::pair<ChunkId, uint64_t> &b : back) {
            retry.push_back(b.first);
        }
        std::vector<ChunkId> again;
        for (const ChunkId &c : retry) {
            if (r_sdf.admit(c, OWNER, 0, 0, words) == SDFBase::Admit::LATER) {
                again.push_back(c);
            }
        }
        retry.swap(again);
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    return r_sdf.parked_count(OWNER) == 0 && retry.empty();
}

// The LOD-p_lod chunks whose span meets [p_lo, p_hi] (LOD0 voxels).
std::vector<ChunkId> chunks_over(const VoxelPos &p_lo, const VoxelPos &p_hi, int p_lod, const ChunkParam &p_param) {
    const ChunkId a = chunk_at_voxel(p_lo, p_lod, p_param), b = chunk_at_voxel(p_hi, p_lod, p_param);
    const int64_t step = int64_t(2) << p_lod;
    std::vector<ChunkId> out;
    for (int64_t x = a.x; x <= b.x; x += step) {
        for (int64_t y = a.y; y <= b.y; y += step) {
            for (int64_t z = a.z; z <= b.z; z += step) {
                out.push_back(ChunkId(x, y, z));
            }
        }
    }
    return out;
}

// Whether an instance whose box test passes at p_voxel at LOD p_lod is dropped at p_lod + 1 (the band rule), the streamed
// ones included.
bool dropped_at(const SDFCascade &p_sdf, const std::shared_ptr<const InstanceSnapshot> &p_snap, const VoxelPos &p_voxel, int p_lod) {
    thread_local std::vector<uint32_t> list;
    const RecordView records = p_sdf.point_list(*p_snap, p_voxel, p_lod, list);
    WorldPos p;
    p.voxel = p_voxel;
    for (uint32_t i : list) {
        const TableRecord &rec = records[i];
        // A moment blob is listed at one LOD by design, not dropped: its step is the skirt's to hide.
        if (rec.op != Combine::ACCUM && !rec.listed_at(p_lod + 1) && in_influence_box(rec, p, p_lod)) {
            return true;
        }
    }
    return false;
}

} // namespace

TEST_CASE("SDFCascade table: ground_at is the bare cascade's top solid voxel") {
    ScatteredCascade t(0);
    for (int i = 0; i < 48; i++) {
        const int64_t x = int64_t(i) * 977 - 20000;
        const int64_t z = int64_t(i) * -613 + 5000;
        const int64_t y = t.sdf.ground_at(x, z).y;
        CHECK(!(bare_sdf(t.sdf, VoxelPos(x, y, z), 0) > 0.0f));
        CHECK(bare_sdf(t.sdf, VoxelPos(x, y + 1, z), 0) > 0.0f);
        CHECK(t.sdf.ground_at(x, z).y == y); // deterministic
    }
}

TEST_CASE("SDFCascade table: the scatter stands on the ground, folded at lattice points; far from it the bare cascade, bit for bit") {
    ScatteredCascade t(200);
    const auto snap = t.sdf.get_snapshot();
    // One site per cell of the ground: about the count asked.
    CHECK(snap->records.size() >= 150);
    CHECK(snap->records.size() <= 250);
    std::vector<uint32_t> list;
    int64_t pillars = 0;
    for (uint32_t i = 0; i < snap->records.size(); i++) {
        const TableRecord &rec = snap->records[i];
        if (rec.def != 1) {
            continue; // the rocks are half sunk: solid either way
        }
        // Halfway up a pillar (a capsule on its column's ground): air in the bare cascade, solid with the table.
        WorldPos p;
        p.voxel = rec.anchor + VoxelPos(0, int64_t(0.5f * rec.params[0]), 0);
        CHECK(bare_sdf(t.sdf, p.voxel, 0) > 0.0f);
        CHECK(t.sdf.evaluate(p, 0) < 0.0f);
        snap->point_candidates(p.voxel, 0, list);
        CHECK(std::find(list.begin(), list.end(), i) != list.end());
        pillars++;
    }
    MESSAGE("scatter: ", snap->records.size(), " records, ", pillars, " pillars");
    CHECK(pillars * 10 >= int64_t(snap->records.size()) * 3); // about half: the site's variant picks
    CHECK(pillars * 10 <= int64_t(snap->records.size()) * 7);
    // 4 km away, at LODs 0-6: no candidate, the bare sdf().
    for (int lod = 0; lod <= 6; lod++) {
        for (int i = 0; i < 16; i++) {
            WorldPos p;
            const int64_t step = int64_t(1) << lod;
            p.voxel = VoxelPos((8000 + i * 37) * step, t.sdf.ground_at(8000 * step, 0).y / step * step, (i * 53 - 400) * step);
            snap->point_candidates(p.voxel, lod, list);
            CHECK(list.empty());
            CHECK(float_bits(t.sdf.evaluate(p, lod)) == float_bits(bare_sdf(t.sdf, p.voxel, lod)));
        }
    }
}

TEST_CASE("SDFCascade table: two neighbour chunks give the same bits at the samples they share, whatever their lists") {
    // The fold only finishes (renormalises) where an instance's box test passes: a pure function of the point, so a
    // seam sample is the same from either chunk even when one of them lists an instance the other doesn't.
    ScatteredCascade t(400, 120.0);
    const auto snap = t.sdf.get_snapshot();
    std::vector<uint32_t> list_a, list_b;
    int64_t compared = 0, different_lists = 0, folded = 0;
    for (int lod = 0; lod <= 4; lod++) {
        const int64_t step = int64_t(1) << lod;
        for (uint32_t r = 0; r < snap->records.size(); r += 9) {
            const TableRecord &rec = snap->records[r];
            const ChunkId a = chunk_at_voxel(rec.anchor, lod, t.param);
            for (int axis = 0; axis < 3; axis++) {
                ChunkId b = a;
                b[axis] += int64_t(2) << lod;
                snap->chunk_list(a, list_a);
                snap->chunk_list(b, list_b);
                different_lists += list_a != list_b ? 1 : 0;
                VoxelPos lo_a, hi_a, lo_b, hi_b;
                chunk_sample_box(a, t.param, lo_a, hi_a);
                chunk_sample_box(b, t.param, lo_b, hi_b);
                VoxelPos lo, hi;
                for (int c = 0; c < 3; c++) {
                    lo[c] = WorldCoord::floor_div(MAX(lo_a[c], lo_b[c]) + step - 1, step) * step;
                    hi[c] = MIN(hi_a[c], hi_b[c]);
                }
                // Every shared lattice point on the seam slab, a coarse grid across it.
                const int64_t grid = 7 * step;
                for (int64_t x = lo.x; x <= hi.x; x += axis == 0 ? step : grid) {
                    for (int64_t y = lo.y; y <= hi.y; y += axis == 1 ? step : grid) {
                        for (int64_t z = lo.z; z <= hi.z; z += axis == 2 ? step : grid) {
                            const VoxelPos v(x, y, z);
                            const Float4 sa = sample_with(t.sdf, *snap, list_a, v, lod);
                            const Float4 sb = sample_with(t.sdf, *snap, list_b, v, lod);
                            CHECK(same_bits(sa, sb));
                            folded += float_bits(sa.w) != float_bits(bare_sdf(t.sdf, v, lod)) ? 1 : 0;
                            compared++;
                        }
                    }
                }
            }
        }
    }
    MESSAGE("seams: ", compared, " shared samples, ", folded, " changed by an instance, ", different_lists, " chunk pairs with different lists");
    CHECK(different_lists > 0);
    CHECK(folded > 0);
}

TEST_CASE("SDFCascade table: the skirt covers the ground's step plus the instances' erosion, only with instances") {
    // Next to a rock smooth-unioned into the ground, a dropped octave and the rock's erosion both step between LOD L
    // and L+1 at the same seam: the skirt is their sum. With no instance, the bare cascade's skirt.
    ScatteredCascade bare(0);
    ScatteredCascade rocks(50);
    for (int l = 0; l < 15; l++) {
        const double cell_m = double(bare.param.voxel_size) * std::ldexp(1.0, l);
        CAPTURE(l);
        CHECK(double(rocks.sdf.skirt_depth_m(l)) == doctest::Approx(double(bare.sdf.skirt_depth_m(l)) + Archetype::ERODE_CELLS * cell_m).epsilon(1e-6));
    }
    SDFCascade unbanded;
    unbanded.set_scatter(50, 200.0);
    unbanded.set_instance_band(false);
    unbanded.configure(bare.param, 15);
    CHECK(unbanded.skirt_depth_m(7) == bare.sdf.skirt_depth_m(7)); // no band rule, no erosion
}

TEST_CASE("SDFCascade table: runtime edits on the cascade") {
    ScatteredCascade t(0);
    const int64_t x = 1000, z = -3000;
    const VoxelPos anchor(x, t.sdf.ground_at(x, z).y, z);
    const float params[8] = { 20.0f, 3.0f, 0, 0, 0, 0, 0, 0 };
    WorldPos p;
    p.voxel = anchor + VoxelPos(0, 10, 0);
    CHECK(t.sdf.evaluate(p, 0) > 0.0f);
    const int id = t.sdf.add_instance(1, anchor, nullptr, params, int(Combine::UNION), 0.0f);
    CHECK(id == 0);
    if (id != 0) {
        return;
    }
    CHECK(t.sdf.evaluate(p, 0) < 0.0f);
    CHECK(t.sdf.last_edit_box().valid);
    CHECK(t.sdf.edit_affects_chunk(chunk_at_voxel(p.voxel, 0, t.param)));
    CHECK(!t.sdf.edit_affects_chunk(chunk_at_voxel(p.voxel + VoxelPos(5000, 0, 0), 0, t.param)));
    CHECK(t.sdf.cull_exempt(chunk_at_voxel(p.voxel, 2, t.param), 2, 0.0f));
    CHECK(t.sdf.remove_instance(id));
    CHECK(t.sdf.evaluate(p, 0) > 0.0f);
}

// Edits replace the snapshot on the main thread while the discoverer thread evaluates and runs cull_exempt: a plain
// shared_ptr assignment there freed the snapshot under the reader (SIGSEGV).
TEST_CASE("SDFCascade table: evaluating on another thread while the main thread moves an instance") {
    ScatteredCascade t(2);
    const VoxelPos anchor = t.sdf.get_snapshot()->records[0].anchor;
    std::atomic<bool> stop{ false };
    std::atomic<int64_t> evaluations{ 0 };
    std::thread reader([&]() {
        uint32_t i = 0;
        while (!stop.load()) {
            const int lod = int(i % 3);
            const int64_t mask = ~((int64_t(1) << lod) - 1);
            const VoxelPos v = anchor + VoxelPos(int64_t(i % 17) - 8, int64_t(i % 13) - 6, int64_t(i % 11) - 5);
            WorldPos p;
            p.voxel = VoxelPos(v.x & mask, v.y & mask, v.z & mask);
            (void)t.sdf.evaluate(p, lod);
            (void)t.sdf.cull_exempt(ChunkId(1, 1, 1), 0, 0.0f);
            i++;
            evaluations.fetch_add(1);
        }
    });
    for (int step = 0; step < 2000; step++) {
        CHECK(t.sdf.move_instance(0, anchor + VoxelPos(step % 7, 0, step % 5), nullptr));
    }
    stop.store(true);
    reader.join();
    CHECK(evaluations.load() > 0);
}

TEST_CASE("SDFCascade table: the shapes gallery on the flat preset: cubes holed through, a pipped die, stairs, tunnels") {
    ChunkParam param;
    const double vs = param.voxel_size;
    SDFCascade sdf;
    sdf.preset = SDFCascade::Preset::FLAT;
    sdf.set_scene(SDFTable::Scene::SHAPES);
    sdf.set_instance_band(false); // as VoxelTerrain sets it for the gallery
    sdf.configure(param, 25);
    CHECK(sdf.min_skirt_depth(param, 25) == 0.0f);
    CHECK(sdf.skirt_depth_m(3) == doctest::Approx(double(param.skirt_depth) * vs * 8.0)); // full skirt_depth cells: hard creases
    for (int i = 0; i < sdf.get_library().size(); i++) {
        CHECK(sdf.get_library()[i].program.validate(25).empty());
    }
    const auto snap = sdf.get_snapshot();
    CHECK(snap->records.size() == 11); // 8 cubes, the die, the stairs, the tunnelled cube
    if (snap->records.size() != 11) {
        return;
    }
    for (uint32_t i = 0; i < 11; i++) {
        CHECK(snap->records[i].max_lod == TableRecord::NEVER_DROPPED); // whole at every LOD: no band
        CHECK(snap->records[i].erode == 0.0f);
    }
    // The SDF (metres, LOD 0) at the voxel nearest p_anchor + (p_x, p_y, p_z) metres.
    auto sdf_at = [&](const VoxelPos &p_anchor, double p_x, double p_y, double p_z) {
        WorldPos p;
        p.voxel = p_anchor + VoxelPos(std::llround(p_x / vs), std::llround(p_y / vs), std::llround(p_z / vs));
        return sdf.evaluate(p, 0);
    };
    // Far from the shapes: 10 m above the flat ground.
    const VoxelPos ground(0, sdf.ground_at(0, 0).y, 0);
    CHECK(ground.y == std::llround(SDFCascade::FLAT_GROUND_M / vs));
    CHECK(std::abs(sdf_at(ground, 5000.0, 10.0, 5000.0) - 10.0f) < 1e-3f);
    // Rings around vertical lines: the yaw of an instance doesn't move them.
    auto ring = [&](const VoxelPos &p_centre, double p_radius, double p_y) {
        float worst = -INFINITY;
        for (int a = 0; a < 16; a++) {
            const double phi = a * Math_PI / 8.0;
            worst = MAX(worst, sdf_at(p_centre, p_radius * std::cos(phi), p_y, p_radius * std::sin(phi)));
        }
        return worst;
    };
    // The ladder: air at the centre and up through the top face, solid in the wall around the top hole.
    for (int i = 0; i < 8; i++) {
        CAPTURE(i);
        const VoxelPos c = snap->records[i].anchor;
        const double h = 8.0 * double(1 << i);
        CHECK(sdf_at(c, 0.0, 0.0, 0.0) > 0.0f);
        CHECK(sdf_at(c, 0.0, 0.95 * h, 0.0) > 0.0f);
        CHECK(ring(c, 0.88 * h, 0.92 * h) < 0.0f);
    }
    // The die (half size 6 m): solid, but the top's one pip.
    const VoxelPos die = snap->records[8].anchor;
    CHECK(sdf_at(die, 0.0, 0.0, 0.0) < 0.0f);
    CHECK(sdf_at(die, 0.0, 4.0, 0.0) < 0.0f);
    CHECK(sdf_at(die, 0.0, 5.5, 0.0) > 0.0f);
    // The staircase: step k (2 m deep, along the record's local z) tops out at k + 1 m; local (0, 0, 2k) in world
    // axes is R^T of it.
    const TableRecord &stairs = snap->records[9];
    for (int k = 0; k < 8; k++) {
        CAPTURE(k);
        const double x = double(stairs.rotation[6]) * 2.0 * k, z = double(stairs.rotation[8]) * 2.0 * k;
        CHECK(sdf_at(stairs.anchor, x, k + 0.5, z) < 0.0f);
        CHECK(sdf_at(stairs.anchor, x, k + 1.5, z) > 0.0f);
    }
    // The tunnelled cube (half size 6 m, tunnels 2.5 m): air down the vertical tunnel, solid around it.
    const VoxelPos tunnels = snap->records[10].anchor;
    CHECK(sdf_at(tunnels, 0.0, 0.0, 0.0) > 0.0f);
    CHECK(sdf_at(tunnels, 0.0, 5.0, 0.0) > 0.0f);
    CHECK(ring(tunnels, 4.5, 5.0) < 0.0f);
}

// The band rule makes two LODs see different surfaces (an octave dropped, an instance eroded or dropped between
// them). A skirt pushes its chunk's rim towards air by skirt_depth_m, so at a seam between a LOD L and a LOD L+1
// chunk the field may differ by at most the skirt depth of the chunk whose surface is the outer one, or a crack
// shows. At the samples the two chunks share (Λ_{L+1}) near the LOD-L surface of columns over the scene. An instance
// dropped at L+1 is the exception: the step there has no bound, but what is left of it at L is at most `erode` cells
// thick, so the fine rim's skirt lifts it to air (the old fBm ground's deeper skirts hid that case too).
TEST_CASE("SDFCascade table: the field step between LOD L and L+1 stays within the skirt depth") {
    struct Scene {
        std::string name;
        float voxel;
        int rocks;
        int trees;
        double area_m;
        uint32_t columns;
        double above_m; // above the tallest instance
        ErodeRule rule;
        bool stream;
    };
    // Placed whole (the table's blobs) and streamed (the keys' records and blob tiles).
    const Scene scenes[] = { { "4,000-rock scatter", 0.5f, 4000, 0, 200.0, 300, 12.0, ErodeRule(), false },
        { "4,000-rock scatter, streamed", 0.5f, 4000, 0, 200.0, 300, 12.0, ErodeRule(), true },
        { "forest, streamed", 0.05f, 0, 60, 40.0, 100, 26.0, ErodeRule(), true },
        { "forest, resolve 1, rate 0.5, streamed", 0.05f, 0, 60, 40.0, 100, 26.0, ErodeRule{ 0.5f, 1.0f }, true } };
    for (const Scene &scene : scenes) {
        CAPTURE(scene.name);
        ChunkParam param;
        param.voxel_size = scene.voxel;
        SDFCascade sdf;
        sdf.set_scatter(scene.rocks, scene.area_m);
        sdf.set_forest(scene.trees, scene.area_m);
        sdf.set_stream_sources(scene.stream);
        sdf.set_instance_erode(scene.rule.rate);
        sdf.set_instance_resolve(scene.rule.resolve);
        param.skirt_depth = MAX(param.skirt_depth, sdf.min_skirt_depth(param, 15));
        sdf.configure(param, 15);
        const int64_t window = std::llround(scene.area_m / double(scene.voxel));
        const VoxelPos centre(std::llround(SDFTable::SCENE_X / double(scene.voxel)), 0, std::llround(SDFTable::SCENE_Z / double(scene.voxel)));
        const int64_t above = std::llround(scene.above_m / double(scene.voxel));
        const int64_t ground_c = sdf.ground_at(centre.x, centre.z).y;
        for (int lod = 0; lod <= 6; lod++) {
            CAPTURE(lod);
            if (scene.stream) {
                // The keys of every chunk the columns' samples fall in, at both LODs (the ground within a window's
                // quarter of the centre's).
                const VoxelPos lo(centre.x - window / 2 - 1, ground_c - above - window / 4, centre.z - window / 2 - 1);
                const VoxelPos hi(centre.x + window / 2 + 1, ground_c + above + window / 4, centre.z + window / 2 + 1);
                std::vector<ChunkId> chunks = chunks_over(lo, hi, lod, param);
                const std::vector<ChunkId> coarse = chunks_over(lo, hi, lod + 1, param);
                chunks.insert(chunks.end(), coarse.begin(), coarse.end());
                CHECK(make_resident(sdf, chunks));
            }
            const int64_t unit = int64_t(2) << lod; // Λ_{L+1}
            auto at = [&](const VoxelPos &p_column, int64_t p_y, int p_lod) {
                WorldPos p;
                p.voxel = VoxelPos(p_column.x, p_y, p_column.z);
                return double(sdf.evaluate(p, p_lod));
            };
            const std::shared_ptr<const InstanceSnapshot> snap = sdf.get_snapshot();
            double worst = 0.0;
            int64_t points = 0, drops = 0;
            for (uint32_t i = 0; i < scene.columns; i++) {
                const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ i, uint32_t(lod), 0x5EA4u });
                const VoxelPos column(WorldCoord::floor_div(centre.x + int64_t(h.x % uint32_t(window)) - window / 2, unit) * unit, 0,
                        WorldCoord::floor_div(centre.z + int64_t(h.y % uint32_t(window)) - window / 2, unit) * unit);
                const int64_t ground = sdf.ground_at(column.x, column.z).y;
                // The LOD-L top surface on Λ_{L+1}: scan down.
                int64_t y = WorldCoord::floor_div(ground + above, unit) * unit;
                while (at(column, y - unit, lod) > 0.0 && y > ground - above) {
                    y -= unit;
                }
                for (int64_t d = -2 * unit; d <= 2 * unit; d += unit) {
                    const double fine = at(column, y + d, lod);
                    if (std::fabs(fine) > double(param.voxel_size) * std::ldexp(1.0, lod)) {
                        continue; // only near the surface, where the skirt has to hide the step
                    }
                    const double step = at(column, y + d, lod + 1) - fine;
                    // step > 0: the coarse field is more air there, its surface the inner one: the fine rim must cover.
                    WorldPos shared;
                    shared.voxel = VoxelPos(column.x, y + d, column.z);
                    // The outer chunk's skirt there: its LOD's, plus the moment skirt where a moment blob touches the point.
                    const double depth_m = double(sdf.skirt_depth_at(shared, step > 0.0 ? lod : lod + 1));
                    if (step > depth_m && dropped_at(sdf, snap, VoxelPos(column.x, y + d, column.z), lod)) {
                        // A record listed at L, dropped at L+1: no skirt bounds that step, but the drop leaves at most
                        // `erode` cells of L of it (set_band), which the fine rim's skirt lifts to air: sealed.
                        CHECK(fine + double(sdf.skirt_depth_m(lod)) >= 0.0);
                        drops++;
                        continue;
                    }
                    worst = MAX(worst, std::fabs(step) / depth_m);
                    points++;
                }
            }
            MESSAGE(scene.name, ": LOD ", lod, " | ", lod + 1, ": ", points, " shared samples near the surface, worst step ", worst,
                    " of the outer chunk's skirt depth; ", drops, " past it over an instance dropped at LOD ", lod + 1, ", sealed by the fine rim's skirt");
            CHECK(points > 100);
            CHECK(worst <= 1.0);
        }
    }
}

// An edit only re-meshes the LODs the changed instance is listed at: a small rock added asks for its LOD 0 chunk,
// not for the coarse chunks over it (their lists never held it).
TEST_CASE("SDFCascade table: an edit re-meshes only the LODs its instance is listed at") {
    ScatteredCascade t(0);
    const float params[8] = { 4.0f, 0.4f, 0, 0, 0, 0, 0, 0 }; // a 2 m rock (sphere type 0: radius 4 voxels)
    const VoxelPos at(-104, t.sdf.ground_at(-104, -190).y, -190);
    const int id = t.sdf.add_instance(0, at, nullptr, params, int(Combine::UNION), 0.0f);
    CHECK(id >= 0);
    if (id < 0) {
        return;
    }
    const TableRecord &rec = t.sdf.get_snapshot()->records[id];
    MESSAGE("a radius-4 sphere is listed at LODs [0, ", int(rec.max_lod), ")");
    CHECK(rec.max_lod < 15);
    for (int lod = 0; lod < 15; lod++) {
        CAPTURE(lod);
        CHECK(t.sdf.edit_affects_chunk(chunk_at_voxel(at, lod, t.param)) == (lod < rec.max_lod));
    }
}

// The edit API refuses records the table can't hold: a smooth radius or a size of kilometres made the index place
// ~10^8 entries on the main thread (seconds, then out of memory), and anchors near +-2^63 overflowed the int64 boxes
// (undefined, and the GPU wraps differently). A refused edit changes nothing.
TEST_CASE("SDFCascade table: oversized or far records are refused, the table unchanged") {
    ScatteredCascade t(0);
    const VoxelPos at(-104, 600, -190);
    const float rock[8] = { 4.0f, 0.4f, 0, 0, 0, 0, 0, 0 };
    const int id = t.sdf.add_instance(0, at, nullptr, rock, int(Combine::UNION), 0.0f);
    CHECK(id >= 0);
    if (id < 0) {
        return;
    }
    const size_t count = t.sdf.get_snapshot()->records.size();
    // Each prints its expected error.
    CHECK(t.sdf.add_instance(0, at, nullptr, rock, int(Combine::SMOOTH_UNION), 1e8f) == -1);
    const float huge[8] = { 1e30f, 0.4f, 0, 0, 0, 0, 0, 0 };
    CHECK(t.sdf.add_instance(0, at, nullptr, huge, int(Combine::UNION), 0.0f) == -1);
    const VoxelPos far(int64_t(1) << 61, 0, 0);
    CHECK(t.sdf.add_instance(0, far, nullptr, rock, int(Combine::UNION), 0.0f) == -1);
    CHECK_FALSE(t.sdf.move_instance(id, far, nullptr));
    CHECK(t.sdf.get_snapshot()->records.size() == count);
    CHECK(t.sdf.get_snapshot()->records[id].anchor == at);
    // Big but within 4 top-LOD chunk spans (1.8M voxels at 15 LODs): taken.
    const float big[8] = { 1e5f, 0.4f, 0, 0, 0, 0, 0, 0 };
    CHECK(t.sdf.add_instance(0, at, nullptr, big, int(Combine::SMOOTH_UNION), 1e3f) >= 0);
    CHECK(t.sdf.add_instance(0, VoxelPos(int64_t(1) << 59, 0, 0), nullptr, rock, int(Combine::UNION), 0.0f) >= 0);
    CHECK(t.sdf.move_instance(id, at + VoxelPos(64, 0, 0), nullptr));
}

// Constant cost per chunk: at a fixed density of rocks, the longest list at each LOD doesn't grow with the number of
// rocks (10x more rocks on 10x the area), and from LOD 6 on every rock is eroded away and dropped: empty lists.
TEST_CASE("SDFCascade table: list lengths are flat in the number of instances at a fixed density") {
    int64_t longest[2][13] = {};
    for (int run = 0; run < 2; run++) {
        const int n = run == 0 ? 2000 : 20000;
        const double area = 200.0 * std::sqrt(double(n) / 2000.0);
        ScatteredCascade t(n, area);
        const std::shared_ptr<const InstanceSnapshot> snap = t.sdf.get_snapshot();
        std::vector<uint32_t> list;
        for (int lod = 0; lod <= 12; lod++) {
            for (uint32_t i = 0; i < 48; i++) {
                const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ i, uint32_t(lod), 0x7111u });
                const int64_t x = std::llround((SDFTable::SCENE_X + (double(h.x >> 8) / 16777216.0 - 0.5) * area) / 0.5);
                const int64_t z = std::llround((SDFTable::SCENE_Z + (double(h.y >> 8) / 16777216.0 - 0.5) * area) / 0.5);
                snap->chunk_list(chunk_at_voxel(VoxelPos(x, t.sdf.ground_at(x, z).y, z), lod, t.param), list);
                longest[run][lod] = MAX(longest[run][lod], int64_t(list.size()));
            }
        }
    }
    for (int lod = 0; lod <= 12; lod++) {
        CAPTURE(lod);
        MESSAGE("LOD ", lod, ": longest list ", longest[0][lod], " (2,000 rocks) vs ", longest[1][lod], " (20,000 on 10x the area)");
        if (lod <= 3) {
            // Up to LOD 3 a chunk (112 m) is smaller than the 200 m square: the same density, the same lists.
            CHECK(double(longest[1][lod]) <= 1.5 * double(longest[0][lod]) + 8.0);
        } else if (lod >= 6) {
            CHECK(longest[1][lod] == 0);
        }
    }
}

// Edits go through the overlay (a small index over the edited records, the base kept): after random adds, moves and
// removes, every chunk list and point query equals the one of a whole new index over the same records; past REBASE
// edited records the base is rebuilt. The GPU gets the pages an edit changed.
TEST_CASE("SDFCascade table: the edit overlay lists exactly what a whole new index lists") {
    ScatteredCascade t(3000, 150.0);
    SDFCascade &sdf = t.sdf;
    const std::shared_ptr<const InstanceSnapshot> first = sdf.get_snapshot();
    const int64_t y0 = first->records[0].anchor.y;
    int64_t lists = 0, entries = 0, edits = 0;
    std::vector<uint32_t> touched;
    for (uint32_t step = 0; step < 3000; step++) {
        const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ step, 0xED17u, 0x0FA7u });
        const uint32_t n = sdf.get_snapshot()->records.size();
        const VoxelPos at(-100 + int64_t(h.y % 300u) - 150, y0 + int64_t(h.z % 40u) - 20, -300 + int64_t(h.x % 300u) - 150);
        int id = -1;
        if (h.x % 3u == 0u) {
            const float params[8] = { 3.0f + float(h.y % 6u), 0, 0, 0, 0, 0, 0, 0 };
            id = sdf.add_instance(0, at, nullptr, params, int(h.z % 4u), 1.0f);
        } else if (h.x % 3u == 1u) {
            id = int(h.y % n);
            id = sdf.move_instance(id, at, nullptr) ? id : -1;
        } else {
            id = int(h.z % n);
            id = sdf.remove_instance(id) ? id : -1;
        }
        if (id < 0) {
            continue;
        }
        edits++;
        touched.push_back(uint32_t(id));
        if (h.y % 13u == 0u) {
            // A base built on the pool lands between edits, as in _process (or waited for).
            if (h.z % 2u == 0u) {
                sdf.finish_rebuild();
            } else {
                sdf.poll_rebuild();
            }
        }
        if (step % 200 != 199) {
            continue;
        }
        const std::shared_ptr<const InstanceSnapshot> snap = sdf.get_snapshot();
        const std::shared_ptr<const InstanceSnapshot> whole =
                build_snapshot(snap->library, snap->out_scale, snap->normalize, snap->records, snap->param, snap->max_lod, snap->version);
        std::vector<uint32_t> a, b;
        for (uint32_t i = 0; i < 64; i++) {
            const VoxelPos &v = snap->records[touched[(i * 7919u) % touched.size()]].anchor;
            for (int lod = 0; lod <= 6; lod++) {
                const ChunkId c = chunk_at_voxel(v, lod, t.param);
                snap->chunk_list(c, a);
                whole->chunk_list(c, b);
                CHECK(a == b);
                snap->point_candidates(v + VoxelPos(1, 2, 3), lod, a);
                whole->point_candidates(v + VoxelPos(1, 2, 3), lod, b);
                CHECK(a == b);
                CHECK(snap->near_instance(c, lod) == true); // an instance is there (or was: conservative)
                lists++;
                entries += int64_t(b.size());
            }
        }
        MESSAGE("after ", edits, " edits: overlay ", snap->overlay_ids.size(), " records, base ", std::string(snap->base == first->base ? "kept" : "rebuilt"));
    }
    MESSAGE("overlay vs whole index: ", lists, " chunk lists and point queries, ", entries, " entries");
    sdf.finish_rebuild();
    CHECK(sdf.get_snapshot()->base != first->base); // more than REBASE records edited: rebuilt at least once
    CHECK(entries > 1000);
    // An edit sends its page, not the table.
    const uint32_t before = sdf.get_snapshot()->version;
    const float params[8] = { 4.0f, 0, 0, 0, 0, 0, 0, 0 };
    CHECK(sdf.add_instance(0, VoxelPos(-100, y0, -300), nullptr, params, 0, 0.0f) >= 0);
    SDFGpuTable::UploadCursor now;
    const ov::bytes bytes = sdf.get_gpu_table()->record_bytes(SDFGpuTable::UploadCursor{ before, 0 }, now);
    CHECK(now.snap_version == before + 1);
    CHECK(ov::get_u32(bytes.data()) == 1u);
    CHECK(bytes.size() <= 4 + 8 + int64_t(RecordStore::PAGE) * RECORD_BYTES);
    // Two edits far apart in id (the first record, and an add on the last page) are two runs, not the records between
    // them; consecutive pages are one.
    RecordStore store;
    for (uint32_t i = 0; i < 3 * RecordStore::PAGE; i++) {
        store.push_back(TableRecord(), 1);
    }
    std::vector<std::pair<uint32_t, uint32_t>> runs;
    store.changed_since(1, runs);
    CHECK(runs.empty());
    store.set(5, TableRecord(), 2);
    store.push_back(TableRecord(), 2);
    store.changed_since(1, runs);
    CHECK(runs.size() == 2);
    CHECK((runs.size() == 2 && runs[0] == std::make_pair(0u, RecordStore::PAGE) && runs[1] == std::make_pair(3u * RecordStore::PAGE, 1u)));
    store.set(RecordStore::PAGE + 1, TableRecord(), 3);
    store.changed_since(1, runs);
    CHECK(runs.size() == 2); // pages 0 and 1 merge into one run
    store.changed_since(UINT32_MAX, runs);
    CHECK((runs.size() == 1 && runs[0].second == 3u * RecordStore::PAGE + 1u));
}

// A batch's lists fill the mesher's buffer at most: build_lists keeps the leading chunks whose lists fit whole and
// leaves the rest to a later batch (the viewer's forest drew bare squares where a short batch cut its lists: their
// blobs, last in id order, and every later chunk's whole list). Only one list over the whole buffer is truncated.
TEST_CASE("SDFCascade table: a batch keeps the chunks whose lists fit the buffer whole, the rest wait") {
    ScatteredCascade t(2000);
    const std::shared_ptr<const InstanceSnapshot> snap = t.sdf.get_snapshot();
    SDFGpuTable *table = t.sdf.get_gpu_table();
    std::vector<ChunkId> chunks;
    for (uint32_t r = 0; r < snap->records.size() && chunks.size() < 16; r += 7) {
        const ChunkId c = chunk_at_voxel(snap->records[r].anchor, 3, t.param);
        if (std::find(chunks.begin(), chunks.end(), c) == chunks.end()) {
            chunks.push_back(c);
        }
    }
    std::vector<std::vector<uint32_t>> want(chunks.size());
    size_t total = 0;
    for (size_t i = 0; i < chunks.size(); i++) {
        snap->chunk_list(chunks[i], want[i]);
        total += want[i].size();
    }
    CHECK(chunks.size() >= 8);
    CHECK(want[0].size() > 4);
    if (chunks.size() < 2 || want[0].size() <= 4) {
        return;
    }
    const auto encode = [&](ov::bytes &r_ids) {
        r_ids.resize(chunks.size() * GPUFrame::CHUNK_INPUT_BYTES);
        for (size_t i = 0; i < chunks.size(); i++) {
            GPUFrame::encode_chunk_input(r_ids, i, chunks[i], t.param);
        }
    };
    const int64_t truncated0 = table->get_stats()["list_truncated"].as_int();
    for (const size_t capacity : { total, total / 2, want[0].size() }) {
        CAPTURE(capacity);
        ov::bytes ids, words;
        encode(ids);
        std::vector<std::vector<uint32_t>> palettes;
        const uint32_t kept = table->build_lists(chunks, t.param, ids, words, uint32_t(capacity), &palettes);
        size_t used = 0;
        for (uint32_t i = 0; i < kept; i++) {
            const uint32_t offset = ov::get_u32(ids.data() + (int64_t(i) * GPUFrame::CHUNK_INPUT_BYTES + GPUFrame::CHUNK_LIST_OFFSET_AT));
            const int32_t count = ov::get_s32(ids.data() + (int64_t(i) * GPUFrame::CHUNK_INPUT_BYTES + GPUFrame::CHUNK_LIST_COUNT_AT));
            CHECK(offset == used);
            CHECK(size_t(count) == want[i].size());
            for (int32_t k = 0; k < count && size_t(k) < want[i].size(); k++) {
                CHECK((ov::get_u32(words.data() + (int64_t(offset + uint32_t(k)) * 4)) & SDFGpuTable::LIST_ID_MASK) == want[i][k]);
            }
            used += size_t(count);
        }
        CHECK(kept >= 1);
        CHECK(palettes.size() == kept);
        CHECK(used <= capacity);
        CHECK(int64_t(words.size()) == int64_t(used) * 4);
        if (kept < chunks.size()) {
            CHECK(used + want[kept].size() > capacity); // cut where the next list doesn't fit
        }
        if (capacity == total) {
            CHECK(kept == chunks.size());
        }
    }
    CHECK(table->get_stats()["list_truncated"].as_int() == truncated0);
    // A first list longer than the whole buffer: cut (counted), alone in its batch.
    ov::bytes ids, words;
    encode(ids);
    CHECK(table->build_lists(chunks, t.param, ids, words, uint32_t(want[0].size() - 1)) == 1u);
    CHECK(ov::get_s32(ids.data() + 28) == int32_t(want[0].size() - 1));
    CHECK(table->get_stats()["list_truncated"].as_int() == truncated0 + 1);
}

// A streamed source's chunk: its list (build_lists) is the scene's records, then the streamed ones (slots past the
// table's capacity), then the edits', and folded from the records the GPU holds (record_bytes: the snapshot and the
// placed keys' slots) it gives the CPU's point fold (evaluate) bit for bit at every sample: one compose_list for both,
// and a sample stays a pure function of the point.
TEST_CASE("SDFCascade table: a streamed chunk's list folds like the CPU's points: scene, streamed, then edit records") {
    ChunkParam param;
    param.voxel_size = 0.05f;
    SDFCascade sdf;
    sdf.set_forest(30, 40.0);
    param.skirt_depth = MAX(param.skirt_depth, sdf.min_skirt_depth(param, 18));
    sdf.configure(param, 18);
    const InstanceStream *stream = sdf.get_stream();
    CHECK(stream != nullptr);
    if (stream == nullptr) {
        return;
    }
    const int64_t x = std::llround(SDFTable::SCENE_X / 0.05), z = std::llround(SDFTable::SCENE_Z / 0.05);
    const VoxelPos centre(x, sdf.ground_at(x, z).y, z);
    // An edit over the centre: a sphere smooth-unioned into the canopy, after the streamed records in fold order.
    const uint32_t scene_end = sdf.get_snapshot()->records.size();
    const float sphere[8] = { 30.0f, 0, 0, 0, 0, 0, 0, 0 };
    const int edit = sdf.add_instance(0, centre + VoxelPos(0, 60, 0), nullptr, sphere, int(Combine::SMOOTH_UNION), 8.0f);
    CHECK(edit >= int(scene_end));

    SDFGpuTable *table = sdf.get_gpu_table();
    int64_t streamed = 0, edits = 0, folded = 0, points = 0, mismatches = 0;
    for (int lod = 0; lod <= 6; lod++) {
        CAPTURE(lod);
        const std::vector<ChunkId> chunks{ chunk_at_voxel(centre + VoxelPos(0, 40, 0), lod, param) };
        CHECK(make_resident(sdf, chunks));
        ov::bytes ids, words;
        ids.resize(GPUFrame::CHUNK_INPUT_BYTES);
        GPUFrame::encode_chunk_input(ids, 0, chunks[0], param);
        CHECK(table->build_lists(chunks, param, ids, words, 1u << 20) == 1u); // room for any list
        std::vector<uint32_t> list(ov::get_s32(ids.data() + 28));
        int phase = 0; // 0 scene, 1 streamed, 2 edits: never back
        for (size_t k = 0; k < list.size(); k++) {
            list[k] = ov::get_u32(words.data() + (int64_t(k) * 4)) & SDFGpuTable::LIST_ID_MASK;
            const int p = list[k] < scene_end ? 0 : (list[k] >= stream->slot_base() ? 1 : 2);
            CHECK(p >= phase);
            phase = p;
            streamed += p == 1;
            edits += p == 2;
        }
        // The records as the GPU holds them: every span record_bytes sends from nothing, the slots' after the snapshot's.
        std::vector<TableRecord> gpu(table->instance_capacity());
        const std::shared_ptr<const InstanceSnapshot> snap = sdf.get_snapshot();
        for (uint32_t i = 0; i < snap->records.size(); i++) {
            gpu[i] = snap->records[i];
        }
        std::vector<InstanceStream::Upload> uploads;
        stream->uploads_since(0, uploads);
        for (const InstanceStream::Upload &u : uploads) {
            std::copy(u.content->records.begin(), u.content->records.end(), gpu.begin() + u.first);
        }
        SDFGpuTable::UploadCursor now;
        const ov::bytes spans = table->record_bytes(SDFGpuTable::UploadCursor(), now);
        CHECK(now == table->upload_cursor());
        int64_t at = 4;
        ov::bytes one(RECORD_BYTES);
        for (uint32_t s = 0; s < ov::get_u32(spans.data()); s++) {
            const uint32_t first = ov::get_u32(spans.data() + at) / RECORD_BYTES, count = ov::get_u32(spans.data() + (at + 4)) / RECORD_BYTES;
            for (uint32_t i = 0; i < count; i++) {
                encode_record(one.data(), gpu[first + i]);
                mismatches += std::memcmp(one.data(), spans.data() + at + 8 + int64_t(i) * RECORD_BYTES, RECORD_BYTES) != 0;
            }
            at += 8 + int64_t(count) * RECORD_BYTES;
        }
        // Samples near the surface of the chunk's columns, on its lattice and in its sample box.
        VoxelPos lo, hi;
        chunk_sample_box(chunks[0], param, lo, hi);
        const int64_t unit = int64_t(1) << lod;
        for (uint32_t i = 0; i < 1500; i++) {
            const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ i, uint32_t(lod), 0x57EAu });
            WorldPos p;
            p.voxel.x = WorldCoord::floor_div(lo.x + int64_t(h.x % uint32_t(hi.x - lo.x + 1)), unit) * unit;
            p.voxel.z = WorldCoord::floor_div(lo.z + int64_t(h.y % uint32_t(hi.z - lo.z + 1)), unit) * unit;
            p.voxel.y = WorldCoord::floor_div(sdf.ground_at(p.voxel.x, p.voxel.z).y + int64_t(h.z % 400u) - 40, unit) * unit;
            if (p.voxel.x < lo.x || p.voxel.y < lo.y || p.voxel.z < lo.z || p.voxel.x > hi.x || p.voxel.y > hi.y || p.voxel.z > hi.z) {
                continue;
            }
            float ch[Cascade::CHANNELS];
            sdf.channels_at(p.voxel, lod, ch);
            const float from_list = sdf.sample_sdf(*snap, RecordView(gpu.data()), list.data(), uint32_t(list.size()), ch, p, lod).w;
            const float from_point = sdf.evaluate(p, lod);
            CHECK(float_bits(from_list) == float_bits(from_point));
            folded += from_list != sdf.sdf(ch, p.voxel, lod).w;
            points++;
        }
    }
    MESSAGE("streamed lists: ", points, " samples (", folded, " folded), ", streamed, " streamed and ", edits, " edit entries");
    CHECK(mismatches == 0);
    CHECK(streamed > 100);
    CHECK(edits >= 3);
    CHECK(folded > 500);
    CHECK(sdf.get_gpu_table()->get_stats()["stream"]["lists_missing"].as_int() == 0);
}

// The GLSL 4.50 words (spec 3.6) a shader text can't name a variable with and the cascade's shaders never use as
// keywords: the reserved ones and the qualifiers a short local name could hit. glslang refuses them at pipeline
// creation, and the cascade then falls back to the bare variant without a fold: the fold flag was named `active`.
static std::vector<std::string> glsl_reserved_words_used(const String &p_code) {
    static const std::set<std::string> RESERVED = { "common", "partition", "active", "asm", "class", "union", "enum",
        "typedef", "template", "this", "resource", "goto", "inline", "noinline", "public", "static", "extern",
        "external", "interface", "long", "short", "half", "fixed", "unsigned", "superp", "input", "output", "hvec2",
        "hvec3", "hvec4", "fvec2", "fvec3", "fvec4", "filter", "sizeof", "cast", "namespace", "using",
        "sampler3DRect", "sample", "patch", "subroutine", "flat", "smooth", "noperspective", "invariant", "precision",
        "lowp", "mediump", "highp", "centroid", "attribute", "varying", "coherent", "volatile" };
    std::string code = p_code.utf8().get_data();
    // Comments out (an unterminated one runs to the end).
    for (size_t i = code.find('/'); i != std::string::npos && i + 1 < code.size(); i = code.find('/', i)) {
        if (code[i + 1] != '/' && code[i + 1] != '*') {
            i++;
            continue;
        }
        const bool block = code[i + 1] == '*';
        const size_t end = block ? code.find("*/", i + 2) : code.find('\n', i);
        code.erase(i, end == std::string::npos ? std::string::npos : end - i + (block ? 2 : 0));
    }
    std::vector<std::string> used;
    std::string word;
    for (size_t i = 0; i <= code.size(); i++) {
        const char ch = i < code.size() ? code[i] : ' ';
        if (std::isalnum(uint8_t(ch)) || ch == '_') {
            word += ch;
            continue;
        }
        if (RESERVED.count(word)) {
            used.push_back(word);
        }
        word.clear();
    }
    return used;
}

// The fold variant's workgroups are cubes of samples with a shared list slice and barriers: every placeholder of the
// template is filled (the fold's return after the barriers too), its cube corners are the template's point (one text,
// twice), and the bare variants keep their returns and no barrier (their text is pinned in test_climate_shape.cpp).
// Without whole cubes (p_cubes false) the fold runs per sample, with no barrier.
TEST_CASE("SDFCascade table: the fold variant culls per cube, the bare variants keep their text") {
    ScatteredCascade t(64);
    const String point = "    p.lo = m_lo;\n    p.hi = m_hi;\n    i64_add(p.lo, p.hi, ivec3(i));\n    i64_shl(p.lo, p.hi, m);\n    p.frac = vec3(0.0);\n";
    for (const bool debug : { false, true }) {
        for (const CascadeGLSL::View view : { CascadeGLSL::View::GROUND, CascadeGLSL::View::FIELDS, CascadeGLSL::View::SAMPLES }) {
            for (const bool material : { false, true }) {
                const String fold = godot_str(CascadeGLSL::level_shader(&t.sdf, debug, false, true, material, view));
                const String sample = godot_str(CascadeGLSL::level_shader(&t.sdf, debug, false, true, material, view, false));
                const String bare = godot_str(CascadeGLSL::level_shader(&t.sdf, debug, false, false, material, view));
                for (const String &code : { fold, sample, bare }) {
                    CHECK(!code.contains("/*"));
                    CHECK(glsl_reserved_words_used(code) == std::vector<std::string>());
                    CHECK(code.contains(vformat("local_size_x = %d,", CascadeGLSL::LOCAL_SIZE)));
                }
                CHECK(fold.contains("if (!sample_live) {"));
                CHECK(fold.contains(vformat("shared uint s_fold_word[%d];", CascadeGLSL::LOCAL_SIZE)));
                CHECK(fold.contains("arch_meets_box(group_lo, group_hi, "));
                CHECK(fold.count(point) == 2);
                CHECK(bare.count(point) == 1);
                CHECK(sample.contains("arch_fold(u * s, p, lod, "));
                for (const String &code : { sample, bare }) {
                    CHECK(!code.contains("bool sample_live"));
                    CHECK(!code.contains("barrier()"));
                }
            }
        }
    }
    CHECK(glsl_reserved_words_used("float active; /* sample */ int x; // patch\nuint flat;") == std::vector<std::string>{ "active", "flat" });
    CHECK(glsl_reserved_words_used("uint ok; /* unterminated sample") == std::vector<std::string>());
}
