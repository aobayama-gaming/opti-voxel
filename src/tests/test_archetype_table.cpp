#include "test_common.hpp"

#include "archetype_table.hpp"
#include "cascade_glsl.hpp"
#include "instance_table.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <string>

using namespace Archetype;
using ExactNoise::exact_pow2;
using ExactNoise::float_bits;

namespace {

// p_anchor + an offset in [-p_range, p_range)^3 voxels, fraction in steps of 2^-18.
WorldPos random_point(const VoxelPos &p_anchor, double p_range, uint32_t p_i, uint32_t p_seed) {
    const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ p_i, p_seed, 0x1A57u });
    WorldPos p;
    const uint32_t u[3] = { h.x, h.y, h.z };
    float *frac[3] = { &p.frac.x, &p.frac.y, &p.frac.z };
    for (int a = 0; a < 3; a++) {
        const double v = (double(u[a] >> 8) * std::ldexp(1.0, -24) - 0.5) * 2.0 * p_range;
        const double fl = std::floor(v);
        p.voxel[a] = p_anchor[a] + int64_t(fl);
        *frac[a] = float(std::floor((v - fl) * 262144.0) / 262144.0);
    }
    return p;
}

uint32_t noise_salt(uint32_t p_seed) {
    return p_seed == 0 ? 0u : ExactNoise::pcg3d(ExactNoise::UVec3{ p_seed, 0x5EEDu, 0u }).x;
}

constexpr int ROCK_K = 10;
constexpr uint32_t ROCK_SEED = 0x50C4u;

TableDefPtr table_def(const std::string &p_type) {
    if (p_type == "sphere") {
        return make_sphere_type();
    }
    if (p_type == "box") {
        return make_box_type();
    }
    if (p_type == "capsule") {
        return make_capsule_type();
    }
    if (p_type == "branch") {
        return make_branch_type();
    }
    if (p_type == "cylinder") {
        return make_cylinder_type();
    }
    if (p_type == "rock") {
        return make_rock_type(ROCK_K, ROCK_SEED);
    }
    if (p_type == "holed_cube") {
        return make_swiss_cube_type(1);
    }
    if (p_type == "die") {
        return make_die_type();
    }
    if (p_type == "stairs") {
        return make_stairs_type();
    }
    if (p_type == "tunnel_cube") {
        return make_tunnel_cube_type();
    }
    if (p_type == "basin") {
        return make_basin_type(ROCK_K, ROCK_SEED);
    }
    if (p_type == "island") {
        return make_island_type(ROCK_K, ROCK_SEED);
    }
    return make_swiss_cube_type(4);
}

struct TypeCase {
    std::string name;
    float params[8];
};

// Every library type, with typical params (voxels).
const std::vector<TypeCase> &type_cases() {
    static const std::vector<TypeCase> cases = {
        { "sphere", { 7.0f, 0, 0, 0, 0 } },
        { "box", { 6.0f, 4.0f, 9.0f, 0, 0 } },
        { "capsule", { 20.0f, 3.0f, 0, 0, 0 } },
        { "branch", { 20.0f, 3.0f, -0.1f, 0, 0 } },
        { "branch", { 30.0f, 0.75f, 0.05f, 0, 0 } },
        { "cylinder", { 8.0f, 5.0f, 0, 0, 0 } },
        { "rock", { 12.0f, 2.0f, 0, 0, 0 } },
        { "swiss_cube", { 10.0f, 10.0f, 10.0f, 5.0f, 4.0f } },
        { "holed_cube", { 16.0f, 16.0f, 16.0f, 0.0f, 19.2f } },
        { "die", { 12.0f, 0, 0, 0, 0 } },
        { "stairs", { 8.0f, 2.0f, 2.0f, 0, 0 } },
        { "tunnel_cube", { 12.0f, 5.0f, 0, 0, 0 } },
        { "basin", { 9.0f, 14.0f, 0.5f, 0, 0 } },
        { "island", { 14.0f, 15.4f, 0.739940f, 0.672673f, 25.43f, 21.23f, 0.5f } },
    };
    return cases;
}

} // namespace

// The record carries the frame: the same local point (offset and fraction) gives the same bits at an anchor near 0,
// across the 2^32 low-word wrap and near 2^60, for every type, rotation and LOD (the noise lattice follows the
// anchor, INPUT is the exact sat_diff).
TEST_CASE("Archetype table: perlin3 octaves stay within the bounds the noisy types' boxes use") {
    float vmax = 0.0f;
    float gmax = 0.0f;
    for (uint32_t i = 0; i < 200000; i++) {
        const WorldPos p = random_point(VoxelPos(int64_t(i) << 20, 0, -int64_t(i) * 977), 4096.0, i, 5);
        const ExactNoise::Float4 n = ExactNoise::perlin3(p, 4, i * 31u, Vector3i64());
        vmax = MAX(vmax, std::fabs(n.w));
        gmax = MAX(gmax, std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z));
    }
    MESSAGE("perlin3 max |value| ", vmax, " |grad| per cell ", gmax);
    CHECK(vmax <= PERLIN3_MAX);
    CHECK(gmax <= PERLIN3_GRAD_MAX);
}

TEST_CASE("Archetype table: a library type gives the same bits for the same local point at any anchor") {
    const int64_t P32 = int64_t(1) << 32;
    const VoxelPos anchors[] = { VoxelPos(), VoxelPos(P32 - 3, -P32 + 2, P32 - 1),
        VoxelPos((int64_t(1) << 60) - 1001, 5, -(int64_t(1) << 60) + 999) };
    std::vector<std::array<float, 9>> rots;
    rotations(rots);
    int64_t compared = 0;
    for (const TypeCase &t : type_cases()) {
        const TableDefPtr def = table_def(t.name);
        CHECK(def->program.validate(25).empty());
        for (const std::array<float, 9> &rot : rots) {
            TableRecord rec;
            std::copy(rot.begin(), rot.end(), rec.rotation);
            std::memcpy(rec.params, t.params, sizeof(t.params));
            rec.salt = noise_salt(3);
            for (uint32_t i = 0; i < 300; i++) {
                const WorldPos local = random_point(VoxelPos(), 40.0, i, 0x7AB1u);
                const int lods[3] = { 0, 3, 24 };
                Float4 first[3] = {};
                for (size_t ai = 0; ai < sizeof(anchors) / sizeof(anchors[0]); ai++) {
                    rec.anchor = anchors[ai];
                    const DefRecord dr = rec.def_record();
                    WorldPos p = local;
                    p.voxel = local.voxel + anchors[ai];
                    for (int li = 0; li < 3; li++) {
                        const Float4 r = run(def->program, p, lods[li], &dr);
                        if (ai == 0) {
                            first[li] = r;
                        } else {
                            CAPTURE(t.name);
                            CHECK(same_bits(first[li], r));
                            compared++;
                        }
                    }
                }
            }
        }
    }
    MESSAGE("anchors: ", compared, " comparisons");
}

// The contract the locality rule and the band rule rest on, for every library type in its own frame: outside the
// local box its bounds give, the value is at least the distance to that box, and |grad| <= L everywhere.
TEST_CASE("Archetype table: every library type's box and Lipschitz bound hold") {
    for (const TypeCase &t : type_cases()) {
        const TableDefPtr def = table_def(t.name);
        Float3 lo{}, hi{};
        float lipschitz = 0.0f;
        def->bounds(t.params, lo, hi, lipschitz);
        TableRecord rec;
        std::memcpy(rec.params, t.params, sizeof(t.params));
        rec.salt = noise_salt(5);
        const DefRecord dr = rec.def_record();
        const double box_lo[3] = { lo.x, lo.y, lo.z };
        const double box_hi[3] = { hi.x, hi.y, hi.z };
        double half = 0.0;
        for (int a = 0; a < 3; a++) {
            half = std::max(half, std::max(std::fabs(box_lo[a]), std::fabs(box_hi[a])));
        }
        int64_t outside = 0, violations = 0;
        double worst_margin = INFINITY, worst_slope = 0.0;
        for (uint32_t i = 0; i < 100000; i++) {
            const WorldPos p = random_point(VoxelPos(), 1.5 * half + 4.0, i, 0xB0B5u);
            const Float4 r = run(def->program, p, 0, &dr);
            const double frac[3] = { p.frac.x, p.frac.y, p.frac.z };
            double d2 = 0.0;
            for (int a = 0; a < 3; a++) {
                const double q = double(p.voxel[a]) + frac[a];
                const double e = std::max(std::max(box_lo[a] - q, q - box_hi[a]), 0.0);
                d2 += e * e;
            }
            const double d = std::sqrt(d2);
            if (d > 0.0) {
                outside++;
                worst_margin = std::min(worst_margin, double(r.w) - d);
                violations += double(r.w) < d - 1e-4 * (d + 1.0) ? 1 : 0;
            }
            const double g = std::sqrt(double(r.x) * r.x + double(r.y) * r.y + double(r.z) * r.z);
            worst_slope = std::max(worst_slope, g / double(lipschitz));
            violations += g > double(lipschitz) * (1.0 + 1e-5) ? 1 : 0;
        }
        CAPTURE(t.name);
        MESSAGE(t.name, ": ", outside, " points outside the box (worst value - distance ", worst_margin, "), worst |grad| / L ", worst_slope);
        CHECK(outside > 1000);
        CHECK(violations == 0);
    }
}

TEST_CASE("Archetype table: the locality rule changes F only far from the surface, never its sign") {
    // A random table: every type, random params, rotations, top-level ops and smooth radii.
    DefLibrary lib;
    lib.add(make_sphere_type());
    lib.add(make_box_type());
    lib.add(make_capsule_type());
    lib.add(make_rock_type(8, 0x1234u));
    lib.add(make_branch_type());
    std::vector<TableRecord> records;
    const int64_t P32 = int64_t(1) << 32;
    const int64_t E = 1 << 20;
    for (uint32_t i = 0; i < 24; i++) {
        const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ i, 0x7AB1u, 0x9E37u });
        TableRecord rec;
        rec.def = uint8_t(h.x % uint32_t(lib.size()));
        rec.op = Combine(h.y % 4u);
        rec.k = float(h.z % 8u) + 0.5f;
        rec.inv_k = 1.0f / rec.k;
        const int64_t ax = int64_t(int32_t(ExactNoise::random_offset(i, 0, 1u, E)));
        const int64_t ay = int64_t(int32_t(ExactNoise::random_offset(i, 1, 2u, E)));
        const int64_t az = int64_t(int32_t(ExactNoise::random_offset(i, 2, 3u, E)));
        rec.anchor = VoxelPos(ax + (i % 3 == 0 ? P32 : 0), ay, az);
        const double yaw = double(h.x >> 8) * 1e-3;
        const float cy = float(std::cos(yaw)), sy = float(std::sin(yaw));
        const float rot[9] = { cy, 0, -sy, 0, 1, 0, sy, 0, cy };
        std::copy(rot, rot + 9, rec.rotation);
        switch (rec.def) {
            case 0:
                rec.params[0] = 3.0f + float(i % 5);
                break;
            case 1:
                rec.params[0] = 4.0f;
                rec.params[1] = 2.0f;
                rec.params[2] = 6.0f;
                break;
            case 2:
                rec.params[0] = 10.0f;
                rec.params[1] = 2.0f;
                break;
            case 3:
                rec.params[0] = 6.0f + float(i % 4);
                rec.params[1] = 1.5f;
                break;
            default:
                rec.params[0] = 14.0f;
                rec.params[1] = 3.0f;
                rec.params[2] = -0.12f;
                break;
        }
        rec.salt = noise_salt(i + 1);
        const Bounds wb = record_world_bounds(lib[rec.def], rec);
        for (int a = 0; a < 3; a++) {
            rec.box_lo[a] = int32_t(wb.lo[a] - rec.anchor[a]);
            rec.box_hi[a] = int32_t(wb.hi[a] - rec.anchor[a]);
        }
        rec.lipschitz = wb.lipschitz;
        rec.margin_c = 2.0f * wb.lipschitz + 2.0f;
        records.push_back(rec);
    }
    // A positive ground (air everywhere): a dropped far instance can then undercut the running value and
    // change F, which is what the rule's harmlessness claim is about.
    const Float4 ground{ 0.0f, 0.0f, 0.0f, 1000.0f };
    std::vector<uint32_t> list(records.size());
    for (uint32_t i = 0; i < list.size(); i++) {
        list[i] = i;
    }
    float lmax = 1.0f;
    for (const TableRecord &r : records) {
        lmax = std::max(lmax, r.lipschitz);
    }
    int64_t equal = 0, differ = 0, sign_flips = 0, points = 0;
    for (uint32_t i = 0; i < 60000; i++) {
        // A point near a random record: the rule bites in the halo just outside its box (and points far
        // from every instance, where F is the ground both ways).
        const TableRecord &near = records[i % records.size()];
        const Bounds wb = record_world_bounds(lib[near.def], near);
        double half = 0.0;
        for (int a = 0; a < 3; a++) {
            half = std::max(half, double(wb.hi[a] - wb.lo[a]) * 0.5);
        }
        const double range = half + 4.0 * double(near.margin_c) + 64.0;
        const WorldPos p = i % 5 == 0 ? random_point(VoxelPos(), 1.2e6, i, 0xB0Du) : random_point(near.anchor, range, i, 0xB0Du);
        for (const int lod : { 0, 4, 12 }) {
            const Float4 a = fold_onto(ground, lib, records.data(), list.data(), uint32_t(list.size()), p, lod, true);
            const Float4 b = fold_onto(ground, lib, records.data(), list.data(), uint32_t(list.size()), p, lod, false);
            points++;
            if ((a.w > 0.0f) != (b.w > 0.0f)) {
                sign_flips++;
            }
            const float threshold = 2.0f * lmax * exact_pow2(lod);
            if (float_bits(a.w) == float_bits(b.w)) {
                equal++;
                CHECK(same_bits(a, b));
            } else {
                differ++;
                // Any difference must be far from the surface: the smaller |value| must exceed the band.
                CHECK(std::max(std::fabs(a.w), std::fabs(b.w)) > threshold);
            }
        }
    }
    MESSAGE("locality: ", points, " points, ", equal, " bit-identical, ", differ, " differ, ", sign_flips, " sign flips");
    CHECK(points > 0);
    CHECK(differ > 0); // the rule does drop instances somewhere
    CHECK(sign_flips == 0);
}

TEST_CASE("Archetype table: random add / remove / move keep the index equal to a brute-force fold") {
    DefLibrary lib;
    lib.add(make_sphere_type());
    lib.add(make_box_type());
    lib.add(make_capsule_type());
    const Float4 ground{ 0.0f, 0.0f, 0.0f, 80.0f };
    ChunkParam param;
    param.voxel_size = 1.0f;

    std::vector<TableRecord> records;
    const int64_t E = 1 << 14;
    int64_t checked = 0;
    int64_t near_points = 0;
    for (uint32_t step = 0; step < 400; step++) {
        const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ step, 0xED17u, 0x0BADu });
        switch (h.x % 3u) {
            case 0: { // add
                TableRecord rec;
                rec.def = uint8_t(h.y % uint32_t(lib.size()));
                rec.op = Combine(h.z % 4u);
                rec.k = float(h.z % 6u) + 0.5f;
                rec.inv_k = 1.0f / rec.k;
                rec.anchor = VoxelPos(ExactNoise::random_offset(step, 1, 1u, E), ExactNoise::random_offset(step, 2, 2u, E),
                        ExactNoise::random_offset(step, 3, 3u, E));
                const double yaw = double(h.y >> 8) * 1e-3;
                const float cy = float(std::cos(yaw)), sy = float(std::sin(yaw));
                const float rot[9] = { cy, 0, -sy, 0, 1, 0, sy, 0, cy };
                std::copy(rot, rot + 9, rec.rotation);
                rec.params[0] = 3.0f + float(h.x % 6u);
                rec.params[1] = 2.0f;
                rec.salt = 0;
                const Bounds wb = record_world_bounds(lib[rec.def], rec);
                for (int a = 0; a < 3; a++) {
                    rec.box_lo[a] = int32_t(wb.lo[a] - rec.anchor[a]);
                    rec.box_hi[a] = int32_t(wb.hi[a] - rec.anchor[a]);
                }
                rec.lipschitz = wb.lipschitz;
                rec.margin_c = 2.0f * wb.lipschitz + param.skirt_depth;
                records.push_back(rec);
                break;
            }
            case 1: { // remove a live one
                std::vector<uint32_t> live;
                for (uint32_t i = 0; i < records.size(); i++) {
                    if (!records[i].removed) {
                        live.push_back(i);
                    }
                }
                if (!live.empty()) {
                    records[live[h.y % uint32_t(live.size())]].removed = true;
                }
                break;
            }
            default: { // move a live one
                std::vector<uint32_t> live;
                for (uint32_t i = 0; i < records.size(); i++) {
                    if (!records[i].removed) {
                        live.push_back(i);
                    }
                }
                if (!live.empty()) {
                    TableRecord &rec = records[live[h.y % uint32_t(live.size())]];
                    rec.anchor = VoxelPos(ExactNoise::random_offset(step, 5, 5u, E), ExactNoise::random_offset(step, 6, 6u, E),
                            ExactNoise::random_offset(step, 7, 7u, E));
                    const Bounds wb = record_world_bounds(lib[rec.def], rec);
                    for (int a = 0; a < 3; a++) {
                        rec.box_lo[a] = int32_t(wb.lo[a] - rec.anchor[a]);
                        rec.box_hi[a] = int32_t(wb.hi[a] - rec.anchor[a]);
                    }
                }
                break;
            }
        }

        auto snap = build_snapshot(std::make_shared<const DefLibrary>(lib), 1.0f, false, records, param, 12, step + 1);
        // The whole table in id order, the brute-force reference.
        std::vector<uint32_t> all;
        for (uint32_t i = 0; i < records.size(); i++) {
            if (!records[i].removed) {
                all.push_back(i);
            }
        }
        std::vector<uint32_t> candidates;
        std::vector<uint32_t> have_the_table;
        for (uint32_t i = 0; i < 40; i++) {
            const VoxelPos v(ExactNoise::random_offset(i, step, 11u, E), ExactNoise::random_offset(i, step, 12u, E),
                    ExactNoise::random_offset(i, step, 13u, E));
            WorldPos p;
            p.voxel = v;
            p.frac = Float3{};
            for (const int lod : { 0, 3, 8 }) {
                snap->point_candidates(v, lod, candidates);
                const Float4 indexed = fold_onto(ground, lib, records.data(), candidates.data(), uint32_t(candidates.size()), p, lod, true);
                const Float4 brute = fold_onto(ground, lib, records.data(), all.data(), uint32_t(all.size()), p, lod, true);
                CHECK(same_bits(indexed, brute));
                checked++;
            }
        }
        // Points next to the live instances (random points almost never are): within their influence box.
        for (uint32_t q : all) {
            for (uint32_t j = 0; j < 4; j++) {
                const WorldPos p = random_point(records[q].anchor, 24.0, j, step * 977u + q);
                for (const int lod : { 0, 3, 8 }) {
                    snap->point_candidates(p.voxel, lod, candidates);
                    const Float4 indexed = fold_onto(ground, lib, records.data(), candidates.data(), uint32_t(candidates.size()), p, lod, true);
                    const Float4 brute = fold_onto(ground, lib, records.data(), all.data(), uint32_t(all.size()), p, lod, true);
                    CHECK(same_bits(indexed, brute));
                    for (uint32_t r : all) {
                        if (in_influence_box(records[r], p, lod)) {
                            near_points++;
                            break;
                        }
                    }
                }
            }
        }
        // A few chunks: chunk_list is sorted and unique, and folds the same as the brute force at its own
        // LOD (the duplicate-instance bug the edit run found).
        std::vector<ChunkId> stored_nodes;
        snap->base->stored_at.for_each([&](const ChunkId &p_node, const IndexRange &) { stored_nodes.push_back(p_node); });
        for (size_t c = 0; c < stored_nodes.size() && c < 8; c++) {
            const ChunkId &key = stored_nodes[c];
            std::vector<uint32_t> cl;
            snap->chunk_list(key, cl);
            CHECK(std::is_sorted(cl.begin(), cl.end()));
            CHECK(std::adjacent_find(cl.begin(), cl.end()) == cl.end());
            const int lod = ChunkMath::get_lod(key);
            WorldPos p;
            p.voxel = ChunkMath::chunk_min_voxel(key, param);
            const Float4 from_chunk = fold_onto(ground, lib, records.data(), cl.data(), uint32_t(cl.size()), p, lod, true);
            const Float4 brute = fold_onto(ground, lib, records.data(), all.data(), uint32_t(all.size()), p, lod, true);
            CHECK(same_bits(from_chunk, brute));
        }
    }
    MESSAGE("edits: ", checked, " (index vs brute force) fold comparisons over 400 random add / remove / move steps, ", near_points,
            " points next to an instance");
    CHECK(checked > 0);
    CHECK(near_points > 10000);
}

// The base rebuilt on the pool while edits go on: whether it is taken early (waiting for the task), late, or by an
// edit past hard_overlay, every snapshot lists what a whole new index over its records lists, at its own version.
TEST_CASE("Archetype table: a base rebuilt on the pool lists what a whole new index lists") {
    DefLibrary lib;
    lib.add(make_sphere_type());
    lib.add(make_box_type());
    const std::shared_ptr<const DefLibrary> library = std::make_shared<const DefLibrary>(lib);
    ChunkParam param;
    param.voxel_size = 1.0f;
    const int64_t E = 1 << 12;
    auto place = [&](TableRecord &r_rec, uint32_t p_i, uint32_t p_salt) {
        r_rec.anchor = VoxelPos(ExactNoise::random_offset(p_i, 1, p_salt, E), ExactNoise::random_offset(p_i, 2, p_salt, E),
                ExactNoise::random_offset(p_i, 3, p_salt, E));
        finish_record(lib[r_rec.def], r_rec, param.skirt_depth, false);
    };
    const uint32_t n = 300;
    std::vector<TableRecord> initial(n);
    for (uint32_t i = 0; i < n; i++) {
        initial[i].def = uint8_t(i % 2u);
        initial[i].params[0] = 2.0f + float(i % 5u);
        initial[i].params[1] = 2.0f;
        place(initial[i], i, 7u);
    }
    uint32_t version = 1;
    RecordStore store(initial, version);
    std::shared_ptr<const InstanceSnapshot> snap = build_snapshot(library, 1.0f, false, store, param, 10, version);
    BaseRebuild rebuild(64, 96);
    int64_t starts = 0, early = 0, hard = 0, lists = 0;
    for (uint32_t step = 0; step < 6 * n; step++) {
        // Records in turn, so the overlay grows by one id per edit: a rebuild starts every ~64 edits.
        const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ step, 0xBA5Eu, 0x0A51u });
        const uint32_t id = (step * 7u) % n;
        TableRecord rec = store[id];
        place(rec, step, 11u);
        version++;
        store.set(id, rec, version);
        const bool was_running = rebuild.running();
        const std::shared_ptr<const IndexLayer> base_before = snap->base;
        snap = rebuild.edit(*snap, store, { id }, version);
        starts += !was_running && rebuild.running();
        hard += was_running && !rebuild.running() && snap->base != base_before;
        // Even rebuilds are taken early (waiting for the task), odd ones run until an edit passes hard_overlay.
        if (rebuild.running() && starts % 2 == 0 && h.y % 5u == 0u) {
            snap = rebuild.take(*snap, store, version);
            early++;
        }
        CHECK(snap->version == version);
        if (step % 7u != 0u) {
            continue;
        }
        const std::shared_ptr<const InstanceSnapshot> whole = build_snapshot(library, 1.0f, false, store, param, 10, version);
        std::vector<uint32_t> a, b;
        for (uint32_t i = 0; i < 24; i++) {
            const VoxelPos &v = store[(i * 7919u + step) % n].anchor;
            for (int lod = 0; lod <= 5; lod++) {
                const ChunkId c = ChunkMath::chunk_at_voxel(v, lod, param);
                snap->chunk_list(c, a);
                whole->chunk_list(c, b);
                CHECK(a == b);
                snap->point_candidates(v, lod, a);
                whole->point_candidates(v, lod, b);
                CHECK(a == b);
                CHECK(snap->near_instance(c, lod));
                lists++;
            }
        }
    }
    MESSAGE("pool rebuilds: ", starts, " started, ", early, " taken early, ", hard, " taken past hard_overlay; ", lists, " list comparisons");
    CHECK(starts > 1);
    CHECK(early > 0);
    CHECK(hard > 0);
}

TEST_CASE("Archetype table: encode_record lays the GPU DefRecord out at the documented offsets") {
    TableRecord rec;
    rec.def = 2;
    rec.op = Combine::SMOOTH_SUBTRACT;
    rec.k = 1.5f;
    rec.inv_k = 1.0f / 1.5f;
    rec.anchor = VoxelPos(0x1'0000'0007ll, -5, 3);
    rec.salt = 0xDEADBEEFu;
    rec.rotation[0] = 0.5f;
    rec.rotation[4] = 1.0f;
    rec.rotation[8] = 0.25f;
    rec.params[3] = 9.0f;
    rec.box_lo[0] = -7;
    rec.box_hi[2] = 11;
    rec.margin_c = 6.0f;
    ov::bytes bytes(RECORD_BYTES);
    encode_record(bytes.data(), rec);
    auto decode_u32 = [&bytes](int64_t p_at) { return ov::get_u32(bytes.data() + p_at); };
    auto decode_s32 = [&bytes](int64_t p_at) { return ov::get_s32(bytes.data() + p_at); };
    auto decode_float = [&bytes](int64_t p_at) { return ov::get_f32(bytes.data() + p_at); };
    CHECK(decode_u32(0) == 2u);
    CHECK(decode_u32(4) == uint32_t(Combine::SMOOTH_SUBTRACT));
    CHECK(decode_float(8) == rec.k);
    CHECK(decode_u32(16) == WorldCoord::lo32(rec.anchor.x));
    CHECK(decode_u32(28) == rec.salt);
    CHECK(decode_s32(32) == WorldCoord::hi32(rec.anchor.x));
    CHECK(decode_float(48) == 0.5f);
    CHECK(decode_float(48 + 16 + 4) == 1.0f); // rotation[4] = row 1, col 1
    CHECK(decode_float(96 + 3 * 4) == 9.0f);
    CHECK(decode_s32(128) == -7);
    CHECK(decode_s32(144 + 8) == 11);
    CHECK(decode_float(140) == 6.0f);
}

// A chunk's samples reach past its span (the skirt overhang), and at a coarser LOD than an instance's
// storage LOD its influence box spills into the neighbouring chunks: those must list it too, or two chunks
// see different values at the same point (seams, and skirt walls across a subtracted hole).
TEST_CASE("Archetype table: a chunk lists every instance its samples can see, at every LOD") {
    DefLibrary lib;
    lib.add(make_sphere_type());
    ChunkParam param;
    param.voxel_size = 1.0f;
    int64_t samples = 0, box_hits = 0;
    for (const Combine op : { Combine::UNION, Combine::SUBTRACT }) {
        const Float4 ground{ 0.0f, 0.0f, 0.0f, op == Combine::UNION ? 3.0f : -3.0f };
        for (const float r : { 3.0f, 10.0f }) {
            for (const int64_t gap : { 2, 6, 16, 40 }) {
                // A sphere just below the x = 0 chunk boundary (a boundary at every LOD).
                TableRecord rec;
                rec.op = op;
                rec.params[0] = r;
                rec.anchor = VoxelPos(-(int64_t(r) + gap), 40, 40);
                const Bounds wb = record_world_bounds(lib[0], rec);
                for (int a = 0; a < 3; a++) {
                    rec.box_lo[a] = int32_t(wb.lo[a] - rec.anchor[a]);
                    rec.box_hi[a] = int32_t(wb.hi[a] - rec.anchor[a]);
                }
                rec.margin_c = 2.0f * wb.lipschitz + param.skirt_depth;
                const std::vector<TableRecord> records = { rec };
                auto snap = build_snapshot(std::make_shared<const DefLibrary>(lib), 1.0f, false, records, param, 12, 1);
                const uint32_t all[1] = { 0 };
                for (int lod = 0; lod <= 5; lod++) {
                    const int64_t span = int64_t(param.chunk_voxel_dimension()) << lod;
                    const int64_t unit = int64_t(1) << lod;
                    // The chunk above the boundary and the one holding the sphere.
                    for (const int64_t cx : { int64_t(0), -span }) {
                        const ChunkId c((WorldCoord::floor_div(cx, span) * 2 + 1) * unit, (WorldCoord::floor_div(40, span) * 2 + 1) * unit,
                                (WorldCoord::floor_div(40, span) * 2 + 1) * unit);
                        std::vector<uint32_t> cl;
                        snap->chunk_list(c, cl);
                        for (int i = 0; i < param.chunk_size; i++) {
                            for (int j = 0; j < param.chunk_size; j += 3) {
                                for (int k = 0; k < param.chunk_size; k += 3) {
                                    const WorldPos p = ChunkMath::sample_pos(c, Float3{ float(i), float(j), float(k) }, param);
                                    samples++;
                                    box_hits += in_influence_box(rec, p, lod) ? 1 : 0;
                                    const Float4 a = fold_onto(ground, lib, records.data(), cl.data(), uint32_t(cl.size()), p, lod);
                                    const Float4 b = fold_onto(ground, lib, records.data(), all, 1, p, lod);
                                    CHECK(same_bits(a, b));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    MESSAGE("chunk lists: ", samples, " samples, ", box_hits, " inside the instance's influence box");
    CHECK(box_hits > 10000);
}

namespace {

// Spheres and boxes over six decades of size (1 to 2^20 voxels), unions and subtracts, some smooth, some
// removed, around p_centre: the band rule's test table.
std::vector<TableRecord> band_table(const DefLibrary &p_lib, const VoxelPos &p_centre, int p_count, uint32_t p_seed, bool p_band) {
    std::vector<TableRecord> records;
    for (int i = 0; i < p_count; i++) {
        const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ uint32_t(i), p_seed, 0xBA4Du });
        TableRecord rec;
        rec.def = uint8_t(h.x % 2u);
        rec.op = Combine(h.y % 4u);
        rec.k = (h.y & 0x100u) ? float(h.z % 8u) + 0.5f : 0.0f;
        rec.inv_k = rec.k > 0.0f ? 1.0f / rec.k : 0.0f;
        const float size = std::ldexp(1.0f + float(h.z % 1000u) * 1e-3f, int(h.x >> 8) % 21);
        rec.params[0] = size;
        rec.params[1] = 0.5f * size + 1.0f;
        rec.params[2] = size;
        const int64_t spread = int64_t(size) * 4 + 4096;
        rec.anchor = p_centre + VoxelPos(ExactNoise::random_offset(i, 1, p_seed, spread), ExactNoise::random_offset(i, 2, p_seed, spread), ExactNoise::random_offset(i, 3, p_seed, spread));
        const double yaw = double(h.z >> 8) * 1e-4;
        const float cy = float(std::cos(yaw)), sy = float(std::sin(yaw));
        const float rot[9] = { cy, 0, -sy, 0, 1, 0, sy, 0, cy };
        std::copy(rot, rot + 9, rec.rotation);
        finish_record(p_lib[rec.def], rec, 1.0f, p_band);
        rec.removed = (h.x % 13u) == 0u;
        records.push_back(rec);
    }
    return records;
}

} // namespace

// B3: with the band rule, a chunk's list (and a point's candidates) is still exactly the brute force: every
// record listed at the chunk's LOD whose influence box meets its samples, in id order. The index prunes
// the subtree lists above each instance's drop LOD: that must lose nothing.
TEST_CASE("Archetype band: chunk lists and point candidates equal the brute force at every LOD") {
    DefLibrary lib;
    lib.add(make_sphere_type());
    lib.add(make_box_type());
    ChunkParam param;
    param.voxel_size = 1.0f;
    int64_t lists = 0, entries = 0, dropped_somewhere = 0;
    for (const VoxelPos &centre : { VoxelPos(), VoxelPos(int64_t(1) << 50, -(int64_t(1) << 33), 12345) }) {
        const std::vector<TableRecord> records = band_table(lib, centre, 80, 7u, true);
        auto snap = build_snapshot(std::make_shared<const DefLibrary>(lib), 1.0f, false, records, param, 14, 1);
        for (const TableRecord &r : records) {
            dropped_somewhere += r.max_lod != TableRecord::NEVER_DROPPED ? 1 : 0;
        }
        std::vector<uint32_t> got, want;
        for (uint32_t q = 0; q < records.size(); q++) {
            for (int lod = 0; lod <= 13; lod++) {
                const ChunkId c = ChunkMath::chunk_at_voxel(records[q].anchor, lod, param);
                const int64_t step = int64_t(2) << lod;
                for (int n = 0; n < 27; n += 2) {
                    const ChunkId chunk = c + ChunkId((n % 3 - 1) * step, (n / 3 % 3 - 1) * step, (n / 9 - 1) * step);
                    VoxelPos lo, hi;
                    chunk_sample_box(chunk, param, lo, hi);
                    want.clear();
                    for (uint32_t i = 0; i < records.size(); i++) {
                        if (!records[i].removed && records[i].listed_at(lod) && influence_meets(records[i], lod, lo, hi)) {
                            want.push_back(i);
                        }
                    }
                    snap->chunk_list(chunk, got);
                    CHECK(got == want);
                    lists++;
                    entries += int64_t(want.size());
                }
                // The point query at the anchor and at a box corner.
                for (const VoxelPos &v : { records[q].anchor, records[q].anchor + VoxelPos(records[q].box_hi[0], records[q].box_lo[1], 3) }) {
                    want.clear();
                    for (uint32_t i = 0; i < records.size(); i++) {
                        if (!records[i].removed && records[i].listed_at(lod) && influence_meets(records[i], lod, v, v)) {
                            want.push_back(i);
                        }
                    }
                    snap->point_candidates(v, lod, got);
                    CHECK(got == want);
                }
            }
        }
    }
    MESSAGE("band lists: ", lists, " chunk lists, ", entries, " entries; ", dropped_somewhere, " of 160 instances dropped at some LOD");
    CHECK(dropped_somewhere > 100);
    CHECK(entries > 1000);
}

// B4: the band rule moves the surface only where it erodes or drops an instance. Wherever F with the band rule
// and F with every instance whole at every LOD (the reference) disagree in sign, the point is inside the bare
// box (grown by k) of an instance eroded or dropped at that LOD; and an instance is only dropped once its
// erosion reaches its depth plus k (set_band), so the drop itself moves nothing.
TEST_CASE("Archetype band: the surface moves only inside instances eroded or dropped at the LOD") {
    DefLibrary lib;
    lib.add(make_sphere_type());
    lib.add(make_box_type());
    // The plane y = 0.
    auto ground = [](const WorldPos &p_p) { return Float4{ 0.0f, 1.0f, 0.0f, float(p_p.voxel.y) + p_p.frac.y }; };
    const VoxelPos centre(0, 0, 0);
    const std::vector<TableRecord> banded = band_table(lib, centre, 60, 11u, true);
    const std::vector<TableRecord> reference = band_table(lib, centre, 60, 11u, false);
    std::vector<uint32_t> all;
    for (uint32_t i = 0; i < banded.size(); i++) {
        if (!banded[i].removed) {
            all.push_back(i);
        }
    }
    int64_t points = 0, flips = 0, unexplained = 0;
    for (uint32_t q : all) {
        const TableRecord &near = banded[q];
        double extent = 0.0;
        for (int a = 0; a < 3; a++) {
            extent = std::max(extent, double(near.box_hi[a] - near.box_lo[a]));
        }
        for (int lod = 0; lod <= 20; lod += 2) {
            for (uint32_t j = 0; j < 200; j++) {
                const WorldPos p = random_point(near.anchor, 0.75 * extent + 4.0, j, q * 131u + uint32_t(lod));
                std::vector<uint32_t> listed;
                for (uint32_t i : all) {
                    if (banded[i].listed_at(lod)) {
                        listed.push_back(i);
                    }
                }
                const Float4 fb = fold_onto(ground(p), lib, banded.data(), listed.data(), uint32_t(listed.size()), p, lod);
                const Float4 fr = fold_onto(ground(p), lib, reference.data(), all.data(), uint32_t(all.size()), p, lod);
                points++;
                if ((fb.w > 0.0f) == (fr.w > 0.0f)) {
                    continue;
                }
                flips++;
                bool explained = false;
                for (uint32_t i : all) {
                    const TableRecord &r = banded[i];
                    const bool changed = !r.listed_at(lod) || band_erosion(r.erode, r.erode_from, lod) > 0.0f;
                    if (!changed) {
                        continue;
                    }
                    bool inside = true;
                    for (int a = 0; a < 3; a++) {
                        const double d = double(p.voxel[a] - r.anchor[a]) + double(a == 0 ? p.frac.x : (a == 1 ? p.frac.y : p.frac.z));
                        inside = inside && d >= r.box_lo[a] - r.k - 1.0 && d <= r.box_hi[a] + r.k + 1.0;
                    }
                    explained = explained || inside;
                }
                unexplained += explained ? 0 : 1;
            }
        }
    }
    // Dropped only once fully eroded, and at the first such LOD.
    for (const TableRecord &r : banded) {
        if (r.max_lod != TableRecord::NEVER_DROPPED) {
            const double depth = band_depth(lib[r.def], r);
            CHECK(double(band_erosion(r.erode, r.erode_from, r.max_lod)) >= depth);
            CHECK((r.max_lod == 0 || double(band_erosion(r.erode, r.erode_from, r.max_lod - 1)) < depth));
        }
    }
    MESSAGE("band surface: ", points, " points, ", flips, " sign flips vs the reference, ", unexplained, " unexplained");
    CHECK(flips > 100);
    CHECK(unexplained == 0);
}

// The band depth comes from the local box as well as the world box: a capsule or a branch drops at the same LOD
// however it turns, never later than from its world box alone, and a diagonal one drops earlier than that.
TEST_CASE("Archetype band: a record's drop LOD does not depend on its rotation") {
    std::vector<std::array<float, 9>> rots;
    rotations(rots);
    const double d = 1.0 / std::sqrt(3.0), e = 1.0 / std::sqrt(2.0), f = 1.0 / std::sqrt(6.0);
    // Local y along (1, 1, 1) / sqrt(3): rows (1, -1, 0) / sqrt(2), (1, 1, 1) / sqrt(3), (1, 1, -2) / sqrt(6).
    rots.push_back({ float(e), float(-e), 0.0f, float(d), float(d), float(d), float(f), float(f), float(-2.0 * f) });
    auto first_lod = [](double p_depth) {
        for (int lod = 0; lod < 32; lod++) {
            if (double(ERODE_CELLS) * (std::ldexp(1.0, lod) - 1.0) >= p_depth) {
                return lod;
            }
        }
        return int(TableRecord::NEVER_DROPPED);
    };
    for (const std::string &name : { std::string("capsule"), std::string("branch") }) {
        const TableDefPtr def = table_def(name);
        int lod0 = -1;
        bool earlier = false;
        for (const std::array<float, 9> &rot : rots) {
            TableRecord rec;
            rec.k = 1.5f;
            rec.anchor = VoxelPos(1000, -20, 7);
            std::copy(rot.begin(), rot.end(), rec.rotation);
            rec.params[0] = 60.0f;
            rec.params[1] = 3.0f;
            rec.params[2] = name == "branch" ? -0.03f : 0.0f;
            finish_record(*def, rec, 1.0f, true);
            double side = 1e30;
            for (int a = 0; a < 3; a++) {
                side = std::min(side, double(rec.box_hi[a] - rec.box_lo[a]));
            }
            const int world_lod = first_lod(0.5 * side * double(rec.lipschitz) + double(rec.k));
            CAPTURE(name);
            MESSAGE(name, ": max_lod ", int(rec.max_lod), ", ", world_lod, " from the world box (smallest side ", side, ")");
            CHECK(int(rec.max_lod) <= world_lod);
            earlier = earlier || int(rec.max_lod) < world_lod;
            if (lod0 < 0) {
                lod0 = rec.max_lod;
            }
            CHECK(int(rec.max_lod) == lod0);
        }
        CHECK(earlier);
    }
}

// The discoverer's no-cull rule from the index: near_instance is true wherever the old O(N) scan (every
// instance's bare box against the node box grown by 2 chunk spans) is, at every LOD, band or not. It may say
// yes a node more often (occupancy is per node), never less.
TEST_CASE("Archetype band: near_instance covers every node the brute-force no-cull scan exempts") {
    DefLibrary lib;
    lib.add(make_sphere_type());
    lib.add(make_box_type());
    ChunkParam param;
    param.voxel_size = 1.0f;
    int64_t nodes = 0, exempt = 0, extra = 0;
    for (const VoxelPos &centre : { VoxelPos(), VoxelPos(-(int64_t(1) << 45), 77, int64_t(1) << 38) }) {
        const std::vector<TableRecord> records = band_table(lib, centre, 60, 23u, true);
        auto snap = build_snapshot(std::make_shared<const DefLibrary>(lib), 1.0f, false, records, param, 14, 1);
        for (uint32_t q = 0; q < records.size(); q++) {
            for (int lod = 0; lod <= 14; lod++) {
                const int64_t span = int64_t(param.chunk_voxel_dimension()) << lod;
                const ChunkId c = ChunkMath::chunk_at_voxel(records[q].anchor, lod, param);
                const int64_t step = int64_t(2) << lod;
                for (int n = 0; n < 125; n += 3) {
                    const ChunkId node = c + ChunkId((n % 5 - 2) * step, (n / 5 % 5 - 2) * step, (n / 25 - 2) * step);
                    const VoxelPos min = ChunkMath::chunk_min_voxel(node, param);
                    const VoxelPos lo = min - VoxelPos(2 * span, 2 * span, 2 * span);
                    const VoxelPos hi = min + VoxelPos(3 * span, 3 * span, 3 * span);
                    bool want = false;
                    for (const TableRecord &r : records) {
                        if (r.removed) {
                            continue;
                        }
                        bool meets = true;
                        for (int a = 0; a < 3; a++) {
                            meets = meets && hi[a] >= r.anchor[a] + r.box_lo[a] && lo[a] <= r.anchor[a] + r.box_hi[a];
                        }
                        want = want || meets;
                    }
                    const bool got = snap->near_instance(node, lod);
                    nodes++;
                    exempt += want ? 1 : 0;
                    extra += (got && !want) ? 1 : 0;
                    if (want) {
                        CHECK(got);
                    }
                }
            }
        }
    }
    MESSAGE("no-cull rule: ", nodes, " nodes, ", exempt, " exempt by the brute force, ", extra, " more by the index (per-node occupancy)");
    CHECK(exempt > 1000);
}

// --instance_erode: the record carries its rate, and with resolve 0 a slower rate drops it at the first LOD whose
// erosion at that rate reaches its depth, never before a faster one's (a 6-voxel trunk: LOD 3 at 1 cell per LOD, LOD 4
// at 0.5).
TEST_CASE("Archetype band: the erosion rate sets erode and the drop LOD") {
    const TableDefPtr def = table_def("capsule");
    int previous = -1;
    for (const float rate : { 2.0f, ERODE_CELLS, 0.5f, 0.25f }) {
        TableRecord rec;
        rec.anchor = VoxelPos(0, 0, 0);
        rec.params[0] = 200.0f; // tall: the radius sets the depth
        rec.params[1] = 6.0f;
        finish_record(*def, rec, 1.0f, true, ErodeRule{ rate, 0.0f });
        CAPTURE(rate);
        CHECK(rec.erode == rate);
        const double depth = band_depth(*def, rec);
        const int lod = rec.max_lod;
        CHECK(double(rate) * (std::ldexp(1.0, lod) - 1.0) >= depth);
        CHECK(double(rate) * (std::ldexp(1.0, lod - 1) - 1.0) < depth);
        CHECK(lod >= previous);
        previous = lod;
    }
}

// The band rule's timing: a part is dropped at the later of the first LOD whose cells no longer fit `resolve` times in
// its half thickness (its moments) and the first LOD erosion at `rate` from LOD 0 can make it vanish by; it is whole up
// to the latest onset from which that erosion still reaches its depth at the drop, then shrinks by at most `rate` cells
// per LOD (under the skirt), and the drop comes at the first LOD whose erosion reaches its depth.
TEST_CASE("Archetype band: a part stays whole while its cells resolve it and erodes only before its drop") {
    const TableDefPtr def = table_def("capsule");
    // Worked cases, a tall capsule of radius r: depth = r + 1.04 (its rotation margin; L = 1, k = 0), half thickness
    // ~1.118 r. r 6 (depth 7.04, half 6.7), rate 1: erosion can reach 7.04 by LOD 4 (2^4 - 1); resolve 1 keeps it to
    // LOD 2 (8 > 6.7 at LOD 3): drop 4, onset 3 (16 - 8 >= 7.04). Resolve 0.25: kept to LOD 4, drop 5, onset 4. Rate
    // 0.5: 0.5 (16 - 1) >= 7.04, drop 4, onset 0 (0.5 (16 - 2) < 7.04). r 40 (41.04, 44.7): drop 6, onset 4 (64 - 16).
    // r 0.75 (1.79): drop 2 (2^2 - 1 >= 1.79), onset 1 (4 - 2).
    struct Case {
        float radius;
        ErodeRule rule;
        float erode_from;
        int max_lod;
    };
    const Case worked[] = { { 6.0f, ErodeRule{ 1.0f, 1.0f }, 8.0f, 4 }, { 6.0f, ErodeRule{ 1.0f, 0.25f }, 16.0f, 5 },
        { 6.0f, ErodeRule{ 0.5f, 1.0f }, 1.0f, 4 }, { 40.0f, ErodeRule{ 1.0f, 1.0f }, 16.0f, 6 }, { 0.75f, ErodeRule{ 1.0f, 1.0f }, 2.0f, 2 } };
    for (const Case &c : worked) {
        TableRecord rec;
        rec.params[0] = 400.0f; // tall: the radius sets the depth
        rec.params[1] = c.radius;
        finish_record(*def, rec, 1.0f, true, c.rule);
        CAPTURE(c.radius);
        CAPTURE(c.rule.rate);
        CAPTURE(c.rule.resolve);
        CHECK(rec.erode_from == c.erode_from);
        CHECK(int(rec.max_lod) == c.max_lod);
    }
    // The defining properties, over radii from a voxel to 1,000 and the rules around the default.
    int64_t records = 0, kept_by_resolve = 0;
    for (const float rate : { 0.5f, 1.0f, 2.0f }) {
        for (const float resolve : { 0.0f, 0.5f, 1.0f, 2.0f }) {
            for (double r = 1.0; r < 1000.0; r *= 1.37) {
                TableRecord rec;
                rec.params[0] = 4000.0f;
                rec.params[1] = float(r);
                finish_record(*def, rec, 1.0f, true, ErodeRule{ rate, resolve });
                CAPTURE(rate);
                CAPTURE(resolve);
                CAPTURE(r);
                CHECK(rec.erode == rate);
                const double depth = band_depth(*def, rec);
                const double half = record_moments(*def, rec).half_thickness();
                int budget = 0;
                while (double(rate) * (std::ldexp(1.0, budget) - 1.0) < depth) {
                    budget++;
                }
                int resolved = 0;
                while (resolve > 0.0f && double(resolve) * std::ldexp(1.0, resolved) <= half) {
                    resolved++;
                }
                CHECK(int(rec.max_lod) == MAX(budget, resolved));
                kept_by_resolve += resolved > budget ? 1 : 0;
                const int m = rec.max_lod;
                CHECK(double(band_erosion(rec.erode, rec.erode_from, m)) >= depth);
                CHECK((m == 0 || double(band_erosion(rec.erode, rec.erode_from, m - 1)) < depth));
                // The latest onset: one LOD later would not reach the depth at the drop.
                if (2.0f * rec.erode_from < exact_pow2(m)) {
                    CHECK(double(band_erosion(rec.erode, 2.0f * rec.erode_from, m)) < depth);
                }
                for (int lod = 0; lod < m; lod++) {
                    const float e0 = band_erosion(rec.erode, rec.erode_from, lod);
                    const float e1 = band_erosion(rec.erode, rec.erode_from, lod + 1);
                    CHECK((exact_pow2(lod) > rec.erode_from || e0 == 0.0f)); // whole up to the onset
                    CHECK(e1 - e0 <= rec.erode * exact_pow2(lod)); // at most `rate` cells of LOD L: under the skirt
                }
                records++;
            }
        }
    }
    MESSAGE("band timing: ", records, " records, ", kept_by_resolve, " kept past the erosion budget by resolve");
    CHECK(kept_by_resolve > 0);
}

// The level pass's fold tests each list word once per cube of FOLD_CUBE^3 samples (arch_meets_box, mirrored by
// in_influence_cube) and folds only the words that pass: a record that any sample of the cube would fold must pass, or
// that sample loses it. Random records over six decades of size, margins up to the INPUT_LIMIT clamp, anchors near 0,
// across the 2^31 low-word wrap and near +-2^60, cubes at LODs 0-14 near them and past the sat_diff saturation.
TEST_CASE("Archetype table: a cube's box test keeps every record a sample of the cube folds") {
    const int64_t bases[5] = { 0, (int64_t(1) << 31) - 7, -(int64_t(1) << 31), (int64_t(1) << 60) - 12345, -(int64_t(1) << 60) };
    constexpr uint32_t SEED = 0xC0BEu;
    int64_t misses = 0, kept = 0, cube_only = 0, rejected = 0, saturated_kept = 0;
    for (uint32_t i = 0; i < 40000; i++) {
        const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ i, SEED, 0xF01Du });
        const int lod = int(i % 15u);
        TableRecord rec;
        rec.anchor = VoxelPos(bases[i % 5u], bases[(i / 5u) % 5u], bases[(i / 25u) % 5u]) +
                VoxelPos(ExactNoise::random_offset(i, 1, SEED, 1 << 20), ExactNoise::random_offset(i, 2, SEED, 1 << 20), ExactNoise::random_offset(i, 3, SEED, 1 << 20));
        for (int a = 0; a < 3; a++) {
            const int32_t size = int32_t(std::ldexp(1.0 + double((h.x >> (8 * a)) & 0xFFu) / 256.0, int((h.y >> (5 * a)) % 21u)));
            rec.box_lo[a] = -int32_t(ExactNoise::pcg3d(ExactNoise::UVec3{ i, uint32_t(a), SEED }).x % uint32_t(size + 1));
            rec.box_hi[a] = rec.box_lo[a] + size;
        }
        rec.set_k(float(h.z % 64u) * 0.25f);
        // Some margins reach the INPUT_LIMIT clamp at the LOD: the influence box then reaches saturated samples.
        rec.margin_c = i % 11u == 0 ? std::ldexp(1.0f, 16 + int(h.z >> 28)) : float((h.z >> 8) % 32u) * 0.25f + 0.5f;
        const int64_t step = int64_t(1) << lod;
        const int64_t reach = std::max({ int64_t(rec.box_hi[0]) - rec.box_lo[0], int64_t(rec.box_hi[1]) - rec.box_lo[1], int64_t(rec.box_hi[2]) - rec.box_lo[2] }) +
                influence_margin(rec, lod) + 4 * step;
        const int64_t range = i % 7u == 3u ? 3 * int64_t(INPUT_LIMIT) : reach;
        VoxelPos lo;
        for (int a = 0; a < 3; a++) {
            lo[a] = (rec.anchor[a] + (int64_t(rec.box_lo[a]) + rec.box_hi[a]) / 2 + ExactNoise::random_offset(i, 4 + a, SEED, range)) & ~(step - 1); // on the LOD's lattice
        }
        constexpr int64_t CUBE = CascadeGLSL::FOLD_CUBE;
        const VoxelPos hi = lo + VoxelPos((CUBE - 1) * step, (CUBE - 1) * step, (CUBE - 1) * step);
        bool any = false, saturated = false;
        for (int64_t s = 0; s < CUBE * CUBE * CUBE; s++) {
            WorldPos p;
            p.voxel = lo + VoxelPos(s % CUBE * step, s / CUBE % CUBE * step, s / (CUBE * CUBE) * step);
            if (in_influence_box(rec, p, lod)) {
                any = true;
                for (int a = 0; a < 3; a++) {
                    const int64_t d = p.voxel[a] - rec.anchor[a];
                    saturated = saturated || d > INPUT_LIMIT || d < -INPUT_LIMIT;
                }
            }
        }
        const bool cube = in_influence_cube(rec, lo, hi, lod);
        misses += any && !cube ? 1 : 0;
        kept += any ? 1 : 0;
        cube_only += cube && !any ? 1 : 0;
        rejected += cube ? 0 : 1;
        saturated_kept += saturated && cube ? 1 : 0;
    }
    MESSAGE("cube test: ", kept, " cubes with a sample inside, ", cube_only, " kept with no sample inside, ", rejected,
            " rejected, ", saturated_kept, " kept through saturated samples");
    CHECK(misses == 0);
    CHECK(kept > 3000);
    CHECK(rejected > 5000);
    CHECK(saturated_kept > 0);
}
