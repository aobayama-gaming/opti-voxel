#include "test_common.hpp"

#include "archetype_instance.hpp"
#include "archetype_table.hpp"
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

// p_anchor + an offset in [-p_range, p_range)^3 voxels, fraction in steps of 2^-18 (as
// test_archetype_instance.cpp).
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

// The compiled (PR 1) equivalent of each library type: the same ops with constant sizes and a local
// frame (compile() re-frames it at the instance's anchor and rotation).
std::shared_ptr<Def> ref_def(const std::string &p_type, const float *p_params) {
    Builder b;
    const Local l = b.local();
    Reg r = -1;
    if (p_type == "sphere") {
        r = b.sphere(l, p_params[0]);
    } else if (p_type == "box") {
        r = b.box(l, p_params[0], p_params[1], p_params[2]);
    } else if (p_type == "capsule") {
        r = b.capsule(l, p_params[0], p_params[1]);
    } else if (p_type == "cylinder") {
        r = b.cylinder(l, p_params[0], p_params[1]);
    } else if (p_type == "rock") {
        Reg noise = b.constant(0.0f);
        for (int i = 0; i < 3; i++) {
            noise = b.add(noise, b.perlin3(ROCK_K + i, ROCK_SEED + uint32_t(i), Vector3i64(), float(std::ldexp(1.0, -i))));
        }
        r = b.add(b.sphere(l, p_params[0]), b.mul(noise, p_params[1]));
    } else { // swiss_cube
        Reg box = b.box(l, p_params[0], p_params[1], p_params[2]);
        for (int i = 0; i < 4; i++) {
            const double a = 2.0 * M_PI * double(i) / 4.0;
            const Local q{ b.sub(l.x, b.constant(float(std::cos(a)) * p_params[3])), l.y,
                b.sub(l.z, b.constant(float(std::sin(a)) * p_params[3])) };
            box = b.max(box, b.neg(b.sphere(q, p_params[4])));
        }
        r = box;
    }
    std::shared_ptr<Def> d = std::make_shared<Def>();
    d->program = b.finish(r);
    return d;
}

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
    if (p_type == "cylinder") {
        return make_cylinder_type();
    }
    if (p_type == "rock") {
        return make_rock_type(ROCK_K, ROCK_SEED);
    }
    return make_swiss_cube_type(4);
}

// A small set of rotations: identity, a yaw, and a full rotation (a rotation of 0.5 rad about (1, 2, 3)).
void rotations(std::vector<std::array<float, 9>> &r_out) {
    r_out.push_back({ 1, 0, 0, 0, 1, 0, 0, 0, 1 });
    const float c = float(std::cos(0.7)), s = float(std::sin(0.7));
    r_out.push_back({ c, 0, -s, 0, 1, 0, s, 0, c });
    const double axis[3] = { 1, 2, 3 };
    const double n = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    const double x = axis[0] / n, y = axis[1] / n, z = axis[2] / n;
    const double a = 0.5, ca = std::cos(a), sa = std::sin(a), t = 1.0 - ca;
    const float m[9] = { float(t * x * x + ca), float(t * x * y - sa * z), float(t * x * z + sa * y), float(t * x * y + sa * z),
        float(t * y * y + ca), float(t * y * z - sa * x), float(t * x * z - sa * y), float(t * y * z + sa * x),
        float(t * z * z + ca) };
    r_out.push_back({ m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8] });
}

} // namespace

TEST_CASE("Archetype table: a table Def matches the compiled (PR 1) program per type, rotation and anchor") {
    struct Type {
        std::string name;
        float params[5];
    };
    const std::vector<Type> types = {
        { "sphere", { 7.0f, 0, 0, 0, 0 } },
        { "box", { 6.0f, 4.0f, 9.0f, 0, 0 } },
        { "capsule", { 20.0f, 3.0f, 0, 0, 0 } },
        { "cylinder", { 8.0f, 5.0f, 0, 0, 0 } },
        { "rock", { 12.0f, 2.0f, 0, 0, 0 } },
        { "swiss_cube", { 10.0f, 10.0f, 10.0f, 5.0f, 4.0f } },
    };
    const int64_t P32 = int64_t(1) << 32;
    const VoxelPos anchors[] = { VoxelPos(), VoxelPos(P32 - 3, -P32 + 2, P32 - 1),
        VoxelPos((int64_t(1) << 60) - 1001, 5, -(int64_t(1) << 60) + 999) };
    std::vector<std::array<float, 9>> rots;
    rotations(rots);
    DefLibrary lib;
    std::vector<int> def_index;
    for (const Type &t : types) {
        def_index.push_back(lib.add(table_def(t.name)));
    }
    int64_t compared = 0;
    for (size_t ti = 0; ti < types.size(); ti++) {
        const Type &t = types[ti];
        const Program &table_prog = lib[def_index[ti]].program;
        CHECK(table_prog.validate(25).is_empty());
        for (size_t ai = 0; ai < sizeof(anchors) / sizeof(anchors[0]); ai++) {
            const VoxelPos &a = anchors[ai];
            for (size_t ri = 0; ri < rots.size(); ri++) {
                const std::array<float, 9> &rot = rots[ri];
                Instance inst;
                inst.def = ref_def(t.name, t.params);
                inst.anchor = a;
                std::copy(rot.begin(), rot.end(), inst.rotation);
                inst.seed = 3;
                Compiled c = compile(inst);
                CHECK(c.error.is_empty());
                const Program &compiled = c.program;
                TableRecord rec;
                rec.def = uint8_t(def_index[ti]);
                rec.anchor = a;
                std::copy(rot.begin(), rot.end(), rec.rotation);
                std::memcpy(rec.params, t.params, sizeof(rec.params));
                rec.salt = noise_salt(inst.seed);
                const DefRecord dr = rec.def_record();
                for (uint32_t i = 0; i < 400; i++) {
                    const WorldPos p = random_point(a, 40.0, i, uint32_t(ti + 1));
                    for (const int lod : { 0, 3, 24 }) {
                        const Float4 x = run(compiled, p, lod);
                        const Float4 y = run(table_prog, p, lod, &dr);
                        CAPTURE(t.name);
                        if (!same_bits(x, y)) {
                            MESSAGE(t.name, " ai=", ai, " ri=", ri, " lod=", lod, ": compiled w=", x.w, " table w=", y.w);
                        }
                        CHECK(same_bits(x, y));
                        compared++;
                    }
                }
            }
        }
    }
    MESSAGE("table vs compiled (value and gradient): ", compared, " comparisons");
}

TEST_CASE("Archetype: a child of a rotated instance turns with its parent") {
    Instance inst;
    inst.def = box_def(20.0f, 20.0f, 20.0f);
    inst.set_rotation_y(0.6);
    Instance::Child c;
    c.instance.def = sphere_def(3.0f);
    c.instance.anchor = VoxelPos(10, 0, 0);
    inst.children.push_back(c);
    Compiled cc = compile(inst);
    CHECK(cc.error.is_empty());
    CHECK(cc.nodes.size() == 2);
    // R = yaw(0.6), world -> local: R^T (10, 0, 0) = (10 cos 0.6, 0, -10 sin 0.6).
    const int64_t cx = std::llround(10.0 * std::cos(0.6));
    const int64_t cz = std::llround(-10.0 * std::sin(0.6));
    const Bounds &child = cc.nodes[0].bounds;
    MESSAGE("child box x [", child.lo.x, ", ", child.hi.x, "] z [", child.lo.z, ", ", child.hi.z, "], centre (", cx, ", ", cz, ")");
    CHECK(cx != 10); // the axis-aligned placement would be (10, 0, 0)
    const int64_t ccx = (child.lo.x + child.hi.x) / 2;
    const int64_t ccz = (child.lo.z + child.hi.z) / 2;
    CHECK(ccx - cx <= 2);
    CHECK(cx - ccx <= 2);
    CHECK(ccz - cz <= 2);
    CHECK(cz - ccz <= 2);
    // The old (world-axes) placement would centre the child at (10, 0, 0), far from the rotation.
    const bool old_center = (ccx == 10) && (ccz == 0);
    CHECK_FALSE(old_center);
}

TEST_CASE("Archetype table: the locality rule changes F only far from the surface, never its sign") {
    // A random table: every type, random params, rotations, top-level ops and smooth radii.
    DefLibrary lib;
    lib.add(make_sphere_type());
    lib.add(make_box_type());
    lib.add(make_capsule_type());
    lib.add(make_rock_type(8, 0x1234u));
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
            default:
                rec.params[0] = 6.0f + float(i % 4);
                rec.params[1] = 1.5f;
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
    Builder gb;
    const Program ground = gb.finish(gb.constant(1000.0f));
    std::vector<uint32_t> list(records.size());
    for (uint32_t i = 0; i < list.size(); i++) {
        list[i] = i;
    }
    float lmax = 1.0f;
    for (const TableRecord &r : records) {
        lmax = MAX(lmax, r.lipschitz);
    }
    int64_t equal = 0, differ = 0, sign_flips = 0, points = 0;
    for (uint32_t i = 0; i < 60000; i++) {
        // A point near a random record: the rule bites in the halo just outside its box (and points far
        // from every instance, where F is the ground both ways).
        const TableRecord &near = records[i % records.size()];
        const Bounds wb = record_world_bounds(lib[near.def], near);
        double half = 0.0;
        for (int a = 0; a < 3; a++) {
            half = MAX(half, double(wb.hi[a] - wb.lo[a]) * 0.5);
        }
        const double range = half + 4.0 * double(near.margin_c) + 64.0;
        const WorldPos p = i % 5 == 0 ? random_point(VoxelPos(), 1.2e6, i, 0xB0Du) : random_point(near.anchor, range, i, 0xB0Du);
        for (const int lod : { 0, 4, 12 }) {
            const Float4 a = fold(ground, lib, records.data(), list.data(), uint32_t(list.size()), p, lod, true);
            const Float4 b = fold(ground, lib, records.data(), list.data(), uint32_t(list.size()), p, lod, false);
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
                CHECK(MAX(std::fabs(a.w), std::fabs(b.w)) > threshold);
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
    Builder gb;
    const Program ground = gb.finish(gb.constant(80.0f));
    ChunkParam param;
    param.voxel_size = 1.0f;

    std::vector<TableRecord> records;
    const int64_t E = 1 << 14;
    int64_t checked = 0;
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

        auto snap = build_snapshot(std::make_shared<const DefLibrary>(lib), ground, 1.0f, false, records, param, 12, step + 1);
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
                snap->point_candidates(v, candidates);
                const Float4 indexed = fold(ground, lib, records.data(), candidates.data(), uint32_t(candidates.size()), p, lod, true);
                const Float4 brute = fold(ground, lib, records.data(), all.data(), uint32_t(all.size()), p, lod, true);
                CHECK(same_bits(indexed, brute));
                checked++;
            }
        }
        // A few chunks: chunk_list is sorted and unique, and folds the same as the brute force at its own
        // LOD (the duplicate-instance bug the edit run found).
        int chunk_checks = 0;
        for (const KeyValue<ChunkId, std::vector<uint32_t>> &kv : snap->stored_at) {
            if (chunk_checks >= 8) {
                break;
            }
            chunk_checks++;
            std::vector<uint32_t> cl;
            snap->chunk_list(kv.key, cl);
            CHECK(std::is_sorted(cl.begin(), cl.end()));
            CHECK(std::adjacent_find(cl.begin(), cl.end()) == cl.end());
            const int lod = ChunkMath::get_lod(kv.key);
            WorldPos p;
            p.voxel = ChunkMath::chunk_min_voxel(kv.key, param);
            const Float4 from_chunk = fold(ground, lib, records.data(), cl.data(), uint32_t(cl.size()), p, lod, true);
            const Float4 brute = fold(ground, lib, records.data(), all.data(), uint32_t(all.size()), p, lod, true);
            CHECK(same_bits(from_chunk, brute));
        }
    }
    MESSAGE("edits: ", checked, " (index vs brute force) fold comparisons over 400 random add / remove / move steps");
    CHECK(checked > 0);
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
    PackedByteArray bytes;
    bytes.resize(RECORD_BYTES);
    encode_record(bytes, 0, rec);
    CHECK(bytes.decode_u32(0) == 2u);
    CHECK(bytes.decode_u32(4) == uint32_t(Combine::SMOOTH_SUBTRACT));
    CHECK(bytes.decode_float(8) == rec.k);
    CHECK(bytes.decode_u32(16) == WorldCoord::lo32(rec.anchor.x));
    CHECK(bytes.decode_u32(28) == rec.salt);
    CHECK(bytes.decode_s32(32) == WorldCoord::hi32(rec.anchor.x));
    CHECK(bytes.decode_float(48) == 0.5f);
    CHECK(bytes.decode_float(48 + 16 + 4) == 1.0f); // rotation[4] = row 1, col 1
    CHECK(bytes.decode_float(96 + 3 * 4) == 9.0f);
    CHECK(bytes.decode_s32(128) == -7);
    CHECK(bytes.decode_s32(144 + 8) == 11);
    CHECK(bytes.decode_float(140) == 6.0f);
}
