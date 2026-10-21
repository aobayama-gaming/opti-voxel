#include "test_common.hpp"

#include "forest_stream.hpp"

#include <chrono>
#include <cmath>
#include <thread>

using namespace Archetype;

namespace {

// A 5 cm forest of about 400 trees on gentle hills far from the origin, and chunks of LODs 0-9 over its middle.
struct StreamFixture {
    static constexpr double VOXEL = 0.05;
    std::shared_ptr<const ForestField> field;
    ChunkParam param;
    VoxelPos origin = VoxelPos(int64_t(1) << 33, 1700, -(int64_t(1) << 20));
    std::vector<ChunkId> chunks;

    StreamFixture() {
        param.voxel_size = float(VOXEL);
        param.skirt_depth = 4.0f;
        auto lib = std::make_shared<DefLibrary>();
        ForestField::Setup s;
        s.types = add_tree_types(*lib, VOXEL);
        s.blob_type = lib->add(make_moment_blob_type());
        s.library = lib;
        s.voxel = VOXEL;
        s.finish = RecordFinish{ param.skirt_depth, true, ErodeRule() };
        s.layers.min_occupancy = MOMENT_MATERIAL_ISO;
        s.lod_count = 15;
        s.max_extent = int64_t(1) << 29;
        s.generator.side = 100;
        s.generator.zone = VoxelBox{ origin - VoxelPos(1000, 400, 1000), origin + VoxelPos(999, 400, 999), true };
        const VoxelPos o = origin;
        s.ground = [o](const VoxelPos &p_v, int) {
            const double tau = 6.283185307179586, amp = 60.0, wave = 2400.0;
            const double x = double(p_v.x - o.x), z = double(p_v.z - o.z);
            const double w = double(p_v.y - o.y) - amp * std::sin(tau * x / wave) * std::cos(tau * z / wave);
            return ExactNoise::Float4{ float(-amp * tau / wave * std::cos(tau * x / wave) * std::cos(tau * z / wave)), 1.0f,
                float(amp * tau / wave * std::sin(tau * x / wave) * std::sin(tau * z / wave)), float(w) };
        };
        s.up = [](const VoxelPos &) { return UP_Y; };
        field = std::make_shared<const ForestField>(s);
        for (const int lod : { 0, 2, 4, 6, 7, 9 }) {
            for (const int64_t dx : { int64_t(-300), int64_t(0), int64_t(300) }) {
                const ChunkId c = ChunkMath::chunk_at_voxel(origin + VoxelPos(dx, 0, -dx / 2), lod, param);
                if (std::find(chunks.begin(), chunks.end(), c) == chunks.end()) {
                    chunks.push_back(c);
                }
            }
        }
    }
    static const StreamFixture &get() {
        static const StreamFixture fixture;
        return fixture;
    }

    // The chunk's streamed records computed straight from the field: its keys' records meeting its samples, in key order.
    std::vector<TableRecord> direct(const ChunkId &p_chunk) const {
        std::vector<FieldKey> keys;
        field->chunk_keys(p_chunk, param, keys);
        VoxelPos lo, hi;
        chunk_sample_box(p_chunk, param, lo, hi);
        std::vector<TableRecord> out;
        for (const FieldKey &k : keys) {
            FieldContent c;
            field->content(k, c);
            std::vector<uint32_t> hit;
            c.meeting(lo, hi, ChunkMath::get_lod(p_chunk), hit);
            for (const uint32_t i : hit) {
                out.push_back(c.records[i]);
            }
        }
        return out;
    }
    // The records all of a chunk's keys hold (what its residency takes in slots).
    int64_t key_records(const ChunkId &p_chunk) const {
        std::vector<FieldKey> keys;
        field->chunk_keys(p_chunk, param, keys);
        int64_t n = 0;
        for (const FieldKey &k : keys) {
            FieldContent c;
            field->content(k, c);
            n += int64_t(c.records.size());
        }
        return n;
    }
};

// Frames (poll, then ready for each chunk, as the meshers ask) until every chunk is ready; false past p_frames.
bool settle(ForestStream &r_stream, const std::vector<ChunkId> &p_chunks, int p_frames = 20000) {
    for (int f = 0; f < p_frames; f++) {
        r_stream.poll();
        bool all = true;
        for (const ChunkId &c : p_chunks) {
            all = r_stream.ready(c) && all; // every chunk asks, as a mesher's pops do
        }
        if (all) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

// The chunk's streamed list read back from the slots.
std::vector<TableRecord> streamed(const ForestStream &p_stream, const ChunkId &p_chunk) {
    std::vector<uint32_t> ids;
    p_stream.chunk_list(p_chunk, ids);
    std::vector<TableRecord> out;
    for (const uint32_t id : ids) {
        out.push_back(p_stream.slots()[id - p_stream.slot_base()]);
    }
    return out;
}

bool same_records(const std::vector<TableRecord> &p_a, const std::vector<TableRecord> &p_b) {
    if (p_a.size() != p_b.size()) {
        return false;
    }
    for (size_t i = 0; i < p_a.size(); i++) {
        const TableRecord &a = p_a[i], &b = p_b[i];
        if (a.def != b.def || a.anchor != b.anchor || !std::equal(a.params, a.params + 8, b.params) || !std::equal(a.rotation, a.rotation + 9, b.rotation) ||
                a.max_lod != b.max_lod || a.min_lod != b.min_lod || a.material != b.material || a.op != b.op) {
            return false;
        }
    }
    return true;
}

} // namespace

TEST_CASE("ForestStream: a ready chunk's list is its keys' records, in key order, from the slots") {
    const StreamFixture &fx = StreamFixture::get();
    ForestStream stream(fx.field, fx.param, 1000, uint32_t(1) << 22);
    CHECK(settle(stream, fx.chunks));
    int64_t records = 0;
    for (const ChunkId &c : fx.chunks) {
        CAPTURE(ChunkMath::get_lod(c));
        const std::vector<TableRecord> want = fx.direct(c);
        CHECK(same_records(streamed(stream, c), want));
        records += int64_t(want.size());
    }
    const Dictionary stats = stream.get_stats();
    MESSAGE(fx.chunks.size(), " chunks, ", records, " records listed; ", int64_t(stats["keys_placed"]), " keys placed in ", int64_t(stats["slots_used"]),
            " slots, ", int64_t(stats["computed"]), " computed (", double(stats["compute_ms_avg"]), " ms each)");
    CHECK(records > 1000);
    CHECK(int64_t(stats["lists_missing"]) == 0);
    CHECK(int64_t(stats["evicted"]) == 0);
}

TEST_CASE("ForestStream: under a small slot pool the keys are evicted and computed again, the same records") {
    const StreamFixture &fx = StreamFixture::get();
    int64_t biggest = 0;
    for (const ChunkId &c : fx.chunks) {
        biggest = MAX(biggest, fx.key_records(c));
    }
    // Room for about two chunks' keys: the next chunk evicts the keys the previous ones no longer ask for.
    ForestStream stream(fx.field, fx.param, 0, uint32_t(2 * biggest));
    int64_t checked = 0;
    for (int pass = 0; pass < 2; pass++) {
        for (size_t n = 0; n < fx.chunks.size(); n++) {
            const ChunkId &c = fx.chunks[pass == 0 ? n : fx.chunks.size() - 1 - n];
            CAPTURE(ChunkMath::get_lod(c));
            CHECK(settle(stream, { c }));
            CHECK(same_records(streamed(stream, c), fx.direct(c)));
            checked++;
            for (uint64_t f = 0; f <= ForestStream::GRACE_FRAMES; f++) {
                stream.poll(); // nobody asks: the chunk's keys age past the grace
            }
        }
    }
    const Dictionary stats = stream.get_stats();
    MESSAGE(checked, " chunk lists checked with ", 2 * biggest, " slots: ", int64_t(stats["evicted"]), " keys evicted, ", int64_t(stats["computed"]),
            " computed, peak ", int64_t(stats["slots_peak"]), " slots");
    CHECK(int64_t(stats["evicted"]) > 0);
    CHECK(int64_t(stats["slots_full"]) == 0);
    CHECK(int64_t(stats["lists_missing"]) == 0);
    CHECK(int64_t(stats["slots_peak"]) <= 2 * biggest);
}

TEST_CASE("ForestStream: a point's records are its keys', resident or computed on demand") {
    const StreamFixture &fx = StreamFixture::get();
    ForestStream resident(fx.field, fx.param, 0, uint32_t(1) << 22), cold(fx.field, fx.param, 0, uint32_t(1) << 22);
    CHECK(settle(resident, fx.chunks));
    int64_t points = 0, records = 0;
    for (const ChunkId &c : fx.chunks) {
        const int lod = ChunkMath::get_lod(c);
        VoxelPos lo, hi;
        chunk_sample_box(c, fx.param, lo, hi);
        for (int i = 0; i < 8; i++) {
            const int64_t unit = int64_t(1) << lod;
            const ExactNoise::UVec3 h = ExactNoise::pcg3d(ExactNoise::UVec3{ uint32_t(i), uint32_t(lod), 0x57EAu });
            VoxelPos p;
            for (int a = 0; a < 3; a++) {
                p[a] = WorldCoord::floor_div(lo[a] + int64_t((&h.x)[a] % uint32_t(hi[a] - lo[a] + 1)), unit) * unit;
            }
            std::vector<FieldKey> keys;
            fx.field->keys(p, p, lod, keys);
            std::vector<TableRecord> want, got_resident, got_cold;
            for (const FieldKey &k : keys) {
                FieldContent content;
                fx.field->content(k, content);
                std::vector<uint32_t> hit;
                content.meeting(p, p, lod, hit);
                for (const uint32_t j : hit) {
                    want.push_back(content.records[j]);
                }
            }
            resident.point_records(p, lod, got_resident);
            cold.point_records(p, lod, got_cold);
            CHECK(same_records(got_resident, want));
            CHECK(same_records(got_cold, want));
            points++;
            records += int64_t(want.size());
        }
    }
    MESSAGE(points, " points, ", records, " records; ", int64_t(cold.get_stats()["evaluate_computed"]), " keys computed on demand");
    CHECK(records > 0);
}
