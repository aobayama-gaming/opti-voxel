#include "test_common.hpp"

#include "debug_log.hpp"
#include "instance_stream.hpp"

#include <map>

using namespace Archetype;

namespace {

// A cheap source: one sphere of 4-8 voxels per 16-voxel cell site on a flat ground far from the origin, no blobs. The
// stream runs its keys inline (the base executor), so a key computed when asked for lands at the next poll().
struct StreamFixture {
    std::shared_ptr<const ScatterSource> source;
    ChunkParam param;
    VoxelPos origin = VoxelPos(int64_t(1) << 33, 300, -(int64_t(1) << 20));

    StreamFixture() {
        auto lib = std::make_shared<DefLibrary>();
        const int sphere = lib->add(make_sphere_type());
        ScatterSource::Setup s;
        s.library = lib;
        s.finish = RecordFinish{ 4.0f, true, ErodeRule() };
        s.max_extent = int64_t(1) << 29;
        CellScatter &c = s.scatter;
        c.salt = 0x57E4Au;
        c.side = 16;
        c.zone = VoxelBox{ origin - VoxelPos(512, 32, 512), origin + VoxelPos(511, 32, 511), true };
        const int64_t oy = origin.y;
        c.ground = [oy](const VoxelPos &p_v, int) { return ExactNoise::Float4{ 0.0f, 1.0f, 0.0f, float(p_v.y - oy) }; };
        c.up = [](const VoxelPos &) { return UP_Y; };
        c.generate = [sphere](const Site &p_site, const UpAxis &, std::vector<TableRecord> &r_out) {
            TableRecord r;
            r.def = uint8_t(sphere);
            r.anchor = p_site.base;
            r.params[0] = 4.0f + float(p_site.seed % 5u);
            r_out.push_back(r);
        };
        source = std::make_shared<ScatterSource>(s);
    }
    static const StreamFixture &get() {
        static const StreamFixture fixture;
        return fixture;
    }

    std::unique_ptr<InstanceStream> stream(InstanceStream::Settings p_settings) const {
        return std::make_unique<InstanceStream>(std::vector<std::shared_ptr<const ScatterSource>>{ source }, param, p_settings, std::make_unique<StreamExecutor>());
    }
    ChunkId chunk(int64_t p_x, int64_t p_z, int p_lod = 0) const { return ChunkMath::chunk_at_voxel(origin + VoxelPos(p_x, 0, p_z), p_lod, param); }

    // A key's content as the source computes it alone.
    SourceContent content(const SourceKey &p_key) const {
        SourceContent c;
        source->content(p_key, {}, c);
        return c;
    }
    // The chunk's records as its keys hold them, in key order (what its list must name).
    std::vector<TableRecord> expected(const ChunkId &p_chunk) const {
        std::vector<SourceKey> keys;
        source->chunk_keys(p_chunk, param, keys);
        VoxelPos lo, hi;
        chunk_sample_box(p_chunk, param, lo, hi);
        std::vector<TableRecord> out;
        for (const SourceKey &k : keys) {
            const SourceContent c = content(k);
            std::vector<uint32_t> hit;
            c.meeting(lo, hi, ChunkMath::get_lod(p_chunk), hit);
            for (const uint32_t i : hit) {
                out.push_back(c.records[i]);
            }
        }
        return out;
    }
    // All the records of the chunk's keys: the slots it takes.
    uint32_t slots(const ChunkId &p_chunk) const {
        std::vector<SourceKey> keys;
        source->chunk_keys(p_chunk, param, keys);
        uint32_t n = 0;
        for (const SourceKey &k : keys) {
            n += uint32_t(content(k).records.size());
        }
        return n;
    }
};

// The GPU table's streamed records, as a reader of the placement log keeps them.
struct Mirror {
    uint64_t cursor = 0;
    std::vector<TableRecord> records; // by record id - slot_base

    void update(const InstanceStream &p_stream) {
        std::vector<InstanceStream::Upload> ups;
        p_stream.uploads_since(cursor, ups);
        cursor = p_stream.placements();
        for (const InstanceStream::Upload &u : ups) {
            const uint32_t at = u.first - p_stream.slot_base();
            records.resize(std::max(records.size(), size_t(at + u.content->records.size())));
            std::copy(u.content->records.begin(), u.content->records.end(), records.begin() + at);
        }
    }
};

bool same(const TableRecord &p_a, const TableRecord &p_b) {
    return p_a.def == p_b.def && p_a.anchor == p_b.anchor && p_a.params[0] == p_b.params[0] && p_a.max_lod == p_b.max_lod && p_a.min_lod == p_b.min_lod;
}

// Whether the chunk's list, read through the mirror, names exactly its keys' records in key order.
bool lists_right(const StreamFixture &p_fx, const InstanceStream &p_stream, const Mirror &p_mirror, const ChunkId &p_chunk) {
    std::vector<uint32_t> ids;
    p_stream.chunk_list(p_chunk, ids);
    const std::vector<TableRecord> want = p_fx.expected(p_chunk);
    if (ids.size() != want.size()) {
        return false;
    }
    for (size_t i = 0; i < ids.size(); i++) {
        const uint32_t at = ids[i] - p_stream.slot_base();
        if (ids[i] < p_stream.slot_base() || at >= p_mirror.records.size() || !same(p_mirror.records[at], want[i])) {
            return false;
        }
    }
    return true;
}

int64_t stat(const InstanceStream &p_stream, const char *p_name) {
    return int64_t(p_stream.get_stats()[p_name]);
}

} // namespace

TEST_CASE("Instance stream: parked chunks come back once their keys land, in park order, and list their keys' records") {
    const StreamFixture &fx = StreamFixture::get();
    CHECK(fx.source->profile().lod_end >= 2);
    if (!(fx.source->profile().lod_end >= 2)) {
        return;
    }
    InstanceStream::Settings settings;
    settings.slot_base = 1000;
    std::unique_ptr<InstanceStream> stream = fx.stream(settings);
    Mirror mirror;

    // A LOD-1 chunk's part tile reads the LOD-0 tiles under it: two polls. A LOD-0 chunk's tiles: one.
    const ChunkId coarse = fx.chunk(-200, -200, 1), fine = fx.chunk(200, 200), other = fx.chunk(-200, 200);
    CHECK(stream->admit(coarse, 0, 11) == InstanceStream::PARKED);
    CHECK(stream->admit(fine, 0, 22) == InstanceStream::PARKED);
    CHECK(stream->admit(fine, 0, 23) == InstanceStream::PARKED); // asked again: still parked, the new tag
    CHECK(stream->admit(fine, 7, 70) == InstanceStream::PARKED); // another mesher's: parked apart
    CHECK(stream->parked_count(0) == 2);
    CHECK(stream->parked_count(7) == 1);
    std::vector<std::pair<ChunkId, uint64_t>> back;
    stream->take_ready(0, back);
    CHECK(back.empty());

    stream->poll();
    CHECK(stream->admit(other, 0, 33) == InstanceStream::PARKED);
    stream->take_ready(0, back);
    CHECK(back.size() == 1);
    if (!(back.size() == 1)) {
        return;
    }
    CHECK(back[0] == std::make_pair(fine, uint64_t(23)));
    stream->take_ready(7, back); // to its owner only
    CHECK(back.size() == 1);
    CHECK(stream->parked_count(7) == 0);

    // The coarse chunk lands with the next poll, the other one too: handed back in park order.
    stream->poll();
    stream->take_ready(0, back);
    CHECK(back.size() == 2);
    if (!(back.size() == 2)) {
        return;
    }
    CHECK(back[0] == std::make_pair(coarse, uint64_t(11)));
    CHECK(back[1] == std::make_pair(other, uint64_t(33)));

    mirror.update(*stream);
    for (const ChunkId &c : { coarse, fine, other }) {
        CHECK(stream->admit(c, 0, 0) == InstanceStream::READY);
        CHECK(lists_right(fx, *stream, mirror, c));
    }
    CHECK_FALSE(fx.expected(fine).empty());
    CHECK(stat(*stream, "lists_missing") == 0);
    CHECK(stat(*stream, "chunks_parked") == 0);

    // The CPU evaluation's records: the resident keys' (a voxel of the fine chunk), else computed and dropped.
    std::vector<TableRecord> cpu;
    stream->point_records(fx.origin + VoxelPos(200, 0, 200), 0, cpu);
    CHECK(stat(*stream, "point_computed") == 0);
    stream->point_records(fx.origin + VoxelPos(400, 0, -400), 0, cpu);
    CHECK(stat(*stream, "point_computed") > 0);
}

TEST_CASE("Instance stream: pinned keys and keys in their grace are never evicted, the others are") {
    const StreamFixture &fx = StreamFixture::get();
    InstanceStream::Settings settings;
    settings.entries = 0; // every key past its grace and unpinned goes at each poll
    std::unique_ptr<InstanceStream> stream = fx.stream(settings);
    Mirror mirror;

    const ChunkId graced = fx.chunk(300, 300), parked = fx.chunk(-300, 300);
    CHECK(stream->admit(graced, 0, 2) == InstanceStream::PARKED);
    stream->poll();
    std::vector<std::pair<ChunkId, uint64_t>> back;
    stream->take_ready(0, back);
    CHECK(back.size() == 1);
    if (!(back.size() == 1)) {
        return;
    }
    CHECK(stream->admit(graced, 0, 2) == InstanceStream::READY);

    // Within the grace, nothing goes; the mirror's lists stay whole.
    for (uint64_t f = 0; f < InstanceStream::GRACE_FRAMES; f++) {
        stream->poll();
    }
    mirror.update(*stream);
    CHECK(stat(*stream, "evicted") == 0);
    CHECK(lists_right(fx, *stream, mirror, graced));

    // A chunk parked over keys not computed yet pins them, landed or not; the graced chunk's keys go once its grace is over.
    CHECK(stream->admit(parked, 0, 3) == InstanceStream::PARKED);
    for (uint64_t f = 0; f < 4; f++) {
        stream->poll();
    }
    std::vector<SourceKey> keys;
    fx.source->chunk_keys(parked, fx.param, keys);
    CHECK(stat(*stream, "evicted") > 0);
    CHECK(stat(*stream, "keys") == int64_t(keys.size()));
    stream->take_ready(0, back);
    CHECK(back.size() == 1);
    if (!(back.size() == 1)) {
        return;
    } // its keys landed and stayed: pinned
    CHECK(back[0].first == parked);
    mirror.update(*stream);
    CHECK(stream->admit(parked, 0, 3) == InstanceStream::READY);
    CHECK(lists_right(fx, *stream, mirror, parked));
    // The graced chunk's keys were evicted: asked again, they are computed again (the same records).
    CHECK(stream->admit(graced, 0, 2) == InstanceStream::PARKED);
    stream->poll();
    stream->take_ready(0, back);
    mirror.update(*stream);
    CHECK(stream->admit(graced, 0, 2) == InstanceStream::READY);
    CHECK(lists_right(fx, *stream, mirror, graced));
}

TEST_CASE("Instance stream: evicted keys' slots are reused, and the placement log keeps a reader's mirror right") {
    const StreamFixture &fx = StreamFixture::get();
    const ChunkId a = fx.chunk(-300, -300), b = fx.chunk(300, 300), c = fx.chunk(-300, 300);
    InstanceStream::Settings settings;
    settings.slot_base = 1 << 20;
    settings.slot_capacity = std::max({ fx.slots(a), fx.slots(b), fx.slots(c) }) + 8; // one chunk's keys at a time
    settings.log = 4;
    std::unique_ptr<InstanceStream> stream = fx.stream(settings);
    Mirror live, late; // updated every step / once at the end (past the log: a replay)

    std::vector<std::pair<ChunkId, uint64_t>> back;
    for (const ChunkId &chunk : { a, b, c }) {
        CHECK(stream->admit(chunk, 0, 0) == InstanceStream::PARKED);
        stream->poll();
        stream->take_ready(0, back);
        CHECK(back.size() == 1);
        if (!(back.size() == 1)) {
            return;
        }
        CHECK(stream->admit(chunk, 0, 0) == InstanceStream::READY);
        live.update(*stream);
        CHECK(lists_right(fx, *stream, live, chunk));
        CHECK(stat(*stream, "slots_used") <= int64_t(settings.slot_capacity));
        for (uint64_t f = 0; f <= InstanceStream::GRACE_FRAMES; f++) {
            stream->poll(); // its batch is done: its keys may go
        }
    }
    CHECK(stat(*stream, "evicted") > 0);
    CHECK(stat(*stream, "slots_full") == 0);
    CHECK(live.records.size() <= settings.slot_capacity);
    late.update(*stream);
    CHECK(lists_right(fx, *stream, late, c));
}

TEST_CASE("Instance stream: with the slots full, a chunk answers LATER until room comes back (no deadlock)") {
    const StreamFixture &fx = StreamFixture::get();
    const ChunkId a = fx.chunk(-300, -300), b = fx.chunk(300, 300);
    InstanceStream::Settings settings;
    settings.slot_capacity = std::max(fx.slots(a), fx.slots(b)) + 8;
    std::unique_ptr<InstanceStream> stream = fx.stream(settings);
    Mirror mirror;

    // Both park, their keys land together: the first parked takes the slots, the other is handed back without them.
    CHECK(stream->admit(a, 0, 1) == InstanceStream::PARKED);
    CHECK(stream->admit(b, 0, 2) == InstanceStream::PARKED);
    stream->poll();
    std::vector<std::pair<ChunkId, uint64_t>> back;
    stream->take_ready(0, back);
    CHECK(back.size() == 2);
    if (!(back.size() == 2)) {
        return;
    }
    CHECK(stream->admit(a, 0, 1) == InstanceStream::READY);
    CHECK(stream->admit(b, 0, 2) == InstanceStream::LATER);
    CHECK(stat(*stream, "slots_full") > 0);
    CHECK(stat(*stream, "chunks_parked") == 0); // nothing pinned while it waits

    // Once a's grace is over, b's keys take its slots.
    int frames = 0;
    InstanceStream::Admit answer = InstanceStream::LATER;
    for (; frames < 4 * int(InstanceStream::GRACE_FRAMES) && answer != InstanceStream::READY; frames++) {
        stream->poll();
        stream->take_ready(0, back);
        answer = stream->admit(b, 0, 2);
    }
    CHECK(answer == InstanceStream::READY);
    CHECK(frames > int(InstanceStream::GRACE_FRAMES) - 2);
    mirror.update(*stream);
    CHECK(lists_right(fx, *stream, mirror, b));
}

TEST_CASE("Instance stream: on Godot's worker pool, parked chunks come back with the same lists") {
    const StreamFixture &fx = StreamFixture::get();
    std::unique_ptr<InstanceStream> stream = std::make_unique<InstanceStream>(std::vector<std::shared_ptr<const ScatterSource>>{ fx.source }, fx.param,
            InstanceStream::Settings(), StreamExecutor::pool());
    const std::vector<ChunkId> chunks = { fx.chunk(-200, -200, 1), fx.chunk(200, 200), fx.chunk(-200, 200), fx.chunk(300, -300, 1) };
    for (const ChunkId &c : chunks) {
        CHECK(stream->admit(c, 0, 0) == InstanceStream::PARKED);
    }
    std::vector<std::pair<ChunkId, uint64_t>> back, all;
    const uint64_t t0 = now_usec();
    while (all.size() < chunks.size() && now_usec() - t0 < 30000000) {
        stream->poll();
        stream->take_ready(0, back);
        all.insert(all.end(), back.begin(), back.end());
    }
    CHECK(all.size() == chunks.size());
    Mirror mirror;
    mirror.update(*stream);
    for (const ChunkId &c : chunks) {
        CHECK(stream->admit(c, 0, 0) == InstanceStream::READY);
        CHECK(lists_right(fx, *stream, mirror, c));
    }
}
