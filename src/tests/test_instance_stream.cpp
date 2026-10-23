#include "test_common.hpp"

#include "test_stream_common.hpp"

#include "debug_log.hpp"
#include "instance_stream.hpp"

#include <algorithm>
#include <map>

using namespace Archetype;
using StreamTest::Mirror;

namespace {

// The inline executor, held: while held, reap() runs nothing, so a poll() turns the frame without landing a key. Released,
// a poll()'s drain takes at most `steps` jobs (-1: all), the rest left for the next poll().
struct HeldExecutor : StreamExecutor {
    bool held = true;
    int64_t steps = -1;
    void reap(bool p_wait) override {
        if (!held || p_wait) {
            StreamExecutor::reap(p_wait);
        }
    }
    bool more() override {
        if (steps == 0) {
            return false;
        }
        steps -= steps > 0;
        return true;
    }
};

// A cheap source: one sphere of 4-8 voxels per 16-voxel cell site on a flat ground far from the origin, no blobs. The
// stream runs its keys on the thread calling poll() (the base executor), so a key asked for lands by the next poll().
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

    std::unique_ptr<InstanceStream> stream(InstanceStream::Settings p_settings, std::unique_ptr<StreamExecutor> p_executor = nullptr) const {
        if (!p_executor) {
            p_executor = std::make_unique<StreamExecutor>();
        }
        return std::make_unique<InstanceStream>(std::vector<std::shared_ptr<const ScatterSource>>{ source }, param, p_settings, std::move(p_executor));
    }
    // A stream on a held executor (r_held: to release it).
    std::unique_ptr<InstanceStream> held_stream(HeldExecutor *&r_held) const {
        auto executor = std::make_unique<HeldExecutor>();
        r_held = executor.get();
        return stream(InstanceStream::Settings(), std::move(executor));
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

// A field of computed_by_key's entry p_key ("sites 0"), 0 if none was computed.
int64_t key_stat(const InstanceStream &p_stream, const char *p_key, const char *p_field) {
    const Dictionary by = p_stream.get_stats()["computed_by_key"];
    return by.has(p_key) ? int64_t(Dictionary(by[p_key])[p_field]) : 0;
}

// p_keys and their inputs, each once.
size_t keys_and_inputs(const StreamFixture &p_fx, std::vector<SourceKey> p_keys) {
    std::vector<SourceKey> in;
    for (size_t i = 0, n = p_keys.size(); i < n; i++) {
        p_fx.source->inputs(p_keys[i], in);
        p_keys.insert(p_keys.end(), in.begin(), in.end());
    }
    std::sort(p_keys.begin(), p_keys.end());
    return size_t(std::unique(p_keys.begin(), p_keys.end()) - p_keys.begin());
}

// The chunk's keys and their inputs, each once.
size_t keys_and_inputs(const StreamFixture &p_fx, const ChunkId &p_chunk) {
    std::vector<SourceKey> keys;
    p_fx.source->chunk_keys(p_chunk, p_fx.param, keys);
    return keys_and_inputs(p_fx, keys);
}

// The chunks' keys and their inputs, each once.
size_t keys_and_inputs(const StreamFixture &p_fx, const std::vector<ChunkId> &p_chunks) {
    std::vector<SourceKey> all, one;
    for (const ChunkId &c : p_chunks) {
        p_fx.source->chunk_keys(c, p_fx.param, one);
        all.insert(all.end(), one.begin(), one.end());
    }
    return keys_and_inputs(p_fx, all);
}

// The chunk's own keys (what admit() creates); r_reading: those with inputs.
size_t own_keys(const StreamFixture &p_fx, const ChunkId &p_chunk, size_t *r_reading = nullptr) {
    std::vector<SourceKey> keys, in;
    p_fx.source->chunk_keys(p_chunk, p_fx.param, keys);
    if (r_reading) {
        *r_reading = 0;
        for (const SourceKey &k : keys) {
            p_fx.source->inputs(k, in);
            *r_reading += !in.empty();
        }
    }
    return keys.size();
}

// The keys in the order they were placed: a listed key is placed as it lands, so with the inline executor (one drain task,
// every job of the stream's queue) it is the order the jobs ran in. Keys without records aren't logged.
std::vector<SourceKey> placement_order(const InstanceStream &p_stream) {
    std::vector<InstanceStream::Upload> ups;
    p_stream.uploads_since(0, ups);
    std::vector<SourceKey> out;
    for (const InstanceStream::Upload &u : ups) {
        out.push_back(u.key);
    }
    return out;
}

// The first and last position in p_order of the chunk's keys ({-1, -1}: none placed).
std::pair<int, int> placed_span(const StreamFixture &p_fx, const ChunkId &p_chunk, const std::vector<SourceKey> &p_order) {
    std::vector<SourceKey> keys;
    p_fx.source->chunk_keys(p_chunk, p_fx.param, keys);
    std::pair<int, int> span(-1, -1);
    for (int i = 0; i < int(p_order.size()); i++) {
        if (std::find(keys.begin(), keys.end(), p_order[i]) != keys.end()) {
            span.first = span.first < 0 ? i : span.first;
            span.second = i;
        }
    }
    return span;
}

// Whether every key of p_before's chunk was placed before any of p_after's (both placed).
bool placed_before(const StreamFixture &p_fx, const ChunkId &p_before, const ChunkId &p_after, const std::vector<SourceKey> &p_order) {
    const std::pair<int, int> a = placed_span(p_fx, p_before, p_order), b = placed_span(p_fx, p_after, p_order);
    return a.first >= 0 && b.first >= 0 && a.second < b.first;
}

// Whether no two of the chunks share a key or an input.
bool disjoint(const StreamFixture &p_fx, const std::vector<ChunkId> &p_chunks) {
    std::vector<SourceKey> all, one;
    size_t each = 0;
    for (const ChunkId &c : p_chunks) {
        p_fx.source->chunk_keys(c, p_fx.param, one);
        all.insert(all.end(), one.begin(), one.end());
        each += keys_and_inputs(p_fx, c);
    }
    return keys_and_inputs(p_fx, all) == each;
}

// A scan's set_tag rank: no chunk wanted.
int64_t forget_all(const ChunkId &) {
    return -1;
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

    // A key and the keys waiting on it land within one poll: a part tile starts once the site tiles it grows land, on the
    // same drain.
    const ChunkId coarse = fx.chunk(-200, -200, 1), fine = fx.chunk(200, 200), other = fx.chunk(-200, 200);
    CHECK(stream->admit(coarse, 0, 11, 0) == InstanceStream::PARKED);
    CHECK(stream->admit(fine, 0, 22, 0) == InstanceStream::PARKED);
    CHECK(stream->admit(fine, 0, 23, 0) == InstanceStream::PARKED); // asked again: still parked, the new tag
    CHECK(stream->admit(fine, 7, 70, 0) == InstanceStream::PARKED); // another mesher's: parked apart
    CHECK(stream->parked_count(0) == 2);
    CHECK(stream->parked_count(7) == 1);
    std::vector<std::pair<ChunkId, uint64_t>> back;
    stream->take_ready(0, back);
    CHECK(back.empty());

    stream->poll();
    CHECK(stream->admit(other, 0, 33, 0) == InstanceStream::PARKED);
    stream->take_ready(0, back);
    CHECK(back.size() == 2); // handed back in park order
    if (!(back.size() == 2)) {
        return;
    }
    CHECK(back[0] == std::make_pair(coarse, uint64_t(11)));
    CHECK(back[1] == std::make_pair(fine, uint64_t(23)));
    stream->take_ready(7, back); // to its owner only
    CHECK(back.size() == 1);
    CHECK(stream->parked_count(7) == 0);

    // Parked after the poll: back with the next one.
    stream->poll();
    stream->take_ready(0, back);
    CHECK(back.size() == 1);
    if (!(back.size() == 1)) {
        return;
    }
    CHECK(back[0] == std::make_pair(other, uint64_t(33)));
    // Each parked chunk came back on the poll after its park, whose reap lands the keys before the frame turns: 0 frames,
    // 4 hand-backs.
    const Dictionary park_frames = stream->get_stats()["park_frames"];
    CHECK(int64_t(park_frames["count"]) == 4);
    CHECK(int64_t(park_frames["max"]) == 0);
    CHECK(stat(*stream, "admit_keys_created_max") > 0);
    CHECK(stat(*stream, "keys_created") >= stat(*stream, "admit_keys_created"));
    CHECK(stat(*stream, "jobs_queued_max") > 0);

    mirror.update(*stream);
    for (const ChunkId &c : { coarse, fine, other }) {
        CHECK(stream->admit(c, 0, 0, 0) == InstanceStream::READY);
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

    // The camera's queries compute nothing on their thread: a key not resident is left out, asked for by the next poll,
    // and once it has landed (the one after) the records are point_records'. A grid of voxels away from the chunks asked for.
    const int64_t computed = stat(*stream, "point_computed");
    size_t records = 0;
    for (int64_t z = -600; z <= -200; z += 100) {
        for (int64_t x = 200; x <= 600; x += 100) {
            const VoxelPos v = fx.origin + VoxelPos(x, 0, z);
            std::vector<TableRecord> view, point;
            stream->view_records(v, 0, view);
            stream->poll();
            stream->poll();
            view.clear();
            stream->view_records(v, 0, view);
            stream->point_records(v, 0, point);
            CHECK(view.size() == point.size());
            records += point.size();
        }
    }
    CHECK(stat(*stream, "point_computed") == computed); // every key resident by then
    CHECK(stat(*stream, "view_missing") > 0);
    CHECK(records > 0);
}

TEST_CASE("Instance stream: pinned keys and keys in their grace are never evicted, the others are") {
    const StreamFixture &fx = StreamFixture::get();
    InstanceStream::Settings settings;
    settings.entries = 0; // every key past its grace and unpinned goes at each poll
    settings.walks = 0;
    std::unique_ptr<InstanceStream> stream = fx.stream(settings);
    Mirror mirror;

    const ChunkId graced = fx.chunk(300, 300), parked = fx.chunk(-300, 300);
    CHECK(stream->admit(graced, 0, 2, 0) == InstanceStream::PARKED);
    stream->poll();
    std::vector<std::pair<ChunkId, uint64_t>> back;
    stream->take_ready(0, back);
    CHECK(back.size() == 1);
    if (!(back.size() == 1)) {
        return;
    }
    CHECK(stream->admit(graced, 0, 2, 0) == InstanceStream::READY);

    // Within the grace, no listed key goes (its site tiles may: asked for a frame before); the mirror's lists stay whole.
    for (uint64_t f = 0; f < InstanceStream::GRACE_FRAMES; f++) {
        stream->poll();
    }
    mirror.update(*stream);
    CHECK(key_stat(*stream, "parts 0", "evicted") == 0);
    CHECK(lists_right(fx, *stream, mirror, graced));

    // A chunk parked over keys not computed yet pins them, landed or not; the graced chunk's keys go once its grace is over.
    CHECK(stream->admit(parked, 0, 3, 0) == InstanceStream::PARKED);
    for (uint64_t f = 0; f < 4; f++) {
        stream->poll();
    }
    CHECK(stat(*stream, "evicted") > 0);
    CHECK(stat(*stream, "keys") == int64_t(keys_and_inputs(fx, parked))); // its site tiles still in their grace
    stream->take_ready(0, back);
    CHECK(back.size() == 1);
    if (!(back.size() == 1)) {
        return;
    } // its keys landed and stayed: pinned
    CHECK(back[0].first == parked);
    mirror.update(*stream);
    CHECK(stream->admit(parked, 0, 3, 0) == InstanceStream::READY);
    CHECK(lists_right(fx, *stream, mirror, parked));
    // The evictions per kind and LOD add up to the total.
    const Dictionary by_key = stream->get_stats()["computed_by_key"];
    int64_t evicted = 0;
    const Array kinds = by_key.keys();
    for (int64_t i = 0; i < kinds.size(); i++) {
        evicted += int64_t(Dictionary(by_key[kinds[i]])["evicted"]);
    }
    CHECK(evicted == stat(*stream, "evicted"));
    CHECK(key_stat(*stream, "sites 0", "evicted") > 0); // the walks' cap evicts the site tiles
    // The graced chunk's keys were evicted: asked again, they are computed again (the same records).
    CHECK(stream->admit(graced, 0, 2, 0) == InstanceStream::PARKED);
    stream->poll();
    stream->take_ready(0, back);
    mirror.update(*stream);
    CHECK(stream->admit(graced, 0, 2, 0) == InstanceStream::READY);
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
        CHECK(stream->admit(chunk, 0, 0, 0) == InstanceStream::PARKED);
        stream->poll();
        stream->take_ready(0, back);
        CHECK(back.size() == 1);
        if (!(back.size() == 1)) {
            return;
        }
        CHECK(stream->admit(chunk, 0, 0, 0) == InstanceStream::READY);
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
    CHECK(stream->admit(a, 0, 1, 0) == InstanceStream::PARKED);
    CHECK(stream->admit(b, 0, 2, 0) == InstanceStream::PARKED);
    stream->poll();
    std::vector<std::pair<ChunkId, uint64_t>> back;
    stream->take_ready(0, back);
    CHECK(back.size() == 2);
    if (!(back.size() == 2)) {
        return;
    }
    CHECK(stream->admit(a, 0, 1, 0) == InstanceStream::READY);
    CHECK(stream->admit(b, 0, 2, 0) == InstanceStream::LATER);
    CHECK(stat(*stream, "slots_full") > 0);
    CHECK(stat(*stream, "chunks_parked") == 0); // nothing pinned while it waits

    // Once a's grace is over, b's keys take its slots.
    int frames = 0;
    InstanceStream::Admit answer = InstanceStream::LATER;
    for (; frames < 4 * int(InstanceStream::GRACE_FRAMES) && answer != InstanceStream::READY; frames++) {
        stream->poll();
        stream->take_ready(0, back);
        answer = stream->admit(b, 0, 2, 0);
    }
    CHECK(answer == InstanceStream::READY);
    CHECK(frames > int(InstanceStream::GRACE_FRAMES) - 2);
    mirror.update(*stream);
    CHECK(lists_right(fx, *stream, mirror, b));
}

TEST_CASE("Instance stream: past the pending keys' cap a chunk needing a new key answers LATER, then goes through") {
    const StreamFixture &fx = StreamFixture::get();
    const ChunkId a = fx.chunk(-300, -300), b = fx.chunk(300, 300);
    InstanceStream::Settings settings;
    settings.pending = 1;
    std::unique_ptr<InstanceStream> stream = fx.stream(settings);
    Mirror mirror;

    // A frame asks for one new key and what it reads, then answers LATER: the next frame, once it landed, asks for another.
    CHECK(stream->admit(a, 0, 1, 0) == InstanceStream::LATER);
    const int64_t first = stat(*stream, "keys");
    CHECK(stream->admit(b, 0, 2, 0) == InstanceStream::LATER);
    CHECK(stat(*stream, "keys") == first); // nothing new asked for past the cap
    std::vector<std::pair<ChunkId, uint64_t>> back;
    InstanceStream::Admit answer_a = InstanceStream::LATER, answer_b = InstanceStream::LATER;
    for (int frame = 0; frame < 64 && (answer_a != InstanceStream::READY || answer_b != InstanceStream::READY); frame++) {
        stream->poll();
        stream->take_ready(0, back);
        answer_a = stream->admit(a, 0, 1, 0);
        answer_b = stream->admit(b, 0, 2, 0);
    }
    CHECK(answer_a == InstanceStream::READY);
    CHECK(answer_b == InstanceStream::READY);
    mirror.update(*stream);
    CHECK(lists_right(fx, *stream, mirror, a));
    CHECK(lists_right(fx, *stream, mirror, b));
}

TEST_CASE("Instance stream: making room for inputs evicts the records not placed, never a placed key or a site tile") {
    const StreamFixture &fx = StreamFixture::get();
    // c's keys read site tiles (no records); the camera's view of a voxel far from it asks for a LOD-0 tile not listed: its
    // records are the only inputs.
    const ChunkId c = fx.chunk(-300, -300, 2);
    InstanceStream::Settings settings;
    settings.input_records = 0;
    std::unique_ptr<InstanceStream> stream = fx.stream(settings);
    std::vector<std::pair<ChunkId, uint64_t>> back;
    InstanceStream::Admit answer = InstanceStream::LATER;
    for (int frame = 0; frame < 16 && answer != InstanceStream::READY; frame++) {
        answer = stream->admit(c, 0, 1, 0);
        stream->poll();
        stream->take_ready(0, back);
    }
    CHECK(answer == InstanceStream::READY);
    std::vector<TableRecord> view;
    stream->view_records(fx.origin + VoxelPos(400, 0, 400), 0, view);
    stream->poll(); // asks for it
    stream->poll(); // it lands
    CHECK(stat(*stream, "input_records") > 0);
    const int64_t computed = stat(*stream, "computed");
    for (int frame = 0; frame < 2 * int(InstanceStream::GRACE_FRAMES); frame++) {
        stream->poll();
    }
    CHECK(stat(*stream, "input_records") == 0);
    CHECK(key_stat(*stream, "parts 0", "evicted") > 0); // the viewed tile went
    CHECK(key_stat(*stream, "sites 0", "evicted") == 0); // the site tiles stay: evicting them frees no record
    CHECK(stream->admit(c, 0, 1, 0) == InstanceStream::READY);
    CHECK(stat(*stream, "computed") == computed);
}

// Evicting for input records takes the oldest key holding them and moves nothing else: the entries cap then still finds the
// least recently used keys with records first, and the walks have a cap of their own (two chains: neither cap evicts the
// other's keys).
TEST_CASE("Instance stream: the entries cap evicts the oldest keys with records, the walks' cap the oldest walks") {
    const StreamFixture &fx = StreamFixture::get();
    const ChunkId a = fx.chunk(-300, -300, 2);
    const VoxelPos v1 = fx.origin + VoxelPos(400, 0, 400), v2 = fx.origin + VoxelPos(400, 0, -400);
    // The keys with records and the walks of a and of v1 and v2's views (their inputs included).
    const auto split = [&fx](std::vector<SourceKey> p_keys, size_t &r_records, size_t &r_walks) {
        std::vector<SourceKey> in;
        for (size_t i = 0, n = p_keys.size(); i < n; i++) {
            fx.source->inputs(p_keys[i], in);
            p_keys.insert(p_keys.end(), in.begin(), in.end());
        }
        std::sort(p_keys.begin(), p_keys.end());
        p_keys.erase(std::unique(p_keys.begin(), p_keys.end()), p_keys.end());
        r_walks = size_t(std::count_if(p_keys.begin(), p_keys.end(), [](const SourceKey &p_k) { return p_k.walk(); }));
        r_records = p_keys.size() - r_walks;
    };
    std::vector<SourceKey> a_keys, v1_keys, v2_keys;
    fx.source->chunk_keys(a, fx.param, a_keys);
    fx.source->keys(v1, v1, 0, v1_keys);
    fx.source->keys(v2, v2, 0, v2_keys);
    size_t a_records = 0, a_walks = 0, v1_records = 0, v1_walks = 0, v2_records = 0, v2_walks = 0;
    split(a_keys, a_records, a_walks);
    split(v1_keys, v1_records, v1_walks);
    split(v2_keys, v2_records, v2_walks);
    // What the assertions below rest on (a fixture change breaking it fails here, for this reason).
    CHECK((a_walks > 0 && v2_walks > 0 && v2_records <= a_records && v2_walks <= a_walks));
    if (!(a_walks > 0 && v2_walks > 0 && v2_records <= a_records && v2_walks <= a_walks)) {
        return;
    }
    for (const bool walks_cap : { false, true }) {
        CAPTURE(walks_cap);
        InstanceStream::Settings settings;
        settings.input_records = 0;
        // One cap at what a and v1 hold, the other wide: v2 goes over the first only.
        settings.entries = walks_cap ? InstanceStream::ENTRIES_MAX : a_records + v1_records;
        settings.walks = walks_cap ? a_walks + v1_walks : InstanceStream::WALKS_MAX;
        std::unique_ptr<InstanceStream> stream = fx.stream(settings);
        std::vector<std::pair<ChunkId, uint64_t>> back;
        InstanceStream::Admit answer = InstanceStream::LATER;
        for (int frame = 0; frame < 16 && answer != InstanceStream::READY; frame++) {
            answer = stream->admit(a, 0, 1, 0);
            stream->poll();
            stream->take_ready(0, back);
        }
        CHECK(answer == InstanceStream::READY);
        for (uint64_t f = 0; f < InstanceStream::GRACE_FRAMES + 2; f++) {
            stream->poll(); // a's keys out of their grace
        }
        // v1's tile lands unplaced: the inputs' eviction finds it in its grace (and leaves a's keys where they are).
        std::vector<TableRecord> view;
        stream->view_records(v1, 0, view);
        stream->poll();
        stream->poll();
        CHECK(stat(*stream, "input_records") > 0);
        CHECK(stat(*stream, "evicted") == 0);
        // v2's keys go over one cap: a's keys of that chain go, the least recently used, never v1's or v2's in their grace,
        // never the other chain's.
        stream->view_records(v2, 0, view);
        stream->poll();
        stream->poll();
        const int64_t sites = key_stat(*stream, "sites 0", "evicted");
        const int64_t parts = key_stat(*stream, "parts 0", "evicted") + key_stat(*stream, "parts 2", "evicted");
        CHECK(stat(*stream, "evicted") > 0);
        CHECK(stat(*stream, "evicted") == sites + parts);
        CHECK((walks_cap ? parts : sites) == 0);
        if (walks_cap) {
            CHECK(stat(*stream, "walks_ready") == int64_t(settings.walks));
        }
    }
}

// A LOD sharing the LOD-0 tile side has no part key of its own: its chunks read the LOD-0 keys and keep what is listed at
// their LOD.
TEST_CASE("Instance stream: a chunk of a LOD with the LOD-0 tile side lists the LOD-0 keys' records listed at its LOD") {
    const StreamFixture &fx = StreamFixture::get();
    const ScatterSource &f = *fx.source;
    int lod = 0;
    while (lod + 1 < f.profile().lod_end && f.part_tile_side(lod + 1) == f.part_tile_side(0)) {
        lod++;
    }
    CHECK(lod >= 1);
    CHECK(f.part_key_lod(lod) == 0);
    CHECK(f.part_key_lod(lod + 1) == lod + 1);
    const ChunkId chunk = fx.chunk(200, -200, lod);
    std::vector<SourceKey> keys;
    f.chunk_keys(chunk, fx.param, keys);
    CHECK_FALSE(keys.empty());
    CHECK(std::all_of(keys.begin(), keys.end(), [](const SourceKey &p_k) { return p_k.kind == SourceKey::PARTS && p_k.lod == 0; }));

    std::unique_ptr<InstanceStream> stream = fx.stream(InstanceStream::Settings());
    std::vector<std::pair<ChunkId, uint64_t>> back;
    CHECK(stream->admit(chunk, 0, 0, 0) == InstanceStream::PARKED);
    stream->poll();
    stream->take_ready(0, back);
    CHECK(stream->admit(chunk, 0, 0, 0) == InstanceStream::READY);
    CHECK(key_stat(*stream, vformat("parts %d", lod).utf8().get_data(), "count") == 0);
    Mirror mirror;
    mirror.update(*stream);
    CHECK(lists_right(fx, *stream, mirror, chunk));
    // Its list: the LOD-0 keys' records listed at its LOD, and not all of them.
    std::vector<uint32_t> ids;
    stream->chunk_list(chunk, ids);
    CHECK_FALSE(ids.empty());
    size_t all = 0;
    for (const SourceKey &k : keys) {
        all += fx.content(k).records.size();
    }
    CHECK(ids.size() < all);
    CHECK(std::all_of(ids.begin(), ids.end(), [&](uint32_t p_id) { return mirror.records[p_id - stream->slot_base()].listed_at(lod); }));
}

TEST_CASE("Instance stream: on Godot's worker pool, parked chunks come back with the same lists") {
    const StreamFixture &fx = StreamFixture::get();
    std::unique_ptr<InstanceStream> stream = std::make_unique<InstanceStream>(std::vector<std::shared_ptr<const ScatterSource>>{ fx.source }, fx.param,
            InstanceStream::Settings(), StreamExecutor::pool());
    const std::vector<ChunkId> chunks = { fx.chunk(-200, -200, 1), fx.chunk(200, 200), fx.chunk(-200, 200), fx.chunk(300, -300, 1) };
    for (const ChunkId &c : chunks) {
        CHECK(stream->admit(c, 0, 0, 0) == InstanceStream::PARKED);
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
        CHECK(stream->admit(c, 0, 0, 0) == InstanceStream::READY);
        CHECK(lists_right(fx, *stream, mirror, c));
    }
}

TEST_CASE("Instance stream: jobs run by rank, then in the order asked") {
    const StreamFixture &fx = StreamFixture::get();
    const ChunkId a = fx.chunk(-300, -300), b = fx.chunk(300, 300), c = fx.chunk(-300, 300);
    CHECK(disjoint(fx, { a, b, c }));
    std::unique_ptr<InstanceStream> stream = fx.stream(InstanceStream::Settings());
    CHECK(stream->admit(a, 0, 0, 5) == InstanceStream::PARKED);
    CHECK(stream->admit(b, 0, 0, 1) == InstanceStream::PARKED);
    CHECK(stream->admit(c, 0, 0, 3) == InstanceStream::PARKED);
    stream->poll();
    const std::vector<SourceKey> order = placement_order(*stream);
    CHECK(placed_before(fx, b, c, order));
    CHECK(placed_before(fx, c, a, order));
    CHECK(stat(*stream, "jobs_reranked") == 0);
}

// admit() runs on the main thread: it creates the chunk's own keys only. Their first job, on a drain task, lists their
// inputs and asks for them at the key's rank; the key runs once they have landed, and the chunk comes back.
TEST_CASE("Instance stream: admit creates a chunk's own keys, a drain task asks for their inputs; a dropped chunk drops their jobs") {
    const StreamFixture &fx = StreamFixture::get();
    const ScatterSource &f = *fx.source;
    int lod = 1; // part keys of its own: their inputs are the site tiles inside them
    while (lod < f.profile().lod_end && f.part_key_lod(lod) != lod) {
        lod++;
    }
    const ChunkId c = fx.chunk(-200, -200, lod), d = fx.chunk(200, 200, lod);
    CHECK(disjoint(fx, { c, d }));
    size_t c_reading = 0, d_reading = 0;
    const size_t c_own = own_keys(fx, c, &c_reading), d_own = own_keys(fx, d, &d_reading);
    CHECK((c_reading > 0 && d_reading > 0 && keys_and_inputs(fx, c) > c_own));
    HeldExecutor *held = nullptr;
    std::unique_ptr<InstanceStream> stream = fx.held_stream(held);
    CHECK(stream->admit(c, 0, 1, 3) == InstanceStream::PARKED);
    CHECK(stat(*stream, "keys") == int64_t(c_own));
    CHECK(stat(*stream, "admit_keys_created_max") == int64_t(c_own));
    stream->poll(); // held: nothing runs
    CHECK(stat(*stream, "keys_expanded") == 0);

    // One job per key: those with inputs list and ask for them, the others are computed.
    held->held = false;
    held->steps = int64_t(c_own);
    stream->poll();
    CHECK(stat(*stream, "keys_expanded") == int64_t(c_reading));
    CHECK(stat(*stream, "keys") == int64_t(keys_and_inputs(fx, c)));
    CHECK(stat(*stream, "admit_keys_created") == int64_t(c_own));
    std::vector<std::pair<ChunkId, uint64_t>> back;
    stream->take_ready(0, back);
    CHECK(back.empty());
    held->steps = -1;
    stream->poll();
    stream->take_ready(0, back);
    CHECK(back.size() == 1);
    if (!(back.size() == 1)) {
        return;
    }
    CHECK(back[0] == std::make_pair(c, uint64_t(1)));
    Mirror mirror;
    mirror.update(*stream);
    // READY: the records of its keys, a bound of its list.
    uint64_t words = 0;
    CHECK(stream->admit(c, 0, 1, 3, &words) == InstanceStream::READY);
    CHECK(lists_right(fx, *stream, mirror, c));
    std::vector<uint32_t> ids;
    stream->chunk_list(c, ids);
    CHECK(words == fx.slots(c));
    CHECK(words >= ids.size());
    CHECK(!ids.empty());

    // A chunk forgotten once its keys listed their inputs: the waiting keys go, and their inputs' queued jobs.
    held->steps = int64_t(d_own);
    CHECK(stream->admit(d, 0, 1, 3) == InstanceStream::PARKED);
    stream->poll();
    CHECK(stat(*stream, "keys") == int64_t(keys_and_inputs(fx, c) + keys_and_inputs(fx, d)));
    stream->set_tag(0, 2, forget_all, [](const ChunkId &) {});
    const int64_t d_inputs = int64_t(keys_and_inputs(fx, d) - d_own);
    CHECK(stat(*stream, "jobs_dropped") == d_inputs);
    CHECK(stat(*stream, "keys_dropped") == d_inputs + int64_t(d_reading)); // the keys without inputs landed already
    CHECK(stat(*stream, "keys") == int64_t(keys_and_inputs(fx, c) + d_own - d_reading));
    CHECK(stat(*stream, "lists_missing") == 0);
}

// The pending cap counts the keys chunks and the camera ask for: the inputs their expansions ask for (a coarse part tile's
// site tiles: thousands) don't turn the next chunk's admit into LATER.
TEST_CASE("Instance stream: an expansion's pending inputs don't count against the pending cap") {
    const StreamFixture &fx = StreamFixture::get();
    const ScatterSource &f = *fx.source;
    int lod = 1;
    while (lod < f.profile().lod_end && f.part_key_lod(lod) != lod) {
        lod++;
    }
    const ChunkId a = fx.chunk(-200, -200, lod), b = fx.chunk(300, 300);
    CHECK(disjoint(fx, { a, b }));
    size_t a_reading = 0;
    const size_t a_own = own_keys(fx, a, &a_reading), b_own = own_keys(fx, b);
    const size_t a_inputs = keys_and_inputs(fx, a) - a_own;
    // a's waiting keys and b's fill the cap exactly; a's waiting keys and their inputs alone are past it.
    InstanceStream::Settings settings;
    settings.pending = a_reading + b_own;
    CHECK((a_own <= settings.pending && a_reading + a_inputs > settings.pending));
    HeldExecutor *held = nullptr;
    auto executor = std::make_unique<HeldExecutor>();
    held = executor.get();
    std::unique_ptr<InstanceStream> stream = fx.stream(settings, std::move(executor));
    held->held = false;
    held->steps = int64_t(a_own); // a's keys list their inputs, none of which runs
    CHECK(stream->admit(a, 0, 1, 1) == InstanceStream::PARKED);
    stream->poll();
    CHECK(stat(*stream, "keys_pending_inputs") == int64_t(a_inputs));
    CHECK(stat(*stream, "keys_pending_asked") == int64_t(a_reading));
    CHECK(stream->admit(b, 0, 1, 2) == InstanceStream::PARKED);
    CHECK(stat(*stream, "admit_later_pending") == 0);
    // Past the cap of asked keys, a chunk needing a new key answers LATER, counted.
    const ChunkId c = fx.chunk(-300, 300);
    CHECK(disjoint(fx, { a, b, c }));
    CHECK(stream->admit(c, 0, 1, 3) == InstanceStream::LATER);
    CHECK(stat(*stream, "admit_later_pending") == 1);
    held->steps = -1;
    stream->poll();
    stream->poll();
    std::vector<std::pair<ChunkId, uint64_t>> back;
    stream->take_ready(0, back);
    CHECK(back.size() == 2);
    CHECK(stat(*stream, "keys_pending_asked") == 0);
    CHECK(stat(*stream, "keys_pending_inputs") == 0);
}

TEST_CASE("Instance stream: an input not landed runs at the best rank of the keys waiting for it") {
    const StreamFixture &fx = StreamFixture::get();
    const ScatterSource &f = *fx.source;
    int lod = 1; // the first LOD with part keys of its own (their inputs: the site tiles inside them)
    while (lod < f.profile().lod_end && f.part_key_lod(lod) != lod) {
        lod++;
    }
    CHECK(lod < f.profile().lod_end);
    // A coarse chunk asked first at rank 9 creates the site tiles a fine chunk inside it reads; a rank-5 chunk away from
    // both; then the fine chunk at rank 2: its shared site tiles move ahead of the rank-5 chunk's keys.
    const ChunkId coarse = fx.chunk(200, 200, lod), fine = fx.chunk(200, 200), away = fx.chunk(-400, -400);
    CHECK(disjoint(fx, { coarse, away }));
    CHECK(disjoint(fx, { fine, away }));
    CHECK_FALSE(disjoint(fx, { coarse, fine })); // they share site tiles
    HeldExecutor *held = nullptr;
    std::unique_ptr<InstanceStream> stream = fx.held_stream(held);
    held->held = false;
    held->steps = int64_t(own_keys(fx, coarse)); // its keys list their inputs: the site tiles, queued at 9
    CHECK(stream->admit(coarse, 0, 0, 9) == InstanceStream::PARKED);
    stream->poll();
    CHECK(stat(*stream, "keys") == int64_t(keys_and_inputs(fx, coarse)));
    CHECK(stream->admit(away, 0, 0, 5) == InstanceStream::PARKED);
    CHECK(stream->admit(fine, 0, 0, 2) == InstanceStream::PARKED);
    held->steps = -1;
    stream->poll();
    CHECK(stat(*stream, "jobs_reranked") > 0); // the fine chunk's keys asked for the shared site tiles at 2
    const std::vector<SourceKey> order = placement_order(*stream);
    CHECK(placed_before(fx, fine, away, order));
    CHECK(placed_before(fx, away, coarse, order));
}

TEST_CASE("Instance stream: a parked chunk asked again with a better rank moves its queued jobs ahead") {
    const StreamFixture &fx = StreamFixture::get();
    const ChunkId x = fx.chunk(-300, -300), y = fx.chunk(300, 300);
    CHECK(disjoint(fx, { x, y }));
    std::unique_ptr<InstanceStream> stream = fx.stream(InstanceStream::Settings());
    CHECK(stream->admit(x, 0, 0, 7) == InstanceStream::PARKED);
    CHECK(stream->admit(y, 0, 0, 4) == InstanceStream::PARKED);
    CHECK(stream->admit(x, 0, 0, 1) == InstanceStream::PARKED);
    CHECK(stat(*stream, "jobs_reranked") > 0);
    stream->poll();
    CHECK(placed_before(fx, x, y, placement_order(*stream)));
}

TEST_CASE("Instance stream: set_tag forgets the owner's chunks parked under an older tag the scan doesn't want, and drops their jobs") {
    const StreamFixture &fx = StreamFixture::get();
    const ChunkId old_chunk = fx.chunk(-300, -300), current = fx.chunk(300, 300), other = fx.chunk(-300, 300);
    CHECK(disjoint(fx, { old_chunk, current, other }));
    HeldExecutor *held = nullptr;
    std::unique_ptr<InstanceStream> stream = fx.held_stream(held);
    CHECK(stream->admit(old_chunk, 0, 1, 1) == InstanceStream::PARKED);
    CHECK(stream->admit(current, 0, 2, 5) == InstanceStream::PARKED);
    CHECK(stream->admit(other, 7, 1, 5) == InstanceStream::PARKED); // another owner's: its tags are its own
    stream->poll(); // a frame later, nothing landed

    std::vector<ChunkId> kept;
    stream->set_tag(0, 2, forget_all, [&](const ChunkId &p_chunk) { kept.push_back(p_chunk); });
    CHECK(kept.empty());
    CHECK(stream->parked_count(0) == 1);
    CHECK(stream->parked_count(7) == 1);
    CHECK(stat(*stream, "stale_handed_back") == 1);
    // None of its keys had started: all erased with their queued jobs (their inputs not listed yet: admit() creates a
    // chunk's own keys only).
    CHECK(stat(*stream, "keys_dropped") == int64_t(own_keys(fx, old_chunk)));
    CHECK(stat(*stream, "jobs_dropped") == int64_t(own_keys(fx, old_chunk)));
    CHECK(stat(*stream, "keys") == int64_t(own_keys(fx, current) + own_keys(fx, other)));

    held->held = false;
    stream->poll();
    std::vector<std::pair<ChunkId, uint64_t>> back;
    stream->take_ready(0, back); // the forgotten chunk never comes back
    CHECK(back.size() == 1);
    if (!(back.size() == 1)) {
        return;
    }
    CHECK(back[0] == std::make_pair(current, uint64_t(2)));
    CHECK(placed_span(fx, old_chunk, placement_order(*stream)).first == -1);
    const Dictionary park_frames = stream->get_stats()["park_frames"];
    CHECK(int64_t(park_frames["count"]) == 2);

    // Asked again, it is computed from scratch, the same.
    CHECK(stream->admit(old_chunk, 0, 2, 1) == InstanceStream::PARKED);
    stream->poll();
    stream->take_ready(0, back);
    Mirror mirror;
    mirror.update(*stream);
    CHECK(stream->admit(old_chunk, 0, 2, 1) == InstanceStream::READY);
    CHECK(lists_right(fx, *stream, mirror, old_chunk));
}

TEST_CASE("Instance stream: set_tag drops no key asked for this frame (a chunk answered LATER keeps what it asked for)") {
    const StreamFixture &fx = StreamFixture::get();
    std::unique_ptr<InstanceStream> stream = fx.stream(InstanceStream::Settings());
    CHECK(stream->admit(fx.chunk(-300, -300), 0, 1, 1) == InstanceStream::PARKED);
    stream->set_tag(0, 2, forget_all, [](const ChunkId &) {}); // the same frame
    CHECK(stat(*stream, "stale_handed_back") == 1);
    CHECK(stat(*stream, "keys_dropped") == 0);
}

TEST_CASE("Instance stream: set_tag keeps parked the chunks the scan still wants, under the new tag at the new rank") {
    const StreamFixture &fx = StreamFixture::get();
    const ChunkId wanted = fx.chunk(-300, -300), dropped = fx.chunk(300, 300), fresh = fx.chunk(-300, 300);
    CHECK(disjoint(fx, { wanted, dropped, fresh }));
    std::unique_ptr<InstanceStream> stream = fx.stream(InstanceStream::Settings());
    CHECK(stream->admit(wanted, 0, 1, 4) == InstanceStream::PARKED);
    CHECK(stream->admit(dropped, 0, 1, 1) == InstanceStream::PARKED);
    CHECK(stream->admit(fresh, 0, 2, 2) == InstanceStream::PARKED); // the new scan's
    std::vector<ChunkId> kept;
    // The camera turned: the kept chunk is now in the cone, rank 1.
    stream->set_tag(0, 2, [&](const ChunkId &p_chunk) { return p_chunk == wanted ? int64_t(1) : int64_t(-1); }, [&](const ChunkId &p_chunk) { kept.push_back(p_chunk); });
    CHECK(kept == std::vector<ChunkId>{ wanted });
    CHECK(stream->parked_count(0) == 2);
    CHECK(stat(*stream, "jobs_reranked") > 0);
    stream->poll();
    CHECK(placed_before(fx, wanted, fresh, placement_order(*stream))); // its queued jobs moved ahead
    std::vector<std::pair<ChunkId, uint64_t>> back;
    stream->take_ready(0, back);
    CHECK(back.size() == 2);
    if (!(back.size() == 2)) {
        return;
    }
    CHECK(back[0] == std::make_pair(wanted, uint64_t(2)));
}

// A waiting key dropped while its input is still to run: the input lands for the keys still waiting, and never reaches
// the dropped one (it left the input's dependents).
TEST_CASE("Instance stream: a dropped waiting key leaves its inputs' dependents; a shared input lands for the others") {
    const StreamFixture &fx = StreamFixture::get();
    const ScatterSource &f = *fx.source;
    int lod = 1;
    while (lod < f.profile().lod_end && f.part_key_lod(lod) != lod) {
        lod++;
    }
    const ChunkId coarse = fx.chunk(200, 200, lod), fine = fx.chunk(200, 200);
    CHECK_FALSE(disjoint(fx, { coarse, fine })); // they share site tiles
    HeldExecutor *held = nullptr;
    std::unique_ptr<InstanceStream> stream = fx.held_stream(held);
    CHECK(stream->admit(coarse, 0, 1, 1) == InstanceStream::PARKED);
    CHECK(stream->admit(fine, 0, 2, 1) == InstanceStream::PARKED);
    // Both chunks' keys list their inputs (queued after them, at the same rank), none of which runs yet.
    held->held = false;
    held->steps = int64_t(own_keys(fx, coarse) + own_keys(fx, fine));
    stream->poll();
    held->held = true;
    CHECK(stat(*stream, "keys") == int64_t(keys_and_inputs(fx, { coarse, fine })));
    stream->set_tag(0, 2, forget_all, [](const ChunkId &) {}); // the fine chunk is the new scan's already
    // The coarse part tiles (waiting) and the site tiles only they read went; the shared ones stay, still to run.
    CHECK(stat(*stream, "keys_dropped") > stat(*stream, "jobs_dropped"));
    CHECK(stat(*stream, "jobs_dropped") > 0);
    CHECK(stat(*stream, "keys") == int64_t(keys_and_inputs(fx, fine)));
    held->held = false;
    held->steps = -1;
    stream->poll();
    stream->poll(); // the drain task the last one left queued
    std::vector<std::pair<ChunkId, uint64_t>> back;
    stream->take_ready(0, back);
    CHECK(back.size() == 1);
    Mirror mirror;
    mirror.update(*stream);
    CHECK(stream->admit(fine, 0, 2, 2) == InstanceStream::READY);
    CHECK(lists_right(fx, *stream, mirror, fine));
    // The coarse chunk asked again: computed again, its shared site tiles resident.
    CHECK(stream->admit(coarse, 0, 2, 1) == InstanceStream::PARKED);
    stream->poll();
    stream->take_ready(0, back);
    mirror.update(*stream);
    CHECK(stream->admit(coarse, 0, 2, 1) == InstanceStream::READY);
    CHECK(lists_right(fx, *stream, mirror, coarse));
}

TEST_CASE("Instance stream: a walk key off its grid walks nothing (a site tile above LOD 0, proxies on a part layer)") {
    const StreamFixture &fx = StreamFixture::get();
    const ScatterSource &f = *fx.source;
    const int64_t side = f.part_tile_side(0);
    const VoxelPos tile(WorldCoord::floor_div(fx.origin.x, side), WorldCoord::floor_div(fx.origin.y, side), WorldCoord::floor_div(fx.origin.z, side));
    const SourceKey sites{ f.setup().id, SourceKey::SITES, 0, tile };
    CHECK_FALSE(fx.content(sites).sites.empty()); // the fixture's ground crosses it
    CHECK(fx.content(SourceKey{ f.setup().id, SourceKey::SITES, 1, tile }).sites.empty());
    CHECK(f.part_layer(0));
    CHECK(fx.content(SourceKey{ f.setup().id, SourceKey::PROXIES, 0, tile }).sites.empty());
}
