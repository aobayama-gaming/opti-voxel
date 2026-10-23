#pragma once

#include "test_common.hpp"

#include "instance_stream.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

// What the stream tests share (test_instance_stream.cpp, test_scatter_source.cpp).
namespace StreamTest {

// The GPU table's streamed records, as a reader of the placement log keeps them.
struct Mirror {
    uint64_t cursor = 0;
    std::vector<Archetype::TableRecord> records; // by record id - slot_base

    void update(const Archetype::InstanceStream &p_stream) {
        std::vector<Archetype::InstanceStream::Upload> ups;
        p_stream.uploads_since(cursor, ups);
        cursor = p_stream.placements();
        for (const Archetype::InstanceStream::Upload &u : ups) {
            const uint32_t at = u.first - p_stream.slot_base();
            records.resize(std::max(records.size(), size_t(at + u.content->records.size())));
            std::copy(u.content->records.begin(), u.content->records.end(), records.begin() + at);
        }
    }
};

// Every field of two records, bit for bit (the padding left out).
inline bool same_bits(const Archetype::TableRecord &p_a, const Archetype::TableRecord &p_b) {
    const auto eq = [](const auto &p_x, const auto &p_y) { return std::memcmp(&p_x, &p_y, sizeof(p_x)) == 0; };
    return eq(p_a.def, p_b.def) && eq(p_a.removed, p_b.removed) && eq(p_a.op, p_b.op) && eq(p_a.k, p_b.k) && eq(p_a.inv_k, p_b.inv_k) &&
            eq(p_a.anchor, p_b.anchor) && eq(p_a.rotation, p_b.rotation) && eq(p_a.params, p_b.params) && eq(p_a.salt, p_b.salt) &&
            eq(p_a.box_lo, p_b.box_lo) && eq(p_a.box_hi, p_b.box_hi) && eq(p_a.margin_c, p_b.margin_c) && eq(p_a.lipschitz, p_b.lipschitz) &&
            eq(p_a.max_lod, p_b.max_lod) && eq(p_a.min_lod, p_b.min_lod) && eq(p_a.erode, p_b.erode) && eq(p_a.erode_from, p_b.erode_from) &&
            eq(p_a.material, p_b.material);
}

} // namespace StreamTest
