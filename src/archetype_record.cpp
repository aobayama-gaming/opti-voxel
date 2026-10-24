// The archetype table's records: the record store's pages and the locality rule (archetype_record.hpp).
#include "archetype_record.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace Archetype {

using ExactNoise::exact_pow2;

DefRecord TableRecord::def_record() const {
    DefRecord r;
    r.anchor = anchor;
    std::memcpy(r.rotation, rotation, sizeof(r.rotation));
    std::memcpy(r.params, params, sizeof(r.params));
    r.salt = salt;
    return r;
}

RecordStore::RecordStore(const std::vector<TableRecord> &p_records, uint32_t p_version) {
    for (const TableRecord &r : p_records) {
        push_back(r, p_version);
    }
}

RecordStore::Page &RecordStore::own(uint32_t p_page) {
    if (pages[p_page].use_count() > 1) {
        pages[p_page] = std::make_shared<Page>(*pages[p_page]); // a snapshot shares it: copy on write
    }
    return *pages[p_page];
}

void RecordStore::push_back(const TableRecord &p_record, uint32_t p_version) {
    if (count % PAGE == 0) {
        pages.push_back(std::make_shared<Page>());
        pages.back()->r.reserve(PAGE);
    }
    Page &page = own(uint32_t(pages.size() - 1));
    page.r.push_back(p_record);
    page.version = std::max(page.version, p_version);
    count++;
}

void RecordStore::set(uint32_t p_i, const TableRecord &p_record, uint32_t p_version) {
    Page &page = own(p_i / PAGE);
    page.r[p_i % PAGE] = p_record;
    page.version = std::max(page.version, p_version);
}

void RecordStore::changed_since(uint32_t p_version, std::vector<std::pair<uint32_t, uint32_t>> &r_runs) const {
    r_runs.clear();
    for (uint32_t p = 0; p < pages.size(); p++) {
        if (p_version != UINT32_MAX && pages[p]->version <= p_version) {
            continue;
        }
        const uint32_t first = p * PAGE;
        const uint32_t end = std::min(count, (p + 1) * PAGE);
        if (!r_runs.empty() && r_runs.back().first + r_runs.back().second == first) {
            r_runs.back().second = end - r_runs.back().first;
        } else {
            r_runs.push_back({ first, end - first });
        }
    }
}

int influence_margin(const TableRecord &p_record, int p_lod) {
    // Clamped before the conversion: a float above INT_MAX converts to anything (CPU) or saturates (GPU).
    const float m = std::min(p_record.k + p_record.margin_c * exact_pow2(p_lod), float(INPUT_LIMIT));
    return int(std::ceil(m)) + 2;
}

bool in_influence_cube(const TableRecord &p_record, const VoxelPos &p_lo, const VoxelPos &p_hi, int p_lod) {
    const int m = influence_margin(p_record, p_lod);
    for (int a = 0; a < 3; a++) {
        const int64_t dlo = WorldCoord::sat_diff(p_lo[a], p_record.anchor[a], INPUT_LIMIT);
        const int64_t dhi = WorldCoord::sat_diff(p_hi[a], p_record.anchor[a], INPUT_LIMIT);
        if (dhi < int64_t(p_record.box_lo[a]) - m || dlo > int64_t(p_record.box_hi[a]) + m) {
            return false;
        }
    }
    return true;
}

bool in_influence_box(const TableRecord &p_record, const WorldPos &p_pos, int p_lod) {
    return in_influence_cube(p_record, p_pos.voxel, p_pos.voxel, p_lod);
}

} // namespace Archetype
