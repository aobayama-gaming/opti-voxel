#include "chunk_memory.hpp"

#include <algorithm>

void VoxelState::add_scan_chunk(std::unique_ptr<LODHashSet> p_scanned_chunk, const ScanDelta &p_delta) {
    p_scanned_chunk->max_lod = dirty_chunk.max_lod;
    if (verify) {
        _verify_delta(*p_scanned_chunk, p_delta);
    }
    scanned_chunk = std::move(p_scanned_chunk);

    // The loaded set doesn't change here, so the other dirty chunks keep their anchors. With nothing loaded
    // (the first scan: the whole scan enters) there is no anchor to look for.
    const bool any_loaded = !loaded_chunk.is_empty();
    dirty_chunk.reserve(dirty_chunk.size() + p_delta.entered.size());
    for (const ChunkId &id : p_delta.left) {
        if (dirty_chunk.erase(id)) {
            _unanchor(id);
        } else {
            leaving.insert(id);
        }
    }
    for (const ChunkId &id : p_delta.entered) {
        if (loaded_chunk.has(id)) {
            leaving.erase(id);
        } else {
            // Must be generated.
            dirty_chunk.insert(id);
            if (any_loaded) {
                _anchor(id);
            }
        }
    }
}

void VoxelState::add_new_loaded(ChunkOutput &new_output) {
    const ChunkId id = new_output.chunk_id;
    // A re-meshed chunk (an edit) can land after its destruction was marked: drop the pending copy, so
    // it can't be freed twice or leak its new quads. The next update_destroy re-adds it if it is still
    // leaving. (Also covered by "only scanned chunks are re-meshed".)
    if (to_destroy_chunk.has(id)) {
        to_destroy_chunk.erase(id);
    }
    ChunkOutput *old = loaded_chunk.getptr(id);
    if (old) {
        // Loaded again: the sets don't change.
        old->free();
        *old = new_output;
        return;
    }

    if (dirty_chunk.erase(id)) {
        _unanchor(id);
    }
    loaded_chunk.insert(id, new_output);

    if (!scanned_chunk->has(id)) {
        // A batch that was in flight when the chunk left the scan.
        leaving.insert(id);
        // The scan is an octree cut, so only a chunk outside it can have dirty descendants.
        if (ChunkMath::get_lod(id) > 0 && !dirty_chunk.is_empty()) {
            _rescan_below(id);
        }
    }
}

void VoxelState::update_destroy() {
    to_destroy_chunk.clear();
    stats.leaving_max = std::max(stats.leaving_max, int64_t(leaving.size()));
    for (const ChunkId &id : leaving) {
        // Split: a dirty descendant still needs it.
        if (stall_count.has(id)) {
            continue;
        }
        // Merge: an ancestor replacing it is still being generated.
        ChunkId parent_id = id;
        if (dirty_chunk.find_ancestor(parent_id) >= 0) {
            continue;
        }
        const ChunkOutput *output = loaded_chunk.getptr(id);
        if (output) {
            to_destroy_chunk.insert(id, *output);
        }
    }
}

void VoxelState::compute_destroy_full(ChunkHashMap &r_to_destroy, StallCountMap *r_stall_count) const {
    // A loaded chunk stays while one of its descendants is still being generated (split).
    StallCountMap local_stall;
    StallCountMap &stall_chunk = r_stall_count ? *r_stall_count : local_stall;
    stall_chunk.clear();
    for (const ChunkId &scan_pos : dirty_chunk) {
        ChunkId parent_id = scan_pos;
        if (loaded_chunk.find_ancestor(parent_id) >= 0) {
            int32_t *count = stall_chunk.getptr(parent_id);
            if (count) {
                (*count)++;
            } else {
                stall_chunk.insert(parent_id, 1);
            }
        }
    }

    r_to_destroy.clear();

    // Anything loaded but not stalled can be safely destroyed.
    for (ChunkHashMap::ConstIterator it = loaded_chunk.begin(); it != loaded_chunk.end(); ++it) {
        if (scanned_chunk->has(it->first)) {
            continue;
        }

        if (stall_chunk.has(it->first)) {
            continue;
        }

        // ... and stays while an ancestor replacing it is still being generated (merge).
        ChunkId parent_id = it->first;
        if (dirty_chunk.find_ancestor(parent_id) >= 0) {
            continue;
        }

        r_to_destroy.insert(it->first, it->second);
    }
}

int64_t VoxelState::verify_against_full() {
    ChunkHashMap full_destroy;
    StallCountMap full_stall;
    compute_destroy_full(full_destroy, &full_stall);

    int64_t mismatches = 0;
    mismatches += full_destroy.size() == to_destroy_chunk.size() ? 0 : 1;
    for (const auto &kv : full_destroy) {
        mismatches += to_destroy_chunk.has(kv.first) ? 0 : 1;
    }
    mismatches += full_stall.size() == stall_count.size() ? 0 : 1;
    for (const auto &kv : full_stall) {
        const int32_t *count = stall_count.getptr(kv.first);
        mismatches += count && *count == kv.second ? 0 : 1;
    }
    // dirty = scan - loaded, leaving = loaded - scan.
    int64_t dirty_expected = 0;
    for (const ChunkId &id : *scanned_chunk) {
        const bool dirty = !loaded_chunk.has(id);
        dirty_expected += dirty ? 1 : 0;
        mismatches += dirty_chunk.has(id) == dirty ? 0 : 1;
    }
    mismatches += int64_t(dirty_chunk.size()) == dirty_expected ? 0 : 1;
    int64_t leaving_expected = 0;
    for (const auto &kv : loaded_chunk) {
        const bool out = !scanned_chunk->has(kv.first);
        leaving_expected += out ? 1 : 0;
        mismatches += leaving.has(kv.first) == out ? 0 : 1;
    }
    mismatches += int64_t(leaving.size()) == leaving_expected ? 0 : 1;

    stats.verify_checks++;
    stats.verify_mismatches += mismatches;
    return mismatches;
}

void VoxelState::destroy_marked() {
    for (const auto &kv : to_destroy_chunk) {
        ChunkOutput output = kv.second;
        output.free();
        loaded_chunk.erase(kv.first);
        leaving.erase(kv.first);
        // No dirty chunk has it as anchor (update_destroy's split rule), so no anchor changes.
        if (verify && stall_count.has(kv.first)) {
            stats.verify_mismatches++;
        }
    }
    to_destroy_chunk.clear();
}

void VoxelState::clear() {
    const int max_lod = dirty_chunk.max_lod;
    dirty_chunk.clear();
    scanned_chunk = std::make_unique<LODHashSet>();
    loaded_chunk.clear();
    to_destroy_chunk.clear();
    dirty_anchor.clear();
    stall_count.clear();
    leaving.clear();
    set_max_lod(max_lod);
}

void VoxelState::_anchor(const ChunkId &p_dirty) {
    ChunkId parent_id = p_dirty;
    if (loaded_chunk.find_ancestor(parent_id) >= 0) {
        dirty_anchor.insert(p_dirty, parent_id);
        _stall_add(parent_id, 1);
    }
}

void VoxelState::_unanchor(const ChunkId &p_dirty) {
    const ChunkId *anchor = dirty_anchor.getptr(p_dirty);
    if (anchor) {
        _stall_add(*anchor, -1);
        dirty_anchor.erase(p_dirty);
    }
}

void VoxelState::_stall_add(const ChunkId &p_loaded, int32_t p_count) {
    int32_t *count = stall_count.getptr(p_loaded);
    if (!count) {
        stall_count.insert(p_loaded, p_count);
        return;
    }
    *count += p_count;
    if (*count == 0) {
        stall_count.erase(p_loaded);
    }
}

void VoxelState::_rescan_below(const ChunkId &p_loaded) {
    const uint64_t t0 = now_usec();
    const int lod = ChunkMath::get_lod(p_loaded);
    const int64_t half = int64_t(1) << lod; // a LOD-L chunk spans its centre +- 2^L ids
    int32_t moved = 0;
    for (const ChunkId &id : dirty_chunk) {
        if (ChunkMath::get_lod(id) >= lod || ABS(id.x - p_loaded.x) >= half || ABS(id.y - p_loaded.y) >= half || ABS(id.z - p_loaded.z) >= half) {
            continue; // not a descendant
        }
        ChunkId *anchor = dirty_anchor.getptr(id);
        if (anchor) {
            if (ChunkMath::get_lod(*anchor) < lod) {
                continue; // a loaded chunk between the two is nearer
            }
            _stall_add(*anchor, -1);
            *anchor = p_loaded;
        } else {
            dirty_anchor.insert(id, p_loaded);
        }
        moved++;
    }
    if (moved > 0) {
        _stall_add(p_loaded, moved);
    }
    stats.rescans++;
    stats.rescan_reanchored += moved;
    stats.rescan_usec += now_usec() - t0;
}

void VoxelState::_verify_delta(const LODHashSet &p_new, const ScanDelta &p_delta) {
    const LODHashSet &old_scan = *scanned_chunk;
    int64_t mismatches = 0;
    for (const ChunkId &id : p_delta.left) {
        mismatches += old_scan.has(id) && !p_new.has(id) ? 0 : 1;
    }
    for (const ChunkId &id : p_delta.entered) {
        mismatches += !old_scan.has(id) && p_new.has(id) ? 0 : 1;
    }
    // With both checks, equal sizes make new = old - left + entered.
    mismatches += int64_t(p_new.size()) == int64_t(old_scan.size()) - int64_t(p_delta.left.size()) + int64_t(p_delta.entered.size()) ? 0 : 1;
    stats.delta_mismatches += mismatches;

    int64_t violations = 0;
    for (const ChunkId &id : p_new) {
        ChunkId parent_id = id;
        violations += p_new.find_ancestor(parent_id) >= 0 ? 1 : 0;
    }
    stats.octree_violations += violations;
    if (mismatches > 0 || violations > 0) {
        ov::report_error(__FILE__, __LINE__, ov::format("VoxelState: scan delta check: %d delta mismatches, %d scanned chunks with a scanned ancestor.", mismatches, violations));
    }
}
