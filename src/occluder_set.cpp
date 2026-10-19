#include "occluder_set.hpp"

#include <algorithm>
#include <cmath>

void OccluderSet::merge_blocks(const std::vector<uint8_t> &p_solid, int p_n, std::vector<BlockBox> &r_boxes) {
    std::vector<uint8_t> used(p_solid.size(), 0);
    const auto free_solid = [&](int x, int y, int z) {
        const size_t i = (size_t(z) * p_n + y) * p_n + x;
        return p_solid[i] && !used[i];
    };
    for (int z = 0; z < p_n; z++) {
        for (int y = 0; y < p_n; y++) {
            for (int x = 0; x < p_n; x++) {
                if (!free_solid(x, y, z)) {
                    continue;
                }
                int x1 = x + 1;
                while (x1 < p_n && free_solid(x1, y, z)) {
                    x1++;
                }
                const auto row_free = [&](int py, int pz) {
                    for (int i = x; i < x1; i++) {
                        if (!free_solid(i, py, pz)) {
                            return false;
                        }
                    }
                    return true;
                };
                int y1 = y + 1;
                while (y1 < p_n && row_free(y1, z)) {
                    y1++;
                }
                int z1 = z + 1;
                const auto slab_free = [&](int pz) {
                    for (int j = y; j < y1; j++) {
                        if (!row_free(j, pz)) {
                            return false;
                        }
                    }
                    return true;
                };
                while (z1 < p_n && slab_free(z1)) {
                    z1++;
                }
                for (int k = z; k < z1; k++) {
                    for (int j = y; j < y1; j++) {
                        for (int i = x; i < x1; i++) {
                            used[(size_t(k) * p_n + j) * p_n + i] = 1;
                        }
                    }
                }
                r_boxes.push_back({ x, y, z, x1, y1, z1 });
            }
        }
    }
}

void OccluderSet::set(const ChunkId &p_chunk, std::vector<Box> p_boxes) {
    if (p_boxes.empty()) {
        boxes.erase(p_chunk);
        return;
    }
    boxes[p_chunk] = std::move(p_boxes);
}

void OccluderSet::erase(const ChunkId &p_chunk) {
    boxes.erase(p_chunk);
}

int64_t OccluderSet::box_count() const {
    int64_t n = 0;
    for (const KeyValue<ChunkId, std::vector<Box>> &kv : boxes) {
        n += int64_t(kv.value.size());
    }
    return n;
}

OccluderSet::BuildStats OccluderSet::build(OcclusionMap &r_map, const VoxelPos &p_eye_base, const OcclusionMap::Vec3 &p_eye, double p_margin) const {
    using Vec3 = OcclusionMap::Vec3;
    struct Item {
        double nearest;
        Vec3 lo, hi;
    };
    BuildStats stats;
    std::vector<Item> items;
    for (const KeyValue<ChunkId, std::vector<Box>> &kv : boxes) {
        for (const Box &b : kv.value) {
            stats.boxes++;
            Item it;
            double lo[3], hi[3], gap2 = 0.0;
            bool empty = false;
            bool holds_eye = true;
            for (int a = 0; a < 3; a++) {
                lo[a] = double(b.lo[a] - p_eye_base[a]) - p_eye[a] + p_margin;
                hi[a] = double(b.hi[a] - p_eye_base[a]) - p_eye[a] - p_margin;
                empty = empty || !(lo[a] < hi[a]);
                holds_eye = holds_eye && lo[a] <= 0.0 && hi[a] >= 0.0;
                const double g = std::max({ lo[a], -hi[a], 0.0 });
                gap2 += g * g;
            }
            if (empty) {
                stats.vanished++;
                continue;
            }
            if (holds_eye) {
                stats.holding_eye++;
                continue;
            }
            it.nearest = std::sqrt(gap2);
            it.lo = { lo[0], lo[1], lo[2] };
            it.hi = { hi[0], hi[1], hi[2] };
            items.push_back(it);
        }
    }
    std::sort(items.begin(), items.end(), [](const Item &a, const Item &b) { return a.nearest < b.nearest; });
    r_map.clear();
    for (const Item &it : items) {
        if (r_map.box_hidden(it.lo, it.hi)) {
            stats.already_hidden++;
            continue;
        }
        stats.rasterised++;
        stats.written += r_map.add_box(it.lo, it.hi) ? 1 : 0;
    }
    return stats;
}
