#include "cloud_climate.hpp"

#include "cascade.hpp"
#include "trace.hpp"

#include "ov/half.hpp"
#include "ov/math.hpp"

namespace CloudClimate {

bool any(const Climate::Table &p_table) {
    for (const Climate::Biome &b : p_table.biomes) {
        if (b.clouds.set) {
            return true;
        }
    }
    return false;
}

Texel of_weights(const Climate::Table &p_table, const Climate::Float4 p_w[Climate::MAX_BIOMES]) {
    Texel t;
    for (size_t b = 0; b < p_table.biomes.size(); b++) {
        const Climate::BiomeClouds &c = p_table.biomes[b].clouds;
        if (c.set) {
            const float w = ov::clamp(p_w[b].w, 0.0f, 1.0f);
            t.cover += w * c.cover;
            t.type += w * c.type;
            t.weight += w;
        }
    }
    return t;
}

Texel at(const Climate::Table &p_table, const Cascade::Spec &p_spec, const VoxelPos &p_voxel, int p_m) {
    const int m = ov::max(0, ov::min(p_m, p_spec.top_level(true)));
    Climate::Float4 fields[Climate::MAX_FIELDS];
    Climate::fields_at(p_table, p_spec, WorldCoord::floor_div(p_voxel, int64_t(1) << m), m, fields);
    Climate::Float4 w[Climate::MAX_BIOMES];
    Climate::weights(p_table, fields, w);
    return of_weights(p_table, w);
}

Map::Map(const BaseSurface &p_base, int p_cell_shift, double p_bottom_m, double p_top_m, double p_voxel_m) :
        base(p_base), shift(p_cell_shift) {
    const double margin = double(MARGIN) * std::ldexp(1.0, shift);
    low_vox = p_bottom_m / p_voxel_m - margin;
    high_vox = p_top_m / p_voxel_m + margin;
}

Vector3i64 Map::cell_of(const VoxelPos &p_voxel) const {
    return WorldCoord::floor_div(p_voxel, int64_t(1) << shift);
}

Vector3i64 Map::centre_for(const VoxelPos &p_camera) const {
    return cell_of(base.on_base(p_camera));
}

bool Map::covers(const Vector3i64 &p_centre, const Vector3i64 &p_cell) {
    const int64_t reach = SIZE / 8;
    return std::abs(p_cell.x - p_centre.x) <= reach && std::abs(p_cell.y - p_centre.y) <= reach && std::abs(p_cell.z - p_centre.z) <= reach;
}

VoxelPos Map::corner(const Vector3i64 &p_centre) const {
    const int64_t half = SIZE / 2;
    return VoxelPos((p_centre.x - half) << shift, (p_centre.y - half) << shift, (p_centre.z - half) << shift);
}

std::vector<uint16_t> Map::build(const Vector3i64 &p_centre, const SampleFn &p_sample, float &r_max_cover) {
    OV_TRACE("clouds", "climate_map");
    std::vector<uint16_t> texels(size_t(SIZE) * SIZE * SIZE * 4, 0);
    ov::HashMap<Vector3i64, Texel, Vector3i64Hasher> next;
    const int64_t cell = int64_t(1) << shift;
    // The base cell's texel: this build's, the last build's, or a sample at its centre.
    auto texel_of = [&](const Vector3i64 &p_cell) {
        if (const Texel *seen = next.getptr(p_cell)) {
            return *seen;
        }
        const Texel *old = cache.getptr(p_cell);
        const Texel t = old ? *old : p_sample(VoxelPos(p_cell.x * cell + cell / 2, p_cell.y * cell + cell / 2, p_cell.z * cell + cell / 2), shift);
        next.insert(p_cell, t);
        return t;
    };
    const VoxelPos c0 = corner(p_centre);
    r_max_cover = 0.0f;
    for (int k = 0; k < SIZE; k++) {
        for (int j = 0; j < SIZE; j++) {
            for (int i = 0; i < SIZE; i++) {
                const VoxelPos centre(c0.x + i * cell + cell / 2, c0.y + j * cell + cell / 2, c0.z + k * cell + cell / 2);
                const double altitude = base.altitude(centre);
                if (altitude < low_vox || altitude > high_vox) {
                    continue;
                }
                const Texel t = texel_of(cell_of(base.on_base(centre)));
                uint16_t *o = &texels[((size_t(k) * SIZE + j) * SIZE + i) * 4];
                o[0] = ov::float_to_half(t.cover);
                o[1] = ov::float_to_half(t.type);
                o[2] = ov::float_to_half(t.weight);
                r_max_cover = ov::max(r_max_cover, t.cover);
            }
        }
    }
    cache = std::move(next);
    return texels;
}

Stream::Stream(const Map &p_map, Map::SampleFn p_sample) :
        map(p_map), sample(std::move(p_sample)) {
    worker = ov::executor().start_worker("clouds_climate", *this);
}

Stream::~Stream() {
    worker->stop(); // the build running finishes, before the members it uses go
}

bool Stream::step() {
    Vector3i64 centre;
    uint64_t id;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!busy || ready) {
            return false;
        }
        centre = next_centre;
        id = next_id;
    }
    OV_TRACE("clouds", "climate_build");
    Trace::flow(Trace::Kind::FLOW_STEP, "climate_map", id);
    float max_cover = 0.0f;
    std::vector<uint16_t> texels = map.build(centre, sample, max_cover);
    {
        std::lock_guard<std::mutex> lock(mutex);
        next_texels = std::move(texels);
        next_max_cover = max_cover;
        ready = true;
    }
    built.notify_all();
    return false;
}

void Stream::finish() {
    if (!ov::executor().threaded()) {
        ov::executor().pump(); // steps the worker
        return;
    }
    std::unique_lock<std::mutex> lock(mutex);
    built.wait(lock, [this]() { return !busy || ready; });
}

bool Stream::update(const VoxelPos &p_camera, std::vector<uint16_t> &r_texels) {
    OV_TRACE("clouds", "climate_update");
    const Vector3i64 cell = map.centre_for(p_camera);
    bool landed = false;
    bool ask = false;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (ready) {
            landed = true;
            busy = false;
            ready = false;
            has_live = true;
            live_centre = next_centre;
            live_max_cover = next_max_cover;
            live_build = next_id;
            r_texels = std::move(next_texels);
        }
        if (!busy && (!has_live || !Map::covers(live_centre, cell))) {
            busy = true;
            next_centre = cell;
            next_id = ++asked;
            ask = true;
            Trace::flow(Trace::Kind::FLOW_START, "climate_map", next_id);
        }
    }
    if (ask) {
        worker->wake();
    }
    // A camera that left the live map (a teleport, before its build lands) reads none rather than its stretched edge.
    const int64_t inside = Map::SIZE / 2 - 1;
    holds_camera = has_live && std::abs(cell.x - live_centre.x) < inside && std::abs(cell.y - live_centre.y) < inside &&
            std::abs(cell.z - live_centre.z) < inside;
    return landed;
}

} // namespace CloudClimate
