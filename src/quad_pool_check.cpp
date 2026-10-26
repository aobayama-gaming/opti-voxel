#include "quad_pool_check.hpp"

#include "debug_log.hpp"
#include "gpu_shader.hpp"
#include "page_format.hpp"

#include "ov/json.hpp"
#include "ov/math.hpp"
#include "ov/on_error.hpp"

#include <cmath>
#include <utility>
#include <vector>

namespace {

constexpr const char CHECK_COMPUTE[] = R"glsl(#version 450
layout(local_size_x = 64) in;

layout(set = 0, binding = 0, rgba16ui) uniform restrict writeonly uimage2D staging;
layout(set = 0, binding = 1, std430) restrict buffer Counter { uint counter[]; };
layout(push_constant, std430) uniform Params {
    vec4 origin_size; // xyz origin, w sheet size
    uvec4 dims;       // x = side
    ivec4 chunk;      // xyz: the chunk the corners are stored relative to
    vec4 grid;        // xyz: world position of its vertex (0, 0, 0), w: its cell size
} params;

/*GLSL_PAGE_FORMAT*/

float height(vec2 xz) { return params.origin_size.y + 3.0 * sin(xz.x * 0.3) * cos(xz.y * 0.3); }
vec3 surf(vec2 xz) { return vec3(xz.x, height(xz), xz.y); }
vec3 surf_normal(vec2 xz) {
    float dx = 0.9 * cos(xz.x * 0.3) * cos(xz.y * 0.3);
    float dz = -0.9 * sin(xz.x * 0.3) * sin(xz.y * 0.3);
    return normalize(vec3(-dx, 1.0, -dz));
}

void put_corner(uint slot, uint k, vec2 xz) {
    vec3 local = (surf(xz) - params.grid.xyz) / params.grid.w;
    store_corner(slot, k, encode_corner(local, pack_normal_oct(surf_normal(xz)), 0u, page_pos_scale(32)));
}

void main() {
    uint side = params.dims.x;
    uint i = gl_GlobalInvocationID.x;
    if (i == 0u) {
        counter[1] = (side * side + 63u) / 64u; // used groups of 64 quads
    }
    if (i >= side * side) {
        return;
    }
    uint qx = i % side;
    uint qz = i / side;
    float step = params.origin_size.w / float(side);
    vec2 base = params.origin_size.xz + vec2(qx, qz) * step;
    vec2 c1 = base + vec2(step, 0.0);
    vec2 c2 = base + vec2(0.0, step);
    if (qx >= side / 2u) { // swapped winding on the right half
        vec2 t = c1; c1 = c2; c2 = t;
    }
    uint slot = atomicAdd(counter[0], 1u);
    if (slot % QUAD_GROUP == 0u) {
        store_group_header(slot / QUAD_GROUP, params.chunk.xyz, 0u);
    }
    put_corner(slot, 0u, base);
    put_corner(slot, 1u, c1);
    put_corner(slot, 2u, c2);
    put_corner(slot, 3u, base + vec2(step, step));
}
)glsl";

// godot's Vector3 float arithmetic (the tolerances were tuned on it).
float distance(const Float3 &p_a, const Float3 &p_b) {
    const float dx = p_b.x - p_a.x;
    const float dy = p_b.y - p_a.y;
    const float dz = p_b.z - p_a.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

} // namespace

bool QuadPoolCheck::start(ov::gpu::Device *p_device, PageHost *p_host, const Settings &p_settings) {
    settings = p_settings;
    const int side = settings.side;
    OV_ERR_FAIL_COND_V_MSG(side * side > settings.page_capacity, false, "QuadPoolCheck: the sheet does not fit in one page.");
    OV_ERR_FAIL_COND_V_MSG(KILL_B_OFFSET + KILL_B_COUNT > uint32_t(side * side), false, "QuadPoolCheck: the sheet is too small for the kill test.");
    device = p_device;

    // Corners are stored relative to one chunk: the smallest one whose vertex grid holds the sheet (its box from
    // origin - (0, 5, 0), size (sheet, 10, sheet)).
    const Float3 &o = settings.sheet_origin;
    const float s = settings.sheet_size;
    const Float3 sheet_lo{ o.x, o.y - 5.0f, o.z };
    const Float3 sheet_hi{ sheet_lo.x + s, sheet_lo.y + 10.0f, sheet_lo.z + s };
    sheet_chunk = ChunkMath::chunk_at_render(Float3{ sheet_lo.x + s * 0.5f, sheet_lo.y + 5.0f, sheet_lo.z + s * 0.5f }, chunk_param);
    for (int lod = 0; lod < 24; lod++) {
        const Float3 lo = ChunkMath::grid_to_render(sheet_chunk, Float3{}, chunk_param);
        const float last = float(chunk_param.chunk_size - 1);
        const Float3 hi = ChunkMath::grid_to_render(sheet_chunk, Float3{ last, last, last }, chunk_param);
        if (lo.x <= sheet_lo.x && lo.y <= sheet_lo.y && lo.z <= sheet_lo.z && hi.x >= sheet_hi.x && hi.y >= sheet_hi.y && hi.z >= sheet_hi.z) {
            break;
        }
        sheet_chunk = ChunkMath::get_parent_from_child(sheet_chunk);
    }

    pool = new QuadPagePool();
    pool->initialize(device, p_host, uint32_t(settings.page_capacity), uint32_t(1 + settings.extra_pages));
    page = pool->acquire_page();
    pool->set_page_aabb(page, RenderBox{ sheet_lo, Float3{ s, 10.0f, s } });
    device->post([this]() { _rt_run(); });
    return true;
}

void QuadPoolCheck::_rt_read_texture(int p_stage) {
    rt_readbacks_in_flight++;
    device->read_texture_async(pool->rt_get_page_rd(page).texture, [this, p_stage](const ov::bytes &p_data) {
        mutex.lock();
        texture_data[p_stage] = p_data;
        readback_ready = p_stage == 1 || !command_data.empty();
        mutex.unlock();
        _rt_readback_done();
    });
}

void QuadPoolCheck::_rt_run() {
    rt_pipeline = ov::gpu::compile(*device, "quad_pool_check", ov::replace_all(CHECK_COMPUTE, "/*GLSL_PAGE_FORMAT*/", PageFormat::glsl_encode()));
    OV_ERR_FAIL_COND_MSG(!rt_pipeline.is_valid(), "QuadPoolCheck: the shader failed to compile.");
    rt_staging = device->create_texture(QuadPagePool::TEX_FORMAT, QuadPagePool::TEX_WIDTH, pool->get_page_rows(),
            ov::gpu::TEXTURE_STORAGE | ov::gpu::TEXTURE_COPY_SRC | ov::gpu::TEXTURE_COPY_DST);
    device->clear_texture(rt_staging);
    rt_counter = device->create_buffer(ov::gpu::BufferKind::STORAGE, 16);
    rt_set = device->create_bind_set(rt_pipeline, { ov::gpu::Binding::storage_image(0, rt_staging), { 1, ov::gpu::BufferKind::STORAGE, rt_counter } });

    const Float3 grid = ChunkMath::grid_to_render(sheet_chunk, Float3{}, chunk_param);
    const ov::ivec3 sheet_chunk_lo = WorldCoord::lo32_i(sheet_chunk);
    uint8_t push[64] = {};
    ov::put_f32(push + 0, settings.sheet_origin.x);
    ov::put_f32(push + 4, settings.sheet_origin.y);
    ov::put_f32(push + 8, settings.sheet_origin.z);
    ov::put_f32(push + 12, settings.sheet_size);
    ov::put_u32(push + 16, uint32_t(settings.side));
    ov::put_s32(push + 32, sheet_chunk_lo.x);
    ov::put_s32(push + 36, sheet_chunk_lo.y);
    ov::put_s32(push + 40, sheet_chunk_lo.z);
    ov::put_f32(push + 48, grid.x);
    ov::put_f32(push + 52, grid.y);
    ov::put_f32(push + 56, grid.z);
    ov::put_f32(push + 60, chunk_param.voxel_size * float(ChunkMath::get_lod_size(sheet_chunk)));

    const uint32_t quad_count = uint32_t(settings.side * settings.side);
    device->clear_buffer(rt_counter, 0, 16);
    device->dispatch(rt_pipeline, rt_set, push, sizeof(push), (quad_count + 63) / 64, 1, 1);

    // The whole point: GPU -> GPU, the CPU never sees the quads or the count.
    pool->rt_copy_page_from(page, rt_staging, 0);
    const QuadPagePool::PageRD page_rd = pool->rt_get_page_rd(page);
    device->copy_buffer(rt_counter, page_rd.command, 4, QuadPagePool::INDIRECT_INSTANCE_COUNT_OFFSET, 4);

    // Debug verification only.
    _rt_read_texture(0);
    rt_readbacks_in_flight++;
    device->read_buffer_async(page_rd.command, 0, 20, [this](const ov::bytes &p_data) {
        mutex.lock();
        command_data = p_data;
        readback_ready = !texture_data[0].empty();
        mutex.unlock();
        _rt_readback_done();
    });
    VLOG(1, "QuadPoolCheck: dispatched ", quad_count, " quads into page ", page);
}

void QuadPoolCheck::_rt_kill() {
    pool->rt_kill_range(page, KILL_A_OFFSET, KILL_A_COUNT);
    pool->rt_kill_range(page, KILL_B_OFFSET, KILL_B_COUNT);
    _rt_read_texture(1);
}

void QuadPoolCheck::step() {
    mutex.lock();
    const bool ready = readback_ready;
    readback_ready = false;
    mutex.unlock();
    if (!ready) {
        return;
    }
    if (current_stage == 0) {
        ov::Value r = ov::Value::object();
        _verify_write(r);
        mutex.lock();
        results = r;
        mutex.unlock();
        current_stage = 1;
    } else if (current_stage == 2) {
        ov::Value r = results;
        _verify_kills(r);
        r["ok"] = r["write_ok"].as_bool() && r["kill_ok"].as_bool();
        r["pool"] = pool->get_stats();
        current_stage = 3;
        mutex.lock();
        results = r;
        mutex.unlock();
        VLOG(1, "QuadPoolCheck report: ", ov::to_json(r));
    }
}

void QuadPoolCheck::_verify_write(ov::Value &r) {
    mutex.lock();
    const ov::bytes bytes = texture_data[0];
    const ov::bytes command = command_data;
    mutex.unlock();
    const uint16_t *data = reinterpret_cast<const uint16_t *>(bytes.data());
    const uint32_t rows = uint32_t(bytes.size() / PageFormat::ROW_BYTES);
    const float scale = PageFormat::pos_scale(chunk_param.chunk_size);
    // One fixed-point step, plus float rounding of the world <-> grid transforms.
    const double pos_tolerance = chunk_param.voxel_size * double(ChunkMath::get_lod_size(sheet_chunk)) / scale + 1e-3;
    const int side = settings.side;
    const Float3 &origin = settings.sheet_origin;

    const uint32_t quad_count = uint32_t(side * side);
    const uint32_t index_count = command.size() >= 8 ? ov::get_u32(command.data()) : 0;
    const uint32_t group_count = command.size() >= 8 ? ov::get_u32(command.data() + 4) : 0;
    r["expected_quads"] = quad_count;
    r["expected_groups"] = (quad_count + 63) / 64;
    r["indirect_index_count"] = index_count;
    r["indirect_group_count"] = group_count;
    r["texture_bytes_read"] = int64_t(bytes.size());
    r["sheet_chunk"] = WorldCoord::to_string(sheet_chunk);

    const float step = settings.sheet_size / float(side);
    auto height = [&](float x, float z) { return origin.y + 3.0f * std::sin(x * 0.3f) * std::cos(z * 0.3f); };
    auto surf = [&](float x, float z) { return Float3{ x, height(x, z), z }; };
    auto surf_normal = [&](float x, float z) {
        const float dx = 0.9f * std::cos(x * 0.3f) * std::cos(z * 0.3f);
        const float dz = -0.9f * std::sin(x * 0.3f) * std::sin(z * 0.3f);
        return Float3{ -dx, 1.0f, -dz }.normalized();
    };

    // Quads land in arbitrary slots (atomicAdd), so match each slot to its grid cell by corner A.
    double max_pos_err = 0.0;
    double max_normal_err_deg = 0.0;
    int bad = 0;
    std::vector<int> seen(quad_count, 0);
    int bad_headers = 0;
    const uint32_t used_groups = (quad_count + PageFormat::QUAD_GROUP - 1) / PageFormat::QUAD_GROUP;
    for (uint32_t g = 0; g < used_groups && g < rows; g++) {
        bad_headers += PageFormat::header_chunk(PageFormat::row(data, g)) != WorldCoord::lo32_i(sheet_chunk) ? 1 : 0;
    }
    auto world_of = [&](uint32_t slot, int k) {
        const ov::ivec3 chunk = PageFormat::header_chunk(PageFormat::row(data, slot / PageFormat::QUAD_GROUP));
        return ChunkMath::grid_to_render_lo32(chunk, PageFormat::decode_local(PageFormat::corner(data, slot, k), scale), chunk_param);
    };
    for (uint32_t slot = 0; slot < quad_count && slot / PageFormat::QUAD_GROUP < rows; slot++) {
        if (PageFormat::is_zero(PageFormat::corner(data, slot, 0))) {
            bad++;
            continue;
        }
        const Float3 a = world_of(slot, 0);
        const int qx = int(std::lround((a.x - origin.x) / step));
        const int qz = int(std::lround((a.z - origin.z) / step));
        if (qx < 0 || qz < 0 || qx >= side || qz >= side) {
            bad++;
            continue;
        }
        seen[size_t(qz * side + qx)]++;
        std::pair<float, float> c[4] = {
            { origin.x + float(qx) * step, origin.z + float(qz) * step },
            { origin.x + float(qx + 1) * step, origin.z + float(qz) * step },
            { origin.x + float(qx) * step, origin.z + float(qz + 1) * step },
            { origin.x + float(qx + 1) * step, origin.z + float(qz + 1) * step },
        };
        if (qx >= side / 2) {
            std::swap(c[1], c[2]);
        }
        for (int k = 0; k < 4; k++) {
            max_pos_err = ov::max(max_pos_err, double(distance(world_of(slot, k), surf(c[k].first, c[k].second))));
            const Float3 n = PageFormat::decode_normal(PageFormat::corner(data, slot, k));
            const Float3 want = surf_normal(c[k].first, c[k].second);
            const double ang = ov::rad_to_deg(std::acos(ov::clamp(n.x * want.x + n.y * want.y + n.z * want.z, -1.0f, 1.0f)));
            max_normal_err_deg = ov::max(max_normal_err_deg, ang);
        }
    }
    int missing = 0;
    int duplicated = 0;
    for (const int n : seen) {
        missing += n == 0 ? 1 : 0;
        duplicated += n > 1 ? 1 : 0;
    }
    // Slots past the sheet must be zero (the tail of the last group is drawn), and so must the rows
    // past it.
    int dirty_tail = 0;
    for (uint32_t slot = quad_count; slot < used_groups * PageFormat::QUAD_GROUP; slot++) {
        for (int k = 0; k < 4; k++) {
            dirty_tail += PageFormat::is_zero(PageFormat::corner(data, slot, k)) ? 0 : 1;
        }
    }
    for (size_t i = size_t(used_groups) * PageFormat::TEX_WIDTH * PageFormat::TEXEL_CHANNELS; i < bytes.size() / 2; i++) {
        dirty_tail += data[i] != 0 ? 1 : 0;
    }
    r["max_position_error"] = max_pos_err;
    r["position_tolerance"] = pos_tolerance;
    r["bad_group_headers"] = bad_headers;
    r["max_normal_error_deg"] = max_normal_err_deg;
    r["bad_quads"] = bad;
    r["missing_cells"] = missing;
    r["duplicated_cells"] = duplicated;
    r["dirty_tail_floats"] = dirty_tail;
    r["write_ok"] = group_count == (quad_count + 63) / 64 && index_count == 64 * 6 && max_pos_err < pos_tolerance &&
            max_normal_err_deg < PageFormat::NORMAL_MAX_ERROR_DEG && bad == 0 && bad_headers == 0 && missing == 0 && duplicated == 0 && dirty_tail == 0;
}

void QuadPoolCheck::_verify_kills(ov::Value &r) {
    mutex.lock();
    const ov::bytes before_bytes = texture_data[0];
    const ov::bytes after_bytes = texture_data[1];
    mutex.unlock();
    const uint16_t *before = reinterpret_cast<const uint16_t *>(before_bytes.data());
    const uint16_t *after = reinterpret_cast<const uint16_t *>(after_bytes.data());
    // Kills zero whole rows (groups), headers included.
    auto killed = [](uint32_t g) {
        const uint32_t q = g * PageFormat::QUAD_GROUP;
        return (q >= KILL_A_OFFSET && q < KILL_A_OFFSET + KILL_A_COUNT) || (q >= KILL_B_OFFSET && q < KILL_B_OFFSET + KILL_B_COUNT);
    };
    int not_zeroed = 0;
    int damaged = 0;
    const uint32_t row_channels = PageFormat::TEX_WIDTH * PageFormat::TEXEL_CHANNELS;
    const uint32_t rows = uint32_t(ov::min(before_bytes.size(), after_bytes.size()) / PageFormat::ROW_BYTES);
    for (uint32_t g = 0; g < rows; g++) {
        for (uint32_t i = g * row_channels; i < (g + 1) * row_channels; i++) {
            if (killed(g)) {
                not_zeroed += after[i] != 0 ? 1 : 0;
            } else {
                damaged += after[i] != before[i] ? 1 : 0;
            }
        }
    }
    r["kill_not_zeroed_channels"] = not_zeroed;
    r["kill_damaged_channels"] = damaged;
    r["kill_ok"] = rows > 0 && not_zeroed == 0 && damaged == 0;
}

ov::Value QuadPoolCheck::report() {
    const std::lock_guard<std::mutex> lock(mutex);
    return results;
}

void QuadPoolCheck::run_kill_test() {
    OV_ERR_FAIL_COND_MSG(current_stage != 1, "QuadPoolCheck: run_kill_test() needs a verified write first.");
    current_stage = 2;
    device->post([this]() { _rt_kill(); });
}

void QuadPoolCheck::_rt_readback_done() {
    rt_readbacks_in_flight--;
    if (rt_destroying && rt_readbacks_in_flight == 0) {
        delete this;
    }
}

void QuadPoolCheck::destroy() {
    if (pool != nullptr) {
        pool->destroy();
        pool = nullptr;
    }
    device->post([this]() {
        for (const ov::gpu::Handle h : { ov::gpu::Handle(rt_set), ov::gpu::Handle(rt_pipeline), ov::gpu::Handle(rt_staging), ov::gpu::Handle(rt_counter) }) {
            if (h.is_valid()) {
                device->free(h);
            }
        }
        rt_destroying = true;
        if (rt_readbacks_in_flight == 0) {
            delete this;
        }
    });
}
