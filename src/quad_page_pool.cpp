#include "quad_page_pool.hpp"

#include "debug_log.hpp"

#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/rd_texture_format.hpp>
#include <godot_cpp/classes/rd_texture_view.hpp>

static constexpr const char PAGE_SHADER[] = R"gdshader(
shader_type spatial;
// Positions are absolute world coordinates pulled from the page texture; the MultiMesh instance
// transforms are unused (left zero), so transform to view space ourselves.
render_mode skip_vertex_transform, cull_back;

// (RenderingServer::shader_set_code does not run the preprocessor: no #define here.)
const int QUAD_GROUP = 64;
const int TEX_WIDTH = 1024;

uniform sampler2D quad_tex : filter_nearest, repeat_disable;
// 0 shaded, 1 normals, 2 lod, 3 page, 4 shaded + quad wireframe, 5 chunk hash,
// 6 group (INSTANCE_ID), 7 quad index within the group (VERTEX_ID / 4)
uniform int debug_view = 0;
uniform float voxel_size = 0.5;
uniform float chunk_world_dimension = 14.0;
instance uniform int page_id = 0;

varying vec3 v_world_normal;
varying vec2 v_quad_uv;
varying flat float v_lod;
varying flat vec3 v_quad_corner0;
varying flat ivec2 v_ids;

vec3 oct_decode24(float packed) {
    uint p = uint(packed);
    vec2 f = vec2(float(p & 4095u), float(p >> 12u)) / 4095.0 * 2.0 - 1.0;
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = max(-n.z, 0.0);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}

vec4 corner_texel(int t) {
    return texelFetch(quad_tex, ivec2(t % TEX_WIDTH, t / TEX_WIDTH), 0);
}

vec3 hash3(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}

void vertex() {
    int corner = VERTEX_ID % 4; // base quad corners A, B, C, D
    int quad_texel = (INSTANCE_ID * QUAD_GROUP + VERTEX_ID / 4) * 4;
    vec4 d = corner_texel(quad_texel + corner);
    // Empty / killed quads are all-zero texels: every corner lands on the origin -> zero area.
    vec3 world = d.xyz;
    vec3 n = d.w > 0.0 ? oct_decode24(d.w - 1.0) : vec3(0.0, 1.0, 0.0);

    v_world_normal = n;
    v_ids = ivec2(INSTANCE_ID, VERTEX_ID / 4);
    v_quad_uv = vec2(float(corner & 1), float(corner >> 1));
    v_lod = 0.0;
    v_quad_corner0 = world;
    if (debug_view == 2 || debug_view == 5) {
        vec3 a = corner_texel(quad_texel).xyz;
        vec3 b = corner_texel(quad_texel + 1).xyz;
        vec3 c = corner_texel(quad_texel + 2).xyz;
        vec3 e = corner_texel(quad_texel + 3).xyz;
        float edge = 0.25 * (length(b - a) + length(c - a) + length(e - b) + length(e - c));
        v_lod = round(log2(max(edge, 1e-6) / voxel_size));
        v_quad_corner0 = a;
    }

    VERTEX = (VIEW_MATRIX * vec4(world, 1.0)).xyz;
    NORMAL = normalize(mat3(VIEW_MATRIX) * n);
}

void fragment() {
    vec3 albedo = vec3(1.0);
    vec3 emission = vec3(0.0);
    if (debug_view == 1) {
        albedo = vec3(0.0);
        emission = v_world_normal * 0.5 + 0.5;
    } else if (debug_view == 2) {
        albedo = vec3(0.0);
        emission = hash3(vec3(v_lod * 7.0 + 1.0)) * 0.8 + 0.2;
    } else if (debug_view == 3) {
        albedo = vec3(0.0);
        emission = hash3(vec3(float(page_id) + 0.5)) * 0.8 + 0.2;
    } else if (debug_view == 5) {
        float size = chunk_world_dimension * exp2(v_lod);
        albedo = vec3(0.0);
        emission = hash3(floor(v_quad_corner0 / size) + v_lod * 13.0) * 0.8 + 0.2;
    } else if (debug_view == 6) {
        albedo = vec3(0.0);
        emission = hash3(vec3(float(v_ids.x) + 0.5)) * 0.8 + 0.2;
    } else if (debug_view == 7) {
        albedo = vec3(0.0);
        emission = vec3(float(v_ids.y) / 63.0, 0.2, 1.0 - float(v_ids.y) / 63.0);
    }
    if (debug_view == 4) {
        float d = min(min(v_quad_uv.x, 1.0 - v_quad_uv.x), min(v_quad_uv.y, 1.0 - v_quad_uv.y));
        if (d < fwidth(d) * 1.2) {
            albedo = vec3(0.05);
        }
    }
    ALBEDO = albedo;
    EMISSION = emission;
    ROUGHNESS = 1.0;
}
)gdshader";

void QuadPagePool::initialize(RID p_scenario, uint32_t p_page_capacity, uint32_t p_initial_pages, float p_voxel_size) {
    ERR_FAIL_COND_MSG(p_page_capacity % (TEX_WIDTH / 4) != 0 || p_page_capacity % QUAD_GROUP != 0,
            "QuadPagePool: page capacity must be a multiple of 256 quads.");
    RenderingServer *rs = RenderingServer::get_singleton();
    scenario = p_scenario;
    page_capacity = p_page_capacity;
    voxel_size = p_voxel_size;

    // Shared base mesh: QUAD_GROUP quads, corners A B C D = vertices 4q+0..3, triangles A,B,D / A,D,C
    // (== the legacy emit_quad positive winding). Positions are placeholders: the shader ignores them.
    PackedVector3Array vertices;
    PackedInt32Array indices;
    for (uint32_t q = 0; q < QUAD_GROUP; q++) {
        vertices.push_back(Vector3(0, 0, q));
        vertices.push_back(Vector3(1, 0, q));
        vertices.push_back(Vector3(0, 1, q));
        vertices.push_back(Vector3(1, 1, q));
        for (int i : { 0, 1, 3, 0, 3, 2 }) {
            indices.push_back(q * 4 + i);
        }
    }
    Array arrays;
    arrays.resize(Mesh::ARRAY_MAX);
    arrays[Mesh::ARRAY_VERTEX] = vertices;
    arrays[Mesh::ARRAY_INDEX] = indices;
    base_mesh = rs->mesh_create();
    rs->mesh_add_surface_from_arrays(base_mesh, RenderingServer::PRIMITIVE_TRIANGLES, arrays);

    shader = rs->shader_create();
    rs->shader_set_code(shader, PAGE_SHADER);
    material_params["voxel_size"] = voxel_size;

    for (uint32_t i = 0; i < p_initial_pages; i++) {
        free_pages.push_back(_create_page());
    }
}

int QuadPagePool::_create_page() {
    RenderingServer *rs = RenderingServer::get_singleton();
    Page page;
    page.material = rs->material_create();
    rs->material_set_shader(page.material, shader);
    Array keys = material_params.keys();
    for (int i = 0; i < keys.size(); i++) {
        rs->material_set_param(page.material, keys[i], material_params[keys[i]]);
    }
    rs->material_set_param(page.material, "debug_view", debug_view);

    page.multimesh = rs->multimesh_create();
    // Order matters: the indirect command buffer is only created by set_mesh when the data was
    // allocated with use_indirect first.
    rs->multimesh_allocate_data(page.multimesh, page_capacity / QUAD_GROUP, RenderingServer::MULTIMESH_TRANSFORM_2D,
            false, false, true);
    rs->multimesh_set_mesh(page.multimesh, base_mesh);
    // Without CPU data the engine cannot compute bounds; set by the user of the page.
    rs->multimesh_set_custom_aabb(page.multimesh, AABB(Vector3(-1, -1, -1), Vector3(2, 2, 2)));

    page.instance = rs->instance_create2(page.multimesh, scenario);
    rs->instance_geometry_set_material_override(page.instance, page.material);
    rs->instance_set_visible(page.instance, false);

    const int index = pages.size();
    pages.push_back(page);
    rs->instance_geometry_set_shader_parameter(page.instance, "page_id", index);

    // Texture creation must happen on the render thread; it also zeroes the group count (a fresh
    // page would otherwise draw every group).
    rs->call_on_render_thread(callable_mp(this, &QuadPagePool::_rt_create_page).bind(index, page.multimesh, page.material));
    return index;
}

void QuadPagePool::_rt_create_page(int p_page, RID p_multimesh, RID p_material) {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    RenderingServer *rs = RenderingServer::get_singleton();

    Ref<RDTextureFormat> format;
    format.instantiate();
    format->set_format(RenderingDevice::DATA_FORMAT_R32G32B32A32_SFLOAT);
    format->set_width(TEX_WIDTH);
    format->set_height(get_page_rows());
    format->set_usage_bits(RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT | RenderingDevice::TEXTURE_USAGE_CAN_COPY_TO_BIT |
            RenderingDevice::TEXTURE_USAGE_CAN_COPY_FROM_BIT | RenderingDevice::TEXTURE_USAGE_CAN_UPDATE_BIT);
    Ref<RDTextureView> view;
    view.instantiate();

    if (!rt_zero_texture.is_valid()) {
        rt_zero_texture = rd->texture_create(format, view);
        rd->texture_clear(rt_zero_texture, Color(0, 0, 0, 0), 0, 1, 0, 1);
    }

    if ((int)rt_pages.size() <= p_page) {
        rt_pages.resize(p_page + 1);
    }
    PageRT &page = rt_pages[p_page];
    page.texture = rd->texture_create(format, view);
    rd->texture_clear(page.texture, Color(0, 0, 0, 0), 0, 1, 0, 1);
    // We are on the render thread: these RenderingServer calls execute immediately.
    page.rs_texture = rs->texture_rd_create(page.texture);
    rs->material_set_param(p_material, "quad_tex", page.rs_texture);
    page.command = rs->multimesh_get_command_buffer_rd_rid(p_multimesh);
    rt_set_group_count(p_page, p_multimesh, 0);
}

void QuadPagePool::_rt_reset_page(int p_page, RID p_multimesh) {
    rt_set_group_count(p_page, p_multimesh, 0);
}

int QuadPagePool::acquire_page() {
    int page;
    if (free_pages.is_empty()) {
        page = _create_page();
        growth_events++;
        VLOG(2, "QuadPagePool: grew to ", (int64_t)pages.size(), " pages");
    } else {
        page = free_pages[free_pages.size() - 1];
        free_pages.remove_at(free_pages.size() - 1);
    }
    pages[page].in_use = true;
    pages_in_use++;
    RenderingServer::get_singleton()->instance_set_visible(pages[page].instance, true);
    return page;
}

void QuadPagePool::release_page(int p_page) {
    ERR_FAIL_INDEX(p_page, (int)pages.size());
    ERR_FAIL_COND(!pages[p_page].in_use);
    RenderingServer *rs = RenderingServer::get_singleton();
    pages[p_page].in_use = false;
    pages_in_use--;
    rs->instance_set_visible(pages[p_page].instance, false);
    rs->call_on_render_thread(callable_mp(this, &QuadPagePool::_rt_reset_page).bind(p_page, pages[p_page].multimesh));
    free_pages.push_back(p_page);
}

void QuadPagePool::set_page_aabb(int p_page, const AABB &p_aabb) {
    RenderingServer::get_singleton()->multimesh_set_custom_aabb(pages[p_page].multimesh, p_aabb);
}

void QuadPagePool::set_material_param(const StringName &p_name, const Variant &p_value) {
    material_params[p_name] = p_value;
    for (const Page &page : pages) {
        RenderingServer::get_singleton()->material_set_param(page.material, p_name, p_value);
    }
}

void QuadPagePool::set_debug_view(int p_view) {
    debug_view = p_view;
    set_material_param("debug_view", p_view);
}

QuadPagePool::PageRD QuadPagePool::rt_get_page_rd(int p_page, RID p_multimesh) {
    PageRD out;
    if (p_page < (int)rt_pages.size()) {
        out.texture = rt_pages[p_page].texture;
        out.command = rt_pages[p_page].command;
    }
    return out;
}

void QuadPagePool::rt_set_group_count(int p_page, RID p_multimesh, uint32_t p_groups) {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    const PageRD page_rd = rt_get_page_rd(p_page, p_multimesh);
    ERR_FAIL_COND_MSG(!page_rd.command.is_valid(), "QuadPagePool: page has no indirect command buffer.");
    PackedByteArray data;
    data.resize(4);
    data.encode_u32(0, p_groups);
    rd->buffer_update(page_rd.command, INDIRECT_INSTANCE_COUNT_OFFSET, 4, data);
}

void QuadPagePool::rt_copy_page_from(int p_page, RID p_multimesh, RID p_src, uint32_t p_src_row) {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    const PageRD page_rd = rt_get_page_rd(p_page, p_multimesh);
    rd->texture_copy(p_src, page_rd.texture, Vector3(0, p_src_row, 0), Vector3(0, 0, 0),
            Vector3(TEX_WIDTH, get_page_rows(), 1), 0, 0, 0, 0);
}

void QuadPagePool::rt_kill_range(int p_page, RID p_multimesh, uint32_t p_offset, uint32_t p_count) {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    const PageRD page_rd = rt_get_page_rd(p_page, p_multimesh);
    // The texel run [first, end) wraps over rows: copy zeros as up to 3 rectangles.
    uint32_t first = p_offset * 4;
    const uint32_t end = (p_offset + p_count) * 4;
    while (first < end) {
        const uint32_t row = first / TEX_WIDTH;
        const uint32_t col = first % TEX_WIDTH;
        uint32_t width;
        uint32_t height;
        if (col != 0 || end - first < TEX_WIDTH) { // partial row
            width = MIN(TEX_WIDTH - col, end - first);
            height = 1;
        } else { // block of full rows
            width = TEX_WIDTH;
            height = (end - first) / TEX_WIDTH;
        }
        rd->texture_copy(rt_zero_texture, page_rd.texture, Vector3(col, 0, 0), Vector3(col, row, 0),
                Vector3(width, height, 1), 0, 0, 0, 0);
        first += width * height;
    }
}

Dictionary QuadPagePool::get_stats() const {
    Dictionary d;
    d["pages_total"] = (int64_t)pages.size();
    d["pages_in_use"] = (int64_t)pages_in_use;
    d["page_capacity"] = (int64_t)page_capacity;
    d["slots_total"] = (int64_t)pages.size() * page_capacity;
    d["growth_events"] = (int64_t)growth_events;
    d["vram_mb"] = pages.size() * (double)page_capacity * QUAD_BYTES / 1048576.0;
    return d;
}

void QuadPagePool::destroy() {
    RenderingServer *rs = RenderingServer::get_singleton();
    for (Page &page : pages) {
        rs->free_rid(page.instance);
        rs->free_rid(page.multimesh);
        rs->free_rid(page.material);
    }
    pages.clear();
    free_pages.clear();
    rs->free_rid(shader);
    rs->free_rid(base_mesh);
    // Render-thread resources are freed there, after everything queued before; then we die.
    rs->call_on_render_thread(callable_mp(this, &QuadPagePool::_rt_destroy));
}

void QuadPagePool::_rt_destroy() {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    RenderingServer *rs = RenderingServer::get_singleton();
    for (PageRT &page : rt_pages) {
        if (page.rs_texture.is_valid()) {
            rs->free_rid(page.rs_texture);
        }
        if (page.texture.is_valid()) {
            rd->free_rid(page.texture);
        }
    }
    rt_pages.clear();
    if (rt_zero_texture.is_valid()) {
        rd->free_rid(rt_zero_texture);
    }
    memdelete(this);
}
