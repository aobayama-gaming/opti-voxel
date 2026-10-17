#include "quad_page_pool.hpp"

#include "debug_log.hpp"
#include "world_coord.hpp"

#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/rd_texture_format.hpp>
#include <godot_cpp/classes/rd_texture_view.hpp>

static constexpr const char PAGE_SHADER[] = R"gdshader(
shader_type spatial;
// Corners are chunk-local (page_format.hpp): world positions are rebuilt here from the group
// header's chunk id; the MultiMesh instance transforms are unused (left zero), so transform to
// view space ourselves.
render_mode skip_vertex_transform, cull_back;

// (RenderingServer::shader_set_code does not run the preprocessor: no #define here.)
// Must match page_format.hpp.
const int QUAD_GROUP = 64;
const int HEADER_TEXELS = 2;

uniform usampler2D quad_tex : filter_nearest, repeat_disable;
// 0 shaded, 1 normals, 2 lod, 3 page, 4 shaded + quad wireframe, 5 chunk hash,
// 6 group (INSTANCE_ID), 7 quad index within the group (VERTEX_ID / 4)
uniform int debug_view = 0;
uniform float voxel_size = 0.5;
// Sea level in metres above the world origin: below -1e29, no height colours (VoxelTerrain sets it for
// the continents preset).
uniform float sea_level_m = -1e30;
// World origin in chunk-id units, low 32 bits: positions are drawn relative to it.
uniform ivec3 origin_chunk_lo = ivec3(0);
uniform int chunk_size = 32;
uniform int skirt_size = 2;
uniform float pos_scale = 512.0;
// Skirt depth per LOD (metres, SDFBase::skirt_depth_m): v_height adds the skirt push back, so the height
// colours see the unpushed surface and the dip along chunk borders keeps the colour around it.
uniform float skirt_depth_m[32];
instance uniform int page_id = 0;

varying vec3 v_world_normal;
varying vec2 v_quad_uv;
varying flat float v_lod;
varying flat vec3 v_chunk;
varying flat ivec2 v_ids;
varying float v_height; // metres above the world origin (shaded view: contour lines)

vec3 oct_decode22(uint p) {
    vec2 f = vec2(float(p & 2047u), float(p >> 11u)) / 2047.0 * 2.0 - 1.0;
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = max(-n.z, 0.0);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}

vec3 hash3(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}

void vertex() {
    int corner = VERTEX_ID % 4; // base quad corners A, B, C, D
    uvec4 h0 = texelFetch(quad_tex, ivec2(0, INSTANCE_ID), 0);
    uvec4 h1 = texelFetch(quad_tex, ivec2(1, INSTANCE_ID), 0);
    uvec4 c = texelFetch(quad_tex, ivec2(HEADER_TEXELS + VERTEX_ID, INSTANCE_ID), 0);
    ivec3 chunk = ivec3(int(h0.x | (h0.y << 16u)), int(h0.z | (h0.w << 16u)), int(h1.x | (h1.y << 16u)));

    // ChunkMath::vertices_to_world. Empty / killed rows are all zero: every corner lands on the
    // same point -> zero area.
    int lod = chunk.x == 0 ? 0 : findLSB(chunk.x);
    float voxel_dim = float(chunk_size - 2 * skirt_size);
    float chunk_world_dim = voxel_dim * voxel_size;
    float size = float(1 << lod) * chunk_world_dim;
    // Wrapping int32 difference: exact while the chunk is within 2^31 ids of the origin. The origin is a
    // multiple of the root chunk span, so the low word keeps the LOD bit used above.
    ivec3 rel = chunk - origin_chunk_lo;
    vec3 chunk_min = vec3(rel) * (chunk_world_dim * 0.5) - vec3(size * 0.5);
    vec3 local = vec3(float(c.x & 16383u), float(c.y & 16383u), float(c.z & 16383u)) / pos_scale;
    vec3 world = chunk_min + (local - float(skirt_size)) * size / voxel_dim;
    uint packed_normal = (c.x >> 14u) | ((c.y >> 14u) << 2u) | ((c.z >> 14u) << 4u) | (c.w << 6u);
    vec3 n = c != uvec4(0u) ? oct_decode22(packed_normal) : vec3(0.0, 1.0, 0.0);

    v_world_normal = n;
    v_ids = ivec2(INSTANCE_ID, VERTEX_ID / 4);
    v_quad_uv = vec2(float(corner & 1), float(corner >> 1));
    v_lod = float(lod);
    v_chunk = vec3(rel);
    // skirt_sdf pushed the SDF towards air by clamp(-inside) * depth past the chunk bounds, which lowers
    // the surface by push / n.y. The push is linear inside a cell, so on a planar surface the vertex lies
    // on the pushed plane and this gives back the exact height.
    float lo = float(skirt_size);
    vec3 in3 = min(local - lo, lo + voxel_dim - local);
    float push = clamp(-min(in3.x, min(in3.y, in3.z)), 0.0, 1.0) * skirt_depth_m[min(lod, 31)];
    v_height = world.y + push / max(n.y, 0.2);

    VERTEX = (VIEW_MATRIX * vec4(world, 1.0)).xyz;
    NORMAL = normalize(mat3(VIEW_MATRIX) * n);
}

void fragment() {
    // Shaded: grass on gentle slopes, rock on steep ones, faint contour lines every 100 m (mid tones, so
    // the tonemapper keeps the relief readable).
    float up = clamp(v_world_normal.y, 0.0, 1.0);
    vec3 rock = vec3(0.36, 0.32, 0.28);
    vec3 albedo = mix(rock, vec3(0.24, 0.36, 0.16), smoothstep(0.8, 0.95, up));
    float contour_m = 100.0;
    if (sea_level_m > -1e29) {
        // Height colours (dark albedos: the scene's tonemapper brightens them): sea floor blue, darker with
        // depth, beach, lowland green, highland brown, snow; rock on steep land.
        float e = v_height - sea_level_m;
        vec3 steep = vec3(0.20, 0.18, 0.16);
        if (e < 0.0) {
            albedo = mix(vec3(0.005, 0.02, 0.10), vec3(0.04, 0.16, 0.35), smoothstep(-4000.0, 0.0, e));
        } else if (e < 30.0) {
            albedo = vec3(0.45, 0.40, 0.26);
        } else {
            vec3 land = mix(vec3(0.08, 0.20, 0.05), vec3(0.24, 0.18, 0.10), smoothstep(200.0, 1800.0, e));
            land = mix(land, vec3(0.80), smoothstep(2800.0, 3600.0, e));
            albedo = mix(steep, land, smoothstep(0.7, 0.9, up));
        }
        contour_m = 500.0;
    }
    float h = v_height / contour_m;
    float fw = max(fwidth(h), 1e-6);
    float line = abs(fract(h + 0.5) - 0.5) / fw;
    // Contours fade out where they would be denser than ~10 pixels apart.
    float contour = 1.0 - smoothstep(0.03, 0.1, fw);
    albedo *= mix(1.0 - 0.35 * contour, 1.0, smoothstep(0.0, 1.5, line));
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
        albedo = vec3(0.0);
        emission = hash3(v_chunk * 0.37 + v_lod * 13.0) * 0.8 + 0.2;
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
    ERR_FAIL_COND_MSG(p_page_capacity % QUAD_GROUP != 0, "QuadPagePool: page capacity must be a multiple of 64 quads.");
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

int QuadPagePool::_create_page(int p_index) {
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

    int index = p_index;
    if (index < 0) {
        index = pages.size();
        pages.push_back(page);
    } else {
        pages[index] = page;
    }
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
    format->set_format(TEX_FORMAT);
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
    rt_set_group_count(p_page, 0);
}

void QuadPagePool::_rt_reset_page(int p_page) {
    rt_set_group_count(p_page, 0);
}

int QuadPagePool::acquire_page() {
    int page;
    if (free_pages.is_empty()) {
        int reuse = -1;
        if (!trimmed_pages.is_empty()) {
            reuse = trimmed_pages[trimmed_pages.size() - 1];
            trimmed_pages.remove_at(trimmed_pages.size() - 1);
        }
        page = _create_page(reuse);
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
    rs->call_on_render_thread(callable_mp(this, &QuadPagePool::_rt_reset_page).bind(p_page));
    free_pages.push_back(p_page);
}

int QuadPagePool::trim(int p_keep_free, int p_max) {
    RenderingServer *rs = RenderingServer::get_singleton();
    int freed = 0;
    while ((int)free_pages.size() > p_keep_free && freed < p_max) {
        const int index = free_pages[free_pages.size() - 1];
        free_pages.remove_at(free_pages.size() - 1);
        Page &page = pages[index];
        // Render-server frees are queued in order with call_on_render_thread: everything already
        // queued for this page (resets, kills, copies) runs before its texture goes away.
        rs->free_rid(page.instance);
        rs->free_rid(page.multimesh);
        rs->free_rid(page.material);
        page = Page();
        rs->call_on_render_thread(callable_mp(this, &QuadPagePool::_rt_free_page).bind(index));
        trimmed_pages.push_back(index);
        pages_trimmed++;
        freed++;
    }
    return freed;
}

void QuadPagePool::_rt_free_page(int p_page) {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    PageRT &page = rt_pages[p_page];
    RenderingServer::get_singleton()->free_rid(page.rs_texture);
    rd->free_rid(page.texture);
    page = PageRT(); // the command buffer belonged to the (already freed) multimesh
}

void QuadPagePool::set_page_aabb(int p_page, const AABB &p_aabb) {
    RenderingServer::get_singleton()->multimesh_set_custom_aabb(pages[p_page].multimesh, p_aabb);
}

void QuadPagePool::set_material_param(const StringName &p_name, const Variant &p_value) {
    material_params[p_name] = p_value;
    for (const Page &page : pages) {
        if (page.material.is_valid()) {
            RenderingServer::get_singleton()->material_set_param(page.material, p_name, p_value);
        }
    }
}

void QuadPagePool::set_debug_view(int p_view) {
    debug_view = p_view;
    set_material_param("debug_view", p_view);
}

QuadPagePool::PageRD QuadPagePool::rt_get_page_rd(int p_page) {
    PageRD out;
    if (p_page < (int)rt_pages.size()) {
        out.texture = rt_pages[p_page].texture;
        out.command = rt_pages[p_page].command;
    }
    return out;
}

void QuadPagePool::rt_set_group_count(int p_page, uint32_t p_groups) {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    const PageRD page_rd = rt_get_page_rd(p_page);
    ERR_FAIL_COND_MSG(!page_rd.command.is_valid(), "QuadPagePool: page has no indirect command buffer.");
    PackedByteArray data;
    data.resize(4);
    data.encode_u32(0, p_groups);
    rd->buffer_update(page_rd.command, INDIRECT_INSTANCE_COUNT_OFFSET, 4, data);
}

void QuadPagePool::rt_copy_page_from(int p_page, RID p_src, uint32_t p_src_row) {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    const PageRD page_rd = rt_get_page_rd(p_page);
    rd->texture_copy(p_src, page_rd.texture, Vector3(0, p_src_row, 0), Vector3(0, 0, 0),
            Vector3(TEX_WIDTH, get_page_rows(), 1), 0, 0, 0, 0);
}

void QuadPagePool::rt_kill_range(int p_page, uint32_t p_offset, uint32_t p_count) {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    const PageRD page_rd = rt_get_page_rd(p_page);
    if (!page_rd.texture.is_valid()) {
        return; // trimmed
    }
    ERR_FAIL_COND_MSG(p_offset % QUAD_GROUP != 0, "QuadPagePool: kill ranges must start on a group.");
    const uint32_t first_row = p_offset / QUAD_GROUP;
    const uint32_t rows = (p_count + QUAD_GROUP - 1) / QUAD_GROUP;
    rd->texture_copy(rt_zero_texture, page_rd.texture, Vector3(0, 0, 0), Vector3(0, first_row, 0),
            Vector3(TEX_WIDTH, rows, 1), 0, 0, 0, 0);
}

void QuadPagePool::rt_copy_rows(int p_src_page, int p_dst_page, uint32_t p_src_row, uint32_t p_dst_row, uint32_t p_rows) {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    ERR_FAIL_INDEX(p_src_page, (int)rt_pages.size());
    ERR_FAIL_INDEX(p_dst_page, (int)rt_pages.size());
    ERR_FAIL_COND(p_src_page == p_dst_page);
    rd->texture_copy(rt_pages[p_src_page].texture, rt_pages[p_dst_page].texture, Vector3(0, p_src_row, 0),
            Vector3(0, p_dst_row, 0), Vector3(TEX_WIDTH, p_rows, 1), 0, 0, 0, 0);
}

Dictionary QuadPagePool::get_stats() const {
    Dictionary d;
    const int64_t alive = (int64_t)pages.size() - (int64_t)trimmed_pages.size();
    d["pages_total"] = alive;
    d["pages_in_use"] = (int64_t)pages_in_use;
    d["page_capacity"] = (int64_t)page_capacity;
    d["slots_total"] = alive * page_capacity;
    d["growth_events"] = (int64_t)growth_events;
    d["pages_trimmed"] = (int64_t)pages_trimmed;
    d["vram_mb"] = alive * (double)get_page_rows() * PageFormat::ROW_BYTES / 1048576.0;
    return d;
}

void QuadPagePool::destroy() {
    RenderingServer *rs = RenderingServer::get_singleton();
    for (Page &page : pages) {
        if (!page.multimesh.is_valid()) {
            continue; // trimmed
        }
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
