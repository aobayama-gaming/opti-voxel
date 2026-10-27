#include "godot_page_host.hpp"

#include "godot_callable.hpp"
#include "godot_conv.hpp"
#include "page_format.hpp"
#include "page_shading_glsl.hpp"
#include "quad_page_pool.hpp"
#include "voxel_sky.hpp"

#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/variant/aabb.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>

#include <cstring>

using namespace godot;

static constexpr const char PAGE_SHADER[] = R"gdshader(
shader_type spatial;
// Corners are chunk-local (page_format.hpp): world positions are rebuilt here from the group
// header's chunk id; the MultiMesh instance transforms are unused (left zero), so transform to
// view space ourselves.
render_mode skip_vertex_transform, cull_back;

// (RenderingServer::shader_set_code does not run the preprocessor: no #define here.)
/*GLSL_PAGE_DECODE*/

uniform usampler2D quad_tex : filter_nearest, repeat_disable;
// The chunks' palettes (palette_pool.hpp) and the materials' parameters (Material::Library::make_image).
uniform usampler2D palette_tex : filter_nearest, repeat_disable;
uniform sampler2D material_tex : filter_nearest, repeat_disable;
// 0 shaded, 1 normals, 2 lod, 3 page, 4 shaded + quad wireframe, 5 chunk hash,
// 6 group (INSTANCE_ID), 7 quad index within the group (VERTEX_ID / 4), 8 faceted, 9 material, 10 / 11 bark seam
// probe (bench.gd --probe), 12 biome (the materials' debug colours), 13 coverage (the quads a stalled chunk hides are
// drawn magenta, those it still draws orange)
uniform int debug_view = 0;
uniform float voxel_size = 0.5;
// Sea level (VoxelTerrain sets it for the continents preset): on the flat world render metres (above the render origin,
// set again when it moves), on a planet its altitude (0, its base); below -1e29, no height colours. The slope colours,
// the heights and the contour lines read the base's up and altitude (planet_up, planet_altitude: PlanetGLSL).
uniform float sea_level_m = -1e30;
// The render origin in chunk-id units, low 32 bits: positions are drawn relative to it; and its LOD0 voxel, low 32 bits
// (the shading frame's origin). PageDecode's.
uniform ivec3 origin_chunk_lo = ivec3(0);
uniform ivec3 origin_voxel_lo = ivec3(0);
uniform int chunk_size = 32;
uniform int skirt_size = 2;
uniform float pos_scale = 256.0; // PageFormat::pos_scale(chunk_size), set by the mesher
// Skirt depth per LOD (metres, SDFBase::skirt_depth_m): v_height adds the skirt push back, so the height
// colours see the unpushed surface and the dip along chunk borders keeps the colour around it.
uniform float skirt_depth_m[32];
instance uniform int page_id = 0;

varying vec3 v_world_normal;
varying vec2 v_quad_uv;
varying flat float v_lod;
varying flat vec3 v_chunk; // the chunk id's low 10 bits (the chunk hash views: hash3 wants small inputs)
varying flat ivec2 v_ids; // INSTANCE_ID, quad index | coverage << QUAD_INDEX_BITS (0 drawn, 1 stalled, 2 hidden)
// The height colours': the flat world's render metres (the contour lines read it through the shading frame), a planet's
// altitude.
varying float v_height;
// The quad's two materials, from the provoking corner A: x = A's (global id), y = the other one's | its corner mask
// << MATERIAL_ID_BITS (bit k, k = 1..3: corner k is not A's material). The fragment's weight is the mask interpolated
// over v_quad_uv, so the blend adds no interpolant: triangles are smaller than a pixel here, every attribute the
// fragment reads costs setup per triangle.
varying flat ivec2 v_materials;
varying vec3 v_world_pos; // the fragment's position (metres, render space), set by fragment() for light()

vec3 hash3(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}

// The shading frame's origin (shade_frame_glsl.hpp): the render origin's LOD0 voxel, low words.
ivec3 shade_origin_voxel() {
    return origin_voxel_lo;
}

float shade_voxel_m() {
    return voxel_size;
}

// The materials (page_shading_glsl.hpp: palettes, the quad's two materials, bark, leaves, ground detail).
/*GLSL_PAGE_SHADING*/
// The cloud layer's uniforms and functions (VoxelSky::glsl_clouds, the same text as the sky shader's): the sun's
// light in light() is cut where a cloud lies between the fragment and the sun; the leaves' wind sway.
/*GLSL_CLOUDS*/

void vertex() {
    int corner = VERTEX_ID % 4; // base quad corners A, B, C, D
    uvec4 h0 = texelFetch(quad_tex, ivec2(0, INSTANCE_ID), 0);
    uvec4 h1 = texelFetch(quad_tex, ivec2(1, INSTANCE_ID), 0);
    uvec4 c = texelFetch(quad_tex, ivec2(int(HEADER_TEXELS) + VERTEX_ID, INSTANCE_ID), 0);
    ivec3 chunk = page_header_chunk(h0, h1);
    int lod = page_chunk_lod(chunk);
    float voxel_dim = float(chunk_size - 2 * skirt_size);
    ivec3 rel = chunk - origin_chunk_lo;
    vec3 chunk_min = page_chunk_min(rel, lod, voxel_dim, voxel_size);
    vec3 local = page_decode_local(c, pos_scale);
    vec3 world = page_corner_world(chunk_min, lod, local, skirt_size, voxel_dim, voxel_size);
    // Wind: leaf corners sway (wind_sway, a pure function of the position: shared corners agree). Only up to LOD 2:
    // farther the amplitude is under a pixel, and the corner's material costs a palette and a material fetch.
    uint palette_index = page_decode_palette(c);
    if (palette_index != 0u && lod <= 2 && wind_sway_m > 0.0
            && material_pattern(corner_material(h1, palette_index)) == MATERIAL_PATTERN_LEAVES) {
        world += wind_sway(world);
    }
    // Coverage (page_format.hpp, header texel 2): a quad whose 4 corners all lie in hidden sub-cubes collapses to one
    // point, so a cut overlaps the chunks replacing it instead of leaving a gap. Every corner of the quad decides alike.
    // Only the rows flagged in texel 1 read texel 2.
    int coverage = 0;
    if (page_header_has_coverage(h1)) {
        uvec4 h2 = texelFetch(quad_tex, ivec2(int(PAGE_COVERAGE_TEXEL), INSTANCE_ID), 0);
        int first = int(HEADER_TEXELS) + (VERTEX_ID & ~3);
        bool hidden = true;
        for (int k = 0; k < 4; k++) {
            vec3 lk = page_decode_local(texelFetch(quad_tex, ivec2(first + k, INSTANCE_ID), 0), pos_scale);
            hidden = hidden && page_subcube_hidden(h2, page_corner_subcube(lk, skirt_size, voxel_dim));
        }
        coverage = hidden ? 2 : 1;
        if (hidden && debug_view != 13) {
            world = chunk_min;
        }
    }
    vec3 n = c != uvec4(0u) ? page_decode_normal(c) : vec3(0.0, 1.0, 0.0);
    // The quad's materials (quad_materials), from the provoking corner A: only A reads the other corners (the quad's
    // texels are consecutive).
    v_materials = ivec2(0);
    if (corner == 0) {
        v_materials = quad_materials(h1, palette_index, int(HEADER_TEXELS) + VERTEX_ID, INSTANCE_ID);
    }

    v_world_normal = n;
    v_ids = ivec2(INSTANCE_ID, VERTEX_ID / 4 | coverage << int(QUAD_INDEX_BITS));
    v_quad_uv = vec2(float(corner & 1), float(corner >> 1));
    v_lod = float(lod);
    v_chunk = vec3(chunk & ivec3(1023)); // absolute (not rel): the colours stay when the render origin moves
    // The cascade skirt pushed the SDF towards air by clamp(-inside) * depth past the chunk bounds, which lowers
    // the surface by push / n.up. The push is linear inside a cell, so on a planar surface the vertex lies
    // on the pushed plane and this gives back the exact height.
    float lo = float(skirt_size);
    vec3 in3 = min(local - lo, lo + voxel_dim - local);
    float push = clamp(-min(in3.x, min(in3.y, in3.z)), 0.0, 1.0) * skirt_depth_m[min(lod, 31)];
    v_height = (planet_radius_m > 0.0 ? planet_altitude(world) : world.y) + push / max(dot(n, planet_up(world)), 0.2);

    VERTEX = (VIEW_MATRIX * vec4(world, 1.0)).xyz;
    NORMAL = normalize(mat3(VIEW_MATRIX) * n);
}

void fragment() {
    // Shaded: grass on gentle slopes, rock on steep ones, faint contour lines every 100 m (mid tones, so
    // the tonemapper keeps the relief readable).
    vec3 p = (INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;
    vec3 base_up = planet_up(p);
    float up = clamp(dot(v_world_normal, base_up), 0.0, 1.0);
    float flat_w = ground_flat_weight(v_world_normal, base_up);
    vec3 albedo = mix(GROUND_STEEP, GROUND_FLAT, flat_w); // Material::GROUND_STEEP / GROUND_FLAT
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
    // Contour lines at whole multiples of contour_m of absolute height: on the flat world snapped to the shading frame's
    // period, on a planet the altitude itself (a few km at most).
    float h = v_height / contour_m;
    if (planet_radius_m <= 0.0) {
        float contour_period_m;
        h = shade_frame(vec3(0.0, v_height, 0.0), contour_m, contour_period_m).y * (shade_cells(vec3(1.0 / contour_m), contour_period_m).x / contour_period_m);
    }
    float fw = max(fwidth(h), 1e-6);
    float line = abs(fract(h + 0.5) - 0.5) / fw;
    // Contours fade out where they would be denser than ~10 pixels apart.
    float contour = 1.0 - smoothstep(0.03, 0.1, fw);
    v_world_pos = p;
    float footprint = -VERTEX.z * 2.0 / (PROJECTION_MATRIX[1][1] * VIEWPORT_SIZE.y); // metres per pixel
    vec3 shade = ground_detail(p, footprint) * mix(1.0 - 0.35 * contour, 1.0, smoothstep(0.0, 1.5, line));
    albedo *= shade;
    vec3 ground = albedo;
    uint mat_id = fragment_material(v_materials, v_quad_uv, p, voxel_size * exp2(v_lod));
    float roughness;
    vec3 emission;
    material_shading(mat_id, ground, p, v_world_normal, base_up, footprint, flat_w, shade, TIME, wind_sway_m, albedo, roughness, emission);
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
        float quad = float(v_ids.y & int(QUAD_GROUP - 1u));
        emission = vec3(quad / 63.0, 0.2, 1.0 - quad / 63.0);
    } else if (debug_view == 8) {
        // Faceted: one clay colour, each triangle lit by its own plane (the vertex normals, shared across a
        // crease, smear the shading of sharp edges).
        albedo = vec3(0.5);
        emission = vec3(0.0);
        NORMAL = normalize(cross(dFdy(VERTEX), dFdx(VERTEX)));
    } else if (debug_view == 9) {
        albedo = vec3(0.0);
        emission = mat_id == MATERIAL_GROUND ? vec3(0.15) : hash3(vec3(float(mat_id) * 0.61 + 0.5)) * 0.8 + 0.2;
    } else if (debug_view == 12) {
        // Biomes: each material's debug colour (a biome's ground carries its biome's), lit and with the contours so the
        // relief stays readable; dark grey for the rest (instances, the ground without a climate).
        vec4 dbg = mat_id == MATERIAL_GROUND ? vec4(0.0) : texelFetch(material_tex, material_texel(mat_id) + ivec2(MATERIAL_TEXEL_DEBUG, 0), 0);
        albedo = (dbg.a > 0.0 ? dbg.rgb : vec3(0.06)) * shade;
        emission = albedo * 0.3;
        roughness = 1.0;
    } else if (debug_view == 13) {
        int coverage = v_ids.y >> int(QUAD_INDEX_BITS);
        albedo = coverage == 2 ? vec3(0.9, 0.1, 0.9) : (coverage == 1 ? vec3(1.0, 0.5, 0.0) : albedo);
    } else if (debug_view == 10 || debug_view == 11) {
        // Bark seam probe (bench.gd --probe, read by tools/bark_seams.py): bytes, written through the inverse of the
        // sRGB encoding (the bench sets a linear tonemapper, no grade, glow or fog; SPECULAR 0 below), so each
        // channel reads back as the byte. Bark fragments at full detail only, black elsewhere. 10: the grain (smooth, so
        // a jump in it is a seam; the plates' V grooves step like one where they are a few pixels wide), a hash of the
        // chunk, 8 * LOD + 4; 11: log2 of the depth in 12 bits (6 per channel, times 4), the normal's azimuth.
        albedo = vec3(0.0);
        emission = vec3(0.0);
        bool bark = material_pattern(mat_id) == MATERIAL_PATTERN_BARK;
        if (bark && footprint < 0.012) {
            vec3 bytes;
            if (debug_view == 10) {
                float knot, ring;
                float grain = bark_grain(bark_point(p, v_world_normal, base_up), knot, ring);
                bytes = vec3(round(grain * 255.0), 4.0 * floor(cell_hash(v_chunk, uint(v_lod)).x * 64.0),
                        8.0 * v_lod + 4.0);
            } else {
                float depth = floor(clamp((log2(-VERTEX.z) + 4.0) / 16.0, 0.0, 1.0) * 4095.0);
                float azimuth = atan(v_world_normal.z, v_world_normal.x) / 6.2831853 + 0.5;
                bytes = vec3(4.0 * floor(depth / 64.0), 4.0 * mod(depth, 64.0), round(azimuth * 255.0));
            }
            vec3 c = bytes / 255.0;
            emission = mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
        }
    }
    if (debug_view == 4) {
        float d = min(min(v_quad_uv.x, 1.0 - v_quad_uv.x), min(v_quad_uv.y, 1.0 - v_quad_uv.y));
        if (d < fwidth(d) * 1.2) {
            albedo = vec3(0.05);
        }
    }
    ALBEDO = albedo;
    EMISSION = emission;
    ROUGHNESS = roughness;
    if (debug_view == 10 || debug_view == 11) {
        SPECULAR = 0.0; // the probe's bytes are its emission only
    }
}

// Lambert (Godot's default diffuse: LIGHT_COLOR carries the PI; the albedo is applied after), with the sun cut by
// the cloud layer where the fragment's ray to the sun crosses a cloud: the shadow of the cloud the sky shader draws
// there. The terrain's materials have a roughness of 1, so the specular lobe is left out.
void light() {
    float sun = 1.0;
    if (LIGHT_IS_DIRECTIONAL && cloud_shadow > 0.0) {
        vec3 to_sun = mat3(INV_VIEW_MATRIX) * LIGHT;
        vec3 hit;
        // VERTEX is not readable in light(): the fragment's position comes through v_world_pos.
        if (dot(to_sun, planet_up(v_world_pos)) > 0.05 && cloud_hit(v_world_pos, to_sun, hit)) {
            sun = 1.0 - cloud_shadow * cloud_density(cloud_texel(hit));
        }
    }
    DIFFUSE_LIGHT += clamp(dot(NORMAL, LIGHT), 0.0, 1.0) * ATTENUATION * LIGHT_COLOR * sun;
}
)gdshader";

GodotPageHost::GodotPageHost(RID p_scenario, RDDevice *p_device) :
        scenario(p_scenario), device(p_device) {
    RenderingServer *rs = RenderingServer::get_singleton();
    // Shared base mesh: QUAD_GROUP quads, corners A B C D = vertices 4q+0..3, triangles A,B,D / A,D,C
    // (== the legacy emit_quad positive winding). Positions are placeholders: the shader ignores them.
    PackedVector3Array vertices;
    PackedInt32Array indices;
    for (uint32_t q = 0; q < QuadPagePool::QUAD_GROUP; q++) {
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
    rs->shader_set_code(shader,
            String(PAGE_SHADER)
                    .replace("/*GLSL_PAGE_DECODE*/", godot_str(PageFormat::glsl_decode() + Material::glsl_constants()))
                    .replace("/*GLSL_CLOUDS*/", VoxelSky::glsl_clouds())
                    .replace("/*GLSL_PAGE_SHADING*/", godot_str(PageShading::glsl())));
}

void GodotPageHost::destroy() {
    RenderingServer *rs = RenderingServer::get_singleton();
    rs->free_rid(shader);
    rs->free_rid(base_mesh);
    device->post([this]() { delete this; });
}

void GodotPageHost::set_material_param(const StringName &p_name, const Variant &p_value) {
    material_params[p_name] = p_value;
    for (const Page &page : pages) {
        if (page.material.is_valid()) {
            RenderingServer::get_singleton()->material_set_param(page.material, p_name, p_value);
        }
    }
}

void GodotPageHost::create_page(int p_page, uint32_t p_rows) {
    RenderingServer *rs = RenderingServer::get_singleton();
    Page page;
    page.material = rs->material_create();
    rs->material_set_shader(page.material, shader);
    const Array keys = material_params.keys();
    for (int i = 0; i < keys.size(); i++) {
        rs->material_set_param(page.material, keys[i], material_params[keys[i]]);
    }

    page.multimesh = rs->multimesh_create();
    // Order matters: the indirect command buffer is only created by set_mesh when the data was
    // allocated with use_indirect first.
    rs->multimesh_allocate_data(page.multimesh, int32_t(p_rows), RenderingServer::MULTIMESH_TRANSFORM_2D, false, false, true);
    rs->multimesh_set_mesh(page.multimesh, base_mesh);
    // Without CPU data the engine cannot compute bounds; set by the user of the page.
    rs->multimesh_set_custom_aabb(page.multimesh, AABB(Vector3(-1, -1, -1), Vector3(2, 2, 2)));

    page.instance = rs->instance_create2(page.multimesh, scenario);
    rs->instance_geometry_set_material_override(page.instance, page.material);
    rs->instance_set_visible(page.instance, false);
    rs->instance_geometry_set_shader_parameter(page.instance, "page_id", p_page);

    if (int(pages.size()) <= p_page) {
        pages.resize(size_t(p_page) + 1);
    }
    pages[p_page] = page;
    // The render thread's copy, before the core's work on the page (FIFO).
    device->post([this, p_page, page]() {
        if (int(rt_pages.size()) <= p_page) {
            rt_pages.resize(size_t(p_page) + 1);
        }
        rt_pages[p_page].multimesh = page.multimesh;
        rt_pages[p_page].material = page.material;
    });
}

void GodotPageHost::free_page(int p_page) {
    RenderingServer *rs = RenderingServer::get_singleton();
    Page &page = pages[p_page];
    rs->free_rid(page.instance);
    rs->free_rid(page.multimesh);
    rs->free_rid(page.material);
    page = Page();
}

void GodotPageHost::set_page_visible(int p_page, bool p_visible) {
    RenderingServer::get_singleton()->instance_set_visible(pages[p_page].instance, p_visible);
}

void GodotPageHost::set_page_bounds(int p_page, const RenderBox &p_bounds) {
    RenderingServer::get_singleton()->multimesh_set_custom_aabb(pages[p_page].multimesh, AABB(godot_v3(p_bounds.position), godot_v3(p_bounds.size)));
}

void GodotPageHost::set_decode(const PageDecode &p_decode) {
    // Only what changed: a rebase moves the origin alone, and every page has its own material.
    const auto put = [this](bool p_changed, const char *p_name, const Variant &p_value) {
        if (p_changed || !has_decode) {
            set_material_param(p_name, p_value);
        }
    };
    put(p_decode.voxel_size != decode.voxel_size, "voxel_size", p_decode.voxel_size);
    put(p_decode.chunk_size != decode.chunk_size, "chunk_size", p_decode.chunk_size);
    put(p_decode.skirt_size != decode.skirt_size, "skirt_size", p_decode.skirt_size);
    put(p_decode.pos_scale != decode.pos_scale, "pos_scale", p_decode.pos_scale);
    put(!(p_decode.origin_chunk_lo == decode.origin_chunk_lo), "origin_chunk_lo", godot_v3i(p_decode.origin_chunk_lo));
    put(!(p_decode.origin_voxel_lo == decode.origin_voxel_lo), "origin_voxel_lo", godot_v3i(p_decode.origin_voxel_lo));
    decode = p_decode;
    has_decode = true;
}

void GodotPageHost::set_materials(const Material::Library::Image &p_image) {
    PackedByteArray bytes;
    bytes.resize(int64_t(p_image.rgba.size() * sizeof(float)));
    std::memcpy(bytes.ptrw(), p_image.rgba.data(), p_image.rgba.size() * sizeof(float));
    material_texture = ImageTexture::create_from_image(Image::create_from_data(p_image.width, p_image.height, false, Image::FORMAT_RGBAF, bytes));
    set_material_param("material_tex", material_texture->get_rid());
}

void GodotPageHost::set_debug_view(int p_view) {
    set_material_param("debug_view", p_view);
}

ov::gpu::Buffer GodotPageHost::rt_attach_page(int p_page, ov::gpu::Texture p_texture) {
    // On the render thread: these RenderingServer calls execute immediately.
    RenderingServer *rs = RenderingServer::get_singleton();
    PageRT &page = rt_pages[p_page];
    page.rs_texture = rs->texture_rd_create(device->get_rid(p_texture));
    rs->material_set_param(page.material, "quad_tex", page.rs_texture);
    page.command = device->import_buffer(rs->multimesh_get_command_buffer_rd_rid(page.multimesh));
    return page.command;
}

void GodotPageHost::rt_detach_page(int p_page) {
    PageRT &page = rt_pages[p_page];
    if (page.rs_texture.is_valid()) {
        RenderingServer::get_singleton()->free_rid(page.rs_texture);
    }
    device->free(page.command); // the multimesh's (already freed)
    page = PageRT();
}

void GodotPageHost::rt_set_palette_texture(ov::gpu::Texture p_texture) {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rt_palette_texture.is_valid()) {
        rs->free_rid(rt_palette_texture);
        rt_palette_texture = RID();
    }
    if (!p_texture.is_valid()) {
        return;
    }
    rt_palette_texture = rs->texture_rd_create(device->get_rid(p_texture));
    // The pages' materials are set on the main thread.
    const RID texture = rt_palette_texture;
    godot_callable([this, texture]() { set_material_param("palette_tex", texture); }).call_deferred();
}
