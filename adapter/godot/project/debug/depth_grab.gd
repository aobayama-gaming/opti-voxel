extends CompositorEffect
## DepthGrab, bench tooling (`--depth_grab=1`): copies one frame's scene depth, after the opaque pass, to an R32F texture and
## reads it back with the camera that drew it. tools/occlusion_census.py tests every loaded chunk's box against it
## (what a Hi-Z occlusion test of that frame would hide). Every RenderingDevice call runs on the render thread.

## One readback stays under this (many MB in one texture_get_data_async crash Godot 4.6.3, docs/gotchas.md): a bigger
## frame is reduced by k x k blocks first.
const MAX_READBACK_BYTES := 2 * 1024 * 1024

## Each texel keeps the farthest depth of its k x k block (reverse Z: the least), so what the reduced depth hides, the
## full one does too.
const COPY_GLSL := """
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0) uniform sampler2D depth_tex;
layout(r32f, set = 0, binding = 1) uniform restrict writeonly image2D out_img;
layout(push_constant, std430) uniform Params {
	ivec4 block_size; // x: k, yz: the depth's size
} params;
void main() {
	ivec2 p = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(p, imageSize(out_img)))) {
		return;
	}
	int k = params.block_size.x;
	ivec2 last = params.block_size.yz - 1;
	float farthest = 1.0;
	for (int y = 0; y < k; y++) {
		for (int x = 0; x < k; x++) {
			farthest = min(farthest, texelFetch(depth_tex, min(p * k + ivec2(x, y), last), 0).r);
		}
	}
	imageStore(out_img, p, vec4(farthest, 0.0, 0.0, 0.0));
}
"""

var _mutex := Mutex.new()
var _requested := false
var _result := {}  # set on the render thread once the readback lands: size, depth, projection, transform


func _init() -> void:
	effect_callback_type = EFFECT_CALLBACK_TYPE_POST_OPAQUE
	access_resolved_depth = true


## Main thread: grab the next frame drawn.
func request() -> void:
	_mutex.lock()
	_requested = true
	_result = {}
	_mutex.unlock()


## Main thread: once the grab has landed, writes p_dir/depth.f32 (reverse Z, R32F rows) and depth.json (the camera
## that drew it, the terrain's frame) for tools/occlusion_census.py and returns true; false while it is in flight.
func write(p_dir: String, p_camera: Camera3D, p_terrain: Node) -> bool:
	_mutex.lock()
	var grab := _result
	_mutex.unlock()
	if grab.is_empty():
		return false
	var fd := FileAccess.open(p_dir.path_join("depth.f32"), FileAccess.WRITE)
	fd.store_buffer(grab["depth"])
	fd.close()
	var pr: Projection = grab["projection"]
	var t: Transform3D = grab["transform"]
	var v4 := func(v: Vector4) -> Array: return [v.x, v.y, v.z, v.w]
	var v3 := func(v: Vector3) -> Array: return [v.x, v.y, v.z]
	var meta := {
		"width": grab["size"].x, "height": grab["size"].y, "block": grab["block"],
		"projection_columns": [v4.call(pr.x), v4.call(pr.y), v4.call(pr.z), v4.call(pr.w)],
		"basis_columns": [v3.call(t.basis.x), v3.call(t.basis.y), v3.call(t.basis.z)], "origin": v3.call(t.origin),
		"near": p_camera.near, "far": p_camera.far, "voxel_size": p_terrain.voxel_size, "lods": p_terrain.lod_numbers,
		"half_chunk_voxels": p_terrain.get_half_chunk_voxels(),
		"world_origin_voxel": [str(p_terrain.world_origin_x), str(p_terrain.world_origin_y), str(p_terrain.world_origin_z)],
	}
	fd = FileAccess.open(p_dir.path_join("depth.json"), FileAccess.WRITE)
	fd.store_string(JSON.stringify(meta, "  "))
	fd.close()
	print("BENCH: depth_grab %dx%d written" % [grab["size"].x, grab["size"].y])
	return true


func _render_callback(_p_type: int, p_data: RenderData) -> void:
	_mutex.lock()
	var go := _requested
	_requested = false
	_mutex.unlock()
	if not go:
		return
	var rd := RenderingServer.get_rendering_device()
	var buffers: RenderSceneBuffersRD = p_data.get_render_scene_buffers()
	var scene: RenderSceneData = p_data.get_render_scene_data()
	var full: Vector2i = buffers.get_internal_size()
	var k := 1
	while ((full.x + k - 1) / k) * ((full.y + k - 1) / k) * 4 > MAX_READBACK_BYTES:
		k += 1
	var size := Vector2i((full.x + k - 1) / k, (full.y + k - 1) / k)

	var src := RDShaderSource.new()
	src.language = RenderingDevice.SHADER_LANGUAGE_GLSL
	src.source_compute = COPY_GLSL
	var spirv := rd.shader_compile_spirv_from_source(src)
	if spirv.compile_error_compute != "":
		push_error("DepthGrab: " + spirv.compile_error_compute)
		return
	var shader := rd.shader_create_from_spirv(spirv)
	var pipeline := rd.compute_pipeline_create(shader)
	var fmt := RDTextureFormat.new()
	fmt.format = RenderingDevice.DATA_FORMAT_R32_SFLOAT
	fmt.width = size.x
	fmt.height = size.y
	fmt.usage_bits = RenderingDevice.TEXTURE_USAGE_STORAGE_BIT | RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	var out := rd.texture_create(fmt, RDTextureView.new())
	var sampler := rd.sampler_create(RDSamplerState.new())

	var u_depth := RDUniform.new()
	u_depth.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_depth.binding = 0
	u_depth.add_id(sampler)
	u_depth.add_id(buffers.get_depth_layer(0))
	var u_out := RDUniform.new()
	u_out.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_out.binding = 1
	u_out.add_id(out)
	var uset := rd.uniform_set_create([u_depth, u_out], shader, 0)

	var cl := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(cl, pipeline)
	rd.compute_list_bind_uniform_set(cl, uset, 0)
	var params := PackedInt32Array([k, full.x, full.y, 0]).to_byte_array()
	rd.compute_list_set_push_constant(cl, params, params.size())
	rd.compute_list_dispatch(cl, (size.x + 7) / 8, (size.y + 7) / 8, 1)
	rd.compute_list_end()

	var projection: Projection = scene.get_cam_projection()
	var xform: Transform3D = scene.get_cam_transform()
	rd.texture_get_data_async(out, 0, func(p_bytes: PackedByteArray) -> void:
		_mutex.lock()
		_result = {"size": size, "block": k, "depth": p_bytes, "projection": projection, "transform": xform}
		_mutex.unlock()
		rd.free_rid(out)
		rd.free_rid(sampler)
		rd.free_rid(shader))
