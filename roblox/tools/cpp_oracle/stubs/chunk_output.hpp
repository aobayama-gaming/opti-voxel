#pragma once
// Oracle stub: no render resources, only the fields CPU_mesher.cpp touches.
#include <godot_shim.hpp>
#include "debug_log.hpp"
using namespace godot;
struct ChunkOutput {
    Vector3i chunk_id;
    RID mesh_rid;
    RID instance_rid;
    uint32_t triangle_count = 0;
    uint32_t drawn_triangles = 0;
    void free() {}
};
