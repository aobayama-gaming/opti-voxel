#pragma once

#include "godot_cpp/variant/variant.hpp"

using namespace godot;

struct ChunkOutput{
    Vector3i chunk_id;
    RID mesh_rid;
    RID instance_rid;
};