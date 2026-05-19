#pragma once

#include "godot_cpp/variant/variant.hpp"
#include <godot_cpp/classes/rendering_server.hpp>

using namespace godot;

struct ChunkOutput{
    Vector3i chunk_id;
    RID mesh_rid;
    RID instance_rid;

    void free(){
        RenderingServer* rs = RenderingServer::get_singleton();
        rs->free_rid(mesh_rid);
        rs->free_rid(instance_rid);
    }

};