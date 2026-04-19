#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/thread.hpp>
#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/classes/semaphore.hpp>
#include "chunk_math.hpp"
#include "lod_structures.hpp"
#include "sdf_base.hpp"

using namespace godot;

class ChunkDiscoverer : public Object {

private:
    Ref<Thread> worker;


    Ref<Semaphore> semaphore;
    Ref<Mutex> mutex;
    bool exit_thread = false;

    // non thread safe variable
    ChunkParam chunk_param=ChunkParam();
    PackedFloat32Array lod_distances;

    // actively mutexed
    Vector3i camera_chunk;
    LODHashSet scanned_chunks;
    bool scanned_chunks_dirty = false;

    SDFBase * sdf;

    bool _chunk_is_outside_sdf(const Vector3i &chunk_pos);

public:

    void start();
    void stop();

    void set_camera_chunk(Vector3 camera_pos);
    bool get_scanned_chunks(LODHashSet &p_scanned_chunks);

    void set_config(
        ChunkParam &p_chunk_param,
        SDFBase * p_sdf
    );

    void discovery_loop();

    void setup_lod(
        int lod_levels = 4,
        float target_pixel_error = 12.0f, 
        float viewport_height = 1080.0f,
        float fov_rad = 75.0f * (3.14159265f / 180.0f));

    void _recursive_chunk_scan(const Vector3i &camera_chunk,const Vector3i &parent_chunk, LODHashSet &scanned_chunks);

};
