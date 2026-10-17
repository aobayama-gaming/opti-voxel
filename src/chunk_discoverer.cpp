#include "chunk_discoverer.hpp"

#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

using namespace godot;

bool ChunkDiscoverer::_chunk_is_outside_sdf(const ChunkId &chunk_pos){

    const float chunk_size = ChunkMath::world_chunk_size(chunk_pos,chunk_param);
    const float chunk_outer_radius = chunk_size*2 ; //0.8660254f // 2 is extra margin
    const VoxelPos chunk_center = ChunkMath::chunk_center_voxel(chunk_pos,chunk_param);
    // Half the chunk size in voxels: the corners are at the centre +- this.
    const int64_t half_voxels = ChunkMath::get_lod_size(chunk_pos) * chunk_param.half_chunk_voxels();

    for (int corner = 0; corner < 8; ++corner) {
        WorldPos corner_pos;
        corner_pos.voxel = chunk_center + VoxelPos(
            (corner & 1) ? half_voxels : -half_voxels,
            (corner & 2) ? half_voxels : -half_voxels,
            (corner & 4) ? half_voxels : -half_voxels
        );

        const float corner_distance = std::abs(sdf->evaluate(corner_pos));
        if (corner_distance <= chunk_outer_radius) {
            return false;
        }
    }

    return true;

}

void ChunkDiscoverer::start(){
    worker.instantiate();
    mutex.instantiate();
    semaphore.instantiate();

    //UtilityFunctions::print("ChunkDiscoverer : Start loop");

    worker->start(callable_mp(this,&ChunkDiscoverer::discovery_loop),Thread::PRIORITY_NORMAL);
}

void ChunkDiscoverer::stop(){

    mutex->lock();
    exit_thread=true;
    mutex->unlock();

    semaphore->post();

    if (worker.is_valid()) {
        worker->wait_to_finish();
    }

    worker.unref();
}

void ChunkDiscoverer::discovery_loop() {

    //UtilityFunctions::print("ChunkDiscoverer : loop started");

    while(true){

        semaphore->wait();

        mutex->lock();
        bool should_exit = exit_thread;

        ChunkId actual_camera_chunk = camera_chunk;
        // Add other variable import.
        mutex->unlock();

        if(should_exit){
            break;
        }

        const uint64_t t_scan_start = now_usec();
        LODHashSet new_scanned_chunk;

        const int lod_count = lod_distances.size()-1; 

        const int max_lod = lod_count-1; // The maximum LOD level to consider for loading. This is determined by the size of the lod_distances array, which should be set up so that each LOD level corresponds to a distance threshold.

        const int64_t max_lod_radius = int64_t(2) << (max_lod); // The radius of the largest LOD. This is the maximum distance at which chunks will be loaded, so it determines the size of the area to scan.

        const int64_t max_lod_number = MIN(int64_t(std::ceil( lod_distances[lod_count-1] / double(max_lod_radius)))+2, int64_t(max_lod));

        const ChunkId ancestor_chunk = ChunkMath::get_parent_from_child_until(actual_camera_chunk, max_lod);

        const ChunkId search_space = ChunkId(max_lod_number, max_lod_number, max_lod_number) * max_lod_radius;

        const ChunkId start_scan = ancestor_chunk - search_space;

        const ChunkId end_scan = ancestor_chunk + search_space;

        for (int64_t x = start_scan.x; x <= end_scan.x; x += max_lod_radius) {
            for (int64_t y = start_scan.y; y <= end_scan.y; y += max_lod_radius) {
                for (int64_t z = start_scan.z; z <= end_scan.z; z += max_lod_radius) {
                    ChunkId scan(x, y, z);
                    _recursive_chunk_scan(actual_camera_chunk,scan,new_scanned_chunk);
                }
            }
        }

        mutex->lock();
        //add output here
        scanned_chunks = new_scanned_chunk;
        scanned_chunks_dirty = true;
        scan_time.add(now_usec() - t_scan_start);
        last_scan_size = new_scanned_chunk.size();
        mutex->unlock();

        
        //UtilityFunctions::print("scanned chunk count:", (int)new_scanned_chunk.size());

        // print debug 
        // for (const ChunkId &chunk : new_scanned_chunk) {
        //     UtilityFunctions::print("scanned:", chunk);
        // }

    }

}

void ChunkDiscoverer::set_config(
    ChunkParam &p_chunk_param,
    SDFBase* p_sdf
){
    sdf=p_sdf;
    chunk_param= p_chunk_param;
}

bool ChunkDiscoverer::get_scanned_chunks(LODHashSet &p_scanned_chunks){

    mutex->lock();

    bool dirty=scanned_chunks_dirty;
    
    if(dirty){
        p_scanned_chunks = scanned_chunks;
        scanned_chunks_dirty = false;
    }

    mutex->unlock();

    return dirty;

}

void ChunkDiscoverer::setup_lod(
    int lod_levels,
    float target_pixel_error, 
    float viewport_height,
    float fov_rad){


    const float focal_length = viewport_height / (2.0f * std::tan(fov_rad * 0.5f));

    lod_distances.resize(lod_levels + 1);

    float previous_distance = 0.0f;
    for (int i = 0; i < lod_levels + 1; ++i) {
        // Geometric error follows voxel size per LOD (voxel doubles each level).
        const float lod_voxel_size_world = chunk_param.voxel_size* static_cast<float>(1 << i);
        const float geometric_error_world = lod_voxel_size_world * 0.5f;

        // Compute threshold in world units, then convert to chunk-coordinate units.
        // One chunk coordinate step equals CHUNK_DIMENSION / 2 in world units.
        float dist_world = (geometric_error_world * focal_length) / target_pixel_error;
        float dist_chunk = dist_world / (chunk_param.chunk_world_dimension() * 0.5f);

        // Keep thresholds strictly increasing and avoid near-zero first level.
        if (i == 0 && dist_chunk < 1.0f) {
            dist_chunk = 1.0f;
        }
        if (dist_chunk <= previous_distance) {
            dist_chunk = previous_distance + 1.0f;
        }

        lod_distances.set(i, dist_chunk);
        previous_distance = dist_chunk;
    }

}

void ChunkDiscoverer::_recursive_chunk_scan(const ChunkId &camera_chunk,const ChunkId &parent_chunk, LODHashSet &scanned_chunks)
{
    const int current_lod = ChunkMath::get_lod(parent_chunk);

    if(_chunk_is_outside_sdf(parent_chunk)){
        return;
    }

    if (current_lod ==0){
        scanned_chunks.insert(parent_chunk);
        return;
    }

    const double distance_to_player = camera_chunk.distance_to(parent_chunk);

    // Work fully in chunk-coordinate units.
    // Half-extent of current LOD chunk in chunk-coord units is 2^lod.
    const float half_extent = static_cast<float>(1 << current_lod);
    // Bounding sphere radius of the chunk cube in chunk-coord units.
    const float chunk_outer_radius = std::sqrt(3.0f) * half_extent;

    // lod_distances[L] is the distance from which LOD L is within the pixel error: split when closer.
    if (distance_to_player - chunk_outer_radius < lod_distances[current_lod]) {

        const int child_lod = current_lod - 1;
        const int64_t child_size = int64_t(1) << child_lod;

        for (int i = 0; i < 8; ++i) {
            ChunkId child_offset(
                (i & 1) ? child_size : -child_size,
                (i & 2) ? child_size : -child_size,
                (i & 4) ? child_size : -child_size
            );

            _recursive_chunk_scan(
                camera_chunk,
                parent_chunk + child_offset,
                scanned_chunks
            );
        }
    }
    else {
        scanned_chunks.insert(parent_chunk);
    }
}

void ChunkDiscoverer::set_camera_chunk(Vector3 camera_pos){
    /*
    Function to set the camera chunk in a thread-safe manner.
    */

    mutex->lock();
    ChunkId old_camera_chunk = camera_chunk;
    mutex->unlock();

    ChunkId new_camera_chunk = ChunkMath::chunk_at_render(camera_pos,chunk_param);

    if(new_camera_chunk!=old_camera_chunk){

        mutex->lock();
        camera_chunk = new_camera_chunk;
        mutex->unlock();

        semaphore->post();

        VLOG(2, "new chunk:", WorldCoord::to_string(new_camera_chunk));
    }

}
Dictionary ChunkDiscoverer::get_stats() {
    Dictionary d;
    mutex->lock();
    d["scan"] = scan_time.to_dict();
    d["last_scan_size"] = last_scan_size;
    mutex->unlock();
    return d;
}
