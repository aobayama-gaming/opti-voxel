#include "voxel_terrain.hpp"
#include <godot_cpp/classes/world3d.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/mesh.hpp>



void VoxelTerrain::_destroy_chunk() {

    auto *rs = RenderingServer::get_singleton();

    for (ChunkHashMap::Iterator it = visual_mesh_data.to_destroy_chunk.begin();
        it != visual_mesh_data.to_destroy_chunk.end();) {

        const Vector3i chunk_pos = it->key;
        const ChunkOutput output = it->value;

        if (output.mesh_rid.is_valid()) {
            rs->free_rid(output.instance_rid);
            rs->free_rid(output.mesh_rid);
            visual_mesh_data.loaded_chunk.erase(chunk_pos);
            visual_mesh_data.to_destroy_chunk.erase(chunk_pos);

            it = visual_mesh_data.to_destroy_chunk.begin();
        }



        
    }
}

void VoxelTerrain::_bind_methods() {
	//godot::ClassDB::bind_method(D_METHOD("print_type", "variant"), &ExampleClass::print_type);
}

void VoxelTerrain::_ready(){

    const int lod_numbers = 5;
    const int worker_pool = 8;

    visual_mesh_data.set_max_lod(lod_numbers);

    if(Engine::get_singleton()->is_editor_hint()) {
        camera = EditorInterface::get_singleton()->get_editor_viewport_3d()->get_camera_3d();
    }

    chunk_discoverer = memnew(ChunkDiscoverer);

    chunk_discoverer->set_config(chunk_param,sdf);

    chunk_discoverer->setup_lod(
        lod_numbers,
        5.0f
    ); //use base args

    chunk_discoverer->start();

    scenario = get_world_3d()->get_scenario();

    //start mesher pool
    visual_mesh = memnew(MesherPool);

    visual_mesh->start(
        worker_pool,
        lod_numbers,
        scenario,
        sdf,
        chunk_param
    );

    chunk_discoverer->set_camera_chunk(camera->get_global_position());

    //start GPU mesher pool
    gpu_mesher = memnew(GPUMesher);

    gpu_mesher->initialize(
        sdf,
        chunk_param,
        1024, // chunk_batch
        lod_numbers,
        scenario
    );

}

void VoxelTerrain::_process(double delta){

    bool GPU = true;
    //if(exec_time>1){return;}

    //debug
    
    //end of debug
    
    chunk_discoverer->set_camera_chunk(camera->get_global_position());

    LODHashSet new_scanned_chunk;

    bool dirty_scan = chunk_discoverer->get_scanned_chunks(new_scanned_chunk);

    if(dirty_scan){
        //UtilityFunctions::print("Dirty scan checking");

        visual_mesh_data.add_scan_chunk(new_scanned_chunk);
        if(GPU){
            gpu_mesher->send_chunk(visual_mesh_data.dirty_chunk);
        }
        else{
            visual_mesh->send_chunk(visual_mesh_data.dirty_chunk); 
        }

    }

    visual_mesh_data.update_destroy_from_scan();
    _destroy_chunk();

    if(GPU){
        gpu_mesher->process_queue();
        gpu_mesher->get_results(visual_mesh_data);
    }
    else{
        visual_mesh->process_queue();
        visual_mesh->get_results(visual_mesh_data);
    }

    // process the queue and launch worker
    //
    

    //
    

    exec_time++;

}