#include "voxel_terrain.hpp"

void VoxelTerrain::_bind_methods() {
	//godot::ClassDB::bind_method(D_METHOD("print_type", "variant"), &ExampleClass::print_type);
}

void VoxelTerrain::_ready(){

    if(Engine::get_singleton()->is_editor_hint()) {
        camera = EditorInterface::get_singleton()->get_editor_viewport_3d()->get_camera_3d();
    }

    chunk_discoverer = memnew(ChunkDiscoverer);

    chunk_discoverer->set_chunk_param(chunk_param);

    chunk_discoverer->setup_lod(); //use base args

    chunk_discoverer->start();

}

void VoxelTerrain::_process(double delta){
    
    chunk_discoverer->set_camera_chunk(camera->get_global_position());

    LODHashSet new_scanned_chunk;

    bool dirty_scan = chunk_discoverer->get_scanned_chunks(new_scanned_chunk);

    if(dirty_scan){
        UtilityFunctions::print("Dirty scan checking");
    }

}