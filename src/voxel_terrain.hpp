#pragma once

#include "godot_cpp/classes/node3D.hpp"
#include "godot_cpp/classes/wrapped.hpp"
#include "godot_cpp/variant/variant.hpp"

#include "chunk_discoverer.hpp"

#include "godot_cpp/classes/camera3d.hpp"

#include "godot_cpp/classes/editor_interface.hpp"
#include "godot_cpp/classes/sub_viewport.hpp"

#include "mesher_pool.hpp"

#include <vector>

#include "chunk_memory.hpp"

#include "GPU_mesher.hpp"

//debug
#include "sdf_dummy.hpp"

using namespace godot;




class VoxelTerrain : public Node3D {
	GDCLASS(VoxelTerrain, Node3D)

protected:
	static void _bind_methods();

private:

    Camera3D *camera;

	ChunkParam chunk_param{
		32,     // chunk_size
		0.5f,  // voxel_size
		2,      // skirt_size
		true    // use_skirt
	};

    ChunkDiscoverer* chunk_discoverer;

	RID scenario;

	MesherPool* visual_mesh;

	GPUMesher* gpu_mesher;

	SDFBase* sdf = memnew(SDFDummy);

	std::vector<ChunkOutput *> visual_mesh_storage;

	VoxelState visual_mesh_data;

	void _destroy_chunk();

public:

    void _ready() override;
    void _process(double delta) override;

	VoxelTerrain() = default;
	~VoxelTerrain() override = default;

};