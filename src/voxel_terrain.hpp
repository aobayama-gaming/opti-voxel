#pragma once

#include "godot_cpp/classes/node3D.hpp"
#include "godot_cpp/classes/wrapped.hpp"
#include "godot_cpp/variant/variant.hpp"

#include "chunk_discoverer.hpp"

#include "godot_cpp/classes/camera3d.hpp"

#include "godot_cpp/classes/editor_interface.hpp"
#include "godot_cpp/classes/sub_viewport.hpp"

using namespace godot;

class VoxelTerrain : public Node {
	GDCLASS(VoxelTerrain, Node)

protected:
	static void _bind_methods();

private:

    Camera3D *camera;

	ChunkParam chunk_param;

    ChunkDiscoverer* chunk_discoverer;
    

public:

    void _ready() override;
    void _process(double delta) override;

	VoxelTerrain() = default;
	~VoxelTerrain() override = default;

};