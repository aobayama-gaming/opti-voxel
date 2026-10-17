#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/dictionary.hpp>

using namespace godot;

// Test harness for the integer frame (world_coord.hpp, gpu_frame.hpp): builds WorldPos and evaluates
// the terrain SDF on the GPU (local RenderingDevice, the same GLSL snippets as the meshers) for test
// vectors around far world origins and 2^32-voxel boundaries, and checks against the CPU mirrors:
//   - CPU sample_pos == GPU WorldPos and anchor, bit for bit
//   - the same voxel reached from two chunks (neighbour, parent) gives bit-identical GPU results
//   - CPU legacy_rel_voxels == GPU, bit for bit (the high word reaches the drawn terrain through it)
//   - the terrain SDF stays continuous across a low-word wrap
//   - every probe of the probe table (frame_spike.cpp): CPU vs GPU, and pair identity
// run() is synchronous and needs a GPU (not --headless). Run: tools/frame_test.sh
class FrameSpike : public Node {
    GDCLASS(FrameSpike, Node)

protected:
    static void _bind_methods();

public:
    Dictionary run();

    void set_voxel_size(float p_size) { voxel_size = p_size; }
    float get_voxel_size() const { return voxel_size; }

private:
    float voxel_size = 0.5f;
};
