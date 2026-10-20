#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/dictionary.hpp>

using namespace godot;

// Test harness for the integer frame (world_coord.hpp, gpu_frame.hpp) and the exact noise
// (exact_noise.hpp): builds WorldPos and evaluates the probes on the GPU (local RenderingDevice, the
// same GLSL snippets as the meshers) for test vectors around far world origins, 2^32-voxel boundaries
// and absolute points, and checks against the CPU mirrors:
//   - CPU sample_pos == GPU WorldPos and anchor, bit for bit
//   - every probe of the probe table (frame_spike.cpp: hash, octave decomposition, perlin3/2, a 2D
//     and a 3D fBm; archetype_probes.cpp: the archetype IR's shapes, the table probe and fuzzed programs,
//     generated GLSL vs the interpreter): CPU == GPU, bit for bit, at the fixed grid and 1,024
//     hash-random points per origin
//   - the same voxel reached from two chunks (neighbour, parent) gives bit-identical GPU results
//   - the same absolute point gives bit-identical GPU results at every world origin
//   - the noises stay continuous along lines across voxel and cell low-word wraps
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
