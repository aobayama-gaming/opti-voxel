#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/dictionary.hpp>

using namespace godot;

// Test harness of the GPU cascade (cascade_glsl.hpp) against the CPU oracle (Cascade::run), on a local
// RenderingDevice with the meshers' level passes:
//   - G1: every channel of every level and the final sdf_sample of chunks of every LOD (>= 64 per LOD,
//     near 0, +-2^60 and straddling 2^32-voxel boundaries, in mixed-LOD batches): GPU == CPU bit for bit.
//     The passes run one level at a time and each level's scratch is read back before the next one.
//   - G3: on the GPU outputs, at the points a LOD-L chunk shares with its parent (LOD L+1):
//     V_L = flush(V_{L+1} * 2^-order + O_L), exact.
//   - G6 / G7 timing: single-LOD batches of 128 chunks at LOD 0, 4, ... 24, every level of the cascade
//     passes, both channel sets, interleaved runs (GPU timestamps when the device returns them, wall clock
//     of submit + sync otherwise).
//   - F1 / F2: the field points of the site walk (cascade_spike_fields.hpp): ground_unit / ground_field bit for bit,
//     and their GPU ms per batch size.
// run() is synchronous and needs a GPU (not --headless). Run: tools/cascade_test.sh
class CascadeSpike : public Node {
    GDCLASS(CascadeSpike, Node)

protected:
    static void _bind_methods();

public:
    // p_options: lods (25), voxel (0.5), check (true), points (true), fields (true), timing (true), runs (10), sets ("8,4").
    Dictionary run(const Dictionary &p_options);
};
