#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/dictionary.hpp>

using namespace godot;

// The integer frame and exact noise check (FrameCheck, frame_check.hpp) on a local RenderingDevice. run() is
// synchronous and needs a GPU (not --headless). Run: tools/frame_test.sh
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
