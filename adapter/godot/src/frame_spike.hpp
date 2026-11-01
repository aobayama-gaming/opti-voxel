#pragma once

#include "world_file.hpp"

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

    void set_world_path(const String &p_path) { world_path = p_path; }
    String get_world_path() const { return world_path; }

private:
    String world_path = WorldFile::DEFAULT_PATH;
};
