#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/dictionary.hpp>

using namespace godot;

// The GPU cascade check (CascadeCheck, cascade_check.hpp) on a local RenderingDevice. run() is synchronous and needs a
// GPU (not --headless). Run: tools/cascade_test.sh
class CascadeSpike : public Node {
    GDCLASS(CascadeSpike, Node)

protected:
    static void _bind_methods();

public:
    // p_options: CascadeCheck::run's.
    Dictionary run(const Dictionary &p_options);
};
