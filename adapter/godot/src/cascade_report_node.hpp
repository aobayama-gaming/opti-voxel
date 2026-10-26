#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/dictionary.hpp>

using namespace godot;

// CascadeReport (src/cascade_report.hpp) as a node: res://debug/cascade_report.tscn, tools/cascade_report.sh.
class CascadeReportNode : public Node {
    GDCLASS(CascadeReportNode, Node)

protected:
    static void _bind_methods();

public:
    Dictionary run(const String &p_out_dir, int p_scale);
};
