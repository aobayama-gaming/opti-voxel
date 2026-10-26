#include "cascade_report_node.hpp"

#include "cascade_report.hpp"
#include "godot_conv.hpp"

#include <godot_cpp/core/class_db.hpp>

void CascadeReportNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("run", "out_dir", "scale"), &CascadeReportNode::run);
}

Dictionary CascadeReportNode::run(const String &p_out_dir, int p_scale) {
    return godot_dict(CascadeReport::run(ov_str(p_out_dir), p_scale));
}
