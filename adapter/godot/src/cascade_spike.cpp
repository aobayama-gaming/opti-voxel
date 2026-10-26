#include "cascade_spike.hpp"

#include "cascade_check.hpp"
#include "godot_conv.hpp"
#include "rd_device.hpp"

#include <godot_cpp/core/class_db.hpp>

void CascadeSpike::_bind_methods() {
    ClassDB::bind_method(D_METHOD("run", "options"), &CascadeSpike::run);
}

Dictionary CascadeSpike::run(const Dictionary &p_options) {
    const std::unique_ptr<ov::gpu::Device> device = RDDevice::create_local();
    if (!device) {
        Dictionary report;
        report["ok"] = false;
        report["error"] = "no local RenderingDevice (headless or Compatibility renderer)";
        return report;
    }
    return godot_dict(CascadeCheck::run(*device, ov_value(p_options)));
}
