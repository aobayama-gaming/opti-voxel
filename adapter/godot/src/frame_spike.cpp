#include "frame_spike.hpp"

#include "frame_check.hpp"
#include "godot_conv.hpp"
#include "rd_device.hpp"

#include <godot_cpp/core/class_db.hpp>

void FrameSpike::_bind_methods() {
    ClassDB::bind_method(D_METHOD("run"), &FrameSpike::run);
    ClassDB::bind_method(D_METHOD("set_world_path", "path"), &FrameSpike::set_world_path);
    ClassDB::bind_method(D_METHOD("get_world_path"), &FrameSpike::get_world_path);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "world_path"), "set_world_path", "get_world_path");
}

Dictionary FrameSpike::run() {
    const std::unique_ptr<ov::gpu::Device> device = RDDevice::create_local();
    if (!device) {
        Dictionary report;
        report["ok"] = false;
        report["error"] = "no local RenderingDevice (headless or Compatibility renderer)";
        return report;
    }
    return godot_dict(FrameCheck::run(*device, ov_str(world_path)));
}
