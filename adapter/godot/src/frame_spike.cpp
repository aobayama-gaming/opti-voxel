#include "frame_spike.hpp"

#include "frame_check.hpp"
#include "godot_conv.hpp"
#include "rd_device.hpp"

#include <godot_cpp/core/class_db.hpp>

void FrameSpike::_bind_methods() {
    ClassDB::bind_method(D_METHOD("run"), &FrameSpike::run);
    ClassDB::bind_method(D_METHOD("set_voxel_size", "size"), &FrameSpike::set_voxel_size);
    ClassDB::bind_method(D_METHOD("get_voxel_size"), &FrameSpike::get_voxel_size);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "voxel_size"), "set_voxel_size", "get_voxel_size");
}

Dictionary FrameSpike::run() {
    const std::unique_ptr<ov::gpu::Device> device = RDDevice::create_local();
    if (!device) {
        Dictionary report;
        report["ok"] = false;
        report["error"] = "no local RenderingDevice (headless or Compatibility renderer)";
        return report;
    }
    return godot_dict(FrameCheck::run(*device, voxel_size));
}
