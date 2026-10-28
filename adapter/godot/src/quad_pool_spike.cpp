#include "quad_pool_spike.hpp"

#include "godot_conv.hpp"
#include "voxel_sky.hpp"

#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/world3d.hpp>
#include <godot_cpp/core/class_db.hpp>

void QuadPoolSpike::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_side", "side"), &QuadPoolSpike::set_side);
    ClassDB::bind_method(D_METHOD("get_side"), &QuadPoolSpike::get_side);
    ClassDB::bind_method(D_METHOD("set_extra_pages", "pages"), &QuadPoolSpike::set_extra_pages);
    ClassDB::bind_method(D_METHOD("get_extra_pages"), &QuadPoolSpike::get_extra_pages);
    ClassDB::bind_method(D_METHOD("set_page_capacity", "capacity"), &QuadPoolSpike::set_page_capacity);
    ClassDB::bind_method(D_METHOD("get_page_capacity"), &QuadPoolSpike::get_page_capacity);
    ClassDB::bind_method(D_METHOD("set_debug_view", "view"), &QuadPoolSpike::set_debug_view);
    ClassDB::bind_method(D_METHOD("get_report"), &QuadPoolSpike::get_report);
    ClassDB::bind_method(D_METHOD("get_stage"), &QuadPoolSpike::get_stage);
    ClassDB::bind_method(D_METHOD("run_kill_test"), &QuadPoolSpike::run_kill_test);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "side"), "set_side", "get_side");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "extra_pages"), "set_extra_pages", "get_extra_pages");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "page_capacity"), "set_page_capacity", "get_page_capacity");
}

void QuadPoolSpike::_ready() {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    ERR_FAIL_NULL_MSG(rd, "QuadPoolSpike: no RenderingDevice (Compatibility renderer?)");
    device = new RDDevice(rd);
    VoxelSky::add_atmosphere_globals(); // the page shader declares them
    host = new GodotPageHost(get_world_3d()->get_scenario(), device);
    host->set_material_param("voxel_size", ChunkParam().voxel_size); // the page shader's other defaults: chunk 32, skirt 2
    check = new QuadPoolCheck();
    if (!check->start(device, host, settings)) {
        delete check;
        check = nullptr;
    }
}

void QuadPoolSpike::_process(double) {
    if (check) {
        check->step();
    }
}

Dictionary QuadPoolSpike::get_report() {
    return check ? godot_dict(check->report()) : Dictionary();
}

void QuadPoolSpike::run_kill_test() {
    if (check) {
        check->run_kill_test();
    }
}

void QuadPoolSpike::set_debug_view(int p_view) {
    if (host) {
        host->set_debug_view(p_view);
    }
}

void QuadPoolSpike::_exit_tree() {
    if (check) {
        check->destroy(); // the pool and the GPU objects on the render thread, then itself
        check = nullptr;
    }
    if (host) {
        host->destroy();
        host = nullptr;
    }
    if (device) {
        device->post([device = device]() { delete device; });
        device = nullptr;
    }
}
