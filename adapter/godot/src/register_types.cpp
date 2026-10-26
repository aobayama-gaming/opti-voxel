#include "register_types.h"

#include <gdextension_interface.h>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include "cascade_report.hpp"
#include "cascade_spike.hpp"
#include "climate_report.hpp"
#include "frame_spike.hpp"
#include "quad_pool_spike.hpp"
#include "voxel_sky.hpp"
#include "voxel_terrain.hpp"

#include "godot_conv.hpp"
#include "godot_executor.hpp"
#include "ov/files.hpp"
#include "ov/on_error.hpp"
#include "rd_device.hpp"

#include "debug_log.hpp"

#ifdef OPTI_VOXEL_TESTS
#include "tests/unit_tests.hpp"
#endif

using namespace godot;

namespace {
GodotExecutor g_executor;
} // namespace

void initialize_gdextension_types(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
    // The core's engine-free error sink: an ERROR line in the Godot console at the core's file and line.
    ov::set_error_sink([](const char *p_file, int p_line, const char *p_message) {
        _err_print_error("core", p_file, p_line, p_message);
    });
    // Core log lines in the Godot console; core timestamps on the engine's ticks.
    ov::set_log_sink([](int p_level, const char *p_file, int p_line, const char *p_message) {
        if (p_level < 0) {
            _err_print_error("core", p_file, p_line, p_message, false, true);
        } else {
            UtilityFunctions::print(String::utf8(p_message));
        }
    });
    ov::set_clock_source([]() -> uint64_t { return Time::get_singleton()->get_ticks_usec(); });
    // The core's files through FileAccess: res:// and user:// paths.
    ov::set_file_reader([](const ov::str &p_path, ov::str &r_text) {
        const Ref<FileAccess> f = FileAccess::open(godot_str(p_path), FileAccess::READ);
        if (f.is_null()) {
            return false;
        }
        r_text = ov_str(f->get_as_text());
        return true;
    });
    ov::set_file_writer([](const ov::str &p_path, const ov::str &p_text) {
        const Ref<FileAccess> f = FileAccess::open(godot_str(p_path), FileAccess::WRITE);
        if (f.is_null()) {
            return false;
        }
        f->store_string(godot_str(p_text));
        return true;
    });
    // The core's tasks, loops and workers on Godot's pool and threads.
    ov::set_executor(&g_executor);
    ov::gpu::set_local_device_factory(&RDDevice::create_local);
    GDREGISTER_CLASS(VoxelTerrain);
    GDREGISTER_CLASS(VoxelSky);
    GDREGISTER_CLASS(QuadPoolSpike);
    GDREGISTER_CLASS(FrameSpike);
    GDREGISTER_CLASS(CascadeReport);
    GDREGISTER_CLASS(CascadeSpike);
    GDREGISTER_CLASS(ClimateReport);
#ifdef OPTI_VOXEL_TESTS
    GDREGISTER_CLASS(UnitTests);
#endif
}

void uninitialize_gdextension_types(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
    ov::set_executor(nullptr);
    ov::gpu::set_local_device_factory(nullptr);
}

extern "C" {
// Initialization
GDExtensionBool GDE_EXPORT opti_voxel_init(GDExtensionInterfaceGetProcAddress p_get_proc_address, GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization) {
    GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);
    init_obj.register_initializer(initialize_gdextension_types);
    init_obj.register_terminator(uninitialize_gdextension_types);
    init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

    return init_obj.init();
}
}