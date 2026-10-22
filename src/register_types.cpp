#include "register_types.h"

#include <gdextension_interface.h>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

#include "cascade_report.hpp"
#include "cascade_spike.hpp"
#include "climate_report.hpp"
#include "frame_spike.hpp"
#include "quad_pool_spike.hpp"
#include "voxel_terrain.hpp"

#ifdef OPTI_VOXEL_TESTS
#include "tests/unit_tests.hpp"
#endif

using namespace godot;

void initialize_gdextension_types(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
    GDREGISTER_CLASS(VoxelTerrain);
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