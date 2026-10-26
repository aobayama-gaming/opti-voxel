#pragma once

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/callable_method_pointer.hpp>

#include <initializer_list>

using namespace godot;

namespace RDUtils {

inline void _rt_free_rids(const Array &p_rids) {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    for (int i = 0; i < p_rids.size(); i++) {
        const RID rid = p_rids[i];
        if (rid.is_valid()) {
            rd->free_rid(rid);
        }
    }
}

// Frees main-RenderingDevice RIDs on the render thread, in order (free dependents such as uniform
// sets and pipelines before the shader they were created from). The RIDs are bound by value, so
// this is safe to call from a destructor/_exit_tree even though the owner dies before it runs.
inline void free_on_render_thread(std::initializer_list<RID> p_rids) {
    Array rids;
    for (const RID &rid : p_rids) {
        rids.push_back(rid);
    }
    RenderingServer::get_singleton()->call_on_render_thread(callable_mp_static(&_rt_free_rids).bind(rids));
}

inline void _rt_delete_object(uint64_t p_object) {
    memdelete(reinterpret_cast<Object *>(uintptr_t(p_object)));
}

// Deletes an Object on the render thread, after everything already queued there: none of its
// render-thread methods can be running when it goes away (with a separate render thread, deleting it
// from the main thread races them). Not enough for an object with async readbacks in flight: their
// callbacks arrive later and are not skipped once it is freed (see MeshValidator::destroy).
inline void delete_on_render_thread(Object *p_object) {
    RenderingServer::get_singleton()->call_on_render_thread(
            callable_mp_static(&_rt_delete_object).bind(uint64_t(uintptr_t(p_object))));
}

// Debug: with the environment variable OPTI_VOXEL_SHADER_DUMP=<dir>, writes a composed shader source to
// <dir>/<name>_<hash>.glsl (compare the shaders of two builds byte for byte; one file per variant, e.g. the
// debug shapes' origin-relative literals).
inline void dump_shader(const String &p_name, const String &p_code) {
    const String dir = OS::get_singleton()->get_environment("OPTI_VOXEL_SHADER_DUMP");
    if (dir.is_empty()) {
        return;
    }
    Ref<FileAccess> f = FileAccess::open(dir.path_join(p_name + String("_") + String::num_uint64(p_code.hash(), 16) + String(".glsl")), FileAccess::WRITE);
    if (f.is_valid()) {
        f->store_string(p_code);
    }
}

} // namespace RDUtils
