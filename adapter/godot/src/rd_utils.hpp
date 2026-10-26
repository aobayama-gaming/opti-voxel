#pragma once

#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/variant/callable_method_pointer.hpp>

using namespace godot;

namespace RDUtils {

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

} // namespace RDUtils
