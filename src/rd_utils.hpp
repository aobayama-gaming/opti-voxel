#pragma once

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

} // namespace RDUtils
