#pragma once

// The engine's GPU: the compute operations the meshers, the cascade passes and the point paths record, behind one
// interface each adapter implements (adapter/godot: a RenderingDevice; adapter/dawn: a WebGPU device). The engine
// writes GLSL (Vulkan dialect, set 0, push constants of at most PUSH_MAX bytes); a backend compiles it.
//
// Ordering: operations run in the order they are called (an update between two dispatches is seen by the second
// only). A device is used by one thread at a time: the adapter's rules (Godot's main device on its render thread)
// stay the caller's.

#include "ov/bytes.hpp"
#include "ov/str.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace ov::gpu {

// Push constants: WebGPU's immediates guarantee 64 bytes (Dawn's limit on the project's GPUs).
inline constexpr uint32_t PUSH_MAX = 64;

// An object of the device (0: none). Ids are the backend's; the typed wrappers only keep the call sites honest.
struct Handle {
    uint64_t id = 0;
    bool is_valid() const { return id != 0; }
    bool operator==(const Handle &p_other) const { return id == p_other.id; }
    bool operator!=(const Handle &p_other) const { return id != p_other.id; }
};
struct Buffer : Handle {};
struct Pipeline : Handle {};
struct BindSet : Handle {};

enum class BufferKind : uint8_t {
    STORAGE, // std430 storage buffer (readable back, copyable)
    UNIFORM, // std140 uniform buffer
};

// One binding of set 0.
struct Binding {
    uint32_t binding = 0;
    BufferKind kind = BufferKind::STORAGE;
    Buffer buffer;
};

// A GPU timestamp of the last finished submission (Device::timestamps).
struct Timestamp {
    str name;
    uint64_t gpu_ns = 0;
};

// Receives a readback's bytes (empty if it failed).
using Readback = std::function<void(const bytes &p_data)>;

class Device {
public:
    virtual ~Device() = default;

    // A buffer of p_size bytes, p_data (or zeros) as its contents.
    virtual Buffer create_buffer(BufferKind p_kind, uint64_t p_size, const void *p_data = nullptr) = 0;
    virtual void update_buffer(Buffer p_buffer, uint64_t p_offset, uint64_t p_size, const void *p_data) = 0;
    virtual void clear_buffer(Buffer p_buffer, uint64_t p_offset, uint64_t p_size) = 0;
    virtual void copy_buffer(Buffer p_from, Buffer p_to, uint64_t p_from_offset, uint64_t p_to_offset, uint64_t p_size) = 0;

    // A compute pipeline from GLSL (p_name for logs and shader dumps). Invalid, and r_error set, if it fails to compile.
    virtual Pipeline create_compute_pipeline(const str &p_name, const str &p_glsl, str &r_error) = 0;
    // Set 0 of p_pipeline: every binding its shader declares, used or not.
    virtual BindSet create_bind_set(Pipeline p_pipeline, const std::vector<Binding> &p_bindings) = 0;
    virtual void dispatch(Pipeline p_pipeline, BindSet p_set, const void *p_push, uint32_t p_push_size, uint32_t p_x, uint32_t p_y,
            uint32_t p_z) = 0;

    // A named GPU timestamp after the work recorded so far.
    virtual void timestamp(const str &p_name) = 0;
    // The timestamps of the last submission sync() finished (none where the backend has no timestamp queries).
    virtual std::vector<Timestamp> timestamps() = 0;

    // p_size bytes of p_buffer once the work recorded so far is done: delivered on the device's thread, by sync() on a
    // local device, frames later on an engine's main one. The device keeps the request alive; its owner must outlive it.
    virtual void read_buffer_async(Buffer p_buffer, uint64_t p_offset, uint64_t p_size, Readback p_done) = 0;
    // The same, waiting for it.
    virtual bytes read_buffer(Buffer p_buffer, uint64_t p_offset, uint64_t p_size) = 0;

    // A local device: submits the work recorded so far and waits for it (the async readbacks it carried are delivered).
    virtual void sync() = 0;

    virtual void free(Handle p_handle) = 0;
};

// Local devices (the discoverer's, the field service's), owned by the thread that creates them. The adapter installs
// the factory (register_types, the Dawn host); without one, or without a GPU, create_local_device returns null.
using LocalDeviceFactory = std::unique_ptr<Device> (*)();

inline LocalDeviceFactory &local_device_factory() {
    static LocalDeviceFactory s_factory = nullptr;
    return s_factory;
}

inline void set_local_device_factory(LocalDeviceFactory p_factory) {
    local_device_factory() = p_factory;
}

inline std::unique_ptr<Device> create_local_device() {
    return local_device_factory() != nullptr ? local_device_factory()() : nullptr;
}

} // namespace ov::gpu
