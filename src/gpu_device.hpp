#pragma once

// The engine's GPU: the compute operations the meshers, the cascade passes and the point paths record, behind one
// interface each adapter implements (adapter/godot: a RenderingDevice; adapter/dawn: a WebGPU device). The engine
// writes GLSL (Vulkan dialect, set 0, push constants of at most PUSH_MAX bytes); a backend compiles it.
//
// Ordering: operations run in the order they are called (an update between two dispatches is seen by the second
// only). A device is used by one thread at a time, its own (post): Godot's main device lives on the render thread, a
// local device on the thread that created it.

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
struct Texture : Handle {};
struct Pipeline : Handle {};
struct BindSet : Handle {};

enum class BufferKind : uint8_t {
    STORAGE, // std430 storage buffer (readable back, copyable)
    UNIFORM, // std140 uniform buffer
};

// 2D textures: the quad pages, the batch's staging rows, the palettes (PageFormat's texels); the atmosphere's tables
// (half floats, rgba16f).
enum class TextureFormat : uint8_t {
    RGBA16_UINT,
    RGBA16_FLOAT,
};

inline uint32_t texel_bytes(TextureFormat p_format) {
    return p_format == TextureFormat::RGBA16_UINT || p_format == TextureFormat::RGBA16_FLOAT ? 8 : 0;
}

// What a texture is used for (bits): a backend may lay it out for exactly these.
enum TextureUsage : uint32_t {
    TEXTURE_SAMPLED = 1, // read by the host's shaders (the page shader)
    TEXTURE_STORAGE = 2, // a storage image of the engine's shaders
    TEXTURE_COPY_SRC = 4,
    TEXTURE_COPY_DST = 8, // copy_texture into it, clear_texture
    TEXTURE_UPDATE = 16, // update_texture
};

// One binding of set 0: a buffer, or a storage image (image valid).
struct Binding {
    uint32_t binding = 0;
    BufferKind kind = BufferKind::STORAGE;
    Buffer buffer;
    Texture image;

    Binding() = default;
    Binding(uint32_t p_binding, BufferKind p_kind, Buffer p_buffer) :
            binding(p_binding), kind(p_kind), buffer(p_buffer) {}
    static Binding storage_image(uint32_t p_binding, Texture p_image) {
        Binding b;
        b.binding = p_binding;
        b.image = p_image;
        return b;
    }
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

    // A p_width x p_height texture of p_usage (TextureUsage bits), p_data (whole rows, tightly packed) or zeros.
    virtual Texture create_texture(TextureFormat p_format, uint32_t p_width, uint32_t p_height, uint32_t p_usage, const void *p_data = nullptr) = 0;
    // The whole texture's texels (TEXTURE_UPDATE).
    virtual void update_texture(Texture p_texture, const void *p_data, uint64_t p_size) = 0;
    // Every texel to zero (TEXTURE_COPY_DST).
    virtual void clear_texture(Texture p_texture) = 0;
    // A p_width x p_height block of texels from (p_from_x, p_from_y) of p_from to (p_to_x, p_to_y) of p_to.
    virtual void copy_texture(Texture p_from, Texture p_to, uint32_t p_from_x, uint32_t p_from_y, uint32_t p_to_x, uint32_t p_to_y, uint32_t p_width,
            uint32_t p_height) = 0;

    // A compute pipeline from GLSL (p_name for logs and shader dumps). Invalid, and r_error set, if it fails to compile.
    virtual Pipeline create_compute_pipeline(const str &p_name, const str &p_glsl, str &r_error) = 0;
    // Set 0 of p_pipeline: every binding its shader declares, used or not.
    virtual BindSet create_bind_set(Pipeline p_pipeline, const std::vector<Binding> &p_bindings) = 0;
    virtual void dispatch(Pipeline p_pipeline, BindSet p_set, const void *p_push, uint32_t p_push_size, uint32_t p_x, uint32_t p_y,
            uint32_t p_z) = 0;

    // A named GPU timestamp after the work recorded so far.
    virtual void timestamp(const str &p_name) = 0;
    // The timestamps of the last frame the GPU finished (a local device: the last sync()), none where the backend has no
    // timestamp queries, and that frame's number: a reader polling every frame reads each set once.
    virtual std::vector<Timestamp> timestamps() = 0;
    virtual uint64_t timestamps_frame() = 0;

    // p_size bytes of p_buffer (0: to its end) once the work recorded so far is done: delivered on the device's thread, by sync() on a
    // local device, frames later on an engine's main one. The device keeps the request alive; its owner must outlive it.
    virtual void read_buffer_async(Buffer p_buffer, uint64_t p_offset, uint64_t p_size, Readback p_done) = 0;
    // The same, waiting for it.
    virtual bytes read_buffer(Buffer p_buffer, uint64_t p_offset, uint64_t p_size) = 0;
    // A texture's texels (rows tightly packed), like read_buffer_async (TEXTURE_COPY_SRC).
    virtual void read_texture_async(Texture p_texture, Readback p_done) = 0;

    // A local device: submits the work recorded so far and waits for it (the async readbacks it carried are delivered).
    virtual void sync() = 0;

    // Runs p_fn on the device's thread after the work posted before it: Godot's main device, on the render thread (the
    // engine's objects that own main-device resources free them, then themselves, this way); a local device, at once.
    virtual void post(std::function<void()> p_fn) = 0;

    // Frees an object (a pipeline or a buffer: the bind sets using it may go with it, a later free of one is a no-op).
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
