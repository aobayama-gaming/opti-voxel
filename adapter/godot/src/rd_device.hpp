#pragma once

#include "gpu_device.hpp"

#include <godot_cpp/classes/rendering_device.hpp>

#include <map>

// ov::gpu::Device on a Godot RenderingDevice: the main one (used on the render thread, its readbacks delivered
// frames later) or a local one it owns (create_local: the thread that creates it uses it and frees it).
class RDDevice : public ov::gpu::Device {
public:
    // p_rd is not owned.
    explicit RDDevice(godot::RenderingDevice *p_rd) : rd(p_rd) {}
    ~RDDevice() override;

    // A local RenderingDevice of its own (null: headless or the Compatibility renderer). The local device factory.
    static std::unique_ptr<ov::gpu::Device> create_local();

    godot::RenderingDevice *get_rd() const { return rd; }
    // The RID behind a buffer (the adapter's own passes bind engine buffers next to theirs).
    godot::RID get_rid(ov::gpu::Buffer p_buffer) const;

    ov::gpu::Buffer create_buffer(ov::gpu::BufferKind p_kind, uint64_t p_size, const void *p_data) override;
    void update_buffer(ov::gpu::Buffer p_buffer, uint64_t p_offset, uint64_t p_size, const void *p_data) override;
    void clear_buffer(ov::gpu::Buffer p_buffer, uint64_t p_offset, uint64_t p_size) override;
    void copy_buffer(ov::gpu::Buffer p_from, ov::gpu::Buffer p_to, uint64_t p_from_offset, uint64_t p_to_offset, uint64_t p_size) override;
    ov::gpu::Pipeline create_compute_pipeline(const ov::str &p_name, const ov::str &p_glsl, ov::str &r_error) override;
    ov::gpu::BindSet create_bind_set(ov::gpu::Pipeline p_pipeline, const std::vector<ov::gpu::Binding> &p_bindings) override;
    void dispatch(ov::gpu::Pipeline p_pipeline, ov::gpu::BindSet p_set, const void *p_push, uint32_t p_push_size, uint32_t p_x, uint32_t p_y,
            uint32_t p_z) override;
    void timestamp(const ov::str &p_name) override;
    std::vector<ov::gpu::Timestamp> timestamps() override;
    void read_buffer_async(ov::gpu::Buffer p_buffer, uint64_t p_offset, uint64_t p_size, ov::gpu::Readback p_done) override;
    ov::bytes read_buffer(ov::gpu::Buffer p_buffer, uint64_t p_offset, uint64_t p_size) override;
    void sync() override;
    void free(ov::gpu::Handle p_handle) override;

private:
    // A handle's RIDs: a buffer or bind set in `rid`, a pipeline's shader in `shader`.
    struct Entry {
        godot::RID rid;
        godot::RID shader;
    };

    godot::RenderingDevice *rd = nullptr;
    bool owns_rd = false;
    uint64_t next_id = 1;
    std::map<uint64_t, Entry> entries; // by id: freed newest first, dependents before what they use

    ov::gpu::Handle _add(const Entry &p_entry);
    const Entry *_get(ov::gpu::Handle p_handle) const;
};
