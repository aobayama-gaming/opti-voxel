#pragma once

// ov::gpu::Device on Dawn. Commands go into one encoder, submitted when an update must land between two dispatches,
// a readback is asked, or sync() waits. Readbacks are delivered by sync() or poll(). Timestamps: none yet.

#include "dawn_context.hpp"
#include "glsl_compiler.hpp"

#include "gpu_device.hpp"

#include <map>

namespace ov::dawn {

class DawnDevice : public gpu::Device {
public:
    // p_context: initialized, outlives the device.
    explicit DawnDevice(DawnContext &p_context);
    ~DawnDevice() override;

    gpu::Buffer create_buffer(gpu::BufferKind p_kind, uint64_t p_size, const void *p_data) override;
    void update_buffer(gpu::Buffer p_buffer, uint64_t p_offset, uint64_t p_size, const void *p_data) override;
    void clear_buffer(gpu::Buffer p_buffer, uint64_t p_offset, uint64_t p_size) override;
    void copy_buffer(gpu::Buffer p_from, gpu::Buffer p_to, uint64_t p_from_offset, uint64_t p_to_offset, uint64_t p_size) override;
    gpu::Pipeline create_compute_pipeline(const str &p_name, const str &p_glsl, str &r_error) override;
    gpu::BindSet create_bind_set(gpu::Pipeline p_pipeline, const std::vector<gpu::Binding> &p_bindings) override;
    void dispatch(gpu::Pipeline p_pipeline, gpu::BindSet p_set, const void *p_push, uint32_t p_push_size, uint32_t p_x, uint32_t p_y,
            uint32_t p_z) override;
    void timestamp(const str &p_name) override;
    std::vector<gpu::Timestamp> timestamps() override;
    void read_buffer_async(gpu::Buffer p_buffer, uint64_t p_offset, uint64_t p_size, gpu::Readback p_done) override;
    bytes read_buffer(gpu::Buffer p_buffer, uint64_t p_offset, uint64_t p_size) override;
    void sync() override;
    void free(gpu::Handle p_handle) override;

    // Submits the recorded work and delivers the readbacks that finished, without waiting (a host's frame loop).
    void poll();

private:
    struct Entry {
        wgpu::Buffer buffer;
        wgpu::ComputePipeline pipeline;
        wgpu::BindGroupLayout layout; // a pipeline's set 0
        std::vector<ShaderBinding> bindings; // a pipeline's set-0 buffers
        wgpu::BindGroup group;
    };
    // A readback in flight: its staging buffer, mapped when the GPU is done.
    struct Pending {
        wgpu::Buffer staging;
        uint64_t size = 0;
        gpu::Readback done;
        bool mapped = false;
        bool failed = false;
    };

    DawnContext &context;
    wgpu::CommandEncoder encoder;
    bool recorded = false; // the encoder holds commands
    uint64_t next_id = 1;
    std::map<uint64_t, Entry> entries;
    std::vector<std::unique_ptr<Pending>> pending;
    uint32_t push_size_limit = 0;

    gpu::Handle _add(Entry p_entry);
    Entry *_get(gpu::Handle p_handle);
    wgpu::CommandEncoder &_encoder();
    void _submit();
    void _deliver();
};

} // namespace ov::dawn
