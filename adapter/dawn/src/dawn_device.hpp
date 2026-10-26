#pragma once

// ov::gpu::Device on Dawn. Commands go into one encoder, submitted when an update must land between two dispatches,
// a readback is asked, or sync() waits. Readbacks are delivered by sync() or poll(). Timestamps: none yet. The thread
// that uses it is its thread: post() runs at once.

#include "dawn_context.hpp"
#include "glsl_compiler.hpp"

#include "gpu_device.hpp"

#include <map>
#include <memory>

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
    gpu::Texture create_texture(gpu::TextureFormat p_format, uint32_t p_width, uint32_t p_height, uint32_t p_usage, const void *p_data) override;
    void update_texture(gpu::Texture p_texture, const void *p_data, uint64_t p_size) override;
    void clear_texture(gpu::Texture p_texture) override;
    void copy_texture(gpu::Texture p_from, gpu::Texture p_to, uint32_t p_from_x, uint32_t p_from_y, uint32_t p_to_x, uint32_t p_to_y, uint32_t p_width,
            uint32_t p_height) override;
    gpu::Pipeline create_compute_pipeline(const str &p_name, const str &p_glsl, str &r_error) override;
    gpu::BindSet create_bind_set(gpu::Pipeline p_pipeline, const std::vector<gpu::Binding> &p_bindings) override;
    void dispatch(gpu::Pipeline p_pipeline, gpu::BindSet p_set, const void *p_push, uint32_t p_push_size, uint32_t p_x, uint32_t p_y,
            uint32_t p_z) override;
    void timestamp(const str &p_name) override;
    std::vector<gpu::Timestamp> timestamps() override;
    uint64_t timestamps_frame() override { return 0; }
    void read_buffer_async(gpu::Buffer p_buffer, uint64_t p_offset, uint64_t p_size, gpu::Readback p_done) override;
    bytes read_buffer(gpu::Buffer p_buffer, uint64_t p_offset, uint64_t p_size) override;
    void read_texture_async(gpu::Texture p_texture, gpu::Readback p_done) override;
    void sync() override;
    void post(std::function<void()> p_fn) override { p_fn(); }
    void free(gpu::Handle p_handle) override;

    // Submits the recorded work and delivers the readbacks that finished, without waiting (a host's frame loop).
    void poll();

    // The host's side: a buffer an indirect draw reads (the core writes it like any storage buffer), the Dawn objects
    // behind handles (null if unknown).
    gpu::Buffer create_indirect_buffer(uint64_t p_size, const void *p_data);
    wgpu::Buffer get_buffer(gpu::Buffer p_buffer);
    wgpu::Texture get_texture(gpu::Texture p_texture);
    DawnContext &get_context() { return context; }

    // A local device (gpu::set_local_device_factory): its own instance and device, null without an adapter.
    static std::unique_ptr<gpu::Device> create_local();

private:
    struct Entry {
        wgpu::Buffer buffer;
        wgpu::Texture texture;
        uint32_t width = 0; // a texture's
        uint32_t height = 0;
        wgpu::ComputePipeline pipeline;
        wgpu::BindGroupLayout layout; // a pipeline's set 0
        std::vector<ShaderBinding> bindings; // a pipeline's set-0 buffers
        wgpu::BindGroup group;
    };
    // A readback in flight: its staging buffer, mapped when the GPU is done.
    struct Pending {
        wgpu::Buffer staging;
        uint64_t size = 0;
        // A texture's: rows of row_bytes, padded to row_stride in the staging buffer (repacked tight on delivery).
        uint64_t row_bytes = 0;
        uint64_t row_stride = 0;
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
    gpu::Buffer _create_buffer(wgpu::BufferUsage p_usage, uint64_t p_size, const void *p_data);
    void _write_texture(const Entry &p_entry, const void *p_data);
    // A readback of p_staging (its copy recorded and submitted): delivered in request order once mapped.
    void _read(wgpu::Buffer p_staging, uint64_t p_size, uint64_t p_row_bytes, uint64_t p_row_stride, gpu::Readback p_done);
    Entry *_get(gpu::Handle p_handle);
    wgpu::CommandEncoder &_encoder();
    void _submit();
    void _deliver();
};

} // namespace ov::dawn
