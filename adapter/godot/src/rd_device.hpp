#pragma once

#include "gpu_device.hpp"

#include <godot_cpp/classes/rd_texture_format.hpp>
#include <godot_cpp/classes/rendering_device.hpp>

#include <array>
#include <map>
#include <unordered_map>

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
    // The RID behind a buffer or texture (the adapter's own passes and the page host use the engine's objects).
    godot::RID get_rid(ov::gpu::Handle p_handle) const;
    // A buffer the device doesn't own (a MultiMesh's indirect command buffer): free() only forgets it.
    ov::gpu::Buffer import_buffer(const godot::RID &p_rid);

    ov::gpu::Buffer create_buffer(ov::gpu::BufferKind p_kind, uint64_t p_size, const void *p_data) override;
    void update_buffer(ov::gpu::Buffer p_buffer, uint64_t p_offset, uint64_t p_size, const void *p_data) override;
    void clear_buffer(ov::gpu::Buffer p_buffer, uint64_t p_offset, uint64_t p_size) override;
    void copy_buffer(ov::gpu::Buffer p_from, ov::gpu::Buffer p_to, uint64_t p_from_offset, uint64_t p_to_offset, uint64_t p_size) override;
    ov::gpu::Texture create_texture(ov::gpu::TextureFormat p_format, uint32_t p_width, uint32_t p_height, uint32_t p_usage, const void *p_data) override;
    ov::gpu::Texture create_texture_3d(ov::gpu::TextureFormat p_format, uint32_t p_width, uint32_t p_height, uint32_t p_depth, uint32_t p_usage) override;
    void update_texture(ov::gpu::Texture p_texture, const void *p_data, uint64_t p_size) override;
    void clear_texture(ov::gpu::Texture p_texture) override;
    void copy_texture(ov::gpu::Texture p_from, ov::gpu::Texture p_to, uint32_t p_from_x, uint32_t p_from_y, uint32_t p_to_x, uint32_t p_to_y,
            uint32_t p_width, uint32_t p_height) override;
    ov::gpu::Pipeline create_compute_pipeline(const ov::str &p_name, const ov::str &p_glsl, ov::str &r_error) override;
    ov::gpu::BindSet create_bind_set(ov::gpu::Pipeline p_pipeline, const std::vector<ov::gpu::Binding> &p_bindings) override;
    void dispatch(ov::gpu::Pipeline p_pipeline, ov::gpu::BindSet p_set, const void *p_push, uint32_t p_push_size, uint32_t p_x, uint32_t p_y,
            uint32_t p_z) override;
    void timestamp(const ov::str &p_name) override;
    std::vector<ov::gpu::Timestamp> timestamps() override;
    uint64_t timestamps_frame() override;
    void read_buffer_async(ov::gpu::Buffer p_buffer, uint64_t p_offset, uint64_t p_size, ov::gpu::Readback p_done) override;
    ov::bytes read_buffer(ov::gpu::Buffer p_buffer, uint64_t p_offset, uint64_t p_size) override;
    void read_texture_async(ov::gpu::Texture p_texture, ov::gpu::Readback p_done) override;
    void sync() override;
    void post(std::function<void()> p_fn) override;
    void free(ov::gpu::Handle p_handle) override;

    // Device::park: from park on, a pipeline freed is kept by its text, a buffer or a texture made without data by its
    // shape, and one of the same text or shape created takes it instead of being built again (the driver's compile; the
    // mesher's hundreds of MB of scratch: a start's allocations). A buffer taken is cleared or written as a new one would
    // be, a texture keeps its texels (a new one's are undefined). drop_parked frees those not taken.
    void park() override { parking = true; }
    void drop_parked() override;

private:
    // What a parked buffer or texture is taken for: its kind (1: buffer, 2: 2D, 3: 3D texture) with the buffer kind or the
    // format and usage, the sizes.
    using Shape = std::array<uint64_t, 3>;
    // A handle's RIDs: a buffer or bind set in `rid`, a pipeline's shader in `shader`.
    struct Entry {
        godot::RID rid;
        godot::RID shader;
        bool is_set = false; // a uniform set
        bool owned = true; // false: imported (import_buffer), never freed here
        Shape shape{}; // a buffer's or a texture's without data (park's key), zero for the rest
    };

    godot::RenderingDevice *rd = nullptr;
    bool owns_rd = false;
    uint64_t next_id = 1;
    std::map<uint64_t, Entry> entries; // by id: freed newest first, dependents before what they use
    std::unordered_map<uint64_t, ov::str> pipeline_text; // each live pipeline's GLSL, by id (park_pipelines' key)
    std::unordered_multimap<ov::str, Entry> parked; // the pipelines freed while parking, by text
    std::multimap<Shape, Entry> parked_shapes; // the buffers and textures freed while parking, by shape
    bool parking = false;
    godot::RID linear_sampler; // Binding::linear_sampler's, made on first use

    // An entry of p_shape: a parked one taken (its RID valid), or a new one without RID.
    Entry _take_parked(const Shape &p_shape);

    static godot::Ref<godot::RDTextureFormat> texture_format(ov::gpu::TextureFormat p_format, uint32_t p_width, uint32_t p_height, uint32_t p_usage);
    godot::RID _linear_sampler();

    ov::gpu::Handle _add(const Entry &p_entry);
    const Entry *_get(ov::gpu::Handle p_handle) const;
    void _free_entry(const Entry &p_entry);
};
