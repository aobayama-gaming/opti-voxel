#include "rd_device.hpp"

#include "godot_callable.hpp"
#include "godot_conv.hpp"

#include <godot_cpp/classes/rd_shader_source.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_texture_format.hpp>
#include <godot_cpp/classes/rd_texture_view.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/variant/callable_custom.hpp>

#include <cstring>

using namespace godot;

namespace {

PackedByteArray to_bytes(const void *p_data, uint64_t p_size) {
    PackedByteArray out;
    out.resize(int64_t(p_size));
    if (p_size > 0) {
        if (p_data != nullptr) {
            std::memcpy(out.ptrw(), p_data, size_t(p_size));
        } else {
            out.fill(0);
        }
    }
    return out;
}

ov::bytes from_bytes(const PackedByteArray &p_bytes) {
    return ov::bytes(p_bytes.ptr(), p_bytes.ptr() + p_bytes.size());
}

// buffer_get_data_async's callback: hands the bytes to the engine's Readback.
class DeliverReadback : public CallableCustom {
public:
    explicit DeliverReadback(ov::gpu::Readback p_done) : done(std::move(p_done)) {}
    uint32_t hash() const override { return uint32_t(uintptr_t(this)); }
    String get_as_text() const override { return "RDDevice readback"; }
    CompareEqualFunc get_compare_equal_func() const override { return &equal; }
    CompareLessFunc get_compare_less_func() const override { return &less; }
    bool is_valid() const override { return true; }
    ObjectID get_object() const override { return ObjectID(); }
    void call(const Variant **p_arguments, int p_argcount, Variant &, GDExtensionCallError &r_call_error) const override {
        done(p_argcount >= 1 ? from_bytes(*p_arguments[0]) : ov::bytes());
        r_call_error.error = GDEXTENSION_CALL_OK;
    }

private:
    ov::gpu::Readback done;
    static bool equal(const CallableCustom *p_a, const CallableCustom *p_b) { return p_a == p_b; }
    static bool less(const CallableCustom *p_a, const CallableCustom *p_b) { return p_a < p_b; }
};

RenderingDevice::DataFormat rd_format(ov::gpu::TextureFormat p_format) {
    switch (p_format) {
        case ov::gpu::TextureFormat::RGBA16_UINT:
        default:
            return RenderingDevice::DATA_FORMAT_R16G16B16A16_UINT;
    }
}

} // namespace

RDDevice::~RDDevice() {
    for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
        _free_entry(it->second);
    }
    if (owns_rd) {
        memdelete(rd);
    }
}

std::unique_ptr<ov::gpu::Device> RDDevice::create_local() {
    RenderingDevice *local = RenderingServer::get_singleton()->create_local_rendering_device();
    if (local == nullptr) {
        return nullptr;
    }
    std::unique_ptr<RDDevice> device = std::make_unique<RDDevice>(local);
    device->owns_rd = true;
    return device;
}

RID RDDevice::get_rid(ov::gpu::Handle p_handle) const {
    const Entry *entry = _get(p_handle);
    return entry != nullptr ? entry->rid : RID();
}

ov::gpu::Buffer RDDevice::import_buffer(const RID &p_rid) {
    Entry entry;
    entry.rid = p_rid;
    entry.owned = false;
    return ov::gpu::Buffer{ _add(entry) };
}

ov::gpu::Handle RDDevice::_add(const Entry &p_entry) {
    if (!p_entry.rid.is_valid()) {
        return ov::gpu::Handle();
    }
    const uint64_t id = next_id++;
    entries[id] = p_entry;
    return ov::gpu::Handle{ id };
}

void RDDevice::_free_entry(const Entry &p_entry) {
    // Godot frees a uniform set with the buffer or shader it uses: one may be gone already.
    if (!p_entry.owned || (p_entry.is_set && !rd->uniform_set_is_valid(p_entry.rid))) {
        return;
    }
    // A pipeline before its shader.
    for (const RID &rid : { p_entry.rid, p_entry.shader }) {
        if (rid.is_valid()) {
            rd->free_rid(rid);
        }
    }
}

const RDDevice::Entry *RDDevice::_get(ov::gpu::Handle p_handle) const {
    const auto it = entries.find(p_handle.id);
    return it != entries.end() ? &it->second : nullptr;
}

ov::gpu::Buffer RDDevice::create_buffer(ov::gpu::BufferKind p_kind, uint64_t p_size, const void *p_data) {
    const PackedByteArray data = to_bytes(p_data, p_size);
    const RID rid = p_kind == ov::gpu::BufferKind::UNIFORM ? rd->uniform_buffer_create(uint32_t(p_size), data)
                                                           : rd->storage_buffer_create(uint32_t(p_size), data);
    return ov::gpu::Buffer{ _add({ rid, RID() }) };
}

void RDDevice::update_buffer(ov::gpu::Buffer p_buffer, uint64_t p_offset, uint64_t p_size, const void *p_data) {
    rd->buffer_update(get_rid(p_buffer), uint32_t(p_offset), uint32_t(p_size), to_bytes(p_data, p_size));
}

void RDDevice::clear_buffer(ov::gpu::Buffer p_buffer, uint64_t p_offset, uint64_t p_size) {
    rd->buffer_clear(get_rid(p_buffer), uint32_t(p_offset), uint32_t(p_size));
}

void RDDevice::copy_buffer(ov::gpu::Buffer p_from, ov::gpu::Buffer p_to, uint64_t p_from_offset, uint64_t p_to_offset, uint64_t p_size) {
    rd->buffer_copy(get_rid(p_from), get_rid(p_to), uint32_t(p_from_offset), uint32_t(p_to_offset), uint32_t(p_size));
}

ov::gpu::Texture RDDevice::create_texture(ov::gpu::TextureFormat p_format, uint32_t p_width, uint32_t p_height, uint32_t p_usage, const void *p_data) {
    Ref<RDTextureFormat> format;
    format.instantiate();
    format->set_format(rd_format(p_format));
    format->set_width(p_width);
    format->set_height(p_height);
    BitField<RenderingDevice::TextureUsageBits> usage = 0;
    if (p_usage & ov::gpu::TEXTURE_SAMPLED) {
        usage.set_flag(RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT);
    }
    if (p_usage & ov::gpu::TEXTURE_STORAGE) {
        usage.set_flag(RenderingDevice::TEXTURE_USAGE_STORAGE_BIT);
    }
    if (p_usage & ov::gpu::TEXTURE_COPY_SRC) {
        usage.set_flag(RenderingDevice::TEXTURE_USAGE_CAN_COPY_FROM_BIT);
    }
    if (p_usage & ov::gpu::TEXTURE_COPY_DST) {
        usage.set_flag(RenderingDevice::TEXTURE_USAGE_CAN_COPY_TO_BIT);
    }
    if (p_usage & ov::gpu::TEXTURE_UPDATE) {
        usage.set_flag(RenderingDevice::TEXTURE_USAGE_CAN_UPDATE_BIT);
    }
    format->set_usage_bits(usage);
    Ref<RDTextureView> view;
    view.instantiate();
    TypedArray<PackedByteArray> layers;
    if (p_data != nullptr) {
        layers.push_back(to_bytes(p_data, uint64_t(p_width) * p_height * ov::gpu::texel_bytes(p_format)));
    }
    return ov::gpu::Texture{ _add({ rd->texture_create(format, view, layers), RID() }) };
}

void RDDevice::update_texture(ov::gpu::Texture p_texture, const void *p_data, uint64_t p_size) {
    rd->texture_update(get_rid(p_texture), 0, to_bytes(p_data, p_size));
}

void RDDevice::clear_texture(ov::gpu::Texture p_texture) {
    rd->texture_clear(get_rid(p_texture), Color(0, 0, 0, 0), 0, 1, 0, 1);
}

void RDDevice::copy_texture(ov::gpu::Texture p_from, ov::gpu::Texture p_to, uint32_t p_from_x, uint32_t p_from_y, uint32_t p_to_x, uint32_t p_to_y,
        uint32_t p_width, uint32_t p_height) {
    rd->texture_copy(get_rid(p_from), get_rid(p_to), Vector3(p_from_x, p_from_y, 0), Vector3(p_to_x, p_to_y, 0), Vector3(p_width, p_height, 1), 0, 0,
            0, 0);
}

ov::gpu::Pipeline RDDevice::create_compute_pipeline(const ov::str &p_name, const ov::str &p_glsl, ov::str &r_error) {
    const String code = godot_str(p_glsl);
    Ref<RDShaderSource> source;
    source.instantiate();
    source->set_stage_source(RenderingDevice::SHADER_STAGE_COMPUTE, code);
    const Ref<RDShaderSPIRV> spirv = rd->shader_compile_spirv_from_source(source);
    r_error = ov_str(spirv->get_stage_compile_error(RenderingDevice::SHADER_STAGE_COMPUTE));
    if (!r_error.empty()) {
        return ov::gpu::Pipeline();
    }
    const RID shader = rd->shader_create_from_spirv(spirv, godot_str(p_name));
    const RID pipeline = rd->compute_pipeline_create(shader);
    if (!pipeline.is_valid()) {
        r_error = "compute_pipeline_create failed";
        rd->free_rid(shader);
        return ov::gpu::Pipeline();
    }
    return ov::gpu::Pipeline{ _add({ pipeline, shader }) };
}

ov::gpu::BindSet RDDevice::create_bind_set(ov::gpu::Pipeline p_pipeline, const std::vector<ov::gpu::Binding> &p_bindings) {
    const Entry *pipeline = _get(p_pipeline);
    ERR_FAIL_NULL_V(pipeline, ov::gpu::BindSet());
    TypedArray<RDUniform> uniforms;
    for (const ov::gpu::Binding &b : p_bindings) {
        Ref<RDUniform> u;
        u.instantiate();
        if (b.image.is_valid()) {
            u->set_uniform_type(RenderingDevice::UNIFORM_TYPE_IMAGE);
        } else {
            u->set_uniform_type(b.kind == ov::gpu::BufferKind::UNIFORM ? RenderingDevice::UNIFORM_TYPE_UNIFORM_BUFFER : RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER);
        }
        u->set_binding(int32_t(b.binding));
        u->add_id(b.image.is_valid() ? get_rid(b.image) : get_rid(b.buffer));
        uniforms.push_back(u);
    }
    Entry set;
    set.rid = rd->uniform_set_create(uniforms, pipeline->shader, 0);
    set.is_set = true;
    return ov::gpu::BindSet{ _add(set) };
}

void RDDevice::dispatch(ov::gpu::Pipeline p_pipeline, ov::gpu::BindSet p_set, const void *p_push, uint32_t p_push_size, uint32_t p_x, uint32_t p_y,
        uint32_t p_z) {
    const Entry *pipeline = _get(p_pipeline);
    const Entry *set = _get(p_set);
    ERR_FAIL_COND(pipeline == nullptr || set == nullptr || p_push_size > ov::gpu::PUSH_MAX);
    const int64_t list = rd->compute_list_begin();
    rd->compute_list_bind_compute_pipeline(list, pipeline->rid);
    rd->compute_list_bind_uniform_set(list, set->rid, 0);
    if (p_push_size > 0) {
        rd->compute_list_set_push_constant(list, to_bytes(p_push, p_push_size), p_push_size);
    }
    rd->compute_list_dispatch(list, p_x, p_y, p_z);
    rd->compute_list_end();
}

void RDDevice::timestamp(const ov::str &p_name) {
    rd->capture_timestamp(godot_str(p_name));
}

uint64_t RDDevice::timestamps_frame() {
    return rd->get_captured_timestamps_frame();
}

std::vector<ov::gpu::Timestamp> RDDevice::timestamps() {
    std::vector<ov::gpu::Timestamp> out;
    for (uint32_t i = 0; i < rd->get_captured_timestamps_count(); i++) {
        out.push_back({ ov_str(rd->get_captured_timestamp_name(i)), rd->get_captured_timestamp_gpu_time(i) });
    }
    return out;
}

void RDDevice::read_buffer_async(ov::gpu::Buffer p_buffer, uint64_t p_offset, uint64_t p_size, ov::gpu::Readback p_done) {
    const Error err = rd->buffer_get_data_async(get_rid(p_buffer), Callable(memnew(DeliverReadback(p_done))), uint32_t(p_offset), uint32_t(p_size));
    if (err != OK) {
        p_done(ov::bytes());
    }
}

ov::bytes RDDevice::read_buffer(ov::gpu::Buffer p_buffer, uint64_t p_offset, uint64_t p_size) {
    return from_bytes(rd->buffer_get_data(get_rid(p_buffer), uint32_t(p_offset), uint32_t(p_size)));
}

void RDDevice::read_texture_async(ov::gpu::Texture p_texture, ov::gpu::Readback p_done) {
    const Error err = rd->texture_get_data_async(get_rid(p_texture), 0, Callable(memnew(DeliverReadback(p_done))));
    if (err != OK) {
        p_done(ov::bytes());
    }
}

void RDDevice::post(std::function<void()> p_fn) {
    if (owns_rd) {
        p_fn();
        return;
    }
    RenderingServer::get_singleton()->call_on_render_thread(godot_callable(std::move(p_fn)));
}

void RDDevice::sync() {
    ERR_FAIL_COND_MSG(!owns_rd, "RDDevice: sync() is for local devices (the main one is submitted by the engine).");
    rd->submit();
    rd->sync();
}

void RDDevice::free(ov::gpu::Handle p_handle) {
    const auto it = entries.find(p_handle.id);
    if (it == entries.end()) {
        return;
    }
    _free_entry(it->second);
    entries.erase(it);
}
