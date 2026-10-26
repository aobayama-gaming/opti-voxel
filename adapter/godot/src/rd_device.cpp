#include "rd_device.hpp"

#include "godot_conv.hpp"
#include "rd_utils.hpp"

#include <godot_cpp/classes/rd_shader_source.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
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

} // namespace

RDDevice::~RDDevice() {
    for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
        for (const RID &rid : { it->second.rid, it->second.shader }) {
            if (rid.is_valid()) {
                rd->free_rid(rid);
            }
        }
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

RID RDDevice::get_rid(ov::gpu::Buffer p_buffer) const {
    const Entry *entry = _get(p_buffer);
    return entry != nullptr ? entry->rid : RID();
}

ov::gpu::Handle RDDevice::_add(const Entry &p_entry) {
    if (!p_entry.rid.is_valid()) {
        return ov::gpu::Handle();
    }
    const uint64_t id = next_id++;
    entries[id] = p_entry;
    return ov::gpu::Handle{ id };
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

ov::gpu::Pipeline RDDevice::create_compute_pipeline(const ov::str &p_name, const ov::str &p_glsl, ov::str &r_error) {
    const String code = godot_str(p_glsl);
    RDUtils::dump_shader(godot_str(p_name), code);
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
        u->set_uniform_type(b.kind == ov::gpu::BufferKind::UNIFORM ? RenderingDevice::UNIFORM_TYPE_UNIFORM_BUFFER : RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER);
        u->set_binding(int32_t(b.binding));
        u->add_id(get_rid(b.buffer));
        uniforms.push_back(u);
    }
    return ov::gpu::BindSet{ _add({ rd->uniform_set_create(uniforms, pipeline->shader, 0), RID() }) };
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
    // A pipeline before its shader.
    for (const RID &rid : { it->second.rid, it->second.shader }) {
        if (rid.is_valid()) {
            rd->free_rid(rid);
        }
    }
    entries.erase(it);
}
