#include "dawn_device.hpp"

#include "ov/log.hpp"
#include "ov/on_error.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace ov::dawn {

namespace {

// WebGPU copies, clears and maps whole 4-byte words.
uint64_t align4(uint64_t p_size) {
    return (p_size + 3) & ~uint64_t(3);
}

} // namespace

DawnDevice::DawnDevice(DawnContext &p_context) : context(p_context) {
    wgpu::Limits limits{};
    context.device.GetLimits(&limits);
    push_size_limit = std::min<uint32_t>(limits.maxImmediateSize, gpu::PUSH_MAX);
}

DawnDevice::~DawnDevice() {
    sync();
}

gpu::Handle DawnDevice::_add(Entry p_entry) {
    const uint64_t id = next_id++;
    entries[id] = std::move(p_entry);
    return gpu::Handle{ id };
}

DawnDevice::Entry *DawnDevice::_get(gpu::Handle p_handle) {
    const auto it = entries.find(p_handle.id);
    return it != entries.end() ? &it->second : nullptr;
}

wgpu::CommandEncoder &DawnDevice::_encoder() {
    if (!encoder) {
        encoder = context.device.CreateCommandEncoder();
    }
    recorded = true;
    return encoder;
}

void DawnDevice::_submit() {
    if (!recorded) {
        return;
    }
    const wgpu::CommandBuffer commands = encoder.Finish();
    context.queue.Submit(1, &commands);
    encoder = nullptr;
    recorded = false;
}

gpu::Buffer DawnDevice::create_buffer(gpu::BufferKind p_kind, uint64_t p_size, const void *p_data) {
    wgpu::BufferDescriptor desc{};
    desc.size = std::max<uint64_t>(align4(p_size), 4);
    desc.usage = wgpu::BufferUsage::CopyDst |
            (p_kind == gpu::BufferKind::UNIFORM ? wgpu::BufferUsage::Uniform : wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
    desc.mappedAtCreation = true; // zeros, or the data
    Entry entry;
    entry.buffer = context.device.CreateBuffer(&desc);
    OV_ERR_FAIL_COND_V_MSG(!entry.buffer, gpu::Buffer(), ov::format("DawnDevice: no buffer of %d bytes.", int64_t(p_size)));
    if (p_data != nullptr && p_size > 0) {
        std::memcpy(entry.buffer.GetMappedRange(0, desc.size), p_data, size_t(p_size));
    }
    entry.buffer.Unmap();
    return gpu::Buffer{ _add(std::move(entry)) };
}

void DawnDevice::update_buffer(gpu::Buffer p_buffer, uint64_t p_offset, uint64_t p_size, const void *p_data) {
    const Entry *entry = _get(p_buffer);
    OV_ERR_FAIL_COND_MSG(entry == nullptr || !entry->buffer || p_size == 0, "DawnDevice: update of an unknown buffer.");
    // Queue writes land before the next submission's commands: what is recorded so far goes first.
    _submit();
    if (p_size % 4 == 0) {
        context.queue.WriteBuffer(entry->buffer, p_offset, p_data, size_t(p_size));
        return;
    }
    const uint8_t *data = static_cast<const uint8_t *>(p_data);
    bytes padded(data, data + p_size);
    padded.resize(size_t(align4(p_size)), 0);
    context.queue.WriteBuffer(entry->buffer, p_offset, padded.data(), padded.size());
}

void DawnDevice::clear_buffer(gpu::Buffer p_buffer, uint64_t p_offset, uint64_t p_size) {
    const Entry *entry = _get(p_buffer);
    OV_ERR_FAIL_COND_MSG(entry == nullptr || !entry->buffer, "DawnDevice: clear of an unknown buffer.");
    _encoder().ClearBuffer(entry->buffer, p_offset, align4(p_size));
}

void DawnDevice::copy_buffer(gpu::Buffer p_from, gpu::Buffer p_to, uint64_t p_from_offset, uint64_t p_to_offset, uint64_t p_size) {
    const Entry *from = _get(p_from);
    const Entry *to = _get(p_to);
    OV_ERR_FAIL_COND_MSG(from == nullptr || to == nullptr, "DawnDevice: copy of an unknown buffer.");
    _encoder().CopyBufferToBuffer(from->buffer, p_from_offset, to->buffer, p_to_offset, align4(p_size));
}

gpu::Pipeline DawnDevice::create_compute_pipeline(const str &p_name, const str &p_glsl, str &r_error) {
    CompiledShader shader = compile_compute_glsl(p_glsl);
    if (!shader.error.empty()) {
        r_error = shader.error;
        return gpu::Pipeline();
    }
    wgpu::ShaderSourceSPIRV spirv{};
    spirv.codeSize = uint32_t(shader.spirv.size());
    spirv.code = shader.spirv.data();
    // Validation errors (SPIR-V Tint rejects, a layout mismatch) come back through an error scope.
    context.device.PushErrorScope(wgpu::ErrorFilter::Validation);
    wgpu::ShaderModuleDescriptor module_desc{};
    module_desc.nextInChain = &spirv;
    module_desc.label = p_name.c_str();
    const wgpu::ShaderModule module = context.device.CreateShaderModule(&module_desc);

    // The layout of every buffer the shader declares (an automatic one drops the unused ones, which the engine binds).
    std::vector<wgpu::BindGroupLayoutEntry> layout_entries;
    for (const ShaderBinding &b : shader.bindings) {
        wgpu::BindGroupLayoutEntry e{};
        e.binding = b.binding;
        e.visibility = wgpu::ShaderStage::Compute;
        e.buffer.type = b.uniform ? wgpu::BufferBindingType::Uniform
                                  : (b.read_only ? wgpu::BufferBindingType::ReadOnlyStorage : wgpu::BufferBindingType::Storage);
        layout_entries.push_back(e);
    }
    wgpu::BindGroupLayoutDescriptor layout_desc{};
    layout_desc.entryCount = layout_entries.size();
    layout_desc.entries = layout_entries.data();
    Entry entry;
    entry.layout = context.device.CreateBindGroupLayout(&layout_desc);
    entry.bindings = std::move(shader.bindings);

    wgpu::PipelineLayoutDescriptor pipeline_layout_desc{};
    pipeline_layout_desc.bindGroupLayoutCount = 1;
    pipeline_layout_desc.bindGroupLayouts = &entry.layout;
    pipeline_layout_desc.immediateSize = push_size_limit;
    wgpu::ComputePipelineDescriptor desc{};
    desc.label = p_name.c_str();
    desc.layout = context.device.CreatePipelineLayout(&pipeline_layout_desc);
    desc.compute.module = module;
    desc.compute.entryPoint = "main";

    entry.pipeline = context.device.CreateComputePipeline(&desc);
    str scope_error;
    context.instance.WaitAny(context.device.PopErrorScope(wgpu::CallbackMode::WaitAnyOnly,
                                     [&](wgpu::PopErrorScopeStatus, wgpu::ErrorType p_type, wgpu::StringView p_message) {
                                         if (p_type != wgpu::ErrorType::NoError) {
                                             scope_error.assign(p_message.data, p_message.length);
                                         }
                                     }),
            std::numeric_limits<uint64_t>::max());
    if (!scope_error.empty()) {
        r_error = scope_error;
        return gpu::Pipeline();
    }
    r_error.clear();
    return gpu::Pipeline{ _add(std::move(entry)) };
}

gpu::BindSet DawnDevice::create_bind_set(gpu::Pipeline p_pipeline, const std::vector<gpu::Binding> &p_bindings) {
    const Entry *pipeline = _get(p_pipeline);
    OV_ERR_FAIL_COND_V_MSG(pipeline == nullptr, gpu::BindSet(), "DawnDevice: bind set of an unknown pipeline.");
    std::vector<wgpu::BindGroupEntry> group_entries;
    for (const ShaderBinding &sb : pipeline->bindings) {
        const auto it = std::find_if(p_bindings.begin(), p_bindings.end(), [&](const gpu::Binding &b) { return b.binding == sb.binding; });
        OV_ERR_FAIL_COND_V_MSG(it == p_bindings.end(), gpu::BindSet(), ov::format("DawnDevice: the shader's binding %d is not bound.", int64_t(sb.binding)));
        const Entry *buffer = _get(it->buffer);
        OV_ERR_FAIL_COND_V_MSG(buffer == nullptr, gpu::BindSet(), "DawnDevice: binding of an unknown buffer.");
        wgpu::BindGroupEntry e{};
        e.binding = sb.binding;
        e.buffer = buffer->buffer;
        e.size = buffer->buffer.GetSize();
        group_entries.push_back(e);
    }
    wgpu::BindGroupDescriptor desc{};
    desc.layout = pipeline->layout;
    desc.entryCount = group_entries.size();
    desc.entries = group_entries.data();
    Entry entry;
    entry.group = context.device.CreateBindGroup(&desc);
    return gpu::BindSet{ _add(std::move(entry)) };
}

void DawnDevice::dispatch(gpu::Pipeline p_pipeline, gpu::BindSet p_set, const void *p_push, uint32_t p_push_size, uint32_t p_x, uint32_t p_y,
        uint32_t p_z) {
    const Entry *pipeline = _get(p_pipeline);
    const Entry *set = _get(p_set);
    OV_ERR_FAIL_COND_MSG(pipeline == nullptr || set == nullptr || p_push_size > push_size_limit, "DawnDevice: dispatch of an unknown pipeline or set, or too many push bytes.");
    const wgpu::ComputePassEncoder pass = _encoder().BeginComputePass();
    pass.SetPipeline(pipeline->pipeline);
    pass.SetBindGroup(0, set->group);
    if (p_push_size > 0) {
        pass.SetImmediates(0, p_push, p_push_size);
    }
    pass.DispatchWorkgroups(p_x, p_y, p_z);
    pass.End();
}

void DawnDevice::timestamp(const str &p_name) {
    (void)p_name; // TODO: timestamp queries (FeatureName::TimestampQuery), so GPUCostModel and the profiler work on Dawn.
}

std::vector<gpu::Timestamp> DawnDevice::timestamps() {
    return {};
}

void DawnDevice::read_buffer_async(gpu::Buffer p_buffer, uint64_t p_offset, uint64_t p_size, gpu::Readback p_done) {
    const Entry *entry = _get(p_buffer);
    if (entry == nullptr || !entry->buffer || p_size == 0) {
        p_done(bytes());
        return;
    }
    std::unique_ptr<Pending> request = std::make_unique<Pending>();
    wgpu::BufferDescriptor desc{};
    desc.size = align4(p_size);
    desc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    request->staging = context.device.CreateBuffer(&desc);
    request->size = p_size;
    request->done = std::move(p_done);
    _encoder().CopyBufferToBuffer(entry->buffer, p_offset, request->staging, 0, desc.size);
    _submit();
    Pending *target = request.get();
    request->staging.MapAsync(wgpu::MapMode::Read, 0, desc.size, wgpu::CallbackMode::AllowProcessEvents,
            [target](wgpu::MapAsyncStatus p_status, wgpu::StringView) {
                target->mapped = true;
                target->failed = p_status != wgpu::MapAsyncStatus::Success;
            });
    pending.push_back(std::move(request));
}

bytes DawnDevice::read_buffer(gpu::Buffer p_buffer, uint64_t p_offset, uint64_t p_size) {
    bytes out;
    read_buffer_async(p_buffer, p_offset, p_size, [&out](const bytes &p_data) { out = p_data; });
    sync();
    return out;
}

void DawnDevice::_deliver() {
    // In request order: a later readback is never delivered before an earlier one.
    size_t n = 0;
    while (n < pending.size() && pending[n]->mapped) {
        Pending &p = *pending[n];
        bytes data;
        if (!p.failed) {
            const uint8_t *src = static_cast<const uint8_t *>(p.staging.GetConstMappedRange(0, align4(p.size)));
            data.assign(src, src + p.size);
        }
        p.staging.Unmap();
        p.staging.Destroy();
        p.done(data);
        n++;
    }
    pending.erase(pending.begin(), pending.begin() + std::ptrdiff_t(n));
}

void DawnDevice::poll() {
    _submit();
    context.instance.ProcessEvents();
    _deliver();
}

void DawnDevice::sync() {
    _submit();
    context.instance.WaitAny(context.queue.OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly, [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {}),
            std::numeric_limits<uint64_t>::max());
    while (std::any_of(pending.begin(), pending.end(), [](const std::unique_ptr<Pending> &p) { return !p->mapped; })) {
        context.instance.ProcessEvents();
    }
    _deliver();
}

void DawnDevice::free(gpu::Handle p_handle) {
    const auto it = entries.find(p_handle.id);
    if (it == entries.end()) {
        return;
    }
    entries.erase(it); // Dawn keeps an object alive while recorded or submitted work uses it
}

} // namespace ov::dawn
