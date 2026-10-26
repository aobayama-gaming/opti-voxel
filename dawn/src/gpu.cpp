#include "gpu.hpp"

#include "glsl.hpp"

#include <cstdio>
#include <cstring>
#include <limits>

namespace ov::dawn {

namespace {

const char *status_text(wgpu::RequestAdapterStatus p_status) {
    switch (p_status) {
        case wgpu::RequestAdapterStatus::Success:
            return "success";
        case wgpu::RequestAdapterStatus::Unavailable:
            return "unavailable";
        case wgpu::RequestAdapterStatus::Error:
            return "error";
        case wgpu::RequestAdapterStatus::CallbackCancelled:
            return "cancelled";
    }
    return "?";
}

} // namespace

bool GpuContext::initialize(void *p_xdisplay, uint64_t p_xwindow, uint32_t p_width, uint32_t p_height) {
    width = p_width;
    height = p_height;

    wgpu::InstanceFeatureName features[] = { wgpu::InstanceFeatureName::ShaderSourceSPIRV, wgpu::InstanceFeatureName::TimedWaitAny };
    wgpu::InstanceDescriptor instance_desc{};
    instance_desc.requiredFeatureCount = 2;
    instance_desc.requiredFeatures = features;
    instance = wgpu::CreateInstance(&instance_desc);
    if (!instance) {
        std::fprintf(stderr, "dawn: no instance\n");
        return false;
    }

    wgpu::SurfaceSourceXlibWindow xlib{};
    xlib.display = p_xdisplay;
    xlib.window = p_xwindow;
    wgpu::SurfaceDescriptor surface_desc{};
    surface_desc.nextInChain = &xlib;
    surface_desc.label = "optivoxel";
    surface = instance.CreateSurface(&surface_desc);
    if (!surface) {
        std::fprintf(stderr, "dawn: no surface\n");
        return false;
    }

    wgpu::RequestAdapterOptions options{};
    options.powerPreference = wgpu::PowerPreference::HighPerformance;
    options.compatibleSurface = surface;

    struct AdapterResult {
        wgpu::RequestAdapterStatus status = wgpu::RequestAdapterStatus::Error;
        wgpu::Adapter adapter;
        std::string message;
    } result;
    wgpu::Future future = instance.RequestAdapter(
            &options, wgpu::CallbackMode::WaitAnyOnly,
            [&result](wgpu::RequestAdapterStatus p_status, wgpu::Adapter p_adapter, wgpu::StringView p_message) {
                result.status = p_status;
                result.adapter = std::move(p_adapter);
                if (p_message.data != nullptr) {
                    result.message.assign(p_message.data, p_message.length);
                }
            });
    if (instance.WaitAny(future, std::numeric_limits<uint64_t>::max()) != wgpu::WaitStatus::Success ||
            result.status != wgpu::RequestAdapterStatus::Success || !result.adapter) {
        std::fprintf(stderr, "dawn: no adapter (%s): %s\n", status_text(result.status), result.message.c_str());
        return false;
    }
    adapter = std::move(result.adapter);

    wgpu::AdapterInfo info{};
    adapter.GetInfo(&info);
    std::printf("dawn: %s (%s)\n", info.device.data != nullptr ? info.device.data : "?", info.description.data != nullptr ? info.description.data : "");

    wgpu::SupportedFeatures supported{};
    adapter.GetFeatures(&supported);
    wgpu::FeatureName device_features[1];
    size_t device_feature_count = 0;
    for (size_t i = 0; i < supported.featureCount; i++) {
        if (supported.features[i] == wgpu::FeatureName::TextureFormatsTier2) {
            device_features[device_feature_count++] = wgpu::FeatureName::TextureFormatsTier2;
            break;
        }
    }
    wgpu::Limits limits{};
    limits.maxStorageTexturesPerShaderStage = 8; // FSR3's luma pyramid binds six SPD mips plus two more
    wgpu::DeviceDescriptor device_desc{};
    device_desc.label = "optivoxel";
    device_desc.requiredFeatureCount = device_feature_count;
    device_desc.requiredFeatures = device_features;
    device_desc.requiredLimits = &limits;
    device_desc.SetUncapturedErrorCallback([](const wgpu::Device &, wgpu::ErrorType p_type, wgpu::StringView p_message) {
        std::fprintf(stderr, "dawn error (%d): %.*s\n", int(p_type), int(p_message.length), p_message.data);
    });

    struct DeviceResult {
        wgpu::RequestDeviceStatus status = wgpu::RequestDeviceStatus::Error;
        wgpu::Device device;
        std::string message;
    } device_result;
    wgpu::Future device_future = adapter.RequestDevice(
            &device_desc, wgpu::CallbackMode::WaitAnyOnly,
            [&device_result](wgpu::RequestDeviceStatus p_status, wgpu::Device p_device, wgpu::StringView p_message) {
                device_result.status = p_status;
                device_result.device = std::move(p_device);
                if (p_message.data != nullptr) {
                    device_result.message.assign(p_message.data, p_message.length);
                }
            });
    if (instance.WaitAny(device_future, std::numeric_limits<uint64_t>::max()) != wgpu::WaitStatus::Success ||
            device_result.status != wgpu::RequestDeviceStatus::Success || !device_result.device) {
        std::fprintf(stderr, "dawn: no device: %s\n", device_result.message.c_str());
        return false;
    }
    device = std::move(device_result.device);
    queue = device.GetQueue();

    wgpu::SurfaceCapabilities capabilities{};
    if (surface.GetCapabilities(adapter, &capabilities) != wgpu::Status::Success || capabilities.formatCount == 0) {
        std::fprintf(stderr, "dawn: the surface has no format\n");
        return false;
    }
    format = capabilities.formats[0];
    for (size_t i = 0; i < capabilities.formatCount; i++) {
        if (capabilities.formats[i] == wgpu::TextureFormat::BGRA8UnormSrgb || capabilities.formats[i] == wgpu::TextureFormat::RGBA8UnormSrgb) {
            format = capabilities.formats[i];
            break;
        }
    }

    wgpu::SurfaceConfiguration config{};
    config.device = device;
    config.format = format;
    config.usage = wgpu::TextureUsage::RenderAttachment;
    config.width = width;
    config.height = height;
    config.presentMode = wgpu::PresentMode::Fifo;
    config.alphaMode = wgpu::CompositeAlphaMode::Auto;
    surface.Configure(&config);
    return true;
}

void GpuContext::resize(uint32_t p_width, uint32_t p_height) {
    if (p_width == width && p_height == height) {
        return;
    }
    width = p_width;
    height = p_height;
    wgpu::SurfaceConfiguration config{};
    config.device = device;
    config.format = format;
    config.usage = wgpu::TextureUsage::RenderAttachment;
    config.width = width;
    config.height = height;
    config.presentMode = wgpu::PresentMode::Fifo;
    config.alphaMode = wgpu::CompositeAlphaMode::Auto;
    surface.Configure(&config);
}

wgpu::SurfaceTexture GpuContext::acquire() {
    wgpu::SurfaceTexture texture{};
    surface.GetCurrentTexture(&texture);
    return texture;
}

void GpuContext::present(wgpu::SurfaceTexture &p_texture) {
    if (p_texture.texture) {
        surface.Present();
    }
}

wgpu::Buffer create_buffer(const wgpu::Device &p_device, uint64_t p_size, wgpu::BufferUsage p_usage, const char *p_label) {
    wgpu::BufferDescriptor desc{};
    desc.label = p_label != nullptr ? p_label : "buffer";
    desc.size = (p_size + 3) & ~uint64_t(3);
    desc.usage = p_usage;
    desc.mappedAtCreation = false;
    return p_device.CreateBuffer(&desc);
}

void write_buffer(const wgpu::Queue &p_queue, const wgpu::Buffer &p_buffer, const void *p_data, uint64_t p_size, uint64_t p_offset) {
    if (p_size == 0) {
        return;
    }
    p_queue.WriteBuffer(p_buffer, p_offset, p_data, p_size);
}

wgpu::Texture create_texture(const wgpu::Device &p_device, uint32_t p_width, uint32_t p_height, wgpu::TextureFormat p_format,
        wgpu::TextureUsage p_usage, uint32_t p_mips, const char *p_label) {
    wgpu::TextureDescriptor desc{};
    desc.label = p_label != nullptr ? p_label : "texture";
    desc.size = { p_width, p_height, 1 };
    desc.mipLevelCount = p_mips;
    desc.sampleCount = 1;
    desc.dimension = wgpu::TextureDimension::e2D;
    desc.format = p_format;
    desc.usage = p_usage;
    return p_device.CreateTexture(&desc);
}

std::vector<uint8_t> read_buffer(const GpuContext &p_gpu, const wgpu::Buffer &p_buffer, uint64_t p_size) {
    if (p_size == 0) {
        return {};
    }
    wgpu::Buffer staging = create_buffer(p_gpu.device, p_size, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst, "readback");
    wgpu::CommandEncoder encoder = p_gpu.device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(p_buffer, 0, staging, 0, p_size);
    wgpu::CommandBuffer commands = encoder.Finish();
    p_gpu.queue.Submit(1, &commands);

    bool mapped = false;
    wgpu::Future future = staging.MapAsync(wgpu::MapMode::Read, 0, p_size, wgpu::CallbackMode::WaitAnyOnly,
            [&mapped](wgpu::MapAsyncStatus p_status, wgpu::StringView) { mapped = p_status == wgpu::MapAsyncStatus::Success; });
    p_gpu.instance.WaitAny(future, std::numeric_limits<uint64_t>::max());
    if (!mapped) {
        std::fprintf(stderr, "dawn: readback map failed\n");
        return {};
    }
    std::vector<uint8_t> out;
    out.resize(size_t(p_size));
    std::memcpy(out.data(), staging.GetConstMappedRange(0, p_size), size_t(p_size));
    staging.Unmap();
    return out;
}

wgpu::ComputePipeline create_compute_pipeline(const GpuContext &p_gpu, const std::string &p_glsl,
        const wgpu::BindGroupLayout &p_layout, uint32_t p_immediate_size, std::string &r_error, const char *p_label) {
    std::vector<uint32_t> spirv;
    r_error = CompileComputeGLSL(p_glsl, spirv);
    if (!r_error.empty()) {
        return {};
    }

    wgpu::ShaderSourceSPIRV spirv_desc{};
    spirv_desc.codeSize = spirv.size();
    spirv_desc.code = spirv.data();
    wgpu::ShaderModuleDescriptor module_desc{};
    module_desc.label = p_label != nullptr ? p_label : "shader";
    module_desc.nextInChain = &spirv_desc;
    wgpu::ShaderModule module = p_gpu.device.CreateShaderModule(&module_desc);
    if (!module) {
        r_error = "dawn: shader module creation failed";
        return {};
    }

    wgpu::PipelineLayoutDescriptor layout_desc{};
    layout_desc.label = "pipeline layout";
    layout_desc.bindGroupLayoutCount = 1;
    layout_desc.bindGroupLayouts = &p_layout;
    layout_desc.immediateSize = p_immediate_size;
    wgpu::PipelineLayout pipeline_layout = p_gpu.device.CreatePipelineLayout(&layout_desc);

    wgpu::ComputePipelineDescriptor desc{};
    desc.label = p_label != nullptr ? p_label : "pipeline";
    desc.layout = pipeline_layout;
    desc.compute.module = module;
    desc.compute.entryPoint = "main";
    wgpu::ComputePipeline pipeline = p_gpu.device.CreateComputePipeline(&desc);
    if (!pipeline) {
        r_error = "dawn: compute pipeline creation failed";
    }
    return pipeline;
}

} // namespace ov::dawn
