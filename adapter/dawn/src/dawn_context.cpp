#include "dawn_context.hpp"

#include "ov/log.hpp"
#include "ov/on_error.hpp"

#include <limits>
#include <string>

namespace ov::dawn {

namespace {

std::string text(wgpu::StringView p_view) {
    return p_view.data != nullptr ? std::string(p_view.data, p_view.length) : std::string();
}

} // namespace

bool DawnContext::initialize() {
    const wgpu::InstanceFeatureName features[] = { wgpu::InstanceFeatureName::ShaderSourceSPIRV, wgpu::InstanceFeatureName::TimedWaitAny };
    wgpu::InstanceDescriptor instance_desc{};
    instance_desc.requiredFeatureCount = 2;
    instance_desc.requiredFeatures = features;
    instance = wgpu::CreateInstance(&instance_desc);
    OV_ERR_FAIL_COND_V_MSG(!instance, false, "Dawn: no instance.");

    wgpu::RequestAdapterOptions options{};
    options.powerPreference = wgpu::PowerPreference::HighPerformance;
    std::string message;
    const wgpu::Future adapter_future = instance.RequestAdapter(&options, wgpu::CallbackMode::WaitAnyOnly,
            [&](wgpu::RequestAdapterStatus p_status, wgpu::Adapter p_adapter, wgpu::StringView p_message) {
                if (p_status == wgpu::RequestAdapterStatus::Success) {
                    adapter = std::move(p_adapter);
                }
                message = text(p_message);
            });
    instance.WaitAny(adapter_future, std::numeric_limits<uint64_t>::max());
    OV_ERR_FAIL_COND_V_MSG(!adapter, false, "Dawn: no adapter: " + message);

    wgpu::AdapterInfo info{};
    adapter.GetInfo(&info);
    OV_LOG(LOG_INFO, "Dawn: %s (%s)", text(info.device), text(info.description));

    // Everything the adapter offers: the engine's passes bind up to ten storage buffers and push 32 bytes.
    wgpu::Limits limits{};
    adapter.GetLimits(&limits);
    wgpu::DeviceDescriptor device_desc{};
    device_desc.label = "optivoxel";
    device_desc.requiredLimits = &limits;
    device_desc.SetUncapturedErrorCallback([](const wgpu::Device &, wgpu::ErrorType p_type, wgpu::StringView p_message) {
        OV_ERR_PRINT(ov::format("Dawn error (%d): %s", int(p_type), text(p_message)));
    });
    const wgpu::Future device_future = adapter.RequestDevice(&device_desc, wgpu::CallbackMode::WaitAnyOnly,
            [&](wgpu::RequestDeviceStatus p_status, wgpu::Device p_device, wgpu::StringView p_message) {
                if (p_status == wgpu::RequestDeviceStatus::Success) {
                    device = std::move(p_device);
                }
                message = text(p_message);
            });
    instance.WaitAny(device_future, std::numeric_limits<uint64_t>::max());
    OV_ERR_FAIL_COND_V_MSG(!device, false, "Dawn: no device: " + message);
    queue = device.GetQueue();
    return true;
}

} // namespace ov::dawn
