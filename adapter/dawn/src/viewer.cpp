#include "viewer.hpp"

#include "dawn_device.hpp"
#include "dawn_page_host.hpp"
#include "x11_window.hpp"

#include "world.hpp"

#include "ov/files.hpp"
#include "ov/json.hpp"
#include "ov/log.hpp"
#include "ov/png.hpp"

#include <X11/keysym.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>

namespace ov::dawn {

namespace {

constexpr float FOV_RAD = 75.0f * 3.14159265f / 180.0f;
constexpr float NEAR_M = 0.1f;
// Frames drawn after the load is complete before a shot (the last pages' instance counts land).
constexpr int SHOT_SETTLE_FRAMES = 10;
// The Godot player's flight (player.gd): metres per second, radians per pixel, altitudes per second, seconds.
constexpr float BASE_SPEED = 50.0f;
constexpr float MOUSE_SENSITIVITY = 0.002f;
constexpr float FLY_RATE = 1.0f;
constexpr float ALTITUDE_TAU = 0.2f;
constexpr float PITCH_MAX = 1.55f; // short of straight up / down: the view basis needs a horizontal right
constexpr int VIEW_COUNT = 3; // shaded, normals, LOD

Float3 cross(const Float3 &a, const Float3 &b) {
    return Float3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

float dot(const Float3 &a, const Float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Yaw 0 looks along -z (Godot's), pitch up positive.
struct FlyCamera {
    Float3 pos;
    float yaw = 0.0f;
    float pitch = 0.0f;

    Float3 forward() const {
        return Float3{ std::cos(pitch) * -std::sin(yaw), std::sin(pitch), -std::cos(pitch) * std::cos(yaw) };
    }

    // Column-major view-projection: reversed depth (near 1, infinity 0).
    void view_proj(float p_aspect, float r_m[16]) const {
        const Float3 f = forward();
        const Float3 r = cross(f, Float3{ 0.0f, 1.0f, 0.0f }).normalized();
        const Float3 u = cross(r, f);
        const float view[16] = { r.x, u.x, -f.x, 0.0f, r.y, u.y, -f.y, 0.0f, r.z, u.z, -f.z, 0.0f, -dot(r, pos), -dot(u, pos), dot(f, pos), 1.0f };
        const float fy = 1.0f / std::tan(FOV_RAD * 0.5f);
        const float proj[16] = { fy / p_aspect, 0.0f, 0.0f, 0.0f, 0.0f, fy, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, NEAR_M, 0.0f };
        for (int c = 0; c < 4; c++) {
            for (int row = 0; row < 4; row++) {
                float s = 0.0f;
                for (int k = 0; k < 4; k++) {
                    s += proj[k * 4 + row] * view[c * 4 + k];
                }
                r_m[c * 4 + row] = s;
            }
        }
    }
};

// The camera p_above metres above the terrain under it: down the SDF's distance until it is within a metre.
void place_above_ground(const World &p_world, FlyCamera &r_camera, float p_above) {
    for (int i = 0; i < 64; i++) {
        const double pos[3] = { r_camera.pos.x, r_camera.pos.y, r_camera.pos.z };
        const double d = p_world.sdf_at(pos);
        if (std::abs(d) < 1.0) {
            break;
        }
        r_camera.pos.y -= float(d);
    }
    r_camera.pos.y += p_above;
}

wgpu::Texture make_target(const wgpu::Device &p_device, wgpu::TextureFormat p_format, int p_width, int p_height, wgpu::TextureUsage p_usage) {
    wgpu::TextureDescriptor desc{};
    desc.size = { uint32_t(p_width), uint32_t(p_height), 1 };
    desc.format = p_format;
    desc.usage = p_usage;
    return p_device.CreateTexture(&desc);
}

// The pages into p_color / p_depth; with an upscaler, its color and depth, then its passes into p_color.
void draw_frame(const wgpu::Device &p_device, const wgpu::Queue &p_queue, const wgpu::TextureView &p_color, const wgpu::TextureView &p_depth,
        DawnPageHost &r_pages, const PageView &p_view, DawnUpscaler *r_upscaler, float p_frame_seconds) {
    wgpu::RenderPassColorAttachment color{};
    color.view = r_upscaler ? r_upscaler->color_view() : p_color;
    color.loadOp = wgpu::LoadOp::Clear;
    color.storeOp = wgpu::StoreOp::Store;
    color.clearValue = { 0.78, 0.84, 0.93, 1.0 }; // the fog's sky, gamma-encoded
    wgpu::RenderPassDepthStencilAttachment depth{};
    depth.view = r_upscaler ? r_upscaler->depth_view() : p_depth;
    depth.depthLoadOp = wgpu::LoadOp::Clear;
    depth.depthStoreOp = r_upscaler ? wgpu::StoreOp::Store : wgpu::StoreOp::Discard;
    depth.depthClearValue = 0.0f;
    wgpu::RenderPassDescriptor pass_desc{};
    pass_desc.colorAttachmentCount = 1;
    pass_desc.colorAttachments = &color;
    pass_desc.depthStencilAttachment = &depth;
    const wgpu::CommandEncoder encoder = p_device.CreateCommandEncoder();
    const wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&pass_desc);
    r_pages.draw(pass, p_view);
    pass.End();
    if (r_upscaler) {
        r_upscaler->encode(encoder, p_color, p_frame_seconds);
    }
    const wgpu::CommandBuffer commands = encoder.Finish();
    p_queue.Submit(1, &commands);
}

// The offscreen target's pixels, RGBA8 rows tightly packed.
bytes read_target(DawnContext &p_context, const wgpu::Texture &p_target, int p_width, int p_height) {
    const uint32_t row_bytes = uint32_t(p_width) * 4;
    const uint32_t stride = (row_bytes + 255) & ~255u;
    wgpu::BufferDescriptor desc{};
    desc.size = uint64_t(stride) * uint32_t(p_height);
    desc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    const wgpu::Buffer staging = p_context.device.CreateBuffer(&desc);
    wgpu::TexelCopyTextureInfo src{};
    src.texture = p_target;
    wgpu::TexelCopyBufferInfo dst{};
    dst.buffer = staging;
    dst.layout.bytesPerRow = stride;
    dst.layout.rowsPerImage = uint32_t(p_height);
    const wgpu::Extent3D size{ uint32_t(p_width), uint32_t(p_height), 1 };
    const wgpu::CommandEncoder encoder = p_context.device.CreateCommandEncoder();
    encoder.CopyTextureToBuffer(&src, &dst, &size);
    const wgpu::CommandBuffer commands = encoder.Finish();
    p_context.queue.Submit(1, &commands);
    bool ok = false;
    p_context.instance.WaitAny(staging.MapAsync(wgpu::MapMode::Read, 0, desc.size, wgpu::CallbackMode::WaitAnyOnly,
                                       [&](wgpu::MapAsyncStatus p_status, wgpu::StringView) { ok = p_status == wgpu::MapAsyncStatus::Success; }),
            std::numeric_limits<uint64_t>::max());
    bytes out;
    if (!ok) {
        return out;
    }
    const uint8_t *data = static_cast<const uint8_t *>(staging.GetConstMappedRange(0, desc.size));
    for (int y = 0; y < p_height; y++) {
        out.insert(out.end(), data + size_t(y) * stride, data + size_t(y) * stride + row_bytes);
    }
    staging.Unmap();
    return out;
}

} // namespace

int run_viewer(DawnContext &p_context, const ViewerOptions &p_options) {
    gpu::set_local_device_factory(&DawnDevice::create_local); // the discoverer's and the field's devices
    const bool shot = !p_options.shot.empty();
    X11Window window;
    wgpu::Surface surface;
    wgpu::TextureFormat format = wgpu::TextureFormat::RGBA8Unorm;
    if (!shot) {
        if (!window.open(p_options.width, p_options.height, "opti-voxel (Dawn)")) {
            return 1;
        }
        surface = window.create_surface(p_context.instance);
        wgpu::SurfaceCapabilities caps{};
        surface.GetCapabilities(p_context.adapter, &caps);
        OV_ERR_FAIL_COND_V_MSG(caps.formatCount == 0, 1, "Viewer: the surface has no format.");
        format = caps.formats[0];
    }
    const auto configure = [&](int p_width, int p_height) {
        wgpu::SurfaceConfiguration config{};
        config.device = p_context.device;
        config.format = format;
        config.width = uint32_t(p_width);
        config.height = uint32_t(p_height);
        config.presentMode = wgpu::PresentMode::Fifo;
        surface.Configure(&config);
    };
    int width = p_options.width;
    int height = p_options.height;
    if (!shot) {
        configure(width, height);
    }

    std::unique_ptr<DawnUpscaler> upscaler;
    if (p_options.upscale.fsr || p_options.upscale.pixelate > 1) {
        upscaler = std::make_unique<DawnUpscaler>(p_context.device, p_context.instance, format, p_options.upscale, DawnUpscaler::Camera{ NEAR_M, FOV_RAD });
        upscaler->resize(width, height);
        OV_ERR_FAIL_COND_V_MSG(!upscaler->ready(), 1, "Viewer: no upscaler shaders.");
    }
    DawnDevice device(p_context);
    DawnPageHost pages(device, upscaler ? DawnUpscaler::COLOR_FORMAT : format);
    OV_ERR_FAIL_COND_V_MSG(!pages.ready(), 1, "Viewer: no page shader.");

    World world;
    World::Settings &s = world.settings();
    s.pipeline_mode = World::PIPELINE_GPU_DIRECT;
    s.terrain_mode = p_options.terrain_mode;
    s.lod_numbers = p_options.lods;
    s.lod_pixel_error = p_options.lod_error;
    s.lod_pixel_error_back = p_options.lod_error_back;
    s.archetype_trees = p_options.trees;
    s.archetype_tree_area = p_options.tree_area;
    s.climate_path = p_options.climate;
    s.chunk_param.voxel_size = p_options.voxel_size;
    s.debug_view = p_options.view;
    World::Projection projection;
    projection.viewport_height = float(upscaler ? upscaler->render_height() : height); // the LOD's pixels are the 3D's
    projection.fov_rad = FOV_RAD;
    projection.aspect = float(width) / float(height);
    World::Hosts hosts;
    hosts.device = &device;
    hosts.pages = &pages;
    if (!world.start(projection, hosts, nullptr, nullptr)) {
        return 1;
    }
    FlyCamera camera;
    camera.pos = Float3{ p_options.cam_pos[0], p_options.cam_pos[1], p_options.cam_pos[2] };
    camera.pitch = p_options.pitch_deg * 3.14159265f / 180.0f;
    camera.yaw = p_options.yaw_deg * 3.14159265f / 180.0f;
    place_above_ground(world, camera, p_options.above_ground);
    OV_LOG(LOG_INFO, "Viewer: camera at %s m", ov::format("%.1f, %.1f, %.1f", camera.pos.x, camera.pos.y, camera.pos.z));

    wgpu::Texture offscreen;
    if (shot) {
        offscreen = make_target(p_context.device, format, width, height, wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc);
    }
    wgpu::Texture depth = make_target(p_context.device, DawnPageHost::DEPTH_FORMAT, width, height, wgpu::TextureUsage::RenderAttachment);

    using Clock = std::chrono::steady_clock;
    const Clock::time_point start = Clock::now();
    Clock::time_point last = start;
    Clock::time_point last_print = start;
    int frames = 0;
    int settled = -1; // frames since the load completed (a shot)
    int result = 0;
    int view = p_options.view;
    float base_speed = BASE_SPEED;
    float smoothed_altitude = 0.0f; // 0: not read yet
    while (true) {
        const Clock::time_point now = Clock::now();
        const float dt = std::chrono::duration<float>(now - last).count();
        last = now;
        if (!shot) {
            if (!window.poll()) {
                break;
            }
            if (window.resized && window.width > 0 && window.height > 0) {
                width = window.width;
                height = window.height;
                configure(width, height);
                depth = make_target(p_context.device, DawnPageHost::DEPTH_FORMAT, width, height, wgpu::TextureUsage::RenderAttachment);
                if (upscaler) {
                    upscaler->resize(width, height);
                }
            }
            // The Godot player's controls (player.gd): mouse look while captured, WASD / arrows along the yaw, space up,
            // shift down, the wheel scales the base speed, V cycles the views (1-3 pick one).
            camera.yaw -= window.mouse_dx * MOUSE_SENSITIVITY;
            camera.pitch = std::fmax(-PITCH_MAX, std::fmin(PITCH_MAX, camera.pitch - window.mouse_dy * MOUSE_SENSITIVITY));
            base_speed *= std::pow(1.2f, float(window.wheel));
            if (window.pressed(XK_v)) {
                view = (view + 1) % VIEW_COUNT;
                world.set_debug_view(view);
            }
            for (int v = 0; v < VIEW_COUNT; v++) {
                if (window.pressed(XK_1 + unsigned(v))) {
                    view = v;
                    world.set_debug_view(view);
                }
            }
            const auto held = [&](unsigned long p_a, unsigned long p_b) { return float(window.held(p_a) || window.held(p_b)); };
            const float move_f = held(XK_w, XK_Up) - held(XK_s, XK_Down);
            const float move_r = held(XK_d, XK_Right) - held(XK_a, XK_Left);
            const float move_u = float(window.held(XK_space)) - held(XK_Shift_L, XK_Shift_R);
            const Float3 f{ -std::sin(camera.yaw), 0.0f, -std::cos(camera.yaw) };
            const Float3 r{ std::cos(camera.yaw), 0.0f, -std::sin(camera.yaw) };
            Float3 direction{ f.x * move_f + r.x * move_r, move_u, f.z * move_f + r.z * move_r };
            const float length = std::sqrt(dot(direction, direction));
            if (length > 0.0f) {
                // Kilometre-scale relief: FLY_RATE altitudes per second, at least the base speed; only the altitude
                // reading is smoothed, in log space.
                const double here[3] = { camera.pos.x, camera.pos.y, camera.pos.z };
                const float altitude = std::fmax(float(std::abs(world.sdf_at(here))), 0.01f);
                smoothed_altitude = smoothed_altitude <= 0.0f ? altitude
                                                              : smoothed_altitude * std::pow(altitude / smoothed_altitude, 1.0f - std::exp(-dt / ALTITUDE_TAU));
                const float step = std::fmax(base_speed, smoothed_altitude * FLY_RATE) * dt / length;
                camera.pos.x += direction.x * step;
                camera.pos.y += direction.y * step;
                camera.pos.z += direction.z * step;
            }
        }

        const Float3 forward = camera.forward();
        world.frame(&camera.pos, &forward);
        device.poll(); // the meshers' work submitted before the draw reads the pages

        PageView view;
        camera.view_proj(float(width) / float(height), view.view_proj);
        if (upscaler) {
            float unjittered[16];
            std::memcpy(unjittered, view.view_proj, sizeof(unjittered));
            upscaler->begin_frame(unjittered, view.view_proj);
        }
        view.camera = camera.pos;
        view.sun = Float3{ 0.4f, 0.8f, 0.3f }.normalized();
        view.fog_m = float(world.load_radius()) * 0.15f;
        wgpu::SurfaceTexture frame_texture{};
        wgpu::TextureView color;
        if (shot) {
            color = offscreen.CreateView();
        } else {
            surface.GetCurrentTexture(&frame_texture);
            if (frame_texture.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal &&
                    frame_texture.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
                configure(width, height);
                continue;
            }
            color = frame_texture.texture.CreateView();
        }
        draw_frame(p_context.device, p_context.queue, color, depth.CreateView(), pages, view, upscaler.get(), dt);
        frames++;

        if (!shot) {
            surface.Present();
            if (std::chrono::duration<float>(now - last_print).count() > 1.0f) {
                last_print = now;
                std::printf("frame %d  pages drawn %d  pending chunks %lld  camera %.0f %.0f %.0f\n", frames, pages.drawn_pages(),
                        (long long)world.pending_chunk_count(), double(camera.pos.x), double(camera.pos.y), double(camera.pos.z));
                std::fflush(stdout);
            }
            continue;
        }
        if (settled < 0 && world.is_load_complete(&camera.pos)) {
            settled = 0;
        }
        if (settled >= 0) {
            settled++;
        }
        if (settled >= SHOT_SETTLE_FRAMES || frames >= p_options.max_frames) {
            const bytes pixels = read_target(p_context, offscreen, width, height);
            const bytes png = encode_png(uint32_t(width), uint32_t(height), 4, pixels.data());
            const bool written = !pixels.empty() && write_file(p_options.shot, str(png.begin(), png.end()));
            Value report = Value::object();
            report["ok"] = written && settled >= 0;
            report["shot"] = p_options.shot;
            report["load_complete"] = settled >= 0;
            report["frames"] = frames;
            report["seconds"] = std::chrono::duration<double>(Clock::now() - start).count();
            report["pages_drawn"] = pages.drawn_pages();
            report["pending_chunks"] = world.pending_chunk_count();
            std::printf("%s\n", to_json(report).c_str());
            result = report["ok"].as_bool() ? 0 : 1;
            break;
        }
    }
    world.stop();
    device.sync(); // the mesher's last readbacks: it deletes itself once they are in
    return result;
}

} // namespace ov::dawn
