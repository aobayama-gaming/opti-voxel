#include "viewer.hpp"

#include "dawn_device.hpp"
#include "dawn_page_host.hpp"
#include "wgpu_util.hpp"
#ifdef __EMSCRIPTEN__
#include "web_canvas.hpp"

#include <emscripten.h>
#else
#include "x11_window.hpp"
#endif

#include "dig_tool.hpp"
#include "world.hpp"

#include "ov/color.hpp"
#include "ov/files.hpp"
#include "ov/json.hpp"
#include "ov/log.hpp"
#include "ov/png.hpp"

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
// With --fly, a shot's frames of flight from the completed load (one second at the shot's 60 Hz).
constexpr int SHOT_FLY_FRAMES = 60;
// The Godot player's flight (player.gd): metres per second, radians per pixel, altitudes per second, seconds.
constexpr float BASE_SPEED = 50.0f;
constexpr float MOUSE_SENSITIVITY = 0.002f;
constexpr float FLY_RATE = 1.0f;
constexpr float ALTITUDE_TAU = 0.2f;
constexpr float PITCH_MAX = 1.55f; // short of straight up / down: the view basis needs a horizontal right
constexpr int VIEW_COUNT = 3; // shaded, normals, LOD
// The sun of Godot's main scenes (main_scene*.tscn, their DirectionalLight3D: the transform's z column, light_color in
// sRGB, light_energy, light_angular_distance in degrees); the sky light's energy (a stand-in for the sky's light and
// SSAO, tuned on the forest) and the sky's exposure (Godot's is 1: a dull grey-blue at this sun under ACES and the grade).
constexpr Float3 SUN_DIR{ -0.5297686f, 0.4694716f, 0.7063581f };
constexpr ov::Color SUN_COLOR_SRGB{ 1.0f, 0.93f, 0.83f };
constexpr float SUN_ENERGY = 1.3f;
constexpr float SUN_SIZE_DEG = 0.53f;
constexpr float SKY_ENERGY = 0.6f;
constexpr float SKY_EXPOSURE = 3.0f;

// Yaw 0 looks along -z (Godot's), pitch up positive.
struct FlyCamera {
    Float3 pos;
    float yaw = 0.0f;
    float pitch = 0.0f;

    Float3 forward() const {
        return Float3{ std::cos(pitch) * -std::sin(yaw), std::sin(pitch), -std::cos(pitch) * std::cos(yaw) };
    }

    // The view's basis: forward, right (horizontal), up.
    void basis(Float3 &r_f, Float3 &r_r, Float3 &r_u) const {
        r_f = forward();
        r_r = cross(r_f, Float3{ 0.0f, 1.0f, 0.0f }).normalized();
        r_u = cross(r_r, r_f);
    }

    // The view's right and up, each times the tangent of half its field of view: forward + x right + y up is the ray
    // through the clip-space point (x, y).
    void view_rays(float p_aspect, Float3 &r_right, Float3 &r_up) const {
        Float3 f, r, u;
        basis(f, r, u);
        const float t = std::tan(FOV_RAD * 0.5f);
        r_right = Float3{ r.x * t * p_aspect, r.y * t * p_aspect, r.z * t * p_aspect };
        r_up = Float3{ u.x * t, u.y * t, u.z * t };
    }

    // Column-major view-projection: reversed depth (near 1, infinity 0).
    void view_proj(float p_aspect, float r_m[16]) const {
        Float3 f, r, u;
        basis(f, r, u);
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

// The sun's shadow map, then the pages and the sky into p_color / p_depth; with an upscaler, its color and depth, then
// its passes into p_color.
void draw_frame(const wgpu::Device &p_device, const wgpu::Queue &p_queue, const wgpu::TextureView &p_color, const wgpu::TextureView &p_depth,
        DawnPageHost &r_pages, const PageView &p_view, DawnUpscaler *r_upscaler, float p_frame_seconds) {
    wgpu::RenderPassColorAttachment color{};
    color.view = r_upscaler ? r_upscaler->color_view() : p_color;
    color.loadOp = wgpu::LoadOp::Clear;
    color.storeOp = wgpu::StoreOp::Store;
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
    r_pages.encode_shadow(encoder, p_view);
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
#ifdef __EMSCRIPTEN__
    WebCanvas window;
#else
    X11Window window;
#endif
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
    if (!p_options.grade.empty()) {
        str grade;
        if (!read_file(p_options.grade, grade)) {
            OV_LOG(LOG_WARNING, "Viewer: no grade at %s (--grade=PATH or off): ungraded", p_options.grade);
        } else {
            pages.set_grade(grade);
        }
    }

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
    s.discoverer_mode = p_options.discoverer_mode;
    s.walk_gpu = p_options.walk_gpu;
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
    DigTool dig(world);
    OV_LOG(LOG_INFO, "Viewer: camera at %s m", ov::format("%.1f, %.1f, %.1f", camera.pos.x, camera.pos.y, camera.pos.z));

    wgpu::Texture offscreen;
    if (shot) {
        offscreen = make_texture(p_context.device, format, width, height, wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc);
    }
    wgpu::Texture depth = make_texture(p_context.device, DawnPageHost::DEPTH_FORMAT, width, height, wgpu::TextureUsage::RenderAttachment);

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
    // Over the last printed second: the worst frame and where the frames' time went (ms).
    float worst_ms = 0.0f;
    double worst_split[3] = {}; // the worst frame's world, draw, present
    double frame_split[3] = {}; // the last frame's
    double world_ms = 0.0;
    double draw_ms = 0.0;
    double present_ms = 0.0;
    int second_frames = 0;
    const auto ms_since = [](Clock::time_point p_from) { return std::chrono::duration<double, std::milli>(Clock::now() - p_from).count(); };
    while (true) {
        const Clock::time_point now = Clock::now();
        // A shot steps a fixed 60 Hz (the same frames on every run, for a scripted fly).
        const float dt = shot ? 1.0f / 60.0f : std::chrono::duration<float>(now - last).count();
        last = now;
        if (!shot) {
            if (!window.poll()) {
                break;
            }
            if (window.resized && window.width > 0 && window.height > 0) {
                width = window.width;
                height = window.height;
                configure(width, height);
                depth = make_texture(p_context.device, DawnPageHost::DEPTH_FORMAT, width, height, wgpu::TextureUsage::RenderAttachment);
                if (upscaler) {
                    upscaler->resize(width, height);
                }
            }
            // The Godot player's controls (player.gd): mouse look while captured, WASD / arrows along the yaw, space up,
            // shift down, the wheel scales the base speed, V cycles the views (1-3 pick one), right click digs a hole,
            // the left button held sprays.
            camera.yaw -= window.mouse_dx * MOUSE_SENSITIVITY;
            camera.pitch = std::fmax(-PITCH_MAX, std::fmin(PITCH_MAX, camera.pitch - window.mouse_dy * MOUSE_SENSITIVITY));
            base_speed *= std::pow(1.2f, float(window.wheel));
            if (window.button_pressed(MOUSE_RIGHT)) {
                dig.dig(camera.pos, camera.forward());
            }
            if (window.button_held(MOUSE_LEFT)) {
                dig.spray(camera.pos, camera.forward(), dt);
            } else {
                dig.release();
            }
            if (window.pressed(KEY_V)) {
                view = (view + 1) % VIEW_COUNT;
                world.set_debug_view(view);
            }
            for (int v = 0; v < VIEW_COUNT; v++) {
                if (window.pressed(KEY_1 + unsigned(v))) {
                    view = v;
                    world.set_debug_view(view);
                }
            }
            const auto held = [&](unsigned long p_a, unsigned long p_b) { return float(window.held(p_a) || window.held(p_b)); };
            const float move_f = held(KEY_W, KEY_UP) - held(KEY_S, KEY_DOWN);
            const float move_r = held(KEY_D, KEY_RIGHT) - held(KEY_A, KEY_LEFT);
            const float move_u = float(window.held(KEY_SPACE)) - held(KEY_SHIFT_L, KEY_SHIFT_R);
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
#ifdef __EMSCRIPTEN__
        if (shot) {
            emscripten_sleep(0); // the browser's turn: the GPU's callbacks only land between tasks
        }
#endif
        if (!shot || settled >= 0) { // a shot flies from the completed load
            camera.pos.x -= std::sin(camera.yaw) * p_options.fly_speed * dt;
            camera.pos.z -= std::cos(camera.yaw) * p_options.fly_speed * dt;
        }

        const Float3 forward = camera.forward();
        Clock::time_point section = Clock::now();
        world.frame(&camera.pos, &forward);
        device.poll(); // the meshers' work submitted before the draw reads the pages
        frame_split[0] = ms_since(section);
        world_ms += frame_split[0];

        PageView view;
        camera.view_proj(float(width) / float(height), view.view_proj);
        if (upscaler) {
            float unjittered[16];
            std::memcpy(unjittered, view.view_proj, sizeof(unjittered));
            upscaler->begin_frame(unjittered, view.view_proj);
        }
        view.camera = camera.pos;
        view.sun = SUN_DIR.normalized();
        const ov::Color sun_color = SUN_COLOR_SRGB.srgb_to_linear();
        view.sun_color = Float3{ sun_color.r, sun_color.g, sun_color.b };
        view.sun_energy = SUN_ENERGY;
        view.sun_size = float(SUN_SIZE_DEG * WorldCoord::PI / 180.0);
        view.sky_energy = SKY_ENERGY;
        view.sky_exposure = SKY_EXPOSURE;
        view.shadow_m = p_options.shadow_m;
        view.fog_m = float(world.load_radius()) * 0.15f;
        view.forward = forward;
        camera.view_rays(float(width) / float(height), view.view_right, view.view_up);
        const int target_h = upscaler ? upscaler->render_height() : height; // the pages' target
        view.pixel_m = 2.0f * std::tan(FOV_RAD * 0.5f) / float(target_h);
        // A shot stays the same picture; wrapped like Godot's TIME (an hour), so float seconds stay fine.
        view.time_s = shot ? 0.0f : float(std::fmod(ms_since(start) / 1000.0, 3600.0));
        wgpu::SurfaceTexture frame_texture{};
        wgpu::TextureView color;
        if (shot) {
            color = offscreen.CreateView();
        } else {
            section = Clock::now();
            surface.GetCurrentTexture(&frame_texture);
            frame_split[2] = ms_since(section);
            if (frame_texture.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal &&
                    frame_texture.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
                configure(width, height);
                continue;
            }
            color = frame_texture.texture.CreateView();
        }
        section = Clock::now();
        draw_frame(p_context.device, p_context.queue, color, depth.CreateView(), pages, view, upscaler.get(), dt);
        frame_split[1] = ms_since(section);
        draw_ms += frame_split[1];
        frames++;

        if (!shot) {
            section = Clock::now();
            if (window.presents()) {
                surface.Present();
            }
            frame_split[2] += ms_since(section);
            present_ms += frame_split[2];
            second_frames++;
            const float frame_ms = float(ms_since(now)); // input and the 3D included
            if (frame_ms > worst_ms) {
                worst_ms = frame_ms;
                std::memcpy(worst_split, frame_split, sizeof(worst_split));
            }
            const float second = std::chrono::duration<float>(now - last_print).count();
            if (second > 1.0f) {
                const double n = double(second_frames);
                std::printf("fps %.0f  worst %.1f ms (world %.1f draw %.1f present %.1f)  per frame: world %.1f draw %.1f present %.1f ms  pages drawn %d  pending chunks %lld  edits %d  camera %.0f %.0f %.0f\n",
                        n / double(second), double(worst_ms), worst_split[0], worst_split[1], worst_split[2], world_ms / n, draw_ms / n, present_ms / n, pages.drawn_pages(), (long long)world.pending_chunk_count(), dig.edit_count(),
                        double(camera.pos.x), double(camera.pos.y), double(camera.pos.z));
                std::fflush(stdout);
                last_print = now;
                worst_ms = 0.0f;
                world_ms = draw_ms = present_ms = 0.0;
                second_frames = 0;
            }
            continue;
        }
        if (settled < 0 && world.is_load_complete(&camera.pos)) {
            settled = 0;
        }
        if (settled >= 0) {
            settled++;
        }
        if (settled >= (p_options.fly_speed > 0.0f ? SHOT_FLY_FRAMES : SHOT_SETTLE_FRAMES) || frames >= p_options.max_frames) {
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
