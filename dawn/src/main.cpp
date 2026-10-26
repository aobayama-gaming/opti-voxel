#include "gpu.hpp"
#include "renderer.hpp"
#include "terrain.hpp"

#include <X11/Xlib.h>
#include <X11/keysym.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

namespace {

struct Input {
    bool forward = false;
    bool back = false;
    bool left = false;
    bool right = false;
    bool up = false;
    bool down = false;
    bool sprint = false;
};

int g_width = 1280;
int g_height = 720;
bool g_running = true;
bool g_captured = true;
int g_center_x = 0;
int g_center_y = 0;

void set_capture(Display *p_display, Window p_window, bool p_capture) {
    g_captured = p_capture;
    if (p_capture) {
        XGrabPointer(p_display, p_window, True, ButtonPressMask | ButtonReleaseMask | PointerMotionMask,
                GrabModeAsync, GrabModeAsync, p_window, None, CurrentTime);
        XWarpPointer(p_display, None, p_window, 0, 0, 0, 0, g_center_x, g_center_y);
        XDefineCursor(p_display, p_window, None);
    } else {
        XUngrabPointer(p_display, CurrentTime);
        XUndefineCursor(p_display, p_window);
    }
    XFlush(p_display);
}

void handle_key(Input &r_input, KeySym p_key, bool p_down) {
    switch (p_key) {
        case XK_w:
        case XK_W:
            r_input.forward = p_down;
            break;
        case XK_s:
        case XK_S:
            r_input.back = p_down;
            break;
        case XK_a:
        case XK_A:
            r_input.left = p_down;
            break;
        case XK_d:
        case XK_D:
            r_input.right = p_down;
            break;
        case XK_space:
            r_input.up = p_down;
            break;
        case XK_Shift_L:
        case XK_Shift_R:
            r_input.sprint = p_down;
            break;
        case XK_Control_L:
        case XK_Control_R:
            r_input.down = p_down;
            break;
        default:
            break;
    }
}

} // namespace

int main(int argc, char **argv) {
    bool upscale = true;
    bool fsr3 = true;
    bool demo_carve = false;
    bool auto_move = false;
    int pixelate = 2;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--no-fsr") == 0) {
            fsr3 = false;
        } else if (std::strcmp(argv[i], "--fsr3") == 0) {
            fsr3 = true;
        } else if (std::strcmp(argv[i], "--no-upscale") == 0) {
            upscale = false;
        } else if (std::strcmp(argv[i], "--carve") == 0) {
            demo_carve = true;
        } else if (std::strcmp(argv[i], "--auto-move") == 0) {
            auto_move = true;
        } else if (std::strncmp(argv[i], "--pixelate=", 11) == 0) {
            pixelate = std::max(1, std::atoi(argv[i] + 11));
        }
    }

    Display *display = XOpenDisplay(nullptr);
    if (display == nullptr) {
        std::fprintf(stderr, "optivoxel-dawn: no X display\n");
        return 1;
    }
    const int screen = DefaultScreen(display);
    Window window = XCreateSimpleWindow(display, RootWindow(display, screen), 0, 0, g_width, g_height, 0,
            BlackPixel(display, screen), BlackPixel(display, screen));
    XStoreName(display, window, "opti-voxel (Dawn)");
    XSelectInput(display, window, KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask | StructureNotifyMask | ExposureMask);
    Atom wm_delete = XInternAtom(display, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(display, window, &wm_delete, 1);
    XMapWindow(display, window);
    XFlush(display);

    g_center_x = g_width / 2;
    g_center_y = g_height / 2;
    set_capture(display, window, true);

    ov::dawn::GpuContext gpu;
    if (!gpu.initialize(display, uint64_t(window), uint32_t(g_width), uint32_t(g_height))) {
        return 1;
    }

    ov::dawn::Terrain terrain;
    terrain.initialize(0.5f, ov::dawn::Float3{ 0.0f, 60.0f, 0.0f });

    ov::dawn::Renderer renderer;
    if (!renderer.initialize(gpu, terrain)) {
        return 1;
    }

    ov::dawn::Camera camera;
    camera.position = ov::dawn::Float3{ 0.0f, 120.0f, 0.0f };
    camera.yaw = 0.0f;
    camera.pitch = -1.2f;
    {
        ov::dawn::Float3 hit{};
        ov::dawn::Float3 normal{};
        if (terrain.raycast(camera.position, ov::dawn::Float3{ 0.0f, -1.0f, 0.0f }, 300.0f, hit, normal)) {
            camera.position = ov::dawn::Float3{ hit.x, hit.y + 30.0f, hit.z };
        } else {
            camera.position = ov::dawn::Float3{ 0.0f, 30.0f, 0.0f };
        }
    }
    camera.pitch = -0.5f;
    if (demo_carve) {
        terrain.carve_sphere(ov::dawn::Float3{ 0.0f, 0.0f, -8.0f }, 3.0f, true, 0.6f);
        terrain.carve_capsule(ov::dawn::Float3{ -6.0f, 0.0f, -14.0f }, ov::dawn::Float3{ 6.0f, 0.0f, -14.0f }, 2.0f, true, 0.6f);
    }


    Input input;
    float move_speed = 18.0f;
    float mouse_sensitivity = 0.0022f;
    int frame = 0;
    auto last = std::chrono::steady_clock::now();
    double fps = 0.0;

    while (g_running) {
        while (XPending(display) > 0) {
            XEvent event;
            XNextEvent(display, &event);
            switch (event.type) {
                case ClientMessage:
                    if (Atom(event.xclient.data.l[0]) == wm_delete) {
                        g_running = false;
                    }
                    break;
                case ConfigureNotify:
                    if (event.xconfigure.width != g_width || event.xconfigure.height != g_height) {
                        g_width = event.xconfigure.width;
                        g_height = event.xconfigure.height;
                        g_center_x = g_width / 2;
                        g_center_y = g_height / 2;
                        renderer.resize(uint32_t(g_width), uint32_t(g_height));
                    }
                    break;
                case KeyPress:
                case KeyRelease: {
                    const KeySym key = XLookupKeysym(&event.xkey, 0);
                    const bool down = event.type == KeyPress;
                    handle_key(input, key, down);
                    if (down && key == XK_Escape) {
                        set_capture(display, window, !g_captured);
                    }
                    if (down && key == XK_f) {
                        fsr3 = !fsr3;
                    }
                    if (down && key == XK_p) {
                        pixelate = pixelate == 1 ? 2 : (pixelate == 2 ? 4 : 1);
                    }
                    if (down && key == XK_c) {
                        terrain.clear_edits();
                    }
                    break;
                }
                case ButtonPress: {
                    if (!g_captured) {
                        set_capture(display, window, true);
                        break;
                    }
                    const float fx = std::sin(camera.yaw) * std::cos(camera.pitch);
                    const float fy = std::sin(camera.pitch);
                    const float fz = -std::cos(camera.yaw) * std::cos(camera.pitch);
                    const ov::dawn::Float3 forward{ fx, fy, fz };
                    ov::dawn::Float3 hit{};
                    ov::dawn::Float3 normal{};
                    if (terrain.raycast(camera.position, forward, 300.0f, hit, normal)) {
                        if (event.xbutton.button == Button1) {
                            // The big capsule: a 2.5 m radius subtract along the view.
                            const ov::dawn::Float3 a{ hit.x + forward.x * 0.2f, hit.y + forward.y * 0.2f, hit.z + forward.z * 0.2f };
                            const ov::dawn::Float3 b{ hit.x + forward.x * 10.0f, hit.y + forward.y * 10.0f, hit.z + forward.z * 10.0f };
                            terrain.carve_capsule(a, b, 2.5f, true, 0.6f);
                        } else if (event.xbutton.button == Button3) {
                            // The dig: a 0.8 m capsule from the surface into the view.
                            const ov::dawn::Float3 a{ hit.x, hit.y, hit.z };
                            const ov::dawn::Float3 b{ hit.x + forward.x * 6.0f, hit.y + forward.y * 6.0f, hit.z + forward.z * 6.0f };
                            terrain.carve_capsule(a, b, 0.8f, true, 0.25f);
                        }
                    }
                    break;
                }
                case ButtonRelease:
                    break;
                case MotionNotify:
                    if (g_captured) {
                        const int dx = event.xmotion.x - g_center_x;
                        const int dy = event.xmotion.y - g_center_y;
                        camera.yaw += float(dx) * mouse_sensitivity;
                        camera.pitch = std::clamp(camera.pitch - float(dy) * mouse_sensitivity, -1.5f, 1.5f);
                        XWarpPointer(display, None, window, 0, 0, 0, 0, g_center_x, g_center_y);
                    }
                    break;
                default:
                    break;
            }
        }

        const auto now = std::chrono::steady_clock::now();
        const float dt = std::min(std::chrono::duration<float>(now - last).count(), 0.1f);
        last = now;
        fps = fps * 0.95 + (1.0 / std::max(dt, 1e-4f)) * 0.05;

        float mx = 0.0f;
        float my = 0.0f;
        float mz = 0.0f;
        if (input.forward) mz += 1.0f;
        if (input.back) mz -= 1.0f;
        if (input.right) mx += 1.0f;
        if (input.left) mx -= 1.0f;
        if (input.up) my += 1.0f;
        if (input.down) my -= 1.0f;
        if (mx != 0.0f || my != 0.0f || mz != 0.0f) {
            const float len = std::sqrt(mx * mx + my * my + mz * mz);
            mx /= len;
            my /= len;
            mz /= len;
            const float cy = std::cos(camera.yaw);
            const float sy = std::sin(camera.yaw);
            const float cp = std::cos(camera.pitch);
            const float sp = std::sin(camera.pitch);
            const float fx = sy * cp;
            const float fy = sp;
            const float fz = -cy * cp;
            const float rx = cy;
            const float rz = sy;
            const float speed = move_speed * (input.sprint ? 4.0f : 1.0f) * dt;
            camera.position.x += (fx * mz + rx * mx) * speed;
            camera.position.y += (fy * mz + my) * speed;
            camera.position.z += (fz * mz + rz * mx) * speed;
        }

        if (auto_move) {
            static float moved = 0.0f;
            moved += dt;
            if (moved < 3.0f) {
                const float cy = std::cos(camera.yaw);
                const float sy = std::sin(camera.yaw);
                camera.position.x += sy * 4.0f * dt;
                camera.position.z -= cy * 4.0f * dt;
            }
        }
        terrain.update(camera.position, 8);
        renderer.render(camera, upscale, fsr3, pixelate);

        if ((frame++ & 15) == 0) {
            char title[256];
            std::snprintf(title, sizeof(title), "opti-voxel (Dawn)  %.0f fps  %s  pixelate x%d  chunks %d  pending %d  edits %d  yaw %.2f pitch %.2f",
                    fps, renderer.fsr_active() ? "FSR3" : (upscale ? "bilinear" : "native"), pixelate,
                    terrain.loaded_chunks(), terrain.pending_chunks(), terrain.edit_count(), camera.yaw, camera.pitch);
            XStoreName(display, window, title);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    renderer.shutdown();
    XCloseDisplay(display);
    return 0;
}
