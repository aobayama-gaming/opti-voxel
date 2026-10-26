#pragma once

// The viewer's window: plain Xlib (XWayland under a Wayland session), a Dawn surface on it, the keys held and, while the
// mouse is captured (hidden, grabbed, warped back to the centre: Godot's MOUSE_MODE_CAPTURED), its moves.

#include <dawn/webgpu_cpp.h>

#include <set>

namespace ov::dawn {

class X11Window {
public:
    X11Window() = default;
    ~X11Window();
    X11Window(const X11Window &) = delete;
    X11Window &operator=(const X11Window &) = delete;

    // False (an error logged) without a display.
    bool open(int p_width, int p_height, const char *p_title);
    // Handles the pending events: false once the window is closed (or Ctrl+Q). Escape toggles the capture, a click
    // in the window captures, losing the focus releases.
    bool poll();
    wgpu::Surface create_surface(const wgpu::Instance &p_instance) const;

    // A key (an X keysym, lower case: 'w', XK_Shift_L) is held.
    bool held(unsigned long p_keysym) const { return keys.count(p_keysym) != 0; }
    // A key pressed since the last poll.
    bool pressed(unsigned long p_keysym) const { return presses.count(p_keysym) != 0; }

    int width = 0;
    int height = 0;
    bool resized = false; // since the last poll
    float mouse_dx = 0.0f; // pixels moved since the last poll, while captured
    float mouse_dy = 0.0f;
    int wheel = 0; // wheel steps since the last poll, up positive

private:
    void *display = nullptr; // Display *
    unsigned long window = 0;
    unsigned long delete_atom = 0;
    std::set<unsigned long> keys;
    std::set<unsigned long> presses;
    unsigned long blank_cursor = 0;
    bool want_capture = true; // the grab can only succeed once the window is mapped: retried every poll
    bool captured = false;

    void set_capture(bool p_capture);
    void warp_to_centre();
};

} // namespace ov::dawn
