#pragma once

// The viewer's window: plain Xlib (XWayland under a Wayland session), a Dawn surface on it, the keys held and the mouse
// moves while its left button is down.

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
    // Handles the pending events: false once the window is closed (or Escape).
    bool poll();
    wgpu::Surface create_surface(const wgpu::Instance &p_instance) const;

    // A key (an X keysym, lower case: 'w', XK_Shift_L) is held.
    bool held(unsigned long p_keysym) const { return keys.count(p_keysym) != 0; }
    // A key pressed since the last poll.
    bool pressed(unsigned long p_keysym) const { return presses.count(p_keysym) != 0; }

    int width = 0;
    int height = 0;
    bool resized = false; // since the last poll
    float mouse_dx = 0.0f; // pixels moved since the last poll with the left button down
    float mouse_dy = 0.0f;

private:
    void *display = nullptr; // Display *
    unsigned long window = 0;
    unsigned long delete_atom = 0;
    std::set<unsigned long> keys;
    std::set<unsigned long> presses;
    bool dragging = false;
    int last_x = 0;
    int last_y = 0;
};

} // namespace ov::dawn
