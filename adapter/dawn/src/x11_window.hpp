#pragma once

// The viewer's window on Linux: plain Xlib (XWayland under a Wayland session). The mouse is captured like Godot's
// MOUSE_MODE_CAPTURED: hidden, grabbed, warped back to the centre.

#include "window.hpp"

namespace ov::dawn {

class X11Window final : public Window {
public:
    X11Window() = default;
    ~X11Window() override;
    X11Window(const X11Window &) = delete;
    X11Window &operator=(const X11Window &) = delete;

    // False (an error logged) without a display.
    bool open(int p_width, int p_height, const char *p_title) override;
    // Handles the pending events without waiting: false once the window is closed (or Ctrl+Q). Escape toggles the
    // capture, a click in the window captures, losing the focus releases.
    bool poll() override;
    wgpu::Surface create_surface(const wgpu::Instance &p_instance) const override;

private:
    void *display = nullptr; // Display *
    unsigned long window = 0;
    unsigned long delete_atom = 0;
    unsigned long blank_cursor = 0;
    bool want_capture = true; // the grab can only succeed once the window is mapped: retried every poll
    bool captured = false;

    void set_capture(bool p_capture);
    void warp_to_centre();
};

} // namespace ov::dawn
