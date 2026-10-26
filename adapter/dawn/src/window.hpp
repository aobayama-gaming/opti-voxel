#pragma once

// The viewer's window: a Dawn surface, the keys held and, while the mouse is captured, its moves. X11Window on Linux,
// WebCanvas in a browser.

#include <webgpu/webgpu_cpp.h>

#include <set>

namespace ov::dawn {

// The keys the viewer reads, by X keysym (X11Window's native codes; WebCanvas maps the DOM's onto them).
enum Key : unsigned long {
    KEY_SPACE = 0x20,
    KEY_1 = 0x31, // KEY_1 + n for 2, 3...
    KEY_A = 0x61,
    KEY_D = 0x64,
    KEY_S = 0x73,
    KEY_V = 0x76,
    KEY_W = 0x77,
    KEY_LEFT = 0xff51,
    KEY_UP = 0xff52,
    KEY_RIGHT = 0xff53,
    KEY_DOWN = 0xff54,
    KEY_SHIFT_L = 0xffe1,
    KEY_SHIFT_R = 0xffe2,
};

class Window {
public:
    virtual ~Window() = default;

    // False (an error logged) when no window can be made.
    virtual bool open(int p_width, int p_height, const char *p_title) = 0;
    // Waits for the next frame where the platform needs it (a browser's animation frame) and handles the pending
    // events: false once the window is closed.
    virtual bool poll() = 0;
    virtual wgpu::Surface create_surface(const wgpu::Instance &p_instance) const = 0;
    // Whether the frame is shown by surface.Present() (a browser shows the canvas itself).
    virtual bool presents() const { return true; }

    // A key is held (a Key, or any X keysym).
    bool held(unsigned long p_key) const { return keys.count(p_key) != 0; }
    // A key pressed since the last poll.
    bool pressed(unsigned long p_key) const { return presses.count(p_key) != 0; }

    int width = 0;
    int height = 0;
    bool resized = false; // since the last poll
    float mouse_dx = 0.0f; // pixels moved since the last poll, while captured
    float mouse_dy = 0.0f;
    int wheel = 0; // wheel steps since the last poll, up positive

protected:
    std::set<unsigned long> keys;
    std::set<unsigned long> presses;
};

} // namespace ov::dawn
