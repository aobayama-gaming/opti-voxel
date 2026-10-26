#pragma once

// The viewer's window in a browser: the page's <canvas id="canvas">, sized to its CSS box. Input is gathered on the JS
// side (no wasm runs while main is suspended) and handed over by poll(), which first awaits the next animation frame
// (JSPI). A click locks the pointer (the X11 window's capture), the browser's Escape releases it.

#include "window.hpp"

namespace ov::dawn {

class WebCanvas final : public Window {
public:
    bool open(int p_width, int p_height, const char *p_title) override;
    bool poll() override;
    wgpu::Surface create_surface(const wgpu::Instance &p_instance) const override;
    bool presents() const override { return false; }
};

} // namespace ov::dawn
