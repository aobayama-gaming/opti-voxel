#include "web_canvas.hpp"

#include "ov/on_error.hpp"

#include <emscripten.h>

namespace ov::dawn {

namespace {

constexpr int MAX_KEYS = 32;

// Installs the listeners: Module.ovInput holds the keys held (as X keysyms, window.hpp's Key), those pressed, the
// pointer's moves while locked and the wheel steps since the last take. False without the canvas.
EM_JS(int, ov_canvas_init, (const char *p_title), {
    const canvas = document.getElementById('canvas');
    if (!canvas) {
        return 0;
    }
    document.title = UTF8ToString(p_title);
    const input = { held: new Set(), pressed: new Set(), dx: 0, dy: 0, wheel: 0 };
    Module.ovInput = input;
    const keysym = (e) => {
        const c = e.code;
        if (c.startsWith('Key')) return c.charCodeAt(3) + 32; // lower case, like XLookupKeysym's
        if (c.startsWith('Digit')) return c.charCodeAt(5);
        const named = { Space: 0x20, ArrowLeft: 0xff51, ArrowUp: 0xff52, ArrowRight: 0xff53, ArrowDown: 0xff54, ShiftLeft: 0xffe1, ShiftRight: 0xffe2 };
        return named[c] || 0;
    };
    const locked = () => document.pointerLockElement === canvas;
    window.addEventListener('keydown', (e) => {
        const k = keysym(e);
        if (k && (locked() || document.activeElement === canvas || document.activeElement === document.body)) {
            input.held.add(k);
            if (!e.repeat) input.pressed.add(k);
            e.preventDefault();
        }
    });
    window.addEventListener('keyup', (e) => input.held.delete(keysym(e)));
    window.addEventListener('blur', () => input.held.clear());
    canvas.addEventListener('click', () => { if (!locked()) canvas.requestPointerLock(); });
    document.addEventListener('mousemove', (e) => {
        if (locked()) {
            input.dx += e.movementX;
            input.dy += e.movementY;
        }
    });
    canvas.addEventListener('wheel', (e) => {
        input.wheel += e.deltaY < 0 ? 1 : e.deltaY > 0 ? -1 : 0;
        e.preventDefault();
    }, { passive: false });
    return 1;
});

// Resolves on the next animation frame: the browser shows the frame submitted before and runs the GPU callbacks.
EM_ASYNC_JS(void, ov_next_frame, (), { await new Promise((resolve) => requestAnimationFrame(() => resolve())); });

// The canvas's backing size follows its CSS box; then the input since the last take. Returns the number of held keys
// written to p_held (at most MAX_KEYS), pressed ones into p_pressed (their count in *p_pressed_count).
EM_JS(int, ov_canvas_take, (int *p_size, float *p_mouse, int *p_wheel, int *p_held, int *p_pressed, int *p_pressed_count, int p_max), {
    const canvas = document.getElementById('canvas');
    const w = Math.max(1, Math.round(canvas.clientWidth));
    const h = Math.max(1, Math.round(canvas.clientHeight));
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }
    HEAP32[p_size >> 2] = w;
    HEAP32[(p_size >> 2) + 1] = h;
    const input = Module.ovInput;
    HEAPF32[p_mouse >> 2] = input.dx;
    HEAPF32[(p_mouse >> 2) + 1] = input.dy;
    HEAP32[p_wheel >> 2] = input.wheel;
    input.dx = input.dy = input.wheel = 0;
    let n = 0;
    for (const k of input.pressed) {
        if (n < p_max) HEAP32[(p_pressed >> 2) + n++] = k;
    }
    HEAP32[p_pressed_count >> 2] = n;
    input.pressed.clear();
    n = 0;
    for (const k of input.held) {
        if (n < p_max) HEAP32[(p_held >> 2) + n++] = k;
    }
    return n;
});

} // namespace

bool WebCanvas::open(int p_width, int p_height, const char *p_title) {
    OV_ERR_FAIL_COND_V_MSG(!ov_canvas_init(p_title), false, "WebCanvas: no <canvas id=\"canvas\"> in the page.");
    width = p_width;
    height = p_height;
    poll(); // the canvas's own size
    resized = false;
    return true;
}

bool WebCanvas::poll() {
    ov_next_frame();
    int size[2] = {};
    float mouse[2] = {};
    int held_keys[MAX_KEYS] = {};
    int pressed_keys[MAX_KEYS] = {};
    int pressed_count = 0;
    const int held_count = ov_canvas_take(size, mouse, &wheel, held_keys, pressed_keys, &pressed_count, MAX_KEYS);
    resized = size[0] != width || size[1] != height;
    width = size[0];
    height = size[1];
    mouse_dx = mouse[0];
    mouse_dy = mouse[1];
    keys.clear();
    keys.insert(held_keys, held_keys + held_count);
    presses.clear();
    presses.insert(pressed_keys, pressed_keys + pressed_count);
    return true;
}

wgpu::Surface WebCanvas::create_surface(const wgpu::Instance &p_instance) const {
    wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector source{};
    source.selector = "#canvas";
    wgpu::SurfaceDescriptor desc{};
    desc.nextInChain = &source;
    return p_instance.CreateSurface(&desc);
}

} // namespace ov::dawn
