#include "x11_window.hpp"

#include "ov/on_error.hpp"

#include <X11/Xlib.h>
#include <X11/Xutil.h>

namespace ov::dawn {

namespace {

Display *xdisplay(void *p_display) {
    return static_cast<Display *>(p_display);
}

} // namespace

X11Window::~X11Window() {
    if (display != nullptr) {
        XDestroyWindow(xdisplay(display), window);
        XCloseDisplay(xdisplay(display));
    }
}

bool X11Window::open(int p_width, int p_height, const char *p_title) {
    Display *d = XOpenDisplay(nullptr);
    OV_ERR_FAIL_COND_V_MSG(d == nullptr, false, "X11Window: no X display (DISPLAY unset?).");
    display = d;
    width = p_width;
    height = p_height;
    const int screen = DefaultScreen(d);
    window = XCreateSimpleWindow(d, RootWindow(d, screen), 0, 0, unsigned(p_width), unsigned(p_height), 0, BlackPixel(d, screen), BlackPixel(d, screen));
    XSelectInput(d, window, KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask | StructureNotifyMask);
    XStoreName(d, window, p_title);
    Atom atom = XInternAtom(d, "WM_DELETE_WINDOW", False);
    delete_atom = atom;
    XSetWMProtocols(d, window, &atom, 1);
    XMapWindow(d, window);
    XFlush(d);
    return true;
}

bool X11Window::poll() {
    Display *d = xdisplay(display);
    resized = false;
    mouse_dx = 0.0f;
    mouse_dy = 0.0f;
    presses.clear();
    bool open = true;
    while (XPending(d) > 0) {
        XEvent event;
        XNextEvent(d, &event);
        switch (event.type) {
            case KeyPress:
            case KeyRelease: {
                const KeySym sym = XLookupKeysym(&event.xkey, 0);
                if (event.type == KeyPress) {
                    presses.insert(sym);
                    keys.insert(sym);
                    open = open && sym != XK_Escape;
                } else {
                    keys.erase(sym);
                }
                break;
            }
            case ButtonPress:
                dragging = dragging || event.xbutton.button == Button1;
                last_x = event.xbutton.x;
                last_y = event.xbutton.y;
                break;
            case ButtonRelease:
                dragging = dragging && event.xbutton.button != Button1;
                break;
            case MotionNotify:
                if (dragging) {
                    mouse_dx += float(event.xmotion.x - last_x);
                    mouse_dy += float(event.xmotion.y - last_y);
                }
                last_x = event.xmotion.x;
                last_y = event.xmotion.y;
                break;
            case ConfigureNotify:
                if (event.xconfigure.width != width || event.xconfigure.height != height) {
                    width = event.xconfigure.width;
                    height = event.xconfigure.height;
                    resized = true;
                }
                break;
            case ClientMessage:
                open = open && Atom(event.xclient.data.l[0]) != Atom(delete_atom);
                break;
            default:
                break;
        }
    }
    return open;
}

wgpu::Surface X11Window::create_surface(const wgpu::Instance &p_instance) const {
    wgpu::SurfaceSourceXlibWindow source{};
    source.display = display;
    source.window = uint64_t(window);
    wgpu::SurfaceDescriptor desc{};
    desc.nextInChain = &source;
    return p_instance.CreateSurface(&desc);
}

} // namespace ov::dawn
