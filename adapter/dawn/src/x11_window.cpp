#include "x11_window.hpp"

#include "ov/on_error.hpp"

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>

namespace ov::dawn {

namespace {

Display *xdisplay(void *p_display) {
    return static_cast<Display *>(p_display);
}

unsigned mouse_bit(unsigned p_xbutton) {
    return p_xbutton == Button1 ? unsigned(MOUSE_LEFT) : p_xbutton == Button3 ? unsigned(MOUSE_RIGHT)
                                                                              : 0u;
}

} // namespace

X11Window::~X11Window() {
    if (display != nullptr) {
        set_capture(false);
        XFreeCursor(xdisplay(display), blank_cursor);
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
    XSelectInput(d, window, KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask | StructureNotifyMask | FocusChangeMask);
    XStoreName(d, window, p_title);
    Atom atom = XInternAtom(d, "WM_DELETE_WINDOW", False);
    delete_atom = atom;
    XSetWMProtocols(d, window, &atom, 1);
    const char empty = 0;
    const Pixmap pixmap = XCreateBitmapFromData(d, window, &empty, 1, 1);
    XColor black{};
    blank_cursor = XCreatePixmapCursor(d, pixmap, pixmap, &black, &black, 0, 0);
    XFreePixmap(d, pixmap);
    XMapWindow(d, window);
    XFlush(d);
    return true;
}

void X11Window::set_capture(bool p_capture) {
    Display *d = xdisplay(display);
    want_capture = p_capture;
    if (p_capture && !captured) {
        const unsigned int mask = ButtonPressMask | ButtonReleaseMask | PointerMotionMask;
        captured = XGrabPointer(d, window, True, mask, GrabModeAsync, GrabModeAsync, window, blank_cursor, CurrentTime) == GrabSuccess;
        if (captured) {
            XDefineCursor(d, window, blank_cursor);
            warp_to_centre();
        }
    } else if (!p_capture && captured) {
        XUngrabPointer(d, CurrentTime);
        XUndefineCursor(d, window);
        captured = false;
        buttons = 0;
    }
}

void X11Window::warp_to_centre() {
    XWarpPointer(xdisplay(display), None, window, 0, 0, 0, 0, width / 2, height / 2);
}

bool X11Window::poll() {
    Display *d = xdisplay(display);
    if (want_capture && !captured) {
        set_capture(true);
    }
    resized = false;
    mouse_dx = 0.0f;
    mouse_dy = 0.0f;
    wheel = 0;
    presses.clear();
    button_presses = 0;
    bool open = true;
    bool warp = false;
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
                    open = open && !(sym == XK_q && (event.xkey.state & ControlMask) != 0);
                    if (sym == XK_Escape) {
                        set_capture(!captured);
                    }
                } else {
                    keys.erase(sym);
                }
                break;
            }
            case ButtonPress:
                wheel += event.xbutton.button == Button4 ? 1 : event.xbutton.button == Button5 ? -1
                                                                                               : 0;
                if (captured) {
                    buttons |= mouse_bit(event.xbutton.button);
                    button_presses |= mouse_bit(event.xbutton.button);
                } else if (event.xbutton.button == Button1) {
                    set_capture(true);
                }
                break;
            case ButtonRelease:
                buttons &= ~mouse_bit(event.xbutton.button);
                break;
            case MotionNotify:
                // The warp back to the centre comes back as a move to the centre: a zero delta.
                if (captured) {
                    mouse_dx += float(event.xmotion.x - width / 2);
                    mouse_dy += float(event.xmotion.y - height / 2);
                    warp = warp || event.xmotion.x != width / 2 || event.xmotion.y != height / 2;
                }
                break;
            case FocusOut:
                // The grab's own focus events (NotifyGrab / NotifyUngrab) are not a focus change.
                if (event.xfocus.mode == NotifyNormal) {
                    set_capture(false);
                    keys.clear();
                }
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
    if (warp && captured) {
        warp_to_centre();
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
