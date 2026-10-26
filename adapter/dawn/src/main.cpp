// The Dawn adapter's host: brings the core up on a headless Dawn device.

#include "dawn_context.hpp"

#include "ov/log.hpp"

int main() {
    ov::log_level() = ov::LOG_INFO;
    ov::dawn::DawnContext context;
    return context.initialize() ? 0 : 1;
}
