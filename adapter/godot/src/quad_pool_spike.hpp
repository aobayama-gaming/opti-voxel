#pragma once

#include <godot_cpp/classes/node3d.hpp>

#include "godot_page_host.hpp"
#include "quad_pool_check.hpp"
#include "rd_device.hpp"

using namespace godot;

// The GPU -> page path check (QuadPoolCheck, quad_pool_check.hpp) drawn in the scene: the main RenderingDevice and a
// GodotPageHost. res://debug/spike.tscn; run it after touching QuadPagePool.
class QuadPoolSpike : public Node3D {
    GDCLASS(QuadPoolSpike, Node3D)

protected:
    static void _bind_methods();

private:
    QuadPoolCheck::Settings settings;
    RDDevice *device = nullptr;
    GodotPageHost *host = nullptr;
    QuadPoolCheck *check = nullptr;

public:
    void _ready() override;
    void _process(double p_delta) override;
    void _exit_tree() override;

    void set_side(int p_side) { settings.side = p_side; }
    int get_side() const { return settings.side; }
    void set_extra_pages(int p_pages) { settings.extra_pages = p_pages; }
    int get_extra_pages() const { return settings.extra_pages; }
    void set_page_capacity(int p_cap) { settings.page_capacity = p_cap; }
    int get_page_capacity() const { return settings.page_capacity; }
    void set_debug_view(int p_view);

    // Report so far: write results once stage >= 1, kill results once stage == 3.
    Dictionary get_report();
    int get_stage() const { return check ? check->stage() : 0; }
    // Kills two quad ranges and verifies them (call after taking screenshots of the full sheet).
    void run_kill_test();
};
