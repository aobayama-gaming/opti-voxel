#pragma once

#include "chunk_math.hpp"
#include "gpu_device.hpp"
#include "page_host.hpp"
#include "quad_page_pool.hpp"

#include "ov/value.hpp"

#include <mutex>

// The GPU -> page path, independent of the terrain, on the main device and a host that draws the page:
//  1. a compute shader writes a side x side sheet of quads (page_format.hpp: corners relative to one chunk whose vertex
//     grid holds the whole sheet, plus group headers) into a staging image,
//  2. the staging rows are copied into a pool page texture and the GPU writes the used group count into the page's
//     indirect command buffer,
//  3. page texture + command buffer are read back and checked against a CPU computation,
//  4. two group-aligned quad ranges are killed (whole rows, headers included) and checked again.
// The left half of the sheet (qx < side/2) uses the base winding, the right half swaps p1/p2, so a screenshot shows
// which orientation is front-facing. Godot: QuadPoolSpike (res://debug/spike.tscn).
class QuadPoolCheck {
public:
    // Killed ranges (quads), starting on a group like chunks do: 5 groups, then 16.
    static constexpr uint32_t KILL_A_OFFSET = 128, KILL_A_COUNT = 320;
    static constexpr uint32_t KILL_B_OFFSET = 1024, KILL_B_COUNT = 1024;

    struct Settings {
        int side = 64;
        int page_capacity = 16384;
        int extra_pages = 0; // pre-allocated pages, to measure the pool's RAM cost
        Float3 sheet_origin{ -72.0f, 342.0f, -150.0f };
        float sheet_size = 40.0f;
    };

    // Main thread. p_host draws the page and outlives destroy()'s work on the device's thread.
    bool start(ov::gpu::Device *p_device, PageHost *p_host, const Settings &p_settings);
    // Main thread, every frame: verifies what came back.
    void step();
    // Kills two quad ranges and verifies them (after screenshots of the full sheet).
    void run_kill_test();
    // 0 waiting for the write's readback, 1 write verified (screenshots can be taken), 2 waiting for the kills'
    // readback, 3 done.
    int stage() const { return current_stage; }
    // The write's results once stage >= 1, the kills' once stage == 3.
    ov::Value report();
    // Frees the pool and the GPU objects, then this object on the device's thread once its readbacks are in.
    void destroy();

private:
    Settings settings;
    ChunkParam chunk_param; // the page shader's defaults (chunk 32, skirt 2, voxel 0.5)
    ChunkId sheet_chunk; // the chunk the sheet's corners are stored relative to
    ov::gpu::Device *device = nullptr;
    QuadPagePool *pool = nullptr;
    int page = -1;
    int current_stage = 0;

    std::mutex mutex;
    ov::bytes texture_data[2]; // [0] after the write, [1] after the kills
    ov::bytes command_data;
    bool readback_ready = false;
    ov::Value results = ov::Value::object();

    // The device's thread.
    ov::gpu::Pipeline rt_pipeline;
    ov::gpu::Texture rt_staging;
    ov::gpu::Buffer rt_counter;
    ov::gpu::BindSet rt_set;
    int rt_readbacks_in_flight = 0;
    bool rt_destroying = false;

    void _rt_run();
    void _rt_kill();
    void _rt_read_texture(int p_stage);
    void _rt_readback_done();
    void _verify_write(ov::Value &r_report);
    void _verify_kills(ov::Value &r_report);
};
