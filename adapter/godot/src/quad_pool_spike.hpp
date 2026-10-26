#pragma once

#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/classes/node3d.hpp>

#include "chunk_math.hpp"
#include "quad_page_pool.hpp"

using namespace godot;

// Debug/regression node for the GPU -> page path, independent of the terrain:
//  1. a compute shader writes a side x side sheet of quads (page_format.hpp: corners relative to
//     one chunk whose vertex grid holds the whole sheet, plus group headers) into a staging image,
//  2. the staging rows are copied into a pool page texture and the GPU writes the used group count
//     into the page's indirect command buffer,
//  3. page texture + command buffer are read back and checked against a CPU computation,
//  4. two group-aligned quad ranges are killed (whole rows, headers included) and checked again.
// The left half of the sheet (qx < side/2) uses the base winding, the right half swaps p1/p2,
// so a screenshot shows which orientation is front-facing.
class QuadPoolSpike : public Node3D {
    GDCLASS(QuadPoolSpike, Node3D)

protected:
    static void _bind_methods();

private:
    int side = 64;
    int page_capacity = 16384;
    int extra_pages = 0; // pre-allocated pages, to measure pool RAM cost
    Vector3 sheet_origin = Vector3(-72.0f, 342.0f, -150.0f);
    float sheet_size = 40.0f;
    ChunkParam chunk_param; // the page shader's defaults (chunk 32, skirt 2, voxel 0.5)
    ChunkId sheet_chunk; // chunk the sheet's corners are stored relative to

    QuadPagePool *pool = nullptr;
    int page = -1;

    Ref<Mutex> mutex;
    PackedByteArray texture_data[2]; // [0] after write, [1] after kills
    PackedByteArray command_data;
    // 0 waiting for write readback, 1 write verified (screenshots can be taken),
    // 2 waiting for kill readback, 3 done
    int stage = 0;
    bool readback_ready = false;
    Dictionary report;

    // render-thread resources
    RID rt_shader;
    RID rt_pipeline;
    RID rt_staging;
    RID rt_counter;
    RID rt_uniform_set;

    void _rt_run(int p_page);
    void _rt_kill(int p_page);
    void _rt_on_texture(const PackedByteArray &p_data, int p_stage);
    void _rt_on_command(const PackedByteArray &p_data);
    void _verify_write(Dictionary &r);
    void _verify_kills(Dictionary &r);

public:
    // Killed ranges (quads), starting on a group like chunks do: 5 groups, then 16.
    static constexpr uint32_t KILL_A_OFFSET = 128, KILL_A_COUNT = 320;
    static constexpr uint32_t KILL_B_OFFSET = 1024, KILL_B_COUNT = 1024;

    void _ready() override;
    void _process(double p_delta) override;
    void _exit_tree() override;

    void set_side(int p_side) { side = p_side; }
    int get_side() const { return side; }
    void set_extra_pages(int p_pages) { extra_pages = p_pages; }
    int get_extra_pages() const { return extra_pages; }
    void set_page_capacity(int p_cap) { page_capacity = p_cap; }
    int get_page_capacity() const { return page_capacity; }
    void set_debug_view(int p_view);

    // Report so far: write results once stage >= 1, kill results once stage == 3.
    Dictionary get_report();
    int get_stage() const { return stage; }
    // Kills two quad ranges and verifies them (call after taking screenshots of the full sheet).
    void run_kill_test();
};
