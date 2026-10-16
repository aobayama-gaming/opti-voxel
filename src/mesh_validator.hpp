#pragma once

#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/templates/hash_map.hpp>

#include "chunk_math.hpp"
#include "chunk_memory.hpp"
#include "sdf_base.hpp"

#include <vector>

using namespace godot;

class GPUMesher;
class QuadPagePool;

// Debug oracle: checks GPU Direct chunks against the legacy GPU mesher (GPU sampling + CPU
// triangulation), triangle by triangle, in world space.
//
//   start(): pick chunks, launch the legacy mesher on them (its own local RenderingDevice, a hidden
//            scenario, trimmed index buffers) and async-read the page textures holding them.
//   step():  call every frame; returns true once the report is ready.
//
// Legacy quantises positions to 1/255 of a cell, so the tolerance is a few of those.
class MeshValidator : public Object {
public:
    struct Target {
        Vector3i chunk_id;
        int page = -1;
        RID page_multimesh;
        uint32_t offset = 0;
        uint32_t count = 0;
    };

    void start(const std::vector<Target> &p_targets, SDFBase *p_sdf, ChunkParam p_chunk_param, int p_lod_numbers,
            QuadPagePool *p_pool, const String &p_dump_dir);
    bool step();
    Dictionary get_report() const { return report; }

private:
    std::vector<Target> targets;
    SDFBase *sdf = nullptr;
    ChunkParam chunk_param;
    QuadPagePool *pool = nullptr;
    String dump_dir;

    GPUMesher *legacy = nullptr;
    RID legacy_scenario;
    VoxelState legacy_state;
    uint64_t start_usec = 0;

    Ref<Mutex> mutex;
    HashMap<int, PackedByteArray> page_data; // page -> texture bytes
    int pages_requested = 0;
    // Pages still to read back. Requested a few per frame: queuing many MB of async texture
    // downloads in one frame overruns the engine's download staging blocks (crashes in 4.6.3).
    std::vector<std::pair<int, RID>> pages_to_request;
    static constexpr int PAGES_PER_FRAME = 2;

    Dictionary report;
    bool done = false;

    void _rt_read_page(int p_page, RID p_multimesh);
    void _rt_on_page(const PackedByteArray &p_data, int p_page);
    void _compare();
};
