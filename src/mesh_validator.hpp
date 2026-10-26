#pragma once

#include "chunk_math.hpp"
#include "chunk_memory.hpp"
#include "mesh_host.hpp"
#include "palette_pool.hpp"
#include "sdf_base.hpp"

#include "ov/bytes.hpp"
#include "ov/containers.hpp"
#include "ov/value.hpp"

#include <mutex>
#include <vector>

class GPUMesher;
class QuadPagePool;

// Debug oracle: checks GPU Direct chunks against the legacy GPU mesher (GPU sampling + CPU
// triangulation), triangle by triangle, in the chunk's vertex grid (independent of the world origin).
//
//   start(): pick chunks, launch the legacy mesher on them (its own local device, a host that doesn't draw them,
//            trimmed index buffers) and async-read the page textures holding them.
//   step():  call every frame; returns true once the report is ready.
//
// Legacy quantises positions to 1/255 of a cell, so the tolerance is a few of those.
class MeshValidator {
public:
    struct Target {
        ChunkId chunk_id;
        int page = -1;
        uint32_t offset = 0;
        uint32_t count = 0;
        PalettePool::Block palette; // its corners' palette indices name these materials
    };

    // Up to p_count of p_candidates, evenly spaced, up to half of them from the chunks whose palette names more than
    // the ground (they are rare, and material_mismatch needs them).
    static std::vector<Target> pick_targets(const std::vector<Target> &p_candidates, int p_count);

    // p_legacy_host: where the legacy mesher puts its meshes (nothing draws them), alive until destroy().
    void start(const std::vector<Target> &p_targets, SDFBase *p_sdf, ChunkParam p_chunk_param, int p_lod_numbers, QuadPagePool *p_pool,
            MeshHost *p_legacy_host, const ov::str &p_dump_dir, bool p_dual_contouring = false);
    bool step();
    ov::Value get_report() const { return report; }

    // Main thread. Frees the legacy mesher now, then deletes this object on the device's thread once
    // its page readbacks in flight have arrived: their callbacks are not skipped when the object is
    // freed. Safe mid-validation.
    void destroy();

    ~MeshValidator();

private:
    std::vector<Target> targets;
    SDFBase *sdf = nullptr;
    ChunkParam chunk_param;
    QuadPagePool *pool = nullptr;
    ov::str dump_dir;

    GPUMesher *legacy = nullptr;
    MeshHost *legacy_host = nullptr;
    VoxelState legacy_state;
    uint64_t start_usec = 0;

    std::mutex mutex;
    ov::HashMap<int, ov::bytes, ov::DefaultHash> page_data; // page -> texture bytes
    int pages_requested = 0;
    // Pages still to read back. Requested a few per frame: queuing many MB of async texture
    // downloads in one frame overruns the engine's download staging blocks (crashes in 4.6.3).
    std::vector<int> pages_to_request;
    static constexpr int PAGES_PER_FRAME = 2;

    ov::Value report = ov::Value::object();
    bool done = false;

    void _rt_read_page(int p_page);
    void _rt_on_page(const ov::bytes &p_data, int p_page);
    void _rt_destroy();
    void _compare();
    // Stops and frees the legacy mesher (joins its worker: main thread only). Idempotent.
    void _release_legacy();

    // The device's thread only.
    int rt_readbacks_in_flight = 0;
    bool rt_destroying = false;
};
