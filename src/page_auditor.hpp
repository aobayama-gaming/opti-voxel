#pragma once

#include "chunk_math.hpp"
#include "palette_pool.hpp"

#include "ov/bytes.hpp"
#include "ov/containers.hpp"
#include "ov/value.hpp"

#include <mutex>
#include <vector>

class QuadPagePool;

// Debug: reads back pool pages (a few per frame) and classifies every quad slot that the page
// draws, against the CPU's view of which ranges are alive:
//   live      inside a live chunk range, all 4 corners non-zero
//   dead_zero outside live ranges, all zero (killed / never written)   -> costs vertex work only
//   stale     outside live ranges but non-zero                         -> drawn garbage
//   partial   some corners zero, some not                              -> giant triangles to origin
//   broken    inside a live range but has a zero corner
// plus, per live group, whether its header names the chunk that owns it (bad_header otherwise),
// and the longest quad edge among drawn quads.
class PageAuditor {
public:
    struct PageInfo {
        int page = -1;
        struct Range {
            uint32_t offset = 0; // quads
            uint32_t count = 0;
            ChunkId chunk;
            PalettePool::Block palette; // the chunk's (its rows' headers name its offset)
            bool operator<(const Range &p_other) const { return offset < p_other.offset; }
        };
        std::vector<Range> live;
    };

    void start(QuadPagePool *p_pool, const std::vector<PageInfo> &p_pages, const ChunkParam &p_chunk_param);
    bool step();
    ov::Value get_report() const { return report; }
    // Deletes this object on the device's thread once its readbacks in flight have arrived: their
    // callbacks are not skipped when the object is freed. Safe mid-audit.
    void destroy();

private:
    QuadPagePool *pool = nullptr;
    ChunkParam chunk_param;
    std::vector<PageInfo> pages;
    size_t next_request = 0;
    std::mutex mutex;
    ov::HashMap<int, ov::bytes, ov::DefaultHash> data;
    ov::HashMap<int, uint32_t, ov::DefaultHash> groups; // indirect group counts
    ov::Value report = ov::Value::object();
    bool done = false;
    // The device's thread only.
    int rt_readbacks_in_flight = 0;
    bool rt_destroying = false;

    void _rt_read(int p_page);
    void _rt_on_texture(const ov::bytes &p_data, int p_page);
    void _rt_on_command(const ov::bytes &p_data, int p_page);
    void _rt_readback_done();
    void _rt_destroy();
    void _analyse();
};
