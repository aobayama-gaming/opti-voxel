#pragma once

#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/templates/hash_map.hpp>

#include "chunk_math.hpp"

#include <vector>

using namespace godot;

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
class PageAuditor : public Object {
public:
    struct PageInfo {
        int page = -1;
        struct Range {
            uint32_t offset = 0; // quads
            uint32_t count = 0;
            ChunkId chunk;
            bool operator<(const Range &p_other) const { return offset < p_other.offset; }
        };
        std::vector<Range> live;
    };

    void start(QuadPagePool *p_pool, const std::vector<PageInfo> &p_pages, const ChunkParam &p_chunk_param);
    bool step();
    Dictionary get_report() const { return report; }

private:
    QuadPagePool *pool = nullptr;
    ChunkParam chunk_param;
    std::vector<PageInfo> pages;
    size_t next_request = 0;
    Ref<Mutex> mutex;
    HashMap<int, PackedByteArray> data;
    HashMap<int, uint32_t> groups; // indirect group counts
    Dictionary report;
    bool done = false;

    void _rt_read(int p_page);
    void _rt_on_texture(const PackedByteArray &p_data, int p_page);
    void _rt_on_command(const PackedByteArray &p_data, int p_page);
    void _analyse();
};
