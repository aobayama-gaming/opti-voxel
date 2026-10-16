#pragma once

#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/templates/hash_map.hpp>

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
// plus the longest quad edge among drawn quads.
class PageAuditor : public Object {
public:
    struct PageInfo {
        int page = -1;
        RID multimesh;
        std::vector<std::pair<uint32_t, uint32_t>> live; // (offset, count) in quads
    };

    void start(QuadPagePool *p_pool, const std::vector<PageInfo> &p_pages);
    bool step();
    Dictionary get_report() const { return report; }

private:
    QuadPagePool *pool = nullptr;
    std::vector<PageInfo> pages;
    size_t next_request = 0;
    Ref<Mutex> mutex;
    HashMap<int, PackedByteArray> data;
    HashMap<int, uint32_t> groups; // indirect group counts
    Dictionary report;
    bool done = false;

    void _rt_read(int p_page, RID p_multimesh);
    void _rt_on_texture(const PackedByteArray &p_data, int p_page);
    void _rt_on_command(const PackedByteArray &p_data, int p_page);
    void _analyse();
};
