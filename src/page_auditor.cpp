#include "page_auditor.hpp"

#include "chunk_math.hpp"
#include "debug_log.hpp"
#include "page_format.hpp"
#include "quad_page_pool.hpp"

#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rendering_server.hpp>

#include <algorithm>
#include <cmath>

static constexpr int PAGES_PER_FRAME = 2; // larger async texture downloads per frame crash 4.6.3

void PageAuditor::start(QuadPagePool *p_pool, const std::vector<PageInfo> &p_pages, const ChunkParam &p_chunk_param) {
    pool = p_pool;
    chunk_param = p_chunk_param;
    pages = p_pages;
    mutex.instantiate();
    VLOG(1, "PageAuditor: auditing ", (int64_t)pages.size(), " pages");
}

void PageAuditor::_rt_read(int p_page, RID p_multimesh) {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
    const QuadPagePool::PageRD prd = pool->rt_get_page_rd(p_page, p_multimesh);
    rd->texture_get_data_async(prd.texture, 0, callable_mp(this, &PageAuditor::_rt_on_texture).bind(p_page));
    rd->buffer_get_data_async(prd.command, callable_mp(this, &PageAuditor::_rt_on_command).bind(p_page), 0, 20);
}

void PageAuditor::_rt_on_texture(const PackedByteArray &p_data, int p_page) {
    mutex->lock();
    data[p_page] = p_data;
    mutex->unlock();
}

void PageAuditor::_rt_on_command(const PackedByteArray &p_data, int p_page) {
    mutex->lock();
    groups[p_page] = p_data.decode_u32(QuadPagePool::INDIRECT_INSTANCE_COUNT_OFFSET);
    mutex->unlock();
}

bool PageAuditor::step() {
    if (done) {
        return true;
    }
    for (int i = 0; i < PAGES_PER_FRAME && next_request < pages.size(); i++, next_request++) {
        RenderingServer::get_singleton()->call_on_render_thread(callable_mp(this, &PageAuditor::_rt_read)
                        .bind(pages[next_request].page, pages[next_request].multimesh));
    }
    mutex->lock();
    const bool ready = data.size() == pages.size() && groups.size() == pages.size();
    mutex->unlock();
    if (!ready) {
        return false;
    }
    _analyse();
    done = true;
    return true;
}

void PageAuditor::_analyse() {
    int64_t drawn_slots = 0, live = 0, dead_zero = 0, stale = 0, partial = 0, broken = 0, bad_header = 0;
    const float scale = PageFormat::pos_scale(chunk_param.chunk_size);
    double max_edge = 0.0;
    Vector3 max_edge_at;
    int64_t pages_with_stale = 0, pages_with_partial = 0;
    Array samples;
    for (const PageInfo &info : pages) {
        mutex->lock();
        const PackedByteArray bytes = data[info.page];
        const uint32_t page_groups = groups[info.page];
        mutex->unlock();
        const uint16_t *page = reinterpret_cast<const uint16_t *>(bytes.ptr());
        std::vector<PageInfo::Range> ranges = info.live;
        std::sort(ranges.begin(), ranges.end());
        const uint32_t slots = MIN(page_groups, bytes.size() / PageFormat::ROW_BYTES) * QuadPagePool::QUAD_GROUP;
        drawn_slots += slots;
        size_t r = 0;
        bool page_stale = false, page_partial = false;
        for (uint32_t q = 0; q < slots; q++) {
            while (r < ranges.size() && ranges[r].offset + ranges[r].count <= q) {
                r++;
            }
            const bool in_live = r < ranges.size() && q >= ranges[r].offset;
            // Headers carry the low 32 bits of the chunk id.
            const Vector3i chunk = PageFormat::header_chunk(PageFormat::row(page, q / QuadPagePool::QUAD_GROUP));
            if (in_live && q % QuadPagePool::QUAD_GROUP == 0 && chunk != WorldCoord::lo32_i(ranges[r].chunk)) {
                bad_header++;
            }
            int zero_corners = 0;
            Vector3 p[4]; // in the chunk's vertex grid
            for (int k = 0; k < 4; k++) {
                const uint16_t *c = PageFormat::corner(page, q, k);
                p[k] = PageFormat::decode_local(c, scale);
                zero_corners += PageFormat::is_zero(c) ? 1 : 0;
            }
            if (zero_corners > 0 && zero_corners < 4) {
                partial++;
                page_partial = true;
                if (samples.size() < 10) {
                    Dictionary s;
                    s["page"] = info.page;
                    s["slot"] = q;
                    s["in_live_range"] = in_live;
                    s["zero_corners"] = zero_corners;
                    samples.push_back(s);
                }
            } else if (in_live) {
                if (zero_corners == 4) {
                    broken++;
                } else {
                    live++;
                }
            } else if (zero_corners == 4) {
                dead_zero++;
            } else {
                stale++;
                page_stale = true;
            }
            if (zero_corners == 0) {
                const int lod = chunk.x == 0 ? 0 : WorldCoord::ctz64(WorldCoord::lo32(chunk.x));
                const double cell = std::ldexp(double(chunk_param.voxel_size), lod);
                const double e = cell * MAX(MAX(p[0].distance_to(p[1]), p[0].distance_to(p[2])), MAX(p[3].distance_to(p[1]), p[3].distance_to(p[2])));
                if (e > max_edge) {
                    max_edge = e;
                    max_edge_at = ChunkMath::grid_to_render_lo32(chunk, p[0], chunk_param);
                }
            }
        }
        pages_with_stale += page_stale ? 1 : 0;
        pages_with_partial += page_partial ? 1 : 0;
    }
    report["pages"] = (int64_t)pages.size();
    report["drawn_slots"] = drawn_slots;
    report["live"] = live;
    report["dead_zero"] = dead_zero;
    report["stale"] = stale;
    report["partial"] = partial;
    report["broken"] = broken;
    report["bad_header"] = bad_header;
    report["pages_with_stale"] = pages_with_stale;
    report["pages_with_partial"] = pages_with_partial;
    report["max_quad_edge_m"] = max_edge;
    report["max_quad_edge_at"] = max_edge_at;
    report["partial_samples"] = samples;
    report["ok"] = stale == 0 && partial == 0 && broken == 0 && bad_header == 0;
}
