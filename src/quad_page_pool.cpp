#include "quad_page_pool.hpp"

#include "debug_log.hpp"

#include "ov/math.hpp"
#include "ov/on_error.hpp"

#include <cstring>

using namespace ov::gpu;

// The pages' textures: sampled by the host's page shader, copied into (batches, compaction, kills) and out of
// (compaction, the validator's and auditor's readbacks).
static constexpr uint32_t PAGE_USAGE = TEXTURE_SAMPLED | TEXTURE_COPY_DST | TEXTURE_COPY_SRC | TEXTURE_UPDATE;

void QuadPagePool::initialize(Device *p_device, PageHost *p_host, uint32_t p_page_capacity, uint32_t p_initial_pages) {
    OV_ERR_FAIL_COND_MSG(p_page_capacity % QUAD_GROUP != 0, "QuadPagePool: page capacity must be a multiple of 64 quads.");
    device = p_device;
    host = p_host;
    page_capacity = p_page_capacity;
    for (uint32_t i = 0; i < p_initial_pages; i++) {
        free_pages.push_back(_create_page());
    }
}

int QuadPagePool::_create_page(int p_index) {
    int index = p_index;
    if (index < 0) {
        index = int(pages.size());
        pages.push_back(Page());
    }
    pages[index] = Page{ true, false };
    host->create_page(index, get_page_rows());
    // The texture is made on the device's thread; it also zeroes the group count (a fresh page would otherwise draw
    // every group).
    device->post([this, index]() { _rt_create_page(index); });
    // A reused index left trimmed_pages before this call.
    pages_alive_peak = ov::max(pages_alive_peak, _alive_pages());
    return index;
}

void QuadPagePool::_rt_create_page(int p_page) {
    if (!rt_zero_texture.is_valid()) {
        rt_zero_texture = device->create_texture(TEX_FORMAT, TEX_WIDTH, get_page_rows(), PAGE_USAGE);
        device->clear_texture(rt_zero_texture);
    }
    if (int(rt_pages.size()) <= p_page) {
        rt_pages.resize(size_t(p_page) + 1);
    }
    PageRT &page = rt_pages[p_page];
    page.texture = device->create_texture(TEX_FORMAT, TEX_WIDTH, get_page_rows(), PAGE_USAGE);
    device->clear_texture(page.texture);
    page.command = host->rt_attach_page(p_page, page.texture);
    rt_set_group_count(p_page, 0);
}

int QuadPagePool::acquire_page() {
    int page;
    if (free_pages.empty()) {
        int reuse = -1;
        if (!trimmed_pages.empty()) {
            reuse = trimmed_pages.back();
            trimmed_pages.pop_back();
        }
        page = _create_page(reuse);
        growth_events++;
        VLOG(2, "QuadPagePool: grew to ", int64_t(pages.size()), " pages");
    } else {
        page = free_pages.back();
        free_pages.pop_back();
    }
    pages[page].in_use = true;
    pages_in_use++;
    host->set_page_visible(page, true);
    return page;
}

void QuadPagePool::release_page(int p_page) {
    OV_ERR_FAIL_INDEX(p_page, int(pages.size()));
    OV_ERR_FAIL_COND(!pages[p_page].in_use);
    pages[p_page].in_use = false;
    pages_in_use--;
    host->set_page_visible(p_page, false);
    device->post([this, p_page]() { rt_set_group_count(p_page, 0); });
    free_pages.push_back(p_page);
}

int QuadPagePool::trim(int p_keep_free, int p_max) {
    int freed = 0;
    while (int(free_pages.size()) > p_keep_free && freed < p_max) {
        const int index = free_pages.back();
        free_pages.pop_back();
        // The host's frees and the device's posts run in order: everything already posted for this page (resets,
        // kills, copies) runs before its texture goes away.
        host->free_page(index);
        pages[index] = Page();
        device->post([this, index]() { _rt_free_page(index); });
        trimmed_pages.push_back(index);
        pages_trimmed++;
        freed++;
    }
    return freed;
}

void QuadPagePool::_rt_free_page(int p_page) {
    PageRT &page = rt_pages[p_page];
    host->rt_detach_page(p_page);
    device->free(page.texture);
    page = PageRT(); // the command buffer was the host's
}

void QuadPagePool::set_page_aabb(int p_page, const RenderBox &p_aabb) {
    host->set_page_bounds(p_page, p_aabb);
}

QuadPagePool::PageRD QuadPagePool::rt_get_page_rd(int p_page) {
    PageRD out;
    if (p_page < int(rt_pages.size())) {
        out.texture = rt_pages[p_page].texture;
        out.command = rt_pages[p_page].command;
    }
    return out;
}

void QuadPagePool::rt_set_group_count(int p_page, uint32_t p_groups) {
    const PageRD page_rd = rt_get_page_rd(p_page);
    OV_ERR_FAIL_COND_MSG(!page_rd.command.is_valid(), "QuadPagePool: page has no indirect command buffer.");
    OV_ERR_FAIL_COND_MSG(p_groups > get_page_rows(), ov::format("QuadPagePool: group count %d > page rows on page %d", p_groups, p_page));
    device->update_buffer(page_rd.command, INDIRECT_INSTANCE_COUNT_OFFSET, 4, &p_groups);
}

void QuadPagePool::rt_copy_page_from(int p_page, Texture p_src, uint32_t p_src_row) {
    const PageRD page_rd = rt_get_page_rd(p_page);
    OV_ERR_FAIL_COND_MSG(!page_rd.texture.is_valid(), ov::format("QuadPagePool: batch copy into a freed page %d", p_page));
    device->copy_texture(p_src, page_rd.texture, 0, p_src_row, 0, 0, TEX_WIDTH, get_page_rows());
}

void QuadPagePool::rt_kill_range(int p_page, uint32_t p_offset, uint32_t p_count) {
    const PageRD page_rd = rt_get_page_rd(p_page);
    if (!page_rd.texture.is_valid()) {
        return; // trimmed
    }
    OV_ERR_FAIL_COND_MSG(p_offset % QUAD_GROUP != 0, "QuadPagePool: kill ranges must start on a group.");
    const uint32_t first_row = p_offset / QUAD_GROUP;
    const uint32_t rows = (p_count + QUAD_GROUP - 1) / QUAD_GROUP;
    OV_ERR_FAIL_COND_MSG(first_row + rows > get_page_rows(), ov::format("QuadPagePool: kill out of page: page %d rows %d+%d", p_page, first_row, rows));
    device->copy_texture(rt_zero_texture, page_rd.texture, 0, 0, 0, first_row, TEX_WIDTH, rows);
}

void QuadPagePool::rt_copy_rows(int p_src_page, int p_dst_page, uint32_t p_src_row, uint32_t p_dst_row, uint32_t p_rows) {
    OV_ERR_FAIL_INDEX(p_src_page, int(rt_pages.size()));
    OV_ERR_FAIL_INDEX(p_dst_page, int(rt_pages.size()));
    OV_ERR_FAIL_COND(p_src_page == p_dst_page);
    OV_ERR_FAIL_COND_MSG(!rt_pages[p_src_page].texture.is_valid() || !rt_pages[p_dst_page].texture.is_valid(),
            ov::format("QuadPagePool: copy rows on a freed page %d -> %d", p_src_page, p_dst_page));
    OV_ERR_FAIL_COND_MSG(p_src_row + p_rows > get_page_rows() || p_dst_row + p_rows > get_page_rows(),
            ov::format("QuadPagePool: copy rows out of page %d:%d -> %d:%d x %d", p_src_page, p_src_row, p_dst_page, p_dst_row, p_rows));
    device->copy_texture(rt_pages[p_src_page].texture, rt_pages[p_dst_page].texture, 0, p_src_row, 0, p_dst_row, TEX_WIDTH, p_rows);
}

void QuadPagePool::rt_set_hidden(const std::vector<int64_t> &p_writes) {
    const uint32_t rows = get_page_rows();
    constexpr uint32_t ROW_CHANNELS = 2 * PageFormat::TEXEL_CHANNELS; // texels 1 and 2
    rt_hidden_texels.resize(size_t(rows) * ROW_CHANNELS);
    // Rounds of chunks whose rows fit a staging texture: fill it, then copy each chunk's run out. Each round of the
    // call has its own texture: the uploads of one frame all land before its copies run, so a texture updated twice
    // in a frame would hand every copy its last contents.
    const int64_t n = int64_t(p_writes.size());
    size_t round = 0;
    for (int64_t begin = 0; begin < n; round++) {
        if (round == rt_hidden_staging.size()) {
            rt_hidden_staging.push_back(device->create_texture(TEX_FORMAT, 2, rows, TEXTURE_COPY_SRC | TEXTURE_UPDATE));
        }
        const Texture staging = rt_hidden_staging[round];
        uint16_t *texels = rt_hidden_texels.data();
        uint32_t used = 0;
        int64_t end = begin;
        for (; end + HIDDEN_WRITE_WORDS <= n && used + uint32_t(p_writes[end + 2]) <= rows; end += HIDDEN_WRITE_WORDS) {
            const uint64_t header = uint64_t(p_writes[end + 4]);
            uint16_t row_texels[ROW_CHANNELS];
            PageFormat::encode_coverage(row_texels, int32_t(uint32_t(header)), uint32_t(header >> 32), uint64_t(p_writes[end + 3]));
            for (uint32_t r = 0; r < uint32_t(p_writes[end + 2]); r++, used++) {
                std::memcpy(texels + used * ROW_CHANNELS, row_texels, sizeof(row_texels));
            }
        }
        OV_ERR_FAIL_COND_MSG(end == begin, "QuadPagePool: a chunk with more rows than a page.");
        device->update_texture(staging, texels, rt_hidden_texels.size() * sizeof(uint16_t));
        uint32_t at = 0;
        for (int64_t i = begin; i < end; i += HIDDEN_WRITE_WORDS) {
            const uint32_t first_row = uint32_t(p_writes[i + 1]);
            const uint32_t count = uint32_t(p_writes[i + 2]);
            const PageRD page_rd = rt_get_page_rd(int(p_writes[i]));
            if (page_rd.texture.is_valid() && first_row + count <= rows) {
                device->copy_texture(staging, page_rd.texture, 0, at, PageFormat::COVERAGE_TEXEL - 1, first_row, 2, count);
            }
            at += count;
        }
        begin = end;
    }
}

ov::Value QuadPagePool::get_stats() const {
    ov::Value d = ov::Value::object();
    const int64_t alive = _alive_pages();
    d["pages_total"] = alive;
    d["pages_in_use"] = int64_t(pages_in_use);
    d["page_capacity"] = int64_t(page_capacity);
    d["slots_total"] = alive * page_capacity;
    d["growth_events"] = int64_t(growth_events);
    d["pages_trimmed"] = int64_t(pages_trimmed);
    const double page_mb = double(get_page_rows()) * PageFormat::ROW_BYTES / 1048576.0;
    d["vram_mb"] = alive * page_mb;
    d["pages_peak"] = pages_alive_peak;
    d["vram_peak_mb"] = pages_alive_peak * page_mb;
    return d;
}

void QuadPagePool::destroy() {
    for (int i = 0; i < int(pages.size()); i++) {
        if (pages[i].alive) {
            host->free_page(i);
        }
    }
    pages.clear();
    free_pages.clear();
    // The device's resources are freed on its thread, after everything posted before; then we die.
    device->post([this]() { _rt_destroy(); });
}

void QuadPagePool::_rt_destroy() {
    for (int i = 0; i < int(rt_pages.size()); i++) {
        if (rt_pages[i].texture.is_valid()) {
            host->rt_detach_page(i);
            device->free(rt_pages[i].texture);
        }
    }
    rt_pages.clear();
    if (rt_zero_texture.is_valid()) {
        device->free(rt_zero_texture);
    }
    for (const Texture staging : rt_hidden_staging) {
        device->free(staging);
    }
    delete this;
}
