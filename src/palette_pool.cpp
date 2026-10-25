#include "palette_pool.hpp"

#include "material_library.hpp"
#include "quad_page_pool.hpp"

#include <godot_cpp/classes/rd_texture_format.hpp>
#include <godot_cpp/classes/rd_texture_view.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <cstring>

uint64_t PalettePool::batch_bytes(uint32_t p_batch_size) {
    const uint64_t texels = uint64_t(p_batch_size) * PageFormat::PALETTE_TEX_WIDTH; // every chunk a whole palette
    return (uint64_t(p_batch_size) + 1 + texels * BATCH_TEXEL_WORDS) * 4;
}

ov::str PalettePool::glsl_batch() {
    return ov::format(R"glsl(
void write_palette_texel(uint t) {
    if (t < palette_batch[batch.batch_count]) {
        uint at = batch.batch_count + 1u + %du * t;
        imageStore(palette_image, page_palette_texel(palette_batch[at]), page_palette_texel_value(palette_batch[at + 1u], palette_batch[at + 2u]));
    }
}
)glsl",
            BATCH_TEXEL_WORDS);
}

void PalettePool::initialize(uint32_t p_capacity, const std::vector<uint32_t> &p_base) {
    const bool valid = !p_base.empty() && p_base.size() <= PaletteAlloc::MAX_BLOCK;
    if (!valid) {
        ERR_PRINT("PalettePool: a base palette of 1 to MAX_BLOCK entries is required: {GROUND} instead.");
    }
    base = std::make_shared<const std::vector<uint32_t>>(valid ? p_base : std::vector<uint32_t>(1, Material::GROUND));
    alloc.reset(p_capacity, uint32_t(base->size()));
    blocks_live = 0;
    full = 0;
}

PackedByteArray PalettePool::place_batch(std::vector<std::vector<uint32_t>> &p_palettes, std::vector<Block> &r_blocks) {
    const uint32_t chunks = uint32_t(p_palettes.size());
    r_blocks.assign(chunks, Block());
    std::vector<uint32_t> words(chunks + 1, PaletteAlloc::GROUND_BLOCK);
    words[chunks] = 0; // the texel count
    for (uint32_t i = 0; i < chunks; i++) {
        std::vector<uint32_t> &palette = p_palettes[i];
        Block &b = r_blocks[i];
        b.palette = base;
        if (palette.size() <= base->size()) { // the base alone (a chunk palette starts with it): the ground block
            continue;
        }
        const int64_t offset = alloc.alloc(uint32_t(palette.size()));
        if (offset < 0) {
            full++;
            ERR_PRINT_ONCE("PalettePool: no room for a chunk palette (palette_capacity): it draws as its base.");
            continue;
        }
        b.offset = uint32_t(offset);
        blocks_live++;
        words[i] = b.offset;
        for (uint32_t e = 0; e < palette.size(); e += 2) {
            words.push_back(b.offset + e);
            words.push_back(palette[e]);
            words.push_back(e + 1 < palette.size() ? palette[e + 1] : Material::GROUND);
            words[chunks]++;
        }
        b.palette = std::make_shared<const std::vector<uint32_t>>(std::move(palette));
    }
    PackedByteArray bytes;
    bytes.resize(int64_t(words.size()) * 4);
    std::memcpy(bytes.ptrw(), words.data(), words.size() * 4); // little-endian, like the GPU
    return bytes;
}

void PalettePool::release(const Block &p_block) {
    if (p_block.offset == PaletteAlloc::GROUND_BLOCK) {
        return;
    }
    alloc.free(p_block.offset, uint32_t(p_block.palette->size()));
    blocks_live--;
}

Dictionary PalettePool::get_stats() const {
    Dictionary d;
    d["capacity"] = int64_t(alloc.capacity());
    d["used_entries"] = int64_t(alloc.used());
    d["blocks_live"] = blocks_live;
    d["full"] = full;
    d["texture_mb"] = double(alloc.capacity()) * 4.0 / 1048576.0;
    return d;
}

void PalettePool::rt_create(RenderingDevice *p_rd) {
    Ref<RDTextureFormat> format;
    format.instantiate();
    format->set_format(QuadPagePool::TEX_FORMAT); // RGBA16UI, like the pages
    format->set_width(PageFormat::PALETTE_TEX_WIDTH);
    format->set_height(rows());
    format->set_usage_bits(RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT | RenderingDevice::TEXTURE_USAGE_STORAGE_BIT |
            RenderingDevice::TEXTURE_USAGE_CAN_UPDATE_BIT | RenderingDevice::TEXTURE_USAGE_CAN_COPY_TO_BIT); // copy-to: texture_clear
    Ref<RDTextureView> view;
    view.instantiate();
    // Zeroes (an entry reads GROUND), and the ground block's base palette.
    PackedByteArray data;
    data.resize(int64_t(PageFormat::PALETTE_TEX_WIDTH) * rows() * PageFormat::TEXEL_CHANNELS * 2);
    data.fill(0);
    for (size_t e = 0; e < base->size(); e++) {
        PageFormat::encode_palette_entry(reinterpret_cast<uint16_t *>(data.ptrw()), PaletteAlloc::GROUND_BLOCK + uint32_t(e), (*base)[e]);
    }
    TypedArray<PackedByteArray> layers;
    layers.push_back(data);
    texture = p_rd->texture_create(format, view, layers);
    rs_texture = RenderingServer::get_singleton()->texture_rd_create(texture);
}

void PalettePool::rt_free(RenderingDevice *p_rd) {
    if (rs_texture.is_valid()) {
        RenderingServer::get_singleton()->free_rid(rs_texture);
    }
    if (texture.is_valid()) {
        p_rd->free_rid(texture);
    }
    rs_texture = RID();
    texture = RID();
}
