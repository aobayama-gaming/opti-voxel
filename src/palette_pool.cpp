#include "palette_pool.hpp"

#include "material_library.hpp"
#include "page_format.hpp"

#include <godot_cpp/classes/rd_texture_format.hpp>
#include <godot_cpp/classes/rd_texture_view.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/core/error_macros.hpp>

uint64_t PalettePool::batch_bytes(uint32_t p_batch_size) {
    const uint64_t texels = uint64_t(p_batch_size) * PageFormat::PALETTE_TEX_WIDTH; // every chunk a whole palette
    return (uint64_t(p_batch_size) + 1 + texels * BATCH_TEXEL_WORDS) * 4;
}

void PalettePool::initialize(uint32_t p_capacity) {
    alloc.reset(p_capacity);
    blocks_live = 0;
    full = 0;
}

PackedByteArray PalettePool::place_batch(std::vector<std::vector<uint32_t>> &p_palettes, uint32_t p_batch_size, std::vector<Block> &r_blocks) {
    r_blocks.assign(p_palettes.size(), Block());
    std::vector<uint32_t> words(p_batch_size + 1, PaletteAlloc::GROUND_BLOCK);
    uint32_t texels = 0;
    for (size_t i = 0; i < p_palettes.size(); i++) {
        std::vector<uint32_t> &palette = p_palettes[i];
        Block &b = r_blocks[i];
        if (palette.size() > 1) {
            const int64_t offset = alloc.alloc(uint32_t(palette.size()));
            if (offset < 0) {
                full++;
                ERR_PRINT_ONCE("PalettePool: no room for a chunk palette (palette_capacity): it draws as ground.");
            } else {
                b.offset = uint32_t(offset);
                b.entries = uint32_t(palette.size());
                blocks_live++;
                words[i] = b.offset;
                for (uint32_t e = 0; e < b.entries; e += 2) {
                    words.push_back((b.offset + e) / 2);
                    words.push_back(palette[e]);
                    words.push_back(e + 1 < b.entries ? palette[e + 1] : Material::GROUND);
                    words.push_back(0);
                    texels++;
                }
            }
        }
        if (b.offset == PaletteAlloc::GROUND_BLOCK) {
            palette.assign(1, Material::GROUND);
        }
        b.palette = std::make_shared<const std::vector<uint32_t>>(std::move(palette));
    }
    words[p_batch_size] = texels;
    PackedByteArray bytes;
    bytes.resize(int64_t(words.size()) * 4);
    for (size_t i = 0; i < words.size(); i++) {
        bytes.encode_u32(int64_t(i) * 4, words[i]);
    }
    return bytes;
}

void PalettePool::release(const Block &p_block) {
    if (p_block.offset == PaletteAlloc::GROUND_BLOCK) {
        return;
    }
    alloc.free(p_block.offset, p_block.entries);
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
    format->set_format(RenderingDevice::DATA_FORMAT_R16G16B16A16_UINT);
    format->set_width(PageFormat::PALETTE_TEX_WIDTH);
    format->set_height(rows());
    format->set_usage_bits(RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT | RenderingDevice::TEXTURE_USAGE_STORAGE_BIT |
            RenderingDevice::TEXTURE_USAGE_CAN_UPDATE_BIT);
    Ref<RDTextureView> view;
    view.instantiate();
    texture = p_rd->texture_create(format, view);
    p_rd->texture_clear(texture, Color(0, 0, 0, 0), 0, 1, 0, 1);
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
