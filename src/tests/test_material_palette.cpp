#include "test_common.hpp"

#include "archetype_table.hpp"
#include "chunk_palette.hpp"
#include "material_library.hpp"
#include "page_format.hpp"

#include <algorithm>
#include <random>
#include <utility>
#include <vector>

TEST_CASE("Material::Library: built-ins in order, names unique, image layout") {
    Material::Library lib;
    CHECK(lib.size() == Material::BUILTIN_COUNT);
    CHECK(lib.find("ground") == Material::GROUND);
    CHECK(lib.find("lava") == Material::LAVA);
    CHECK(lib.find("nope") == UINT32_MAX);
    const uint32_t moss = lib.add("moss", Material::Params{ Color(0.1, 0.2, 0.1), 0.8f, Color(0, 0, 0), Material::PATTERN_NONE });
    CHECK(moss == Material::BUILTIN_COUNT);
    const Ref<Image> img = lib.make_image();
    CHECK(img->get_format() == Image::FORMAT_RGBAF);
    CHECK(img->get_width() == Material::TEXELS * Material::IMAGE_MATERIALS_PER_ROW);
    CHECK(img->get_height() == 1);
    const Color a = img->get_pixel(Material::TEXELS * int(moss), 0);
    CHECK(a.g == doctest::Approx(0.2));
    CHECK(a.a == doctest::Approx(0.8));
    const Color lava = img->get_pixel(Material::TEXELS * Material::LAVA + 1, 0);
    CHECK(lava.a == float(Material::PATTERN_LAVA));
    CHECK(img->get_pixel(Material::TEXELS * Material::BARK + 1, 0).a == float(Material::PATTERN_BARK));
}

const std::vector<uint32_t> GROUND_BASE{ Material::GROUND };

TEST_CASE("ChunkPalette: ground first, distinct ids in list order, indices point back") {
    const uint32_t mats[] = { 7, Material::GROUND, 3, 7, 3, 9 };
    std::vector<uint32_t> palette;
    std::vector<uint16_t> index;
    CHECK(ChunkPalette::build(GROUND_BASE, mats, 6, palette, index));
    CHECK(palette == std::vector<uint32_t>{ Material::GROUND, 7, 3, 9 });
    for (uint32_t i = 0; i < 6; i++) {
        CHECK(palette[index[i]] == mats[i]);
    }
    CHECK(ChunkPalette::build(GROUND_BASE, nullptr, 0, palette, index));
    CHECK(palette == GROUND_BASE);
}

TEST_CASE("ChunkPalette: a climate's base palette comes first, a record's material in it takes its entry") {
    const std::vector<uint32_t> base{ Material::GROUND, 40, 41, 42 }; // GROUND, then one material per biome
    const uint32_t mats[] = { 7, 41, Material::GROUND, 7 };
    std::vector<uint32_t> palette;
    std::vector<uint16_t> index;
    CHECK(ChunkPalette::build(base, mats, 4, palette, index));
    CHECK(palette == std::vector<uint32_t>{ Material::GROUND, 40, 41, 42, 7 });
    CHECK(index == std::vector<uint16_t>{ 4, 2, 0, 4 });
    CHECK(ChunkPalette::build(base, nullptr, 0, palette, index));
    CHECK(palette == base);
    // A base must start with GROUND (index 0: a sample no record colours, before the pick).
    const std::vector<uint32_t> bad{ 40, Material::GROUND };
    CHECK_FALSE(ChunkPalette::build(bad, mats, 4, palette, index)); // ERR
}

TEST_CASE("ChunkPalette: past MAX_ENTRIES materials the rest draws as ground") {
    std::vector<uint32_t> mats;
    for (uint32_t m = 1; m <= PageFormat::PALETTE_SIZE + 5; m++) {
        mats.push_back(m);
    }
    std::vector<uint32_t> palette;
    std::vector<uint16_t> index;
    CHECK_FALSE(ChunkPalette::build(GROUND_BASE, mats.data(), uint32_t(mats.size()), palette, index));
    CHECK(palette.size() == ChunkPalette::MAX_ENTRIES);
    CHECK(index[ChunkPalette::MAX_ENTRIES - 2] == ChunkPalette::MAX_ENTRIES - 1); // material 510 is the last that fits
    CHECK(index[ChunkPalette::MAX_ENTRIES - 1] == 0);
    CHECK(index.back() == 0);
}

TEST_CASE("ChunkPalette: INHERIT takes the reserved index and no entry") {
    const uint32_t mats[] = { Material::INHERIT, 7, Material::INHERIT };
    std::vector<uint32_t> palette;
    std::vector<uint16_t> index;
    CHECK(ChunkPalette::build(GROUND_BASE, mats, 3, palette, index));
    CHECK(palette == std::vector<uint32_t>{ Material::GROUND, 7 });
    CHECK(index == std::vector<uint16_t>{ uint16_t(ChunkPalette::INHERIT_INDEX), 1, uint16_t(ChunkPalette::INHERIT_INDEX) });
}

TEST_CASE("PaletteAlloc: aligned blocks, the ground block reserved, frees merge back") {
    PaletteAlloc a;
    a.reset(4096);
    CHECK(a.used() == PaletteAlloc::MIN_BLOCK);
    CHECK(PaletteAlloc::block_size(1) == 16);
    CHECK(PaletteAlloc::block_size(17) == 32);
    CHECK(PaletteAlloc::block_size(512) == 512);

    std::mt19937 rng(3);
    std::vector<std::pair<int64_t, uint32_t>> live;
    for (int step = 0; step < 4000; step++) {
        if (live.empty() || rng() % 3 != 0) {
            const uint32_t n = 1 + rng() % PaletteAlloc::MAX_BLOCK;
            const int64_t off = a.alloc(n);
            if (off < 0) {
                continue; // full
            }
            const uint32_t size = PaletteAlloc::block_size(n);
            CHECK(off % size == 0);
            CHECK(off != PaletteAlloc::GROUND_BLOCK);
            CHECK(off + size <= a.capacity());
            for (const std::pair<int64_t, uint32_t> &l : live) { // no overlap
                const uint32_t ls = PaletteAlloc::block_size(l.second);
                CHECK((off + size <= l.first || l.first + ls <= off));
            }
            live.push_back({ off, n });
        } else {
            const size_t i = rng() % live.size();
            a.free(uint32_t(live[i].first), live[i].second);
            live.erase(live.begin() + i);
        }
    }
    for (const std::pair<int64_t, uint32_t> &l : live) {
        a.free(uint32_t(l.first), l.second);
    }
    CHECK(a.used() == PaletteAlloc::MIN_BLOCK);
    // Everything merged back: the largest blocks fit again, all but the half holding the ground block.
    int big = 0;
    while (a.alloc(PaletteAlloc::MAX_BLOCK) >= 0) {
        big++;
    }
    CHECK(big == 4096 / 512 - 1);
}

TEST_CASE("PaletteAlloc: a base palette of 17 entries (16 biomes) reserves a 32-entry ground block") {
    PaletteAlloc a;
    a.reset(4096, 17);
    CHECK(a.used() == 32);
    const int64_t first = a.alloc(1);
    CHECK(first >= 32); // past the ground block
    a.free(uint32_t(first), 1);
    CHECK(a.used() == 32);
}

TEST_CASE("Material::Library: set adds a material once, then replaces its parameters") {
    Material::Library lib;
    Material::Params p;
    p.pattern = Material::PATTERN_GROUND;
    p.debug = Color(1, 0, 0, 1);
    const uint32_t id = lib.set("biome:a", p);
    CHECK(id == Material::BUILTIN_COUNT);
    p.debug = Color(0, 1, 0, 1);
    CHECK(lib.set("biome:a", p) == id);
    CHECK(lib.size() == Material::BUILTIN_COUNT + 1);
    const Ref<Image> img = lib.make_image();
    CHECK(img->get_pixel(Material::TEXELS * int(id) + 1, 0).a == float(Material::PATTERN_GROUND));
    CHECK(img->get_pixel(Material::TEXELS * int(id) + Material::TEXEL_DEBUG, 0).g == doctest::Approx(1.0));
}

TEST_CASE("Fold material: the record that wins the sample gives it its palette index") {
    using namespace Archetype;
    DefLibrary lib;
    lib.add(make_sphere_type());
    std::vector<TableRecord> records;
    auto sphere = [&](int64_t p_x, float p_radius, Combine p_op) {
        TableRecord rec;
        rec.op = p_op;
        rec.anchor = VoxelPos(p_x, 0, 0);
        rec.params[0] = p_radius;
        finish_record(lib[0], rec, 2.0f, false);
        records.push_back(rec);
    };
    sphere(0, 10.0f, Combine::UNION); // palette 1
    sphere(30, 10.0f, Combine::SMOOTH_UNION); // palette 2
    sphere(0, 4.0f, Combine::SUBTRACT); // palette 3: a hole carved in the first
    sphere(30, 4.0f, Combine::SMOOTH_SUBTRACT); // inherits: a hole carved in the second
    records[1].set_k(2.0f);
    records[3].set_k(1.0f);
    const std::vector<uint32_t> list = { 0, 1, 2, 3 };
    const std::vector<uint16_t> palette = { 1, 2, 3, uint16_t(ChunkPalette::INHERIT_INDEX) };
    const Float4 air{ 0.0f, 0.0f, 0.0f, 1000.0f };
    auto material_at = [&](int64_t p_x) {
        WorldPos p;
        p.voxel = VoxelPos(p_x, 0, 0);
        uint32_t m = 0;
        fold_onto(air, lib, records.data(), list.data(), uint32_t(list.size()), p, 0, true, nullptr, palette.data(), &m);
        return m;
    };
    CHECK(material_at(8) == 1); // inside the first sphere, outside the hole
    CHECK(material_at(31) == 2); // inside the smooth-unioned one
    CHECK(material_at(0) == 3); // in the hole: the cutter's face wins
    CHECK(material_at(5) == 3); // the solid side of the hole's face too
    CHECK(material_at(30) == 2); // an inheriting cutter: its face keeps what it cut
    CHECK(material_at(35) == 2);
    CHECK(material_at(500) == 0); // far: no box test passes, ground

    // A record's material: its type's by default; INHERIT only for the cutters, a union that inherits is ground.
    CHECK(record_material(Material::FROM_TYPE, Material::BARK, Combine::UNION) == Material::BARK);
    CHECK(record_material(Material::LAVA, Material::BARK, Combine::UNION) == Material::LAVA);
    CHECK(record_material(Material::FROM_TYPE, Material::INHERIT, Combine::SUBTRACT) == Material::INHERIT);
    CHECK(record_material(Material::FROM_TYPE, Material::INHERIT, Combine::SMOOTH_SUBTRACT) == Material::INHERIT);
    CHECK(record_material(Material::FROM_TYPE, Material::INHERIT, Combine::SMOOTH_UNION) == Material::GROUND);
    CHECK(record_material(Material::INHERIT, Material::BARK, Combine::UNION) == Material::GROUND);

    // The rule itself: the unions take the record where it is lower, the subtractions where its negation is higher.
    const Float4 r{ 0, 0, 0, 1.0f };
    CHECK(record_wins(r, Float4{ 0, 0, 0, 0.5f }, Combine::UNION));
    CHECK(record_wins(r, Float4{ 0, 0, 0, 0.5f }, Combine::SMOOTH_UNION));
    CHECK_FALSE(record_wins(r, Float4{ 0, 0, 0, 1.0f }, Combine::UNION)); // ties keep what was there
    CHECK(record_wins(r, Float4{ 0, 0, 0, -2.0f }, Combine::SUBTRACT));
    CHECK_FALSE(record_wins(r, Float4{ 0, 0, 0, 2.0f }, Combine::SMOOTH_SUBTRACT));
}
