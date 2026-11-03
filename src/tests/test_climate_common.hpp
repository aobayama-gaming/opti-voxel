#pragma once

#include "test_common.hpp"

#include "cascade.hpp"
#include "climate.hpp"
#include "sdf_cascade.hpp"

#include "ov/json.hpp"

// What the climate tests share (test_climate.cpp, test_climate_shape.cpp).

#define CLIMATE_REQUIRE(p_cond) CHECK_OR_RETURN(p_cond)

namespace ClimateTest {

// A JSON text's value (nil if it isn't one).
inline ov::Value json(const ov::str &p_text) {
    ov::str error;
    const ov::Value v = ov::parse_json(p_text, error);
    CHECK(error.empty());
    return v;
}

// A biome's "graph" member playing the template ground (Climate::template_graph) at p_voxel metres, to splice into a
// table's text: a biome without a graph mixes no note (a flat ground).
inline ov::str ground_graph(double p_voxel) {
    return "\"graph\": " + ov::to_json(Climate::template_graph(p_voxel));
}

// Biome p_biome's graph nodes.
inline ov::Value &graph_nodes(ov::Value &r_table, size_t p_biome) {
    return r_table["biomes"].at(p_biome)["graph"]["nodes"];
}

// p_table with every biome's curve node taken out (its output reads the curve's input).
inline ov::Value without_curves(ov::Value p_table) {
    for (size_t b = 0; b < p_table["biomes"].size(); b++) {
        ov::Value &nodes = graph_nodes(p_table, b);
        for (size_t c = 0; c < nodes.size(); c++) {
            if (nodes[c]["kind"].as_str() != "curve") {
                continue;
            }
            for (size_t o = 0; o < nodes.size(); o++) {
                if (nodes[o]["kind"].as_str() == "output" && nodes[o]["height"].as_str() == nodes[c]["id"].as_str()) {
                    nodes.at(o)["height"] = nodes[c]["in"];
                }
            }
            nodes.remove_at(c);
            break;
        }
    }
    return p_table;
}

// Biome p_biome of r_table with a curve node of points p_points (JSON) between its height and its output.
inline void add_curve(ov::Value &r_table, size_t p_biome, const ov::str &p_points) {
    ov::Value &nodes = graph_nodes(r_table, p_biome);
    for (size_t o = 0; o < nodes.size(); o++) {
        if (nodes[o]["kind"].as_str() == "output") {
            ov::Value curve = json(R"({"id": "added_curve", "kind": "curve", "points": )" + p_points + "}");
            curve["in"] = nodes[o]["height"];
            nodes.at(o)["height"] = "added_curve";
            nodes.push_back(curve);
            return;
        }
    }
}

// Every note of r_table with its stretch_y set to p_stretch (1: none, the key removed).
inline void set_stretch(ov::Value &r_table, double p_stretch) {
    for (size_t b = 0; b < r_table["biomes"].size(); b++) {
        ov::Value &nodes = graph_nodes(r_table, b);
        for (size_t n = 0; n < nodes.size(); n++) {
            if (nodes[n]["kind"].as_str() == "note") {
                if (p_stretch == 1.0) {
                    nodes.at(n).erase("stretch_y");
                } else {
                    nodes.at(n)["stretch_y"] = p_stretch;
                }
            }
        }
    }
}

// A graph of one note (p_note: its JSON members after "kind") into the output, through a shape of p_shape's members if
// any: {"graph": ...}'s value.
inline ov::str note_graph(const ov::str &p_note, const ov::str &p_shape = ov::str()) {
    const ov::str shape = p_shape.empty() ? ov::str() : R"({"id": "s", "kind": "shape", "in": "n", )" + p_shape + "}, ";
    return R"("graph": {"nodes": [{"id": "n", "kind": "note", )" + p_note + "}, " + shape + R"({"id": "o", "kind": "output", "height": ")" +
            ov::str(p_shape.empty() ? "n" : "s") + "\"}]}";
}

// An SDFCascade with a climate table (a file, res:// or absolute, or its data) at 5 cm and p_lods LODs; p_skirt_cells
// caps the skirts (ChunkParam::skirt_depth), the default's when negative.
struct ClimateSDF {
    SDFCascade sdf;
    ClimateSDF(const char *p_path, int p_lods, float p_skirt_cells = -1.0f) {
        CHECK(load_table(p_path, sdf.climate_data) == ov::str());
        configure(p_lods, p_skirt_cells);
    }
    ClimateSDF(const ov::Value &p_data, int p_lods, bool p_specialize = true, Climate::Caps p_caps = Climate::Caps()) {
        sdf.climate_data = p_data;
        sdf.specialize = p_specialize;
        sdf.caps = p_caps;
        configure(p_lods, -1.0f);
    }

private:
    void configure(int p_lods, float p_skirt_cells) {
        ChunkParam param;
        param.voxel_size = 0.05f;
        if (p_skirt_cells >= 0.0f) {
            param.skirt_depth = p_skirt_cells;
        }
        sdf.configure(param, p_lods);
    }
};

// Lattice point p_p of Λ_lod as a WorldPos.
inline WorldPos lattice_pos(const Vector3i64 &p_p, int p_lod) {
    return WorldPos{ VoxelPos(p_p.x << p_lod, p_p.y << p_lod, p_p.z << p_lod), Float3() };
}

inline int64_t differing_floats(const float *p_a, const float *p_b, int p_n) {
    int64_t n = 0;
    for (int i = 0; i < p_n; i++) {
        n += ExactNoise::float_bits(p_a[i]) != ExactNoise::float_bits(p_b[i]);
    }
    return n;
}

// Every lattice point of a box on Λ_lod vs Cascade::point_slots, every slot: the box run and the point query agree bit
// for bit (the GPU's chunks and the discoverer's points). p_all_fields: the fields carried everywhere.
inline int64_t box_point_mismatches(const Cascade::Spec &p_spec, const Vector3i64 &p_lo, int p_n, int p_lod, bool p_all_fields = false) {
    Cascade::Box box;
    box.lo[0] = p_lo.x;
    box.lo[1] = p_lo.y;
    box.lo[2] = p_lo.z;
    for (int a = 0; a < 3; a++) {
        box.hi[a] = box.lo[a] + p_n - 1;
    }
    const Cascade::Level l = Cascade::run(p_spec, box, p_lod, nullptr, p_all_fields);
    int64_t bad = 0;
    Cascade::for_each_point(box, [&](const Vector3i64 &p_p, int64_t p_i) {
        float slots[Cascade::SLOTS];
        Cascade::point_slots(p_spec, p_p, p_lod, slots, p_all_fields);
        for (int s = 0; s < Cascade::SLOTS; s++) {
            if (!l.ch[s].empty()) {
                bad += ExactNoise::float_bits(l.ch[s][size_t(p_i)]) != ExactNoise::float_bits(slots[s]);
            }
        }
    });
    return bad;
}

} // namespace ClimateTest
