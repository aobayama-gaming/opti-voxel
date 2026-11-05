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

// The template ground's graph at p_voxel metres (its world file, SDFCascade::template_path).
inline ov::Value template_graph(double p_voxel) {
    ov::Value table;
    CHECK(SDFCascade::template_table(p_voxel, table) == ov::str());
    const ov::Value &t = table;
    return t["biomes"][size_t(0)]["graph"];
}

// A biome's "graph" member playing the template ground at p_voxel metres, to splice into a table's text: a biome
// without a graph mixes no note (a flat ground).
inline ov::str ground_graph(double p_voxel) {
    return "\"graph\": " + ov::to_json(template_graph(p_voxel));
}

// The template ground's nodes but its output: its notes and their mix "ground", each note through a shape of p_shape's
// members if any (a shape on every octave).
inline ov::Value ground_nodes(double p_voxel, const ov::str &p_shape = ov::str()) {
    const ov::Value graph = template_graph(p_voxel);
    ov::Value nodes = ov::Value::array();
    for (const ov::Value &n : graph["nodes"].elements()) {
        if (n["kind"].as_str() == "output") {
            continue;
        }
        ov::Value node = n;
        for (size_t r = 0; !p_shape.empty() && node["kind"].as_str() == "mix" && r < node["rows"].size(); r++) {
            ov::Value shape = json("{" + p_shape + "}");
            shape["id"] = "s_" + node["rows"][r]["in"].as_str();
            shape["kind"] = "shape";
            shape["in"] = node["rows"][r]["in"];
            node["rows"].at(r)["in"] = shape["id"];
            nodes.push_back(shape);
        }
        nodes.push_back(node);
    }
    return nodes;
}

// A biome's "graph" member: the template ground with each note through a shape of p_shape's members, to splice into a
// table's text.
inline ov::str shaped_ground_graph(double p_voxel, const ov::str &p_shape) {
    ov::Value nodes = ground_nodes(p_voxel, p_shape);
    nodes.push_back(json(R"({"id": "o", "kind": "output", "height": "ground"})"));
    ov::Value graph = ov::Value::object();
    graph["nodes"] = nodes;
    return "\"graph\": " + ov::to_json(graph);
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
    ClimateSDF(const ov::Value &p_data, int p_lods, bool p_specialize = true) {
        sdf.climate_data = p_data;
        sdf.specialize = p_specialize;
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

// The biomes' weights at field values p_fields, op for op as the compiler builds them (graph_compile.cpp's
// weight_nodes): each biome but the default the product of its axes' factors in order, the default the fill of their
// sum, each over the total; one biome weighs 1.
inline void weights(const Climate::Table &p_table, const ExactNoise::Float4 *p_fields, ExactNoise::Float4 r_w[Climate::MAX_BIOMES]) {
    using ExactNoise::Float4;
    for (int b = 0; b < Climate::MAX_BIOMES; b++) {
        r_w[b] = Float4{ 0.0f, 0.0f, 0.0f, 0.0f };
    }
    const size_t n = p_table.biomes.size();
    if (n == 1) {
        r_w[0] = Float4{ 0.0f, 0.0f, 0.0f, 1.0f };
        return;
    }
    Float4 sum{ 0.0f, 0.0f, 0.0f, 0.0f };
    for (size_t b = 0; b + 1 < n; b++) {
        Float4 k{ 0.0f, 0.0f, 0.0f, 1.0f };
        for (const Climate::Axis &a : p_table.biomes[b].at) {
            k = Cascade::dmul(k, Climate::axis_factor(a, p_fields[a.field]));
        }
        r_w[b] = k;
        sum = Cascade::dadd(sum, k);
    }
    r_w[n - 1] = Climate::fill(sum);
    const Float4 rcp = Climate::blend_rcp(Cascade::dadd(sum, r_w[n - 1]));
    for (size_t b = 0; b < n; b++) {
        r_w[b] = Climate::blend_weight(r_w[b], rcp);
    }
}

// A v2 program's notes (Terrain::Op::NOTE): their largest amplitude (a LIN's coefficient of one), coarsest octave and count.
inline float note_amp(const Terrain::LevelProgram &p_prog) {
    float amp = 0.0f;
    for (const Terrain::Node &n : p_prog.nodes) {
        for (size_t i = 0; n.op == Terrain::Op::LIN && i < n.in.size(); i++) {
            amp = p_prog.nodes[size_t(n.in[i])].op == Terrain::Op::NOTE ? ov::max(amp, std::fabs(n.k[i])) : amp;
        }
    }
    return amp;
}

inline int note_top_k(const Terrain::LevelProgram &p_prog) {
    int k = 0;
    for (const Terrain::Node &n : p_prog.nodes) {
        k = n.op == Terrain::Op::NOTE ? ov::max(k, n.band.k) : k;
    }
    return k;
}

inline int note_count(const Terrain::LevelProgram &p_prog) {
    int c = 0;
    for (const Terrain::Node &n : p_prog.nodes) {
        c += n.op == Terrain::Op::NOTE ? 1 : 0;
    }
    return c;
}

// The ground's notes alone: the notes the output reads that no field reads (a field's bands are notes too).
inline int ground_note_count(const Terrain::LevelProgram &p_prog) {
    // Nodes come after their inputs: walk back from the roots.
    const auto reached = [&](const std::vector<int> &p_roots) {
        std::vector<bool> r(p_prog.nodes.size(), false);
        for (const int n : p_roots) {
            r[size_t(n)] = true;
        }
        for (size_t i = p_prog.nodes.size(); i-- > 0;) {
            for (const int in : r[i] ? p_prog.nodes[i].in : std::vector<int>()) {
                r[size_t(in)] = true;
            }
        }
        return r;
    };
    const std::vector<bool> ground = reached({ p_prog.output }), fields = reached(p_prog.fields);
    int c = 0;
    for (size_t i = 0; i < p_prog.nodes.size(); i++) {
        c += p_prog.nodes[i].op == Terrain::Op::NOTE && ground[i] && !fields[i] ? 1 : 0;
    }
    return c;
}

} // namespace ClimateTest
