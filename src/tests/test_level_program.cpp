#include "test_climate_common.hpp"

#include "graph_compile.hpp"
#include "level_glsl.hpp"
#include "level_program.hpp"

#include <cmath>
#include <random>
#include <regex>

// Cascade v2 (docs/cascade-v2.md): the compiler's passes and the CPU evaluator.

using namespace Terrain;
using ClimateTest::json;
using ExactNoise::float_bits;

namespace {

// 5 cm voxels, 15 LODs (octaves up to 19), the flat world's cap.
Climate::Ground ground() {
    Climate::Ground g;
    g.voxel_size = 0.05;
    g.k_max = 19;
    g.amp_cap = ExactNoise::exact_pow2(40);
    return g;
}

// A table of one biome whose graph is p_nodes (a JSON list).
ov::Value one_biome(const ov::str &p_nodes) {
    return json(R"({"biomes": [{"name": "b", "graph": {"nodes": )" + p_nodes + "}}]}");
}

// A note node of wavelength p_m metres (octave log2(p_m / 0.05), level 3 less), seed p_seed, p_amp_m metres.
ov::str note(const ov::str &p_id, double p_m, int p_seed, double p_amp_m, const ov::str &p_extra = ov::str()) {
    return ov::format(R"({"id": "%s", "kind": "note", "wavelength_m": %s, "seed": %d, "amp_m": %s%s})", p_id, ov::num(p_m), int64_t(p_seed), ov::num(p_amp_m), p_extra);
}

ov::str compile_table(const ov::Value &p_table, LevelProgram &r_prog, Mode p_mode = Mode::SPECIALISED) {
    Climate::Table table;
    return compile(p_table, ground(), p_mode, table, r_prog);
}

// A test table of the repo (its world file's climate table).
ov::Value test_table(const char *p_path) {
    ov::Value data;
    CHECK(load_table(p_path, data) == ov::str());
    return data;
}

const char *const TABLES[] = { "res://climate/plain.json", "res://climate/shaped_test.json", "res://climate/eight_biomes_test.json", "res://climate/world.json" };

// Every test table of the repo (the CascadeSpike sets' and the GPU checks'), with its v2 hash (the pin below), taken
// when stage E removed the v1 machinery.
struct Pin {
    const char *path;
    uint64_t hash;
};
const Pin PINS[] = {
    { "res://climate/example.json", 14048922634438473485ull },
    { "res://climate/plain.json", 8812832827283825826ull },
    { "res://climate/shaped_test.json", 9767343333093668675ull },
    { "res://climate/fieldless_test.json", 1997498522675870854ull },
    { "res://climate/eight_biomes_test.json", 6180938215590612826ull },
    { "res://climate/notes_test.json", 10293944894637181568ull },
    { "res://climate/programs_test.json", 15627919854828456144ull },
    { "res://climate/graph_test.json", 10291101714587891757ull },
    { "res://climate/caves_test.json", 4148215606099006424ull },
    { "res://climate/world.json", 16376013561544507610ull },
};

Cascade::Box cube(const Vector3i64 &p_lo, int p_n) {
    Cascade::Box b;
    for (int a = 0; a < 3; a++) {
        b.lo[a] = p_lo[a];
        b.hi[a] = p_lo[a] + p_n - 1;
    }
    return b;
}

// The nodes p_node depends on, itself included.
std::vector<bool> depends(const LevelProgram &p_prog, int p_node) {
    std::vector<bool> d(p_prog.nodes.size(), false);
    d[size_t(p_node)] = true;
    for (size_t i = size_t(p_node) + 1; i < p_prog.nodes.size(); i++) {
        for (const int in : p_prog.nodes[i].in) {
            d[i] = d[i] || d[size_t(in)];
        }
    }
    return d;
}

} // namespace

TEST_CASE("Terrain: a node's level is its finest note's, a constant and up have none") {
    LevelProgram prog;
    const ov::str nodes = "[" + note("fine", 3.2, 1, 1.0) + ", " + note("coarse", 25.6, 2, 4.0) +
            R"(, {"id": "s", "kind": "add", "a": "fine", "b": "coarse"}, {"id": "o", "kind": "output", "height": "s"}])";
    CHECK_OR_RETURN(compile_table(one_biome(nodes), prog) == ov::str());
    std::vector<int> notes;
    for (const Node &n : prog.nodes) {
        if (n.op == Op::NOTE) {
            notes.push_back(n.level);
        }
    }
    CHECK(notes == std::vector<int>{ 3, 6 }); // 64 and 512 voxels: octaves 6 and 9
    CHECK(prog.nodes[size_t(prog.output)].level == 3);
    CHECK(prog.top == 6);
    for (const char *kind : { R"({"id": "c", "kind": "constant", "value_m": 2})", R"({"id": "u", "kind": "up"}, {"id": "k", "kind": "constant", "value_m": 2}, {"id": "c", "kind": "min", "a": "u", "b": "k"})" }) {
        CHECK_OR_RETURN(compile_table(one_biome("[" + ov::str(kind) + R"(, {"id": "o", "kind": "output", "height": "c"}])"), prog) == ov::str());
        CHECK(prog.nodes[size_t(prog.output)].level == NO_LEVEL);
        CHECK(prog.top == 0);
    }
    // The flat world: a constant output, the same at every LOD and point.
    CHECK_OR_RETURN(compile_table(one_biome(R"([{"id": "c", "kind": "constant", "value_m": 2}, {"id": "o", "kind": "output", "height": "c"}])"), prog) == ov::str());
    CHECK(point(prog, Vector3i64(3, -7, 11), 4).ground.w == 40.0f);
    CHECK(point(prog, Vector3i64(-5, 2, 0), 0).ground.w == 40.0f);
}

TEST_CASE("Terrain: the gate refuses a warp that isn't strictly coarser than its note") {
    LevelProgram prog;
    const auto table = [](double p_warp_m) {
        return one_biome("[" + note("w", p_warp_m, 7, 1.0) + R"(, {"id": "wp", "kind": "warp", "in": "w", "amount_m": 2}, )" + note("n", 3.2, 1, 1.0, R"(, "warp": "wp")") +
                R"(, {"id": "o", "kind": "output", "height": "n"}])");
    };
    for (const double m : { 3.2, 1.6 }) { // the same octave, a finer one
        const ov::str err = compile_table(table(m), prog);
        CHECK(err.find("node 'n'") != ov::str::npos);
        CHECK(err.find("its warp 'wp' isn't coarser than the note") != ov::str::npos);
        CHECK(prog.nodes.empty());
    }
    CHECK(compile_table(table(6.4), prog) == ov::str());
    // A warp of a ridged (non-linear) coarser node: the gate only asks it be coarser.
    const ov::str ridged = "[" + note("w", 25.6, 7, 1.0) + R"(, {"id": "r", "kind": "shape", "in": "w", "shape": "ridged"}, {"id": "wp", "kind": "warp", "in": "r", "amount_m": 2}, )" +
            note("n", 3.2, 1, 1.0, R"(, "warp": "wp")") + R"(, {"id": "o", "kind": "output", "height": "n"}])";
    CHECK(compile_table(one_biome(ridged), prog) == ov::str());
    // A loop.
    const ov::str loop = R"([{"id": "a", "kind": "add", "a": "b", "b": "b"}, {"id": "b", "kind": "add", "a": "a", "b": "a"}, {"id": "o", "kind": "output", "height": "a"}])";
    CHECK(compile_table(one_biome(loop), prog).find("the graph loops") != ov::str::npos);
}

TEST_CASE("Terrain: a warp's x, y, z ports: one node per axis, each gated") {
    LevelProgram prog;
    const auto warped = [](const ov::str &p_notes, const ov::str &p_warp) {
        return one_biome("[" + p_notes + R"(, {"id": "wp", "kind": "warp", "amount_m": 2)" + p_warp + "}, " + note("n", 3.2, 1, 1.0, R"(, "warp": "wp")") +
                R"(, {"id": "o", "kind": "output", "height": "n"}])");
    };
    // 'in' read with seeds + 0, 1, 2 is the ports on the three notes of those seeds: the same bits.
    const ov::str three = note("a", 25.6, 7, 1.0) + ", " + note("b", 25.6, 8, 1.0) + ", " + note("c", 25.6, 9, 1.0);
    LevelProgram ports;
    CHECK_OR_RETURN(compile_table(warped(note("a", 25.6, 7, 1.0), R"(, "in": "a")"), prog) == ov::str());
    CHECK_OR_RETURN(compile_table(warped(three, R"(, "x": "a", "y": "b", "z": "c")"), ports) == ov::str());
    int64_t bad = 0;
    for (int i = 0; i < 64; i++) {
        const Vector3i64 p(i * 53 - 1700, (i * 7) % 300, i * 31);
        bad += ExactNoise::same_bits(point(prog, p, 0).ground, point(ports, p, 0).ground) ? 0 : 1;
    }
    CHECK(bad == 0);
    // Any node per axis (a ridged one, a constant: no level), but each strictly coarser than the note.
    CHECK(compile_table(warped(three + R"(, {"id": "r", "kind": "shape", "in": "b", "shape": "ridged"}, {"id": "k", "kind": "constant", "value_m": 1})",
                                R"(, "x": "a", "y": "r", "z": "k")"),
                  prog) == ov::str());
    const ov::str fine = compile_table(warped(three + ", " + note("f", 3.2, 4, 1.0), R"(, "x": "a", "y": "f", "z": "c")"), prog);
    CHECK(fine.find("node 'n'") != ov::str::npos);
    CHECK(fine.find("its warp 'wp' isn't coarser than the note (port 'y')") != ov::str::npos);
    // 'in' or all three ports, not both, not some.
    CHECK(compile_table(warped(three, R"(, "in": "a", "x": "a", "y": "b", "z": "c")"), prog).find("the three ports") != ov::str::npos);
    CHECK(compile_table(warped(three, R"(, "x": "a", "y": "b")"), prog).find("the three ports") != ov::str::npos);
}

TEST_CASE("Terrain: a field's ground term is refused") {
    const ov::Value data = json(R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 26214.4, "amp": 1}], "terms": [{"source": "ground", "x": -0.00003}]}],
        "biomes": [{"name": "a", "at": {"t": [0, 1]}}, {"name": "d"}]})");
    LevelProgram prog;
    CHECK(compile_table(data, prog).find("a 'ground' term is gone") != ov::str::npos);
}

TEST_CASE("Terrain: a plain fBm is one channel from top to bottom, in both modes") {
    ov::str nodes = "[";
    ov::str rows;
    for (int k = 3; k <= 14; k++) {
        nodes += note(ov::format("n%d", int64_t(k)), std::ldexp(0.05, k), 9, std::ldexp(0.05 / 8.0, k)) + ", ";
        rows += ov::format(R"(%s{"in": "n%d"})", rows.empty() ? "" : ", ", int64_t(k));
    }
    // A mix of mixes: the inner one is merged into the outer.
    nodes += R"({"id": "inner", "kind": "mix", "rows": [{"in": "n3"}, {"in": "n4", "x": 0.5}]}, )";
    nodes += R"({"id": "m", "kind": "mix", "rows": [)" + rows.substr(rows.find("}, ") + 3) + R"(, {"in": "inner"}]}, {"id": "o", "kind": "output", "height": "m"}])";
    for (const Mode mode : { Mode::SPECIALISED, Mode::GENERIC }) {
        LevelProgram prog;
        CHECK_OR_RETURN(compile_table(one_biome(nodes), prog, mode) == ov::str());
        CHECK(prog.top == 11);
        for (int m = 1; m <= prog.top; m++) {
            CHECK(prog.channels(m) == 1);
        }
        CHECK(prog.channels(0) == 0);
    }
}

TEST_CASE("Terrain: blend factoring pulls a note every biome plays alike out of the weights") {
    const ov::str shared = note("s", 51.2, 3, 2.0);
    const auto graph = [&](int p_seed) {
        return R"({"nodes": [)" + shared + ", " + note("own", 12.8, p_seed, 1.0) +
                R"(, {"id": "m", "kind": "mix", "rows": [{"in": "s"}, {"in": "own"}]}, {"id": "o", "kind": "output", "height": "m"}]})";
    };
    const ov::Value table = json(R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 26214.4, "amp": 1}]}], "biomes": [{"name": "a", "at": {"t": [0.2, 0.5]}, "graph": )" +
            graph(5) + R"(}, {"name": "d", "graph": )" + graph(6) + "}]}");
    LevelProgram spec, gen;
    CHECK_OR_RETURN(compile_table(table, spec, Mode::SPECIALISED) == ov::str());
    CHECK_OR_RETURN(compile_table(table, gen, Mode::GENERIC) == ov::str());
    // The shared note's noise: read past no product with a weight once factored, through each biome's product otherwise.
    const auto through_mul = [](const LevelProgram &p_prog) {
        for (size_t i = 0; i < p_prog.nodes.size(); i++) {
            if (p_prog.nodes[i].op == Op::NOTE && p_prog.nodes[i].band.k == 10) {
                const std::vector<bool> d = depends(p_prog, int(i));
                for (size_t j = 0; j < d.size(); j++) {
                    if (d[j] && p_prog.nodes[j].op == Op::MUL) {
                        return true;
                    }
                }
                return false;
            }
        }
        return true;
    };
    CHECK_FALSE(through_mul(spec));
    CHECK(through_mul(gen));
    // The same ground up to the weights' normalisation error (sum w_b = 1 within a few ulps).
    for (int i = 0; i < 20; i++) {
        const Vector3i64 p(i * 37 - 300, 11 * i, -7 * i);
        CHECK(std::fabs(point(spec, p, 0).ground.w - point(gen, p, 0).ground.w) <= 1e-4f);
    }
}

TEST_CASE("Terrain: a level's channels are a pure function of the point and the level, whatever the run") {
    for (const char *path : { "res://climate/shaped_test.json", "res://climate/world.json" }) {
        LevelProgram prog;
        CHECK_OR_RETURN(compile_table(test_table(path), prog) == ov::str());
        std::vector<Values> a, b;
        run(prog, cube(Vector3i64(-3, 10, 4), 6), 0, &a);
        run(prog, cube(Vector3i64(-1, 4, 0), 4), 2, &b);
        int64_t compared = 0, bad = 0;
        for (const Values &va : a) {
            for (const Values &vb : b) {
                if (va.m != vb.m || va.m <= 2) {
                    continue; // the run to LOD 2 computes every finer node on Λ_2: only the levels above it are shared
                }
                for (size_t n = 0; n < prog.nodes.size(); n++) {
                    if (!va.has(int(n)) || !vb.has(int(n))) {
                        continue;
                    }
                    Cascade::for_each_point(va.box, [&](const Vector3i64 &p_p, int64_t p_i) {
                        if (vb.box.contains(p_p)) {
                            const Float4 x = va.at(int(n), p_i), y = vb.at(int(n), vb.box.index(p_p));
                            bad += ExactNoise::same_bits(x, y) ? 0 : 1;
                            compared++;
                        }
                    });
                }
            }
        }
        CHECK(compared > 0);
        CHECK(bad == 0);
        // V_L: the level above subdivided, then level L's code: the run's last level holds level L + 1's out.
        const Values &last = a.back();
        for (const int n : prog.levels[1].out) {
            CHECK(last.has(n));
        }
    }
}

TEST_CASE("Terrain: a box run and a point query give the same bits, on every test table") {
    for (const char *path : TABLES) {
        for (const Mode mode : { Mode::SPECIALISED, Mode::GENERIC }) {
            LevelProgram prog;
            CHECK_OR_RETURN(compile_table(test_table(path), prog, mode) == ov::str());
            int max_channels = 0;
            for (int m = 0; m <= prog.top; m++) {
                max_channels = ov::max(max_channels, prog.channels(m));
            }
            MESSAGE((std::string(path) + (mode == Mode::GENERIC ? " (generic): " : ": ")), prog.nodes.size(), " nodes, top ", prog.top, ", at most ", max_channels, " channels");
            for (const int lod : { 0, 3 }) {
                const Cascade::Box box = cube(Vector3i64(-1, 5, 2), 3);
                const Values v = run(prog, box, lod);
                int64_t bad = 0;
                Cascade::for_each_point(box, [&](const Vector3i64 &p_p, int64_t p_i) {
                    const Sample s = point(prog, p_p, lod);
                    bad += ExactNoise::same_bits(s.ground, v.at(prog.output, p_i)) ? 0 : 1;
                    bad += s.pick == int(v.at(prog.pick, p_i).w) ? 0 : 1;
                });
                CHECK(bad == 0);
            }
        }
    }
}

// The two modes differ by blend factoring (a note every biome plays alike out of the weights' products): the same ground
// up to the weights' normalisation error, a few float steps of the output's bound. CascadeCheck's --generic=1 checks
// the GPU's generic text against this generic program, bit for bit; this is the bridge to the specialised one.
TEST_CASE("Terrain: the generic and specialised programs of a table agree within float steps of its bound") {
    std::mt19937 rng(29);
    for (const char *path : TABLES) {
        LevelProgram spec, gen;
        CHECK_OR_RETURN(compile_table(test_table(path), spec, Mode::SPECIALISED) == ov::str());
        CHECK_OR_RETURN(compile_table(test_table(path), gen, Mode::GENERIC) == ov::str());
        const size_t o = size_t(spec.output);
        const double bound = ov::max(std::fabs(spec.lo[o]), std::fabs(spec.hi[o]));
        double worst = 0.0;
        int picks = 0;
        for (const int lod : { 0, 2, 5 }) {
            for (int i = 0; i < 32; i++) {
                const int64_t span = int64_t(1) << (18 - lod);
                const Vector3i64 p(int64_t(rng() % uint32_t(span)) - span / 2, int64_t(rng() % 512u) - 256, int64_t(rng() % uint32_t(span)) - span / 2);
                const Sample a = point(spec, p, lod), b = point(gen, p, lod);
                worst = ov::max(worst, std::fabs(double(a.ground.w) - double(b.ground.w)));
                picks += a.pick == b.pick ? 1 : 0;
            }
        }
        MESSAGE(std::string(path), ": worst |specialised - generic| ", worst, " voxels, bound ", bound, "; ", picks, " / 96 picks alike");
        CHECK(worst <= bound * 1e-5 + 1e-4);
        CHECK(picks >= 94); // a pick differs only where two weights tie within their rounding
    }
}

TEST_CASE("Terrain: the step between LOD L and L + 1 stays within the skirt's step, on every test table") {
    std::mt19937 rng(17);
    for (const char *path : TABLES) {
        LevelProgram prog;
        CHECK_OR_RETURN(compile_table(test_table(path), prog) == ov::str());
        for (const int lod : { 0, 1, 3, 6, 10 }) {
            double worst = 0.0;
            for (int i = 0; i < 24; i++) {
                const int64_t span = int64_t(1) << (16 - lod);
                const Vector3i64 p(int64_t(rng() % uint32_t(span)) - span / 2, int64_t(rng() % 256u) - 128, int64_t(rng() % uint32_t(span)) - span / 2);
                const float coarse = point(prog, p, lod + 1).ground.w;
                const float fine = point(prog, Vector3i64(p.x * 2, p.y * 2, p.z * 2), lod).ground.w;
                worst = ov::max(worst, std::fabs(double(fine) - double(coarse)));
            }
            const double step = lod <= prog.top ? prog.step[size_t(lod)] : 0.0;
            CHECK(worst <= step * (1.0 + 1e-6) + 1e-3);
        }
    }
}

TEST_CASE("Terrain: the weights are ClimateTest::weights, op for op, and the pick their largest") {
    // Two fields on the same octaves (one level: there the weights read the fields as ClimateTest::weights does), three
    // biomes, each a note.
    const ov::str bands = R"("bands": [{"wavelength_m": 26214.4, "amp": 1}, {"wavelength_m": 13107.2, "amp": 0.5}])";
    const auto graph = [](int p_seed) { return R"("graph": {"nodes": [)" + note("n", 3.2, p_seed, 1.0) + R"(, {"id": "o", "kind": "output", "height": "n"}]})"; };
    const ov::Value data = json(R"({"fields": [{"name": "t", "seed": 4, )" + bands + R"(}, {"name": "h", "seed": 5, )" + bands + R"(}], "biomes": [)" +
            R"({"name": "a", "at": {"t": [0.2, 0.4], "h": [-0.1, 0.5]}, )" + graph(1) + R"(}, {"name": "b", "at": {"t": [-0.3, 0.3]}, )" + graph(2) + R"(}, {"name": "d", )" + graph(3) +
            "}]}");
    LevelProgram prog;
    Climate::Table table;
    CHECK_OR_RETURN(compile(data, ground(), Mode::SPECIALISED, table, prog) == ov::str());
    const Node &pick = prog.nodes[size_t(prog.pick)];
    CHECK_OR_RETURN(pick.op == Op::PICK);
    CHECK_OR_RETURN(pick.in.size() == table.biomes.size());
    // Each field's node: an axis's input (the axis of the same centre and span).
    std::vector<int> field(table.fields.size(), -1);
    for (const Climate::Biome &b : table.biomes) {
        for (const Climate::Axis &a : b.at) {
            for (const Node &n : prog.nodes) {
                if (n.op == Op::AXIS && n.k[0] == a.centre && n.k[1] == a.inv_span) {
                    field[size_t(a.field)] = n.in[0];
                }
            }
        }
    }
    const int level = prog.nodes[size_t(pick.in[0])].level;
    std::vector<Values> levels;
    run(prog, cube(Vector3i64(-2, 0, 3), 4), 0, &levels);
    const Values &v = levels[size_t(prog.top - level)];
    CHECK_OR_RETURN(v.m == level);
    int64_t bad = 0;
    Cascade::for_each_point(v.box, [&](const Vector3i64 &p_p, int64_t p_i) {
        Float4 f[Climate::MAX_FIELDS] = {};
        for (size_t i = 0; i < field.size(); i++) {
            f[i] = field[i] >= 0 ? v.at(field[i], p_i) : Float4{};
        }
        Float4 w[Climate::MAX_BIOMES];
        ClimateTest::weights(table, f, w);
        for (size_t b = 0; b < table.biomes.size(); b++) {
            bad += ExactNoise::same_bits(w[b], v.at(pick.in[b], p_i)) ? 0 : 1;
        }
        bad += int(v.at(prog.pick, p_i).w) == Climate::pick_of(int(table.biomes.size()), w, p_p, v.m) ? 0 : 1;
    });
    CHECK(bad == 0);
}

TEST_CASE("Terrain: a ridged multifractal parent compiles and weighs its note") {
    // in: a 3.2 m note; parent: a ridged 102.4 m note (its noise shaped without the mean, times 4 m); the weight ramps
    // from 0 at -0.8 m to 1 at 0.4 m of the parent.
    const ov::str nodes = "[" + note("in", 3.2, 1, 1.0) + ", " + note("p", 102.4, 2, 4.0) +
            R"(, {"id": "r", "kind": "shape", "in": "p", "shape": "ridged"}, {"id": "mf", "kind": "multifractal", "in": "in", "parent": "r", "low_m": -0.8, "high_m": 0.4}, )" +
            R"({"id": "o", "kind": "output", "height": "mf"}])";
    LevelProgram prog;
    CHECK_OR_RETURN(compile_table(one_biome(nodes), prog) == ov::str());
    const Node &mf = prog.nodes[size_t(prog.output)];
    CHECK_OR_RETURN(mf.op == Op::MUL);
    const Node &ramp = prog.nodes[size_t(mf.in[1])];
    CHECK_OR_RETURN(ramp.op == Op::CURVE);
    CHECK(prog.nodes[size_t(ramp.in[0])].level == 8); // the parent's level (octave 11), carried to the note's
    CHECK(mf.level == 3);
    int zeros = 0, ones = 0;
    int64_t bad = 0;
    for (int i = 0; i < 200; i++) {
        const Vector3i64 p(i * 53 - 5000, (i * 7) % 300, i * 31);
        std::vector<Values> levels;
        run(prog, Cascade::point_box(p), 0, &levels);
        // The ramp on the parent's lattice, of the parent's value there.
        const Values &parent = levels[size_t(prog.top - 8)];
        Cascade::for_each_point(parent.box, [&](const Vector3i64 &, int64_t p_i) {
            const float w = parent.at(mf.in[1], p_i).w;
            const double expected = ov::clamp((double(parent.at(ramp.in[0], p_i).w) * 0.05 + 0.8) / 1.2, 0.0, 1.0);
            bad += std::fabs(double(w) - expected) <= 1e-5 ? 0 : 1;
            zeros += w == 0.0f ? 1 : 0;
            ones += w == 1.0f ? 1 : 0;
        });
        // The product on the note's lattice, the ramp subdivided to it.
        const Values &v = levels[size_t(prog.top - 3)];
        Cascade::for_each_point(v.box, [&](const Vector3i64 &, int64_t p_i) {
            bad += ExactNoise::same_bits(v.at(prog.output, p_i), Cascade::dmul(v.at(mf.in[0], p_i), v.at(mf.in[1], p_i))) ? 0 : 1;
        });
    }
    CHECK(bad == 0);
    CHECK(zeros > 0);
    CHECK(ones > 0);
}

// Each test table's bits, pinned: the ground and the pick of its specialised program at points of every few LODs, near 0
// and +-2^60 (a change of the compiler or an op that moves one bit fails; re-pin on purpose, saying why).
TEST_CASE("Terrain: every test table's program bits are pinned") {
    for (const Pin &pin : PINS) {
        LevelProgram prog;
        CHECK_OR_RETURN(compile_table(test_table(pin.path), prog) == ov::str());
        uint64_t hash = 0xCBF29CE484222325ull;
        auto mix = [&hash](uint32_t p_v) { hash = (hash ^ p_v) * 0x100000001B3ull; };
        for (const int64_t base : { int64_t(0), int64_t(12345), (int64_t(1) << 60) - 4000, -(int64_t(1) << 60) + 77 }) {
            for (int lod = 0; lod < 15; lod += 3) {
                const int64_t step = int64_t(1) << lod;
                for (int i = -2; i <= 2; i++) {
                    const Sample s = point(prog, Vector3i64(WorldCoord::floor_div(base, step) + i, 5 * i, 2 - i), lod);
                    mix(float_bits(s.ground.x));
                    mix(float_bits(s.ground.y));
                    mix(float_bits(s.ground.z));
                    mix(float_bits(s.ground.w));
                    mix(uint32_t(s.pick));
                }
            }
        }
        CAPTURE(std::string(pin.path));
        MESSAGE(std::string(pin.path), ": hash ", hash);
        CHECK(hash == pin.hash);
    }
}

// A point query's targets need not be carried to its lattice: a field (read by its biomes' axes on its own level, so not
// a channel below it) and a single biome's weight (a constant no level computes) at LOD 0 are the field subdivided and
// the constant, whatever the queries before.
TEST_CASE("Terrain: a point query's targets are their values at its lattice, carried or not") {
    for (const char *path : { "res://climate/eight_biomes_test.json", "res://climate/world.json", "res://climate/plain.json" }) {
        LevelProgram prog;
        CHECK_OR_RETURN(compile_table(test_table(path), prog) == ov::str());
        std::vector<int> targets = prog.fields;
        targets.insert(targets.end(), prog.weights.begin(), prog.weights.end());
        const int count = int(targets.size());
        std::vector<Float4> fine(targets.size()), coarse(targets.size()), other(targets.size());
        int64_t bad = 0, compared = 0;
        for (int i = 0; i < 16; i++) {
            const Vector3i64 p(i * 37 - 300, i * 3 - 20, 11 - i * 29);
            for (int t = 0; t < count; t++) {
                const int level = prog.nodes[size_t(targets[size_t(t)])].level;
                const int m = level == NO_LEVEL ? 0 : level;
                point_nodes(prog, Vector3i64(p.x + 1000, p.y, p.z), 0, targets.data(), count, other.data()); // another point first
                point_nodes(prog, Vector3i64(p.x << m, p.y << m, p.z << m), 0, &targets[size_t(t)], 1, &fine[size_t(t)]);
                point_nodes(prog, p, m, &targets[size_t(t)], 1, &coarse[size_t(t)]);
                bad += fine[size_t(t)].w == coarse[size_t(t)].w ? 0 : 1; // a copy on the shared point: the same value
                compared++;
            }
        }
        CAPTURE(std::string(path));
        CHECK(compared > 0);
        CHECK(bad == 0);
        if (prog.biomes == 1) {
            Float4 w;
            point_nodes(prog, Vector3i64(5, 6, 7), 0, prog.weights.data(), 1, &w);
            CHECK(w.w == 1.0f);
        }
    }
}

// An unbounded change (a note times up) times a node that doesn't change is still unbounded, not inf * 0 = NaN: a NaN
// step under a min or max would read as 0 (ov::max picks the other side) and leave the seam without a skirt.
TEST_CASE("Terrain: a product's step with an unbounded factor stays unbounded") {
    const ov::str nodes = "[" + note("n1", 3.2, 1, 1.0) + ", " + note("n2", 25.6, 2, 1.0) +
            R"(, {"id": "u", "kind": "up"}, {"id": "m1", "kind": "multiply", "a": "u", "b": "n1"}, {"id": "m2", "kind": "multiply", "a": "m1", "b": "n2"},)" +
            R"( {"id": "c", "kind": "constant", "value_m": 10}, {"id": "lo", "kind": "min", "a": "m2", "b": "c"}, {"id": "o", "kind": "output", "height": "lo"}])";
    LevelProgram prog;
    CHECK_OR_RETURN(compile_table(one_biome(nodes), prog) == ov::str());
    CHECK_OR_RETURN(prog.step.size() > 3);
    CHECK(prog.step[3] == INFINITY); // n1's level: its change times an unbounded up
}

// One inlined tp_note per note took the NVIDIA compiler minutes and gigabytes on the example table's 80 notes: the
// specialised text reads every level's notes in one loop, its looped notes one range of the band table.
TEST_CASE("Terrain: the specialised text calls tp_note once, whatever the note count") {
    for (const Pin &pin : PINS) {
        LevelProgram prog;
        CHECK_OR_RETURN(compile_table(test_table(pin.path), prog) == ov::str());
        ov::str err;
        const GpuLayout layout = gpu_layout(prog, err);
        Cascade::ShaderParams params;
        const ov::str text = level_glsl(prog, layout, params);
        size_t calls = 0;
        for (size_t at = text.find("tp_note("); at != ov::str::npos; at = text.find("tp_note(", at + 1)) {
            calls++;
        }
        INFO(pin.path);
        CHECK(calls == 1);
        int64_t notes = 0;
        for (const Node &n : prog.nodes) {
            notes += n.op == Op::NOTE;
        }
        CHECK(int64_t(note_order(prog).size()) == notes);
    }
}

namespace {

// p_prog without its fold: every node computed on every level, as before the constant relaxation.
LevelProgram unfolded(const LevelProgram &p_prog) {
    LevelProgram u = p_prog;
    for (Node &n : u.nodes) {
        n.const_lod = NO_LEVEL;
        n.live_lod = NO_LEVEL;
    }
    return u;
}

// The nodes a run's last level of LOD p_lod computes (not folded, read).
int64_t last_computed(const LevelProgram &p_prog, int p_lod) {
    int64_t n = 0;
    for (const int c : p_prog.last_code(p_lod)) {
        const Node &node = p_prog.nodes[size_t(c)];
        n += p_lod < node.const_lod && p_lod <= node.live_lod ? 1 : 0;
    }
    return n;
}

} // namespace

// The constant relaxation changes what a run computes, never its bits: a box run and a point query of every LOD, folded
// and not, on every test table in both modes.
TEST_CASE("Terrain: the folded program gives the unfolded bits at every LOD, on every test table") {
    for (const Pin &pin : PINS) {
        for (const Mode mode : { Mode::SPECIALISED, Mode::GENERIC }) {
            LevelProgram prog;
            CHECK_OR_RETURN(compile_table(test_table(pin.path), prog, mode) == ov::str());
            const LevelProgram plain = unfolded(prog);
            int64_t bad = 0;
            ov::str counts;
            for (int lod = 0; lod <= prog.top + 1; lod++) {
                const Cascade::Box box = cube(Vector3i64(-1, 3 - lod, 2), 3);
                const Values a = run(prog, box, lod), b = run(plain, box, lod);
                Cascade::for_each_point(box, [&](const Vector3i64 &p_p, int64_t p_i) {
                    bad += ExactNoise::same_bits(a.at(prog.output, p_i), b.at(prog.output, p_i)) ? 0 : 1;
                    bad += float_bits(a.at(prog.pick, p_i).w) == float_bits(b.at(prog.pick, p_i).w) ? 0 : 1;
                    const Sample s = point(prog, p_p, lod);
                    bad += ExactNoise::same_bits(s.ground, b.at(prog.output, p_i)) && s.pick == int(b.at(prog.pick, p_i).w) ? 0 : 1;
                });
                counts += ov::format(" %d/%d", last_computed(prog, lod), last_computed(plain, lod));
            }
            CAPTURE(std::string(pin.path));
            MESSAGE((std::string(pin.path) + (mode == Mode::GENERIC ? " (generic)" : "")), ": last level nodes per LOD, folded/unfolded:", std::string(counts.c_str()));
            CHECK(bad == 0);
        }
    }
}

// The generic text's placeholders are all substituted (one inside another, "STOP" read as "S" + TOP's offset, once
// broke the GPU compile with an undeclared identifier).
TEST_CASE("Terrain: the generic text keeps none of its placeholders") {
    for (const Pin &pin : PINS) {
        LevelProgram prog;
        CHECK_OR_RETURN(compile_table(test_table(pin.path), prog, Mode::GENERIC) == ov::str());
        ov::str err;
        const GpuLayout layout = gpu_layout(prog, err);
        Cascade::ShaderParams params;
        const std::string text = level_glsl(prog, layout, params).c_str();
        INFO(pin.path);
        for (const char *word : { "CASES", "CH_GET", "PUT_OUTPUT", "PUT_CHANNEL", "ROOT_PICK", "MAX_REGISTERS", "MAX_LEVEL", "TOP", "DIR", "OUTPUT", "PICK", "FINEST" }) {
            CHECK_MESSAGE(text.find(word) == std::string::npos, word);
        }
        std::smatch glued; // a placeholder replaced inside a longer one: an upper-case word glued to a number
        CHECK_MESSAGE(!std::regex_search(text, glued, std::regex("\\b[A-Z_]+[0-9]+\\b")), (glued.empty() ? std::string() : glued.str()));
    }
}

// A multifractal's product on a run coarser than its note is 0 whatever its parent: the parent's ramp isn't computed
// there.
TEST_CASE("Terrain: a product by an absent note folds to 0 and its other factor goes unread") {
    const ov::str nodes = "[" + note("in", 3.2, 1, 1.0) + ", " + note("p", 102.4, 2, 4.0) +
            R"(, {"id": "r", "kind": "shape", "in": "p", "shape": "ridged"}, {"id": "mf", "kind": "multifractal", "in": "in", "parent": "r", "low_m": -0.8, "high_m": 0.4}, )" +
            R"({"id": "c", "kind": "constant", "value_m": 2}, {"id": "s", "kind": "add", "a": "mf", "b": "c"}, {"id": "o", "kind": "output", "height": "s"}])";
    for (const Mode mode : { Mode::SPECIALISED, Mode::GENERIC }) {
        LevelProgram prog;
        CHECK_OR_RETURN(compile_table(one_biome(nodes), prog, mode) == ov::str());
        int mf = -1;
        for (size_t i = 0; i < prog.nodes.size(); i++) {
            mf = prog.nodes[i].op == Op::MUL && prog.nodes[i].level == 3 ? int(i) : mf;
        }
        CHECK_OR_RETURN(mf >= 0);
        const Node &mul = prog.nodes[size_t(mf)];
        CHECK(mul.const_lod == 4); // its note (level 3) absent from LOD 4
        CHECK(float_bits(mul.folded) == 0u);
        const int ramp = prog.nodes[size_t(mul.in[0])].op == Op::CURVE ? mul.in[0] : mul.in[1];
        CHECK(prog.nodes[size_t(ramp)].op == Op::CURVE);
        CHECK(prog.nodes[size_t(ramp)].live_lod == 3); // read by the product up to LOD 3 only
        CHECK(prog.nodes[size_t(prog.output)].const_lod == 4); // 0 + 2 m: the whole ground a constant
        CHECK(last_computed(prog, 9) == 0);
    }
}
