#include "test_climate_common.hpp"

#include "cascade_glsl.hpp"
#include "climate.hpp"
#include "sdf_cascade.hpp"

#include <cmath>
#include <random>

using namespace Climate;
using ClimateTest::box_point_mismatches;
using ClimateTest::differing_floats;
using ExactNoise::float_bits;

namespace {

// The ground of an SDFCascade configured at p_voxel metres and p_lods LODs.
struct GroundFixture {
    SDFCascade sdf;
    Ground ground;

    GroundFixture(double p_voxel, int p_lods) {
        ChunkParam param;
        param.voxel_size = float(p_voxel);
        sdf.configure(param, p_lods);
        ground = sdf.climate_ground();
    }
};

// A field's bands alone as a cascade spec (the reference of its group in the combined spec).
Cascade::Spec own_spec(const Field &p_field) {
    Cascade::Spec spec;
    spec.set = Cascade::ChannelSet::LINEAR4;
    spec.bands = p_field.bands;
    return spec;
}

ov::str parse_json(const ov::str &p_json, const Ground &p_ground, Table &r_table) {
    const ov::Value v = ClimateTest::json(p_json);
    CHECK(v.is_object());
    return v.is_object() ? parse(v, p_ground, r_table) : ov::str("not a JSON object");
}

bool subnormal(const Float4 &p_v) {
    return std::fpclassify(p_v.x) == FP_SUBNORMAL || std::fpclassify(p_v.y) == FP_SUBNORMAL || std::fpclassify(p_v.z) == FP_SUBNORMAL ||
            std::fpclassify(p_v.w) == FP_SUBNORMAL;
}

// A random table over p_fields fields (no bands needed: weights read field values only), 2 to MAX_BIOMES biomes.
Table random_table(std::mt19937 &p_rng, int p_fields) {
    std::uniform_real_distribution<float> bound(-1.0f, 1.0f);
    std::uniform_real_distribution<float> fade(0.02f, 0.5f);
    Table t;
    for (int f = 0; f < p_fields; f++) {
        Field field;
        field.name = ov::format("f%d", f);
        t.fields.push_back(field);
    }
    const int biomes = 2 + int(p_rng() % uint32_t(MAX_BIOMES - 1));
    for (int b = 0; b < biomes; b++) {
        Climate::Biome biome;
        biome.name = ov::format("b%d", b);
        const int edges = b + 1 == biomes ? 0 : 1 + int(p_rng() % 4u);
        for (int e = 0; e < edges; e++) {
            const float w = fade(p_rng);
            const bool rising = (p_rng() & 1u) != 0;
            biome.edges.push_back(Edge{ int(p_rng() % uint32_t(p_fields)), bound(p_rng) + (rising ? -w : w), rising ? 1.0f / w : -1.0f / w });
        }
        t.biomes.push_back(biome);
    }
    return t;
}

// The weights in double precision, values only (the reference of the gradients by finite differences).
void reference_weights(const Table &p_t, const double *p_x, double *r_w) {
    double rest = 1.0;
    const size_t n = p_t.biomes.size();
    for (size_t b = 0; b + 1 < n; b++) {
        double r = 1.0;
        for (const Edge &e : p_t.biomes[b].edges) {
            const double s = std::clamp((p_x[e.field] - double(e.anchor)) * double(e.scale), 0.0, 1.0);
            r *= s * s * (3.0 - 2.0 * s);
        }
        r_w[b] = rest * r;
        rest -= r_w[b];
    }
    r_w[n - 1] = rest;
}

} // namespace

TEST_CASE("Climate: make_fbm's octaves are make_octave keyed by their index") {
    const ExactNoise::OctaveSet set = ExactNoise::make_fbm(false, 0xC0FFEEu, 20, 6, 300.0, 0.5);
    for (size_t i = 0; i < set.octaves.size(); i++) {
        const ExactNoise::Octave o = ExactNoise::make_octave(0xC0FFEEu, uint32_t(i), 20 - int(i), set.octaves[i].amp);
        CHECK(o.seed == set.octaves[i].seed);
        CHECK(o.offset == set.octaves[i].offset);
        CHECK(o.k == set.octaves[i].k);
    }
}

TEST_CASE("Climate: the plain preset (the example before shaped bands) parses at 5 cm at any LOD count, its ranges keeping the ground's octaves") {
    ov::Value data;
    CLIMATE_REQUIRE(load("res://climate/plain.json", data) == ov::str());
    GroundFixture g(0.05, 15);
    Table t;
    CHECK(parse(data, g.ground, t) == ov::str());
    CLIMATE_REQUIRE(t.fields.size() == 3);
    CLIMATE_REQUIRE(t.biomes.size() == 7);
    // 3.2 km .. 26 km at 5 cm: k = 16 .. 19, coarsest first, amplitude 0.5 times 0.7 per finer octave; the
    // continental range (52 km ..) is past the ground's top octave (19) at 15 LODs.
    const Field &temp = t.fields[0];
    CLIMATE_REQUIRE(temp.bands.size() == 4);
    CHECK(temp.bands.front().k == 19);
    CHECK(temp.bands.back().k == 16);
    CHECK(temp.bands[1].amp == float(0.5 * 0.7));
    CLIMATE_REQUIRE(temp.terms.size() == 1);
    CHECK(temp.terms[0].source == GROUND);
    CHECK(temp.terms[0].coef == float(-0.00003 * g.ground.voxel_size));
    // Seeds are keyed by k: the same octave of the same field always gets the same lattice.
    CHECK(temp.bands[0].seed == ExactNoise::make_octave(temp.seed, 19u, 19, 1.0f).seed);
    CHECK(t.biomes.back().name == "plain");
    CHECK(t.biomes.back().edges.empty());
    CHECK(t.biomes[0].debug_color == 0xC03818u);
    // Ground multipliers name ground octaves: 0.4 m .. 25 m = k 3 .. 9 and 50 m .. 13 km = k 10 .. 18; the plain's
    // 0.4 m .. 12 m and 25 m .. 30,000 km keep k 3 .. 8 and 9 .. 19 here.
    CHECK(t.biomes[0].amps.size() == 16);
    CHECK(t.biomes.back().amps.size() == 6 + 11);

    // 25 LODs: the continental octaves k = 20 .. 29 join, each at the amplitude it has at any LOD count.
    GroundFixture g25(0.05, 25);
    Table t25;
    CHECK(parse(data, g25.ground, t25) == ov::str());
    CLIMATE_REQUIRE(t25.fields[0].bands.size() == 14);
    CHECK(t25.fields[0].bands.front().k == 29);
    CHECK(t25.fields[0].bands[9].k == 20);
    CHECK(t25.fields[0].bands[9].amp == 0.35f);
    CHECK(t25.fields[0].bands[10].amp == temp.bands[0].amp);
    CHECK(t25.biomes.back().amps.size() == 6 + 21);

    // 14 LODs: the ranges lose k = 19; a single wavelength past the top is refused.
    GroundFixture g14(0.05, 14);
    Table t14;
    CHECK(parse(data, g14.ground, t14) == ov::str());
    CHECK(t14.fields[0].bands.size() == 3);
    Table single;
    const char *past_top = R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 26000, "amp": 1}]}], "biomes": [{"name": "a"}]})";
    CHECK(ov::contains(parse_json(past_top, g14.ground, single), "outside the ground's range"));
}

TEST_CASE("Climate: malformed tables are refused with the reason") {
    GroundFixture g(0.5, 15);
    struct Case {
        const char *json;
        const char *error;
    };
    const Case cases[] = {
        { R"({"fields": [], "biomes": []})", "1 to 16" },
        { R"({"biomes": [{"name": "a", "colour": 1}]})", "unknown key 'colour'" },
        { R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 100, "amp": 1}], "terms": [{"source": "h", "x": 1}]}], "biomes": [{"name": "a"}]})",
                "or an earlier field" },
        { R"({"fields": [{"name": "t"}, {"name": "h", "terms": [{"source": "h", "x": 1}]}], "biomes": [{"name": "a"}]})", "or an earlier field" },
        { R"({"fields": [{"name": "t"}, {"name": "t"}], "biomes": [{"name": "a"}]})", "used twice" },
        { R"({"fields": [{"name": "ground"}], "biomes": [{"name": "a"}]})", "reserved" },
        { R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 100, "amp": 1}, {"wavelength_m": 110, "amp": 1}]}], "biomes": [{"name": "a"}]})",
                "listed twice" },
        { R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 100, "amp": 1e13}]}], "biomes": [{"name": "a"}]})", "amplitude" },
        { R"({"fields": [{"name": "t"}], "biomes": [{"name": "a", "where": [{"field": "t", "min": 0}]}, {"name": "b"}]})", "fade must be > 0" },
        { R"({"fields": [{"name": "t"}], "biomes": [{"name": "a", "where": [{"field": "t", "fade": 1}]}, {"name": "b"}]})", "'min', 'max' or both" },
        { R"({"fields": [{"name": "t"}], "biomes": [{"name": "a"}, {"name": "b"}]})", "only the last biome" },
        { R"({"fields": [{"name": "t"}], "biomes": [{"name": "a", "where": [{"field": "t", "min": 0, "fade": 1}]}]})", "the last biome is the default" },
        { R"({"fields": [{"name": "t"}], "biomes": [{"name": "a", "where": [{"field": "u", "min": 0, "fade": 1}]}, {"name": "b"}]})", "not a field" },
        { R"({"biomes": [{"name": "a", "amp": [{"field": "ground", "wavelength_m": 1, "x": 2}]}]})", "doesn't have it" },
        { R"({"biomes": [{"name": "a", "amp": [{"field": "ground", "from_m": 4000, "to_m": 9000, "x": 50}]}]})", "out of range" },
        { R"({"biomes": [{"name": "a", "amp": [{"field": "ground", "wavelength_m": 4000, "x": -1}]}]})", ">= 0" },
        { R"({"biomes": [{"name": "a", "height_m": 100}]})", "unknown key 'height_m'" },
        { R"({"biomes": [{"name": "a", "debug_color": "red"}]})", "#rrggbb" },
        { R"({"biomes": [{"name": "a", "debug_color": "#-00001"}]})", "#rrggbb" },
        { R"({"fields": [{"name": "t", "bands": [{"from_m": 1, "to_m": 1e30, "amp": 1}]}], "biomes": [{"name": "a"}]})", "outside [0, 62]" },
        { R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 100, "to_m": 500, "amp": 1}]}], "biomes": [{"name": "a"}]})", "not both" },
        { R"({"fields": [{"name": "t", "seed": 1e20}], "biomes": [{"name": "a"}]})", "whole number" },
        { R"({"fields": [{"name": "t", "seed": 7.9}], "biomes": [{"name": "a"}]})", "whole number" },
        { R"({"fields": [{"name": "temp/2"}], "biomes": [{"name": "a"}]})", "identifier" },
        { R"({"fields": [{"name": "t"}], "heights": [{"field": "t", "height_m": 1, "region": "r"}], "biomes": [{"name": "a"}]})", "'r' is not a region" },
        { R"({"fields": [{"name": "t"}], "regions": [{"name": "r", "where": [{"field": "t", "min": 0, "fade": 1}]}],
              "heights": [{"field": "t", "height_m": 1, "region": "r"}], "biomes": [{"name": "a"}]})",
                "a gated term takes a curve" },
        { R"({"fields": [{"name": "t"}], "regions": [{"name": "r", "where": [{"field": "t", "min": 0, "fade": 1}]},
              {"name": "r", "where": [{"field": "t", "max": 0, "fade": 1}]}], "biomes": [{"name": "a"}]})",
                "used twice" },
        { R"({"fields": [{"name": "t"}], "regions": [{"name": "r", "where": []}], "biomes": [{"name": "a"}]})", "at least one edge" },
        { R"({"fields": [{"name": "t"}], "regions": [{"name": "r/2", "where": [{"field": "t", "min": 0, "fade": 1}]}], "biomes": [{"name": "a"}]})",
                "identifier" },
        { R"({"fields": [{"name": "t"}], "biomes": [{"name": "a", "region": "r", "where": [{"field": "t", "min": 0, "fade": 1}]}, {"name": "b"}]})",
                "'r' is not a region" },
        { R"({"fields": [{"name": "t"}], "regions": [{"name": "r", "where": [{"field": "t", "min": 0, "fade": 1}]}],
              "biomes": [{"name": "a", "region": "r"}]})",
                "the last biome is the default" },
    };
    for (const Case &c : cases) {
        CAPTURE(c.json);
        Table t;
        const ov::str err = parse_json(c.json, g.ground, t);
        CHECK(ov::contains(err, c.error));
        CHECK(t.biomes.empty());
    }
    // A multiplier of 0 turns an octave off, and the field count is capped.
    Table t;
    CHECK(parse_json(R"({"biomes": [{"name": "a", "amp": [{"field": "ground", "from_m": 4, "to_m": 100, "x": 0}]}]})", g.ground, t) == ov::str());
    ov::str many = R"({"fields": [)";
    for (int i = 0; i <= MAX_FIELDS; i++) {
        many += ov::format("%s{\"name\": \"f%d\"}", i ? ", " : "", i);
    }
    CHECK(ov::contains(parse_json(many + R"(], "biomes": [{"name": "a"}]})", g.ground, t), ov::format("at most %d", MAX_FIELDS)));
    // A table whose gated list no longer matches its height terms (edited after parse) is refused, not read stale.
    CHECK(parse_json(R"({"fields": [{"name": "t"}], "regions": [{"name": "r", "where": [{"field": "t", "min": 0, "fade": 1}]}],
              "biomes": [{"name": "a"}]})",
                  g.ground, t) == ov::str());
    t.heights.push_back(HeightTerm{ 0, 1.0f, true, make_curve({ -1.0, 1.0 }, { -1.0, 1.0 }), 0 });
    CHECK(ov::contains(validate(t, g.ground), "Table::gated"));
}

// max_ground_mult is validate's cap (climate_report exports it for tools/landforms/match.py): just under it parses,
// just over it is refused, for a plain and a shaped band.
TEST_CASE("Climate: max_ground_mult is the largest multiplier validate accepts") {
    GroundFixture g(0.5, 15);
    for (const char *shape : { "plain", "ridged" }) {
        CAPTURE(shape);
        auto table = [&](double p_x) {
            return ov::format(R"({"biomes": [{"name": "a", "amp": [{"field": "ground", "wavelength_m": 4096, "x": %s, "shape": "%s"}]}]})", p_x,
                    shape);
        };
        Table t;
        CLIMATE_REQUIRE(parse_json(table(1.0), g.ground, t) == ov::str());
        const double cap = max_ground_mult(t, g.ground, t.biomes[0], t.biomes[0].amps[0].k);
        CHECK(cap > 1.0);
        CHECK(parse_json(table(0.99 * cap), g.ground, t) == ov::str());
        CHECK(ov::contains(parse_json(table(1.01 * cap), g.ground, t), "out of range"));
    }
}

TEST_CASE("Climate: weights are a partition of unity, in [0, 1], without denormals, gradients by the chain rule") {
    std::mt19937 rng(0xC11A7Eu);
    std::uniform_real_distribution<float> value(-1.5f, 1.5f);
    std::uniform_real_distribution<float> slope(-2.0f, 2.0f);
    int checked = 0;
    double worst_sum = 0.0;
    double worst_grad = 0.0;
    for (int table = 0; table < 200; table++) {
        const int nf = 1 + int(rng() % uint32_t(MAX_FIELDS));
        const Table t = random_table(rng, nf);
        CHECK(validate(t, Ground()) == ov::str());
        for (int sample = 0; sample < 200; sample++) {
            Float4 x[MAX_FIELDS] = {};
            double xd[MAX_FIELDS] = {};
            double gd[MAX_FIELDS] = {};
            for (int f = 0; f < nf; f++) {
                // Every 4th sample also exercises tiny values (flushed below 2^-64).
                const float tiny = sample % 4 == 0 ? std::ldexp(1.0f, -70 - f) : 0.0f;
                x[f] = Float4{ slope(rng), tiny, 0.0f, value(rng) };
                xd[f] = double(x[f].w);
                gd[f] = double(x[f].x);
            }
            Float4 w[MAX_BIOMES];
            weights(t, x, w);
            double sum = 0.0;
            bool bad = false;
            for (size_t b = 0; b < t.biomes.size(); b++) {
                bad |= subnormal(w[b]) || !(w[b].w >= 0.0f && w[b].w <= 1.0f);
                sum += double(w[b].w);
            }
            CHECK_FALSE(bad);
            worst_sum = std::max(worst_sum, std::fabs(sum - 1.0));
            // d w / dt along x(t) = x + t g, central differences of the double reference.
            const double eps = 1e-6;
            double xp[MAX_FIELDS], xm[MAX_FIELDS], wp[MAX_BIOMES], wm[MAX_BIOMES];
            for (int f = 0; f < nf; f++) {
                xp[f] = xd[f] + eps * gd[f];
                xm[f] = xd[f] - eps * gd[f];
            }
            reference_weights(t, xp, wp);
            reference_weights(t, xm, wm);
            for (size_t b = 0; b < t.biomes.size(); b++) {
                const double fd = (wp[b] - wm[b]) / (2.0 * eps);
                worst_grad = std::max(worst_grad, std::fabs(double(w[b].x) - fd) / (1.0 + std::fabs(fd)));
            }
            checked++;
        }
    }
    CHECK(checked == 40000);
    CHECK(worst_sum < 1e-5);
    CHECK(worst_grad < 2e-3);
}

TEST_CASE("Climate: priority layering, exact 0 / 1 weights, and the pick's ties") {
    Table t;
    Field f;
    f.name = "t";
    t.fields.push_back(f);
    // inner [0.4, 0.6] first, outer [0, 1] second, then the default: fades 0.05.
    auto region = [](const char *p_name, float p_lo, float p_hi) {
        Climate::Biome b;
        b.name = p_name;
        b.edges.push_back(Edge{ 0, p_lo - 0.05f, 1.0f / 0.05f });
        b.edges.push_back(Edge{ 0, p_hi + 0.05f, -1.0f / 0.05f });
        return b;
    };
    t.biomes = { region("inner", 0.4f, 0.6f), region("outer", 0.0f, 1.0f), Climate::Biome() };
    t.biomes[2].name = "default";
    CLIMATE_REQUIRE(validate(t, Ground()) == ov::str());
    auto at = [&](float p_x, Float4 r_w[MAX_BIOMES]) {
        Float4 x[MAX_FIELDS] = {};
        x[0] = Float4{ 1.0f, 0.0f, 0.0f, p_x };
        weights(t, x, r_w);
    };
    Float4 w[MAX_BIOMES];
    const float expect[][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, 1, 0 } };
    const float xs[] = { 0.5f, 0.2f, -0.5f, 0.9f };
    for (int i = 0; i < 4; i++) {
        at(xs[i], w);
        for (int b = 0; b < 3; b++) {
            CAPTURE(i);
            CAPTURE(b);
            CHECK(float_bits(w[b].w) == float_bits(expect[i][b]));
            CHECK(w[b].x == 0.0f);
        }
        CHECK(pick(t, w, Vector3i64(1, 2, 3), 0) == int(std::max_element(expect[i], expect[i] + 3) - expect[i]));
    }
    // Halfway up the outer edge, s = 0.5 gives exactly 0.5: outer and default tie, the hash chooses.
    at(-0.025f, w);
    CHECK(w[1].w == 0.5f);
    CHECK(w[2].w == 0.5f);
    // The tie goes by ExactNoise::hash_cell of the point's LOD0 voxel, high words included: the same point gives the
    // same pick from every lattice that has it.
    int seen[2] = { 0, 0 };
    for (int64_t i = 0; i < 16; i++) {
        const Vector3i64 p(i, -i, i * (int64_t(1) << 33) + 1);
        const int got = pick(t, w, p, 5);
        CHECK(got == 1 + int(ExactNoise::hash_cell(Vector3i64(p.x * 32, p.y * 32, p.z * 32), PICK_SEED) % 2u));
        CHECK(pick(t, w, Vector3i64(p.x * 2, p.y * 2, p.z * 2), 4) == got);
        CHECK(pick(t, w, Vector3i64(p.x * 32, p.y * 32, p.z * 32), 0) == got);
        seen[got - 1]++;
    }
    CHECK(seen[0] > 0);
    CHECK(seen[1] > 0);
}

TEST_CASE("Climate: fields are pure functions of the lattice point, LOD-coherent, terms in list order") {
    GroundFixture g(0.05, 15);
    Table t;
    CLIMATE_REQUIRE(parse_json(R"({"fields": [
            {"name": "a", "bands": [{"from_m": 800, "to_m": 26000, "amp": 1}]},
            {"name": "b", "bands": [{"wavelength_m": 3200, "amp": 0.5}], "terms": [{"source": "a", "x": -0.25}]},
            {"name": "c", "terms": [{"source": "ground", "per_m": 0.001}, {"source": "b", "x": 2}]}],
            "biomes": [{"name": "x"}]})",
                            g.ground, t) == ov::str());
    // No multiplier: the combined spec carries the fields, unmodulated, its ground the bare one.
    const Cascade::Spec spec = combined_spec(t, *g.ground.spec);
    CHECK(spec.fields.size() == 3);
    CHECK(spec.field_range().empty()); // a modulation for the pick's derive only: no modulated band
    // Lattice indices: times 2^13 (the finest point at m = 12, doubled) they stay within +-2^60 voxels.
    const int64_t bases[] = { 0, 12345, (int64_t(1) << 32) - 3, -(int64_t(1) << 47) + 7, (int64_t(1) << 47) - 11 };
    int coherent = 0;
    for (const int64_t base : bases) {
        for (const int m : { 0, 3, 7, 12 }) {
            const Vector3i64 p(base, -base / 3 + 5, 7 - base);
            Float4 r[MAX_FIELDS];
            fields_at(t, spec, p, m, r);
            // Field a is its own spec; b adds -0.25 a; c is 0.001 * 0.05 ground + 2 b, in that order.
            float ch[Cascade::CHANNELS], gc[Cascade::CHANNELS];
            Cascade::point_into(own_spec(t.fields[0]), p, m, ch);
            CHECK(float_bits(r[0].w) == float_bits(ch[Cascade::VALUE]));
            CHECK(float_bits(r[0].x) == float_bits(ch[Cascade::DX]));
            Cascade::point_into(own_spec(t.fields[1]), p, m, ch);
            CHECK(float_bits(r[1].w) == float_bits(Cascade::flush_value(ch[Cascade::VALUE] + r[0].w * -0.25f)));
            Cascade::point_into(*g.ground.spec, p, m, gc);
            const float c0 = Cascade::flush_value(0.0f + gc[Cascade::VALUE] * float(0.001 * g.ground.voxel_size));
            CHECK(float_bits(r[2].w) == float_bits(Cascade::flush_value(c0 + r[1].w * 2.0f)));
            for (int f = 0; f < 3; f++) {
                CHECK_FALSE(subnormal(r[f]));
            }
            CHECK(r[3].w == 0.0f);
            // a and b have no octave finer than 800 m (k = 14, inserted on Λ_11): below it nothing is inserted, so Λ_m's
            // point 2p is a copy of Λ_{m+1}'s p, the same value and half the gradient (lattice units).
            if (m < 11) {
                Float4 fine[MAX_FIELDS];
                fields_at(t, spec, Vector3i64(p.x * 2, p.y * 2, p.z * 2), m, fine);
                Float4 coarse[MAX_FIELDS];
                fields_at(t, spec, p, m + 1, coarse);
                for (int f = 0; f < 2; f++) {
                    CHECK(float_bits(fine[f].w) == float_bits(coarse[f].w));
                    CHECK(float_bits(fine[f].y) == float_bits(coarse[f].y * 0.5f));
                }
                coherent++;
            }
        }
    }
    CHECK(coherent == 15);
}

namespace {

} // namespace

TEST_CASE("Climate: carried fields and a neutral modulation leave the ground bit for bit") {
    GroundFixture g(0.05, 15);
    const Cascade::Spec &bare = *g.ground.spec;
    // Fields without multipliers: carried, the ground untouched.
    Table carried;
    CLIMATE_REQUIRE(parse_json(R"({"fields": [{"name": "t", "bands": [{"from_m": 400, "to_m": 26000, "amp": 1}]}],
            "biomes": [{"name": "a", "where": [{"field": "t", "min": 0, "fade": 0.2}]}, {"name": "b"}]})",
                            g.ground, carried) == ov::str());
    // Only the default biome, x1 on every ground octave: modulated, A = 1 * amp exactly and dA = 0, so n A and
    // n.s (A 2^-3) + n.w dA give the constant band's bits (adding a zero product changes nothing).
    Table neutral;
    CLIMATE_REQUIRE(parse_json(R"({"fields": [{"name": "t", "bands": [{"from_m": 400, "to_m": 26000, "amp": 1}]}],
            "biomes": [{"name": "only", "amp": [{"field": "ground", "from_m": 0.4, "to_m": 30000, "x": 1}]}]})",
                            g.ground, neutral) == ov::str());
    const Cascade::Spec specs[] = { combined_spec(carried, bare), combined_spec(neutral, bare) };
    CHECK(specs[0].field_range().empty()); // the pick's derive only
    CHECK(specs[1].modulation != nullptr);
    int64_t compared = 0, differing = 0;
    for (const Cascade::Spec &spec : specs) {
        for (const int lod : { 0, 2, 6, 11 }) {
            for (const int64_t base : { int64_t(0), int64_t(-77), (int64_t(1) << 32) - 5, -(int64_t(1) << 45) + 3 }) {
                const Vector3i64 p(base, base / 7 + 3, -base + 11);
                float want[Cascade::CHANNELS], got[Cascade::CHANNELS];
                Cascade::point_into(bare, p, lod, want);
                Cascade::point_into(spec, p, lod, got);
                compared++;
                differing += differing_floats(want, got, 4);
            }
        }
    }
    CHECK(compared == 32);
    CHECK(differing == 0);
}

TEST_CASE("Climate: modulated cascade, box runs equal point queries in every slot, and the amplitude follows the biomes") {
    GroundFixture g(0.05, 15);
    ov::Value data;
    CLIMATE_REQUIRE(load("res://climate/plain.json", data) == ov::str());
    Table t;
    CLIMATE_REQUIRE(parse(data, g.ground, t) == ov::str());
    const Cascade::Spec spec = combined_spec(t, *g.ground.spec);
    CLIMATE_REQUIRE(spec.modulation != nullptr);
    CHECK(spec.fields.size() == 3);
    int64_t bad = 0;
    for (const int lod : { 0, 3, 8 }) {
        for (const int64_t base : { int64_t(5), (int64_t(1) << 32) - 2, -(int64_t(1) << 44) }) {
            bad += box_point_mismatches(spec, Vector3i64(base, -3, base / 3), 3, lod);
        }
    }
    CHECK(bad == 0);
    // Ground queries carry the fields only where FieldRange says; with all of them carried the ground is the same bits.
    const Cascade::FieldRange range = spec.field_range();
    CHECK(range.floor == 0); // 0.4 m at 5 cm: k = 3, inserted on Λ_0
    CHECK(range.ceiling == 16); // 26 km: k = 19
    int64_t ground_differs = 0;
    for (const int lod : { 0, 6, 7, 12, 16, 17 }) {
        for (int i = 0; i < 16; i++) {
            const Vector3i64 p(int64_t(i) * 7919 - 50000, 3 - i, int64_t(i) * -104729);
            float part[Cascade::SLOTS], all[Cascade::SLOTS];
            Cascade::point_slots(spec, p, lod, part);
            Cascade::point_slots(spec, p, lod, all, true);
            ground_differs += differing_floats(part, all, 4);
            CHECK((part[Cascade::field_slot(0)] == 0.0f) == !range.carried(lod, lod)); // the fields only where carried
        }
    }
    CHECK(ground_differs == 0);
    // The ground differs from the bare one where a biome multiplies its octaves: heights along a line at Λ_6.
    int differ = 0;
    for (int i = 0; i < 64; i++) {
        const Vector3i64 p(int64_t(i) * 997, 0, int64_t(i) * -1301);
        float a[Cascade::CHANNELS], b[Cascade::CHANNELS];
        Cascade::point_into(*g.ground.spec, p, 6, a);
        Cascade::point_into(spec, p, 6, b);
        differ += float_bits(a[Cascade::VALUE]) != float_bits(b[Cascade::VALUE]);
        CHECK(std::isfinite(b[Cascade::VALUE]));
    }
    CHECK(differ > 48);
}

TEST_CASE("Climate: carried only where the ground needs it, refused where it can't apply") {
    GroundFixture g(0.05, 15);
    // Multipliers on a field's octaves only would never reach the mesh (the fields are carried where the ground is
    // modulated): refused.
    Table t;
    const ov::str fields = R"({"fields": [{"name": "t", "bands": [{"from_m": 400, "to_m": 26000, "amp": 1}]},
            {"name": "h", "bands": [{"from_m": 400, "to_m": 6500, "amp": 1}]}], )";
    CHECK(ov::contains(parse_json(fields + R"("biomes": [{"name": "a", "where": [{"field": "t", "min": 0, "fade": 0.2}], "amp": [{"field": "h", "from_m": 400, "to_m": 6500, "x": 2}]},
                       {"name": "b"}]})",
                               g.ground, t),
            "needs a ground multiplier too"));
    // Fields without any multiplier (a pick only): carried nowhere in the ground's runs (no field scratch, no fields
    // variant), and its top lattice is the ground's.
    CLIMATE_REQUIRE(parse_json(fields + ov::str(R"("biomes": [{"name": "a", "where": [{"field": "t", "min": 0, "fade": 0.2}]}, {"name": "b"}]})"), g.ground, t) ==
            ov::str());
    const Cascade::Spec spec = combined_spec(t, *g.ground.spec);
    CHECK(spec.modulation != nullptr); // the pick's derive, for the chunks' samples (CascadeGLSL::field_range)
    CHECK(spec.field_range().empty());
    CHECK(CascadeGLSL::vec4s_per_point(spec, spec.field_range()) == 1);
    CHECK(spec.top_level() == g.ground.spec->top_level());
    // A flat ground (no band) with fields: no top lattice from them either.
    Cascade::Spec flat = spec;
    flat.bands.clear();
    CHECK(flat.top_level() == -1);
    CHECK(flat.top_level(true) >= 0);
    // A table that can't modulate: an empty one, and multipliers on an 8-channel ground.
    ov::Value empty;
    CHECK(ov::contains(load("res://climate/does_not_exist.json", empty), "cannot read"));
    SDFCascade h8;
    h8.channel_set = Cascade::ChannelSet::HERMITE8;
    CLIMATE_REQUIRE(load("res://climate/plain.json", h8.climate_data) == ov::str());
    ChunkParam param;
    param.voxel_size = 0.05f;
    h8.configure(param, 15);
    CHECK(h8.get_climate() == nullptr);
    CHECK(ov::contains(h8.get_climate_error(), "LINEAR4"));
    CHECK(h8.get_spec().fields.empty());
}

TEST_CASE("Climate: biome ground materials, the base palette, and the per-sample pick (box runs equal point queries)") {
    SDFCascade sdf;
    CLIMATE_REQUIRE(load("res://climate/plain.json", sdf.climate_data) == ov::str());
    ChunkParam param;
    param.voxel_size = 0.05f;
    sdf.configure(param, 15);
    CLIMATE_REQUIRE(sdf.get_climate() != nullptr);
    const Table &t = *sdf.get_climate();
    const std::vector<uint32_t> base = sdf.get_gpu_table()->base_palette();
    CLIMATE_REQUIRE(base.size() == 1 + t.biomes.size());
    CHECK(base[0] == Material::GROUND);
    CHECK(CascadeGLSL::ground_materials(&sdf));
    for (size_t b = 0; b < t.biomes.size(); b++) {
        const Material::Params &m = sdf.get_materials()[base[1 + b]];
        CHECK(sdf.get_materials().name(base[1 + b]) == "biome:" + t.biomes[b].name);
        CHECK(m.pattern == Material::PATTERN_GROUND);
        CHECK(m.debug.a == 1.0f);
    }
    // "#d8b878" (desert, sRGB) as a linear albedo.
    const int desert = 2;
    CHECK(t.biomes[desert].name == "desert");
    CHECK(t.biomes[desert].surface.flat.r == doctest::Approx(ov::Color(216 / 255.0, 0, 0).srgb_to_linear().r));
    // A reconfigure keeps the materials' ids (Library::set) and the chunk passes derive the pick above every sample.
    sdf.configure(param, 15);
    CHECK(sdf.get_gpu_table()->base_palette() == base);
    const Cascade::FieldRange range = CascadeGLSL::field_range(&sdf, false);
    CHECK(range.everywhere());
    CHECK(CascadeGLSL::field_range(&sdf, true).sample == sdf.get_spec().field_range().sample); // point items: the spec's

    // The pick of a box run's level slots (what the GPU's level 0 reads) equals the point query, at several LODs and
    // origins; over a wide coarse slice the example climate shows several biomes.
    const Cascade::Spec &spec = sdf.get_spec();
    int64_t points = 0, mismatches = 0;
    for (const int lod : { 0, 3, 8, 13 }) {
        for (const int64_t origin : { int64_t(0), (int64_t(1) << 40) + 12345, -(int64_t(1) << 33) }) {
            Cascade::Box box;
            for (int a = 0; a < 3; a++) {
                box.lo[a] = (a == 1 ? (sdf.get_datum_y() >> lod) : (origin >> lod)) - 3;
                box.hi[a] = box.lo[a] + 5;
            }
            const Cascade::Level level = Cascade::run(spec, box, lod, nullptr, Cascade::Source(), true);
            Cascade::for_each_point(box, [&](const Vector3i64 &p_p, int64_t p_i) {
                float slots[Cascade::SLOTS];
                level.slots_at(p_i, slots);
                const VoxelPos v(p_p.x << lod, p_p.y << lod, p_p.z << lod);
                points++;
                mismatches += sdf.ground_material(slots) != sdf.ground_material_at(v, lod);
            });
        }
    }
    CHECK(points > 0);
    CHECK(mismatches == 0);
    std::vector<int> seen(t.biomes.size() + 1, 0);
    const int lod = 12;
    for (int i = 0; i < 24; i++) {
        for (int j = 0; j < 24; j++) {
            const VoxelPos v(int64_t(i - 12) << (lod + 3), sdf.get_datum_y() >> lod << lod, int64_t(j - 12) << (lod + 3));
            const uint32_t m = sdf.ground_material_at(v, lod);
            CLIMATE_REQUIRE((m >= 1 && m < base.size()));
            seen[m]++;
        }
    }
    CHECK(std::count_if(seen.begin(), seen.end(), [](int p_n) { return p_n > 0; }) >= 3);

    // Malformed surfaces.
    GroundFixture g(0.05, 15);
    Table bad;
    CHECK(ov::contains(parse_json(R"({"fields": [], "biomes": [{"name": "a", "surface": {"flat": "green"}}]})", g.ground, bad), "#rrggbb"));
    CHECK(ov::contains(parse_json(R"({"fields": [], "biomes": [{"name": "a", "surface": {"roughness": 2}}]})", g.ground, bad), "roughness"));
    CHECK(ov::contains(parse_json(R"({"fields": [], "biomes": [{"name": "a", "surface": {"shiny": 1}}]})", g.ground, bad), "unknown key"));
    // A surface's steep colour: the given flat without its own, the procedural ground's rock without either.
    Table s;
    CLIMATE_REQUIRE(parse_json(R"({"fields": [], "biomes": [{"name": "a", "surface": {"roughness": 0.5}}]})", g.ground, s) == ov::str());
    CHECK(s.biomes[0].surface.steep == Material::GROUND_STEEP);
    CHECK(s.biomes[0].surface.flat == Material::GROUND_FLAT);
    CLIMATE_REQUIRE(parse_json(R"({"fields": [], "biomes": [{"name": "a", "surface": {"flat": "#ff0000"}}]})", g.ground, s) == ov::str());
    CHECK(s.biomes[0].surface.steep == s.biomes[0].surface.flat);
    // A climate without fields (one default biome): no pick to make, so no ground materials (CPU and GPU alike).
    SDFCascade fieldless;
    CLIMATE_REQUIRE(parse_json(R"({"fields": [], "biomes": [{"name": "a"}]})", g.ground, s) == ov::str());
    fieldless.climate_data = ClimateTest::json(R"({"fields": [], "biomes": [{"name": "a"}]})");
    fieldless.configure(param, 15);
    CLIMATE_REQUIRE(fieldless.get_climate() != nullptr);
    CHECK_FALSE(fieldless.has_ground_materials());
    CHECK(fieldless.get_gpu_table()->base_palette() == std::vector<uint32_t>{ Material::GROUND });
    CHECK(fieldless.get_ground_material_glsl(fieldless.get_spec()).empty());
    // Without a climate: the base palette is GROUND alone, and no ground material GLSL.
    SDFCascade plain;
    plain.configure(param, 15);
    CHECK(plain.get_gpu_table()->base_palette() == std::vector<uint32_t>{ Material::GROUND });
    CHECK(plain.get_ground_material_glsl(plain.get_spec()).empty());
    CHECK(ov::contains(Climate::pick_glsl(t, spec), "uint climate_biome(Ch ch)"));
}

// One octave per group and lattice: a second band of one group on one lattice is refused by validate, and the cascade
// ignores it (ERR) in a spec built by hand, box and point alike.
TEST_CASE("Cascade: one band per group and lattice") {
    GroundFixture g(0.05, 15);
    Table t;
    CHECK(ov::contains(parse_json(R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 400, "amp": 1}, {"wavelength_m": 400, "amp": 2}]}], "biomes": [{"name": "a"}]})",
                               g.ground, t),
            "twice"));
    Cascade::Spec one;
    one.set = Cascade::ChannelSet::LINEAR4;
    one.bands = { ExactNoise::make_octave(11u, 0u, 9, 40.0f) };
    Cascade::Spec two = one;
    two.bands.push_back(ExactNoise::make_octave(12u, 0u, 9, 25.0f)); // the same lattice Λ_6
    int64_t mismatches = 0;
    for (int i = 0; i < 32; i++) {
        const Vector3i64 p(i * 37 - 500, i * 11, 900 - i * 23);
        float a[Cascade::CHANNELS], b[Cascade::CHANNELS], c[Cascade::CHANNELS];
        Cascade::point_into(one, p, 2, a);
        Cascade::point_into(two, p, 2, b);
        Cascade::point(two, p, 2, c);
        mismatches += differing_floats(a, b, 4) + differing_floats(a, c, 4);
    }
    CHECK(mismatches == 0);
}

// A biome multiplying the ground with no field: the modulated variant on the GPU too (field_range keyed on the
// modulation, not on the fields), or the GPU draws the bare ground.
TEST_CASE("Climate: a modulation without fields takes the modulated GPU variant") {
    SDFCascade sdf;
    CHECK(load("res://climate/fieldless_test.json", sdf.climate_data) == ov::str());
    ChunkParam param;
    param.voxel_size = 0.05f;
    sdf.configure(param, 15);
    CLIMATE_REQUIRE(sdf.get_climate() != nullptr);
    CHECK(sdf.get_spec().fields.empty());
    CHECK(sdf.get_spec().modulation != nullptr);
    for (const bool points : { false, true }) {
        const Cascade::FieldRange range = CascadeGLSL::field_range(&sdf, points);
        CHECK_FALSE(range.empty());
        CHECK(ov::contains(CascadeGLSL::level_shader(&sdf, false, points, false, !points, CascadeGLSL::View::FIELDS), "cascade_amplitudes("));
    }
}

// A run's level above its last derives its samples' climate and H carries it down (Climate's Modulator): the last level
// carries no field, only the ground and the carry, and its pick is the one its floor parent derived (copied, not
// interpolated).
TEST_CASE("Climate: a run's samples read the climate derived on the level above") {
    ClimateTest::ClimateSDF c("res://climate/example.json", 15);
    CLIMATE_REQUIRE(c.sdf.get_climate() != nullptr);
    const Cascade::Spec &spec = c.sdf.get_spec();
    const SampleCarry sc = sample_carry(*c.sdf.get_climate());
    CLIMATE_REQUIRE(sc.pick >= 0);
    CHECK(spec.carry_groups() > sc.count); // the bands' parts after the samples' terms
    CHECK(spec.field_range().sample); // the example's terraces
    int64_t points = 0, bad = 0;
    for (const int lod : { 0, 4, 9 }) {
        Cascade::Box box;
        for (int a = 0; a < 3; a++) {
            box.lo[a] = a == 1 ? (c.sdf.get_datum_y() >> lod) - 3 : int64_t(-5 - 1000 * lod) >> lod;
            box.hi[a] = box.lo[a] + 6;
        }
        std::vector<Cascade::Level> levels;
        Cascade::run(spec, box, lod, &levels);
        CLIMATE_REQUIRE(levels.size() >= 2);
        const Cascade::Level &last = levels.back();
        const Cascade::Level &above = levels[levels.size() - 2];
        CHECK(last.fields == 0);
        CHECK(last.carry == spec.modulation->carry_used(lod)); // the ground's groups only
        CHECK(last.carry < spec.carry_groups());
        CHECK(above.fields == int(spec.fields.size()));
        Cascade::for_each_point(last.box, [&](const Vector3i64 &p_p, int64_t p_i) {
            const Vector3i64 q(p_p.x >> 1, p_p.y >> 1, p_p.z >> 1);
            const float pick = last.ch[Cascade::carry_slot(sc.pick)][size_t(p_i)];
            points++;
            bad += float_bits(pick) != float_bits(above.ch[Cascade::carry_slot(sc.pick)][size_t(above.box.index(q))]);
        });
    }
    CHECK(points == 3 * 7 * 7 * 7);
    CHECK(bad == 0);
}
