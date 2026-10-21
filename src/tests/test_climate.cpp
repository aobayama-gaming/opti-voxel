#include "test_common.hpp"

#include "climate.hpp"
#include "sdf_cascade.hpp"

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/json.hpp>

#include <cmath>
#include <random>

using namespace Climate;
using ExactNoise::float_bits;

// doctest's REQUIRE needs exceptions: a failed check ends the test case.
#define CLIMATE_REQUIRE(p_cond) \
    do {                        \
        CHECK(p_cond);          \
        if (!(p_cond)) {        \
            return;             \
        }                       \
    } while (0)

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

String parse_json(const String &p_json, const Ground &p_ground, Table &r_table) {
    const Variant v = JSON::parse_string(p_json);
    CHECK(v.get_type() == Variant::DICTIONARY);
    return v.get_type() == Variant::DICTIONARY ? parse(Dictionary(v), p_ground, r_table) : String("not a JSON object");
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
        field.name = vformat("f%d", f);
        t.fields.push_back(field);
    }
    const int biomes = 2 + int(p_rng() % uint32_t(MAX_BIOMES - 1));
    for (int b = 0; b < biomes; b++) {
        Climate::Biome biome;
        biome.name = vformat("b%d", b);
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

TEST_CASE("Climate: the example preset parses at 5 cm and 15 LODs, and is refused where its octaves don't exist") {
    const Ref<FileAccess> f = FileAccess::open("res://climate/example.json", FileAccess::READ);
    CLIMATE_REQUIRE(f.is_valid());
    const String json = f->get_as_text();
    GroundFixture g(0.05, 15);
    Table t;
    CHECK(parse_json(json, g.ground, t) == String());
    CLIMATE_REQUIRE(t.fields.size() == 4);
    CLIMATE_REQUIRE(t.biomes.size() == 7);
    // 3.2 km .. 26 km at 5 cm: k = 16 .. 19, coarsest first, amplitude 0.8 times 0.7 per finer octave.
    const Field &temp = t.fields[0];
    CLIMATE_REQUIRE(temp.spec.bands.size() == 4);
    CHECK(temp.spec.bands.front().k == 19);
    CHECK(temp.spec.bands.back().k == 16);
    CHECK(temp.spec.bands[1].amp == float(0.8 * 0.7));
    CLIMATE_REQUIRE(temp.terms.size() == 1);
    CHECK(temp.terms[0].source == GROUND);
    CHECK(temp.terms[0].coef == float(-0.0003 * g.ground.voxel_size));
    // Seeds are keyed by k: the same octave of the same field always gets the same lattice.
    CHECK(temp.spec.bands[0].seed == ExactNoise::make_octave(temp.seed, 19u, 19, 1.0f).seed);
    CHECK(t.biomes.back().name == "plain");
    CHECK(t.biomes.back().edges.empty());
    CHECK(t.biomes[0].debug_color == 0xC03818u);
    CHECK(t.biomes[0].height == float(200.0 / g.ground.voxel_size));
    // Ground multipliers name ground octaves: 1 km .. 6.5 km = k 14 .. 17.
    CHECK(t.biomes[0].amps.size() == 4);

    GroundFixture g14(0.05, 14);
    Table t14;
    CHECK(parse_json(json, g14.ground, t14).contains("outside the ground's range"));
    CHECK(t14.fields.empty());
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
        { R"({"biomes": [{"name": "a", "height_m": 1e7}]})", "amplitude cap" },
        { R"({"biomes": [{"name": "a", "debug_color": "red"}]})", "#rrggbb" },
        { R"({"biomes": [{"name": "a", "debug_color": "#-00001"}]})", "#rrggbb" },
        { R"({"fields": [{"name": "t", "bands": [{"from_m": 1, "to_m": 1e30, "amp": 1}]}], "biomes": [{"name": "a"}]})", "outside [0, 62]" },
        { R"({"fields": [{"name": "t", "bands": [{"wavelength_m": 100, "to_m": 500, "amp": 1}]}], "biomes": [{"name": "a"}]})", "not both" },
        { R"({"fields": [{"name": "t", "seed": 1e20}], "biomes": [{"name": "a"}]})", "whole number" },
        { R"({"fields": [{"name": "t", "seed": 7.9}], "biomes": [{"name": "a"}]})", "whole number" },
        { R"({"fields": [{"name": "temp/2"}], "biomes": [{"name": "a"}]})", "identifier" },
    };
    for (const Case &c : cases) {
        CAPTURE(c.json);
        Table t;
        const String err = parse_json(c.json, g.ground, t);
        CHECK(err.contains(c.error));
        CHECK(t.biomes.empty());
    }
    // A multiplier of 0 turns an octave off, and the field count is capped.
    Table t;
    CHECK(parse_json(R"({"biomes": [{"name": "a", "amp": [{"field": "ground", "from_m": 4, "to_m": 100, "x": 0}]}]})", g.ground, t) == String());
    String many = R"({"fields": [)";
    for (int i = 0; i <= MAX_FIELDS; i++) {
        many += vformat("%s{\"name\": \"f%d\"}", i ? ", " : "", i);
    }
    CHECK(parse_json(many + R"(], "biomes": [{"name": "a"}]})", g.ground, t).contains("at most 8"));
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
        CHECK(validate(t, Ground()) == String());
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
    CLIMATE_REQUIRE(validate(t, Ground()) == String());
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
                            g.ground, t) == String());
    // Lattice indices: times 2^13 (the finest point at m = 12, doubled) they stay within +-2^60 voxels.
    const int64_t bases[] = { 0, 12345, (int64_t(1) << 32) - 3, -(int64_t(1) << 47) + 7, (int64_t(1) << 47) - 11 };
    int coherent = 0;
    for (const int64_t base : bases) {
        for (const int m : { 0, 3, 7, 12 }) {
            const Vector3i64 p(base, -base / 3 + 5, 7 - base);
            Float4 r[MAX_FIELDS];
            fields_at(t, g.ground, p, m, r);
            // Field a is its own spec; b adds -0.25 a; c is 0.001 * 0.05 ground + 2 b, in that order.
            float ch[Cascade::CHANNELS], gc[Cascade::CHANNELS];
            Cascade::point_into(t.fields[0].spec, p, m, ch);
            CHECK(float_bits(r[0].w) == float_bits(ch[Cascade::VALUE]));
            CHECK(float_bits(r[0].x) == float_bits(ch[Cascade::DX]));
            Cascade::point_into(t.fields[1].spec, p, m, ch);
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
                fields_at(t, g.ground, Vector3i64(p.x * 2, p.y * 2, p.z * 2), m, fine);
                Float4 coarse[MAX_FIELDS];
                fields_at(t, g.ground, p, m + 1, coarse);
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
