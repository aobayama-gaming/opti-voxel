#include "cascade_report.hpp"

#include "cascade.hpp"
#include "cascade_measure.hpp"
#include "cascade_terrain.hpp"

#include "ov/files.hpp"
#include "ov/math.hpp"
#include "ov/on_error.hpp"
#include "ov/png.hpp"
#include "ov/str.hpp"

#include <cmath>

using namespace Cascade;
using namespace CascadeMeasure;

namespace {

constexpr int OCTAVE_K = 7; // wavelength 128 voxels, for the fidelity table and the slices
constexpr int FIDELITY_BOX = 64; // voxels per box side
constexpr int COST_CHUNKS = 200;
constexpr int SLICE_SIZE = 384; // three wavelengths
constexpr int COST_LODS[] = { 24, 20, 16, 12, 8, 4, 0 };

// The measured variants: (insert shift, channel set).
struct Measured {
    int shift;
    ChannelSet set;
};
constexpr Measured VARIANTS[] = { { 2, ChannelSet::HERMITE8 }, { 2, ChannelSet::LINEAR4 }, { 3, ChannelSet::HERMITE8 }, { 3, ChannelSet::LINEAR4 } };

ov::str set_name(ChannelSet p_set) {
    return p_set == ChannelSet::HERMITE8 ? "8 channels" : "4 channels (linear transverse)";
}

ov::str fidelity_section(int p_scale) {
    const int boxes = 16 * p_scale;
    ov::str md = "## Fidelity of one cascaded octave vs analytic perlin3\n\n";
    md += ov::format("Octave k = %d (amplitude 1) alone, cascaded to every voxel of %d random boxes of %d^3 voxels (a quarter of them near "
                     "2^50), compared with perlin3. Normal: 3D-height normal `t = (-hx, 1 - hy, -hz)` at steepness r = amplitude / wavelength. "
                     "Pass: value max <= %s%% and normal <= %s deg at r = 1/8.\n\n",
            OCTAVE_K, boxes, FIDELITY_BOX, ov::num(100.0 * MAX_VALUE_ERROR), ov::num(MAX_NORMAL_DEG));
    md += "| samples / wavelength | channels | points | value max | value rms | slope max (per cell) | normal r=1/16 | r=1/8 | r=1/4 | pass |\n";
    md += "|---|---|---|---|---|---|---|---|---|---|\n";
    for (const Measured &variant : VARIANTS) {
        Spec spec;
        spec.set = variant.set;
        spec.insert_shift = variant.shift;
        const Fidelity f = measure_fidelity(spec, OCTAVE_K, boxes, FIDELITY_BOX, 0x5EEDu);
        md += ov::format("| %d | %s | %d | %s%% | %s%% | %s | %s deg | %s deg | %s deg | %s |\n", 1 << variant.shift, set_name(variant.set), f.points,
                ov::num(100.0 * f.value_max, 3), ov::num(100.0 * f.value_rms, 3), ov::num(f.slope_max, 4), ov::num(f.normal_max_deg[0], 3),
                ov::num(f.normal_max_deg[1], 3), ov::num(f.normal_max_deg[2], 3), f.passes() ? "yes" : "no");
    }
    return md + "\n";
}

ov::str cost_section(int p_scale) {
    const Spec spec8 = CascadeTerrain::earth(ChannelSet::HERMITE8);
    const Spec spec4 = CascadeTerrain::earth(ChannelSet::LINEAR4);
    const int timed = 4 * p_scale;
    ov::str md = ov::format("## Points per chunk (Earth: %d octaves up to k = %d, 8 samples per wavelength, top lattice %d)\n\n", int64_t(spec8.bands.size()),
            spec8.bands.front().k, spec8.top_level());
    md += ov::format("Lattice points of %d random chunks per LOD (the boxes Cascade::run computes); CPU time per chunk over %d evaluated chunks "
                     "(one thread, template_debug build).\n\n",
            COST_CHUNKS, timed);
    md += "| LOD | levels | points per chunk | max | vs LOD 24 | levels L, L+1, L+2, L+3 ... top | ms / chunk, 8 ch | ms / chunk, 4 ch |\n";
    md += "|---|---|---|---|---|---|---|---|\n";
    double base = 0.0;
    for (const int lod : COST_LODS) {
        const Cost c = measure_cost(spec8, lod, COST_CHUNKS, 1, false);
        const Cost t8 = measure_cost(spec8, lod, timed, 2, true);
        const Cost t4 = measure_cost(spec4, lod, timed, 2, true);
        if (base == 0.0) {
            base = c.points_mean;
        }
        ov::str levels;
        for (size_t i = 0; i < c.first_levels.size() && i < 4; i++) {
            levels += ov::cat(i > 0 ? ", " : "", c.first_levels[i]);
        }
        if (c.first_levels.size() > 4) {
            levels += ov::cat(" ... ", c.first_levels.back());
        }
        md += ov::format("| %d | %d | %s | %d | %s | %s | %s | %s |\n", lod, int64_t(c.first_levels.size()), ov::num(c.points_mean, 0), c.points_max,
                ov::num(c.points_mean / base, 4), levels, ov::num(t8.ms_per_chunk, 1), ov::num(t4.ms_per_chunk, 1));
    }
    const int64_t samples = int64_t(ChunkParam().chunk_size) * ChunkParam().chunk_size * ChunkParam().chunk_size;
    md += ov::format("\nEvery octave analytic at every sample instead (%d octaves at LOD 0): %d evaluations per chunk.\n\n", int64_t(spec8.bands.size()),
            samples * int64_t(spec8.bands.size()));
    return md;
}

// A slice of octave OCTAVE_K through the (x, z) plane: analytic | cascaded | error, value (top row) and
// shaded relief at r = 1/8 (bottom row). Returns the PNG file name written in p_dir.
ov::str write_slice(const ov::str &p_dir, const Measured &p_variant) {
    Spec spec;
    spec.set = p_variant.set;
    spec.insert_shift = p_variant.shift;
    spec.bands = ExactNoise::make_fbm(false, 0x51CEu, OCTAVE_K, 1, 1.0, 0.5).octaves;
    const Band &band = spec.bands[0];
    const int n = SLICE_SIZE;
    const Box box{ { 0, 100, 0 }, { n - 1, 100, n - 1 } };
    const Level v = run(spec, box, 0);
    const double light[3] = { 0.408, 0.816, 0.408 };
    ov::bytes data(size_t(3 * n) * (2 * n) * 3);
    auto put = [&](int p_x, int p_y, double p_r, double p_g, double p_b) {
        const int64_t o = (int64_t(p_y) * (3 * n) + p_x) * 3;
        data[o + 0] = uint8_t(ov::clamp(p_r, 0.0, 1.0) * 255.0);
        data[o + 1] = uint8_t(ov::clamp(p_g, 0.0, 1.0) * 255.0);
        data[o + 2] = uint8_t(ov::clamp(p_b, 0.0, 1.0) * 255.0);
    };
    auto put_gray = [&](int p_x, int p_y, double p_g) { put(p_x, p_y, p_g, p_g, p_g); };
    auto shade = [&](const ExactNoise::Float4 &p_t) { return ov::max(0.0, p_t.x * light[0] + p_t.y * light[1] + p_t.z * light[2]); };
    for_each_point(box, [&](const Vector3i64 &p_p, int64_t p_i) {
        const int x = int(p_p.x);
        const int z = int(p_p.z);
        WorldPos pos;
        pos.voxel = p_p;
        const ExactNoise::Float4 a = ExactNoise::perlin3(pos, OCTAVE_K, band.seed, band.offset);
        const float value = v.ch[VALUE][p_i];
        ExactNoise::Float4 tc, ta;
        height_normals(v, p_p, a, OCTAVE_K, STEEPNESS_LOG2[REF_STEEPNESS], tc, ta);
        put_gray(x, z, 0.5 + 0.5 * a.w);
        put_gray(n + x, z, 0.5 + 0.5 * value);
        put(2 * n + x, z, std::fabs(value - a.w) / MAX_VALUE_ERROR, 0.0, 0.0); // full red at the criterion
        put_gray(x, n + z, shade(ta));
        put_gray(n + x, n + z, shade(tc));
        put(2 * n + x, n + z, angle_deg(tc, ta) / MAX_NORMAL_DEG, 0.0, 0.0);
    });
    const ov::str name = ov::format("slice_spw%d_%s.png", 1 << p_variant.shift, p_variant.set == ChannelSet::HERMITE8 ? "8ch" : "4ch");
    const ov::bytes png = ov::encode_png(3 * n, 2 * n, 3, data.data());
    ov::write_file(ov::path_join(p_dir, name), ov::str(png.begin(), png.end()));
    return name;
}

} // namespace

ov::Value CascadeReport::run(const ov::str &p_out_dir, int p_scale) {
    ov::Value report = ov::Value::object();
    report["ok"] = false;
    ov::make_dirs(p_out_dir);
    const int scale = ov::max(p_scale, 1);
    ov::str md = "# Hermite cascade report (CPU reference)\n\n";
    md += fidelity_section(scale);
    md += cost_section(scale);
    md += ov::format("## Slices (octave k = %d, plane y = 100, %d x %d voxels)\n\n", OCTAVE_K, SLICE_SIZE, SLICE_SIZE);
    md += "Top: value (analytic | cascaded | error, full red = the 1% criterion). Bottom: shaded relief at r = 1/8 (analytic | cascaded | "
          "normal error, full red = 1 deg).\n\n";
    ov::Value slices = ov::Value::array();
    for (const Measured &variant : VARIANTS) {
        const ov::str name = write_slice(p_out_dir, variant);
        slices.push_back(ov::path_join(p_out_dir, name));
        md += "- `" + name + "`\n";
    }
    const ov::str path = ov::path_join(p_out_dir, "report.md");
    OV_ERR_FAIL_COND_V_MSG(!ov::write_file(path, md), report, "CascadeReport: cannot write report.md");
    report["report"] = path;
    report["slices"] = slices;
    report["ok"] = true;
    return report;
}
