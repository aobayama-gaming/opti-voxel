#include "climate_report.hpp"

#include "climate.hpp"
#include "parallel_for.hpp"
#include "sdf_cascade.hpp"

#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/json.hpp>
#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/core/class_db.hpp>

#include <cmath>
#include <cstring>
#include <vector>

using namespace Climate;

namespace {

// The slice's values, point-major: fields[i * nf + f], weights[i * nb + b].
struct Sampled {
    int nf = 0;
    int nb = 0;
    std::vector<float> fields;
    std::vector<float> weights;
    std::vector<int> picks;
    std::vector<int64_t> ground_y; // LOD0 voxels: the column's surface, or the datum
    // The ground channels (SDFCascade::ground_channels) at the column's lattice point: height (LOD0 voxels) and its x / z
    // slopes per lattice spacing; fdx / fdz the central differences of the heights of its 4 neighbours at the same y.
    std::vector<float> h, gx, gz, fdx, fdz;
    std::vector<float> dhdy; // dh/dy at the lattice point (per voxel: past 1, the surface folds over)
};

// The slice: pixel (column j, row i) is the column (j - n / 2, i - n / 2) + p_c of Λ_m (x right, z down), at the
// lattice point at or below its ground surface (p_surface: SDFCascade::ground_y), else at the datum's.
Sampled sample(const Table &p_table, const SDFCascade &p_sdf, bool p_surface, int p_m, int p_n, const Vector2i &p_c) {
    Sampled s;
    s.nf = int(p_table.fields.size());
    s.nb = int(p_table.biomes.size());
    const size_t points = size_t(p_n) * p_n;
    s.fields.resize(points * s.nf);
    s.weights.resize(points * s.nb);
    s.picks.resize(points);
    s.ground_y.resize(points);
    s.h.resize(points);
    s.gx.resize(points);
    s.gz.resize(points);
    s.fdx.resize(points);
    s.fdz.resize(points);
    s.dhdy.resize(points);
    const int64_t step = int64_t(1) << p_m;
    parallel_for(uint32_t(p_n), "climate_report", [&](uint32_t p_row) {
        for (int j = 0; j < p_n; j++) {
            const size_t i = size_t(p_row) * p_n + j;
            const int64_t x = int64_t(j) - p_n / 2 + p_c.x;
            const int64_t z = int64_t(p_row) - p_n / 2 + p_c.y;
            s.ground_y[i] = p_surface ? p_sdf.ground_y(x * step, z * step) : p_sdf.get_datum_y();
            const Vector3i64 p(x, WorldCoord::floor_div(s.ground_y[i], step), z);
            Float4 f[MAX_FIELDS];
            Float4 w[MAX_BIOMES];
            float slots[Cascade::SLOTS], ch[Cascade::CHANNELS];
            Cascade::point_slots(p_sdf.get_spec(), p, p_m, slots, true);
            apply_terms(p_table, slots, f); // fields_at
            p_sdf.ground_channels(slots, ch);
            s.h[i] = ch[Cascade::VALUE];
            s.gx[i] = ch[Cascade::DX];
            s.gz[i] = ch[Cascade::DZ];
            s.dhdy[i] = float(double(ch[Cascade::DY]) * std::ldexp(1.0, -p_m));
            auto height = [&](int64_t p_dx, int64_t p_dz) {
                float c[Cascade::CHANNELS];
                p_sdf.channels_at(VoxelPos((p.x + p_dx) * step, p.y * step, (p.z + p_dz) * step), p_m, c);
                return double(c[Cascade::VALUE]);
            };
            s.fdx[i] = float((height(1, 0) - height(-1, 0)) * 0.5);
            s.fdz[i] = float((height(0, 1) - height(0, -1)) * 0.5);
            weights(p_table, f, w);
            s.picks[i] = pick(p_table, w, p, p_m);
            for (int c = 0; c < s.nf; c++) {
                s.fields[i * s.nf + c] = f[c].w;
            }
            for (int b = 0; b < s.nb; b++) {
                s.weights[i * s.nb + b] = w[b].w;
            }
        }
    });
    return s;
}

String save(const String &p_dir, const String &p_name, int p_n, const PackedByteArray &p_rgb) {
    const String path = p_dir.path_join(p_name);
    Image::create_from_data(p_n, p_n, false, Image::FORMAT_RGB8, p_rgb)->save_png(path);
    return path;
}

double channel(uint32_t p_color, int p_c) {
    return double((p_color >> (16 - 8 * p_c)) & 0xFFu);
}

// The biome table (share of the picks, mean weight) and the two maps: debug colours weighted, and the pick's colour.
String biome_section(const Table &p_table, const Sampled &p_s, int p_n, const String &p_dir, Array &r_images) {
    const int64_t points = int64_t(p_n) * p_n;
    std::vector<int64_t> count(size_t(p_s.nb), 0);
    std::vector<double> sum(size_t(p_s.nb), 0.0);
    PackedByteArray blend, picked;
    blend.resize(points * 3);
    picked.resize(points * 3);
    for (int64_t i = 0; i < points; i++) {
        double rgb[3] = { 0.0, 0.0, 0.0 };
        for (int b = 0; b < p_s.nb; b++) {
            const double w = double(p_s.weights[size_t(i) * p_s.nb + b]);
            sum[size_t(b)] += w;
            for (int c = 0; c < 3; c++) {
                rgb[c] += w * channel(p_table.biomes[size_t(b)].debug_color, c);
            }
        }
        const int pb = p_s.picks[size_t(i)];
        count[size_t(pb)]++;
        for (int c = 0; c < 3; c++) {
            blend[i * 3 + c] = uint8_t(CLAMP(std::lround(rgb[c]), 0L, 255L));
            picked[i * 3 + c] = uint8_t(channel(p_table.biomes[size_t(pb)].debug_color, c));
        }
    }
    String md = "| biome | picked | mean weight | debug colour |\n|---|---|---|---|\n";
    for (int b = 0; b < p_s.nb; b++) {
        md += vformat("| %s | %s%% | %s | #%06x |\n", p_table.biomes[size_t(b)].name, String::num(100.0 * double(count[size_t(b)]) / double(points), 1),
                String::num(sum[size_t(b)] / double(points), 3), int64_t(p_table.biomes[size_t(b)].debug_color));
    }
    r_images.push_back(save(p_dir, "biomes_blend.png", p_n, blend));
    r_images.push_back(save(p_dir, "biomes_pick.png", p_n, picked));
    return md;
}

// Where each pair of biomes meets, the spot nearest the slice's centre: a pick and its right or lower neighbour
// differ; the spot is the first of the two (metres, y its ground), with the player flag that spawns above it.
String border_section(const Table &p_table, const Sampled &p_s, int p_n, double p_spacing_m, double p_voxel_m, const Vector2i &p_c) {
    struct Spot {
        int64_t d2 = INT64_MAX;
        size_t i = 0;
    };
    std::vector<Spot> best(size_t(p_s.nb) * p_s.nb);
    for (int r = 0; r < p_n; r++) {
        for (int j = 0; j < p_n; j++) {
            const size_t i = size_t(r) * p_n + j;
            const int64_t dx = j - p_n / 2, dz = r - p_n / 2;
            const int right = j + 1 < p_n ? p_s.picks[i + 1] : p_s.picks[i];
            const int below = r + 1 < p_n ? p_s.picks[i + size_t(p_n)] : p_s.picks[i];
            for (const int o : { right, below }) {
                if (o == p_s.picks[i]) {
                    continue;
                }
                const int a = MIN(o, p_s.picks[i]), b = MAX(o, p_s.picks[i]);
                Spot &spot = best[size_t(a) * p_s.nb + b];
                if (dx * dx + dz * dz < spot.d2) {
                    spot = Spot{ dx * dx + dz * dz, i };
                }
            }
        }
    }
    String md = "| biomes | x m | ground y m | z m | player |\n|---|---|---|---|---|\n";
    for (int a = 0; a < p_s.nb; a++) {
        for (int b = a + 1; b < p_s.nb; b++) {
            const Spot &spot = best[size_t(a) * p_s.nb + b];
            if (spot.d2 == INT64_MAX) {
                continue;
            }
            const int64_t x = int64_t(std::llround(double(int64_t(spot.i % size_t(p_n)) - p_n / 2 + p_c.x) * p_spacing_m));
            const int64_t z = int64_t(std::llround(double(int64_t(spot.i / size_t(p_n)) - p_n / 2 + p_c.y) * p_spacing_m));
            md += vformat("| %s / %s | %d | %d | %d | `--spawn=%d,%d` |\n", p_table.biomes[size_t(a)].name, p_table.biomes[size_t(b)].name, x,
                    int64_t(std::llround(double(p_s.ground_y[spot.i]) * p_voxel_m)), z, x, z);
        }
    }
    return md;
}

// The ground's look and its gradient channels: a hillshade on height colours from the heights' finite differences
// (ground_shaded.png, the geometry) and one from the gradient channels (ground_normals.png, what the mesh's normals
// read), and per biome the gradient channels' error against the finite differences (at the same y): small where the
// gradient is the values' (the differences' own error: bands are sampled 8 times per wavelength), larger where a band
// has a crease the lattice can't resolve or a term leaves its own gradient out (how slope damping was measured and
// dropped, climate.hpp).
String shading_section(const Table &p_table, const Sampled &p_s, int p_n, int p_m, const String &p_dir, Array &r_images) {
    const int64_t points = int64_t(p_n) * p_n;
    double lo = INFINITY, hi = -INFINITY;
    for (int64_t i = 0; i < points; i++) {
        lo = MIN(lo, double(p_s.h[size_t(i)]));
        hi = MAX(hi, double(p_s.h[size_t(i)]));
    }
    const double to_slope = std::ldexp(1.0, -p_m); // per lattice spacing -> per voxel
    std::vector<double> err(size_t(p_s.nb), 0.0), norm(size_t(p_s.nb), 0.0);
    PackedByteArray fd_img, g_img;
    fd_img.resize(points * 3);
    g_img.resize(points * 3);
    // Height colours: land (t in [0, 1]) low green, mid olive and brown, high grey-white; sea (t in [-1, 0): -1 the
    // deepest) light to dark blue.
    auto colour = [&](double p_t, int p_c) {
        static const double STOPS[4][3] = { { 70, 110, 60 }, { 140, 150, 90 }, { 130, 105, 80 }, { 235, 235, 240 } };
        static const double SEA[2][3] = { { 30, 50, 110 }, { 90, 140, 190 } };
        if (p_t < 0.0) {
            const double u = CLAMP(p_t + 1.0, 0.0, 1.0);
            return SEA[0][p_c] + (SEA[1][p_c] - SEA[0][p_c]) * u;
        }
        const double t = CLAMP(p_t, 0.0, 1.0) * 3.0;
        const int k = MIN(int(t), 2);
        return STOPS[k][p_c] + (STOPS[k + 1][p_c] - STOPS[k][p_c]) * (t - k);
    };
    auto shade = [](double p_sx, double p_sz) {
        const double nx = -p_sx, ny = 1.0, nz = -p_sz;
        const double ln = std::sqrt(nx * nx + ny * ny + nz * nz);
        return MAX(0.0, (nx * -0.577 + ny * 0.577 + nz * -0.577) / ln);
    };
    for (int r = 0; r < p_n; r++) {
        for (int j = 0; j < p_n; j++) {
            const size_t i = size_t(r) * p_n + j;
            const double fdx = double(p_s.fdx[i]);
            const double fdz = double(p_s.fdz[i]);
            {
                const size_t b = size_t(p_s.picks[i]);
                err[b] += std::fabs(double(p_s.gx[i]) - fdx) + std::fabs(double(p_s.gz[i]) - fdz);
                norm[b] += std::fabs(fdx) + std::fabs(fdz);
            }
            // Land: its height between 0 (the datum, the sea level) and the slice's top; below the datum, sea blues.
            const double hv = double(p_s.h[i]);
            const double t = hv >= 0.0 ? hv / MAX(hi, 1e-9) : -1.0 - hv / MIN(lo, -1e-9);
            const double sf = shade(fdx * to_slope, fdz * to_slope);
            const double sg = shade(double(p_s.gx[i]) * to_slope, double(p_s.gz[i]) * to_slope);
            for (int c = 0; c < 3; c++) {
                fd_img[i * 3 + c] = uint8_t(CLAMP(std::lround(colour(t, c) * (0.3 + 0.9 * sf)), 0L, 255L));
                g_img[i * 3 + c] = uint8_t(CLAMP(std::lround(colour(t, c) * (0.3 + 0.9 * sg)), 0L, 255L));
            }
        }
    }
    r_images.push_back(save(p_dir, "ground_shaded.png", p_n, fd_img));
    r_images.push_back(save(p_dir, "ground_normals.png", p_n, g_img));
    String md = "Gradient channels vs the heights' finite differences (at the column's y, L1 relative error):\n\n| biome | gradient error |\n|---|---|\n";
    for (int b = 0; b < p_s.nb; b++) {
        md += vformat("| %s | %s |\n", p_table.biomes[size_t(b)].name, norm[size_t(b)] > 0.0 ? String::num(err[size_t(b)] / norm[size_t(b)], 3) : String("-"));
    }
    return md;
}

// The field table (octaves, min / mean / max) and one grey map per field (-max|v| .. +max|v|).
String field_section(const Table &p_table, const Cascade::Spec &p_spec, const Sampled &p_s, int p_n, int p_m, const String &p_dir, Array &r_images) {
    const int64_t points = int64_t(p_n) * p_n;
    String md = "| field | octaves (k) | min | mean | max |\n|---|---|---|---|---|\n";
    for (int f = 0; f < p_s.nf; f++) {
        auto v = [&](int64_t p_i) { return double(p_s.fields[size_t(p_i) * p_s.nf + f]); };
        double lo = INFINITY, hi = -INFINITY, sum = 0.0;
        for (int64_t i = 0; i < points; i++) {
            lo = MIN(lo, v(i));
            hi = MAX(hi, v(i));
            sum += v(i);
        }
        const std::vector<Cascade::Band> &bands = p_table.fields[size_t(f)].bands;
        String ks = bands.empty() ? String("-") : vformat("%d .. %d", bands.back().k, bands.front().k);
        if (bands.empty() || p_spec.insert_level(bands.front()) < p_m) {
            ks += vformat(" (all inserted below lattice %d: only its terms here)", p_m);
        }
        md += vformat("| %s | %s | %s | %s | %s |\n", p_table.fields[size_t(f)].name, ks, String::num(lo, 3), String::num(sum / double(points), 3),
                String::num(hi, 3));
        const double range = MAX(MAX(std::fabs(lo), std::fabs(hi)), 1e-30);
        PackedByteArray grey;
        grey.resize(points * 3);
        for (int64_t i = 0; i < points; i++) {
            grey[i * 3] = grey[i * 3 + 1] = grey[i * 3 + 2] = uint8_t(CLAMP(std::lround(127.5 + 127.5 * v(i) / range), 0L, 255L));
        }
        r_images.push_back(save(p_dir, "field_" + p_table.fields[size_t(f)].name + String(".png"), p_n, grey));
    }
    return md;
}

// The ground under each biome's picks: its height spread and its mean slope between neighbouring columns of the same
// pick (the relief of the octaves longer than about two lattice spacings), and a grey height map.
String relief_section(const Table &p_table, const Sampled &p_s, int p_n, double p_spacing_m, double p_voxel_m, const String &p_dir,
        Array &r_images) {
    const int64_t points = int64_t(p_n) * p_n;
    auto height = [&](int64_t p_i) { return double(p_s.ground_y[size_t(p_i)]) * p_voxel_m; };
    std::vector<double> sum(size_t(p_s.nb), 0.0), sum2(size_t(p_s.nb), 0.0), slope(size_t(p_s.nb), 0.0);
    std::vector<int64_t> count(size_t(p_s.nb), 0), pairs(size_t(p_s.nb), 0);
    double lo = INFINITY, hi = -INFINITY;
    for (int64_t i = 0; i < points; i++) {
        const size_t b = size_t(p_s.picks[size_t(i)]);
        const double h = height(i);
        lo = MIN(lo, h);
        hi = MAX(hi, h);
        sum[b] += h;
        sum2[b] += h * h;
        count[b]++;
        const int64_t col = i % p_n;
        for (const int64_t nb : { col + 1 < p_n ? i + 1 : int64_t(-1), i + p_n < points ? i + p_n : int64_t(-1) }) {
            if (nb >= 0 && size_t(p_s.picks[size_t(nb)]) == b) {
                slope[b] += std::fabs(height(nb) - h) / p_spacing_m;
                pairs[b]++;
            }
        }
    }
    String md = vformat("Ground height over the slice: %s .. %s m.\n\n| biome | mean height m | height std m | mean slope (m/m) |\n|---|---|---|---|\n",
            String::num(lo, 0), String::num(hi, 0));
    for (int b = 0; b < p_s.nb; b++) {
        const double c = double(MAX(count[size_t(b)], int64_t(1)));
        const double mean = sum[size_t(b)] / c;
        md += vformat("| %s | %s | %s | %s |\n", p_table.biomes[size_t(b)].name, String::num(mean, 0),
                String::num(std::sqrt(MAX(sum2[size_t(b)] / c - mean * mean, 0.0)), 0),
                String::num(slope[size_t(b)] / double(MAX(pairs[size_t(b)], int64_t(1))), 3));
    }
    PackedByteArray grey;
    grey.resize(points * 3);
    for (int64_t i = 0; i < points; i++) {
        grey[i * 3] = grey[i * 3 + 1] = grey[i * 3 + 2] = uint8_t(CLAMP(std::lround(255.0 * (height(i) - lo) / MAX(hi - lo, 1e-9)), 0L, 255L));
    }
    r_images.push_back(save(p_dir, "ground_height.png", p_n, grey));
    return md;
}

void store(const String &p_path, const PackedByteArray &p_bytes) {
    const Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::WRITE);
    ERR_FAIL_COND_MSG(f.is_null(), "ClimateReport: cannot write " + p_path);
    f->store_buffer(p_bytes);
}

// A vector's bytes (little endian on every target this builds for).
template <typename T>
void store(const String &p_path, const std::vector<T> &p_v) {
    PackedByteArray bytes;
    bytes.resize(int64_t(p_v.size() * sizeof(T)));
    if (!p_v.empty()) {
        memcpy(bytes.ptrw(), p_v.data(), size_t(bytes.size()));
    }
    store(p_path, bytes);
}

// The heightmap for tools/landforms (compare.py, match.py): heights.f32 (float32 little endian, row-major, row = z, metres
// above the datum), fields.f32 (the fields' values, point-major like Sampled::fields), dhdy.f32 (the ground's dh/dy: the 3D
// noise's vertical slope, past 1 an overhang), picks.u8 (the picked biome) and
// heights.json (n, spacing, the table, the centre in metres, the field and biome names, the cascade constants match.py's
// amplitude cap reads).
void write_heights(const String &p_dir, const Table &p_table, const Sampled &p_s, int p_n, double p_spacing_m, double p_voxel_m, int64_t p_datum_y,
        const String &p_climate, const Vector2i &p_c) {
    std::vector<float> h(p_s.ground_y.size());
    for (size_t i = 0; i < h.size(); i++) {
        h[i] = float(double(p_s.ground_y[i] - p_datum_y) * p_voxel_m);
    }
    store(p_dir.path_join("heights.f32"), h);
    store(p_dir.path_join("fields.f32"), p_s.fields);
    store(p_dir.path_join("dhdy.f32"), p_s.dhdy);
    store(p_dir.path_join("picks.u8"), std::vector<uint8_t>(p_s.picks.begin(), p_s.picks.end()));
    Array field_names, biome_names;
    for (const Field &f : p_table.fields) {
        field_names.push_back(f.name);
    }
    for (const Climate::Biome &b : p_table.biomes) {
        biome_names.push_back(b.name);
    }
    Dictionary meta;
    meta["fields"] = field_names;
    meta["biomes"] = biome_names;
    meta["n"] = p_n;
    meta["spacing_m"] = p_spacing_m;
    meta["climate"] = p_climate;
    meta["center_m"] = Array::make(double(p_c.x) * p_spacing_m, double(p_c.y) * p_spacing_m);
    // For match.py's multiplier cap (shape_bound under the ground's cap).
    meta["amp_cap_m"] = SDFCascade::MAX_AMP_M;
    meta["smooth_abs_e"] = Cascade::SMOOTH_ABS_E;
    meta["noise_abs_mean"] = Cascade::NOISE_ABS_MEAN;
    meta["noise_sq_mean"] = Cascade::NOISE_SQ_MEAN;
    store(p_dir.path_join("heights.json"), JSON::stringify(meta, " ").to_utf8_buffer());
}

} // namespace

void ClimateReport::_bind_methods() {
    ClassDB::bind_method(D_METHOD("run", "out_dir", "climate_path", "voxel_size", "lods", "extent_km", "size", "surface", "center", "lattice"), &ClimateReport::run);
}

Dictionary ClimateReport::run(const String &p_out_dir, const String &p_climate_path, double p_voxel_size, int p_lods, double p_extent_km, int p_size, bool p_surface,
        const Vector2 &p_center, int p_lattice) {
    Dictionary report;
    report["ok"] = false;
    DirAccess::make_dir_recursive_absolute(p_out_dir);
    ChunkParam param;
    param.voxel_size = float(p_voxel_size);
    SDFCascade sdf;
    String err = load(p_climate_path, sdf.climate_data);
    if (err.is_empty()) {
        sdf.configure(param, p_lods);
        err = sdf.get_climate_error();
    }
    if (!err.is_empty()) {
        report["error"] = err;
        ERR_FAIL_V_MSG(report, "ClimateReport: " + err);
    }
    ERR_FAIL_NULL_V_MSG(sdf.get_climate(), report, "ClimateReport: no climate.");
    const Table &table = *sdf.get_climate();
    const Ground ground = sdf.climate_ground();

    const int n = MAX(p_size, 16);
    if (p_lattice > sdf.get_spec().top_level()) {
        report["error"] = vformat("--lattice=%d: above the cascade's top lattice %d.", p_lattice, sdf.get_spec().top_level());
        ERR_FAIL_V_MSG(report, "ClimateReport: " + String(report["error"]));
    }
    const int m = p_lattice >= 0 ? p_lattice : MAX(0, int(std::floor(std::log2(p_extent_km * 1000.0 / n / ground.voxel_size))));
    const double spacing_m = std::ldexp(ground.voxel_size, m);
    const uint64_t t0 = Time::get_singleton()->get_ticks_usec();
    const Vector2i c(int(std::llround(p_center.x / spacing_m)), int(std::llround(p_center.y / spacing_m)));
    const Sampled s = sample(table, sdf, p_surface, m, n, c);
    const double us_per_point = double(Time::get_singleton()->get_ticks_usec() - t0) / (double(n) * n);

    String md = "# Climate report\n\n";
    md += vformat("`%s` at %s m voxels, %d LODs: %d fields, %d biomes. Slice %s, %d x %d points of lattice %d (%s m apart, "
                  "%s km across). CPU: %s us per point (ground trace + fields + weights + pick, over the worker pool).\n\n",
            p_climate_path, String::num(ground.voxel_size, 6), p_lods, s.nf, s.nb, p_surface ? "on the ground surface" : "through the datum", n, n, m, String::num(spacing_m, 2), String::num(n * spacing_m / 1000.0, 1),
            String::num(us_per_point, 2));
    Array images;
    md += biome_section(table, s, n, p_out_dir, images) + String("\n");
    md += border_section(table, s, n, spacing_m, ground.voxel_size, c) + String("\n");
    md += relief_section(table, s, n, spacing_m, ground.voxel_size, p_out_dir, images) + String("\n");
    md += shading_section(table, s, n, m, p_out_dir, images) + String("\n");
    md += field_section(table, sdf.get_spec(), s, n, m, p_out_dir, images);
    if (p_surface) { // a datum slice has no surface heights
        write_heights(p_out_dir, table, s, n, spacing_m, ground.voxel_size, sdf.get_datum_y(), p_climate_path, c);
    }
    md += "\nImages (x right, z down): ";
    for (int64_t i = 0; i < images.size(); i++) {
        md += String(i ? ", " : "") + "`" + String(images[i]).get_file() + String("`");
    }
    const String report_path = p_out_dir.path_join("report.md");
    const Ref<FileAccess> out = FileAccess::open(report_path, FileAccess::WRITE);
    ERR_FAIL_COND_V_MSG(out.is_null(), report, "ClimateReport: cannot write report.md");
    out->store_string(md + String("\n"));
    report["report"] = report_path;
    report["images"] = images;
    report["us_per_point"] = us_per_point;
    report["ok"] = true;
    return report;
}
