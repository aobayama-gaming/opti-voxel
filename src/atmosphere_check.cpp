#include "atmosphere_check.hpp"

#include "atmosphere.hpp"
#include "gpu_shader.hpp"

#include "ov/math.hpp"

#include <cmath>
#include <cstring>

namespace {

using ov::Value;

constexpr double PI = WorldCoord::PI;
constexpr int SUN_STEPS = Atmosphere::SUN_STEPS;
constexpr int FAR_STEPS = Atmosphere::FAR_STEPS;
constexpr int MS_STEPS = Atmosphere::MS_STEPS;
constexpr int MS_DIRS = Atmosphere::MS_DIRS;

struct V3 {
    double x = 0.0, y = 0.0, z = 0.0;
    V3 operator+(const V3 &o) const { return { x + o.x, y + o.y, z + o.z }; }
    V3 operator-(const V3 &o) const { return { x - o.x, y - o.y, z - o.z }; }
    V3 operator*(double s) const { return { x * s, y * s, z * s }; }
    V3 operator*(const V3 &o) const { return { x * o.x, y * o.y, z * o.z }; }
};

double dot(const V3 &a, const V3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

V3 cross(const V3 &a, const V3 &b) {
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

V3 normalize(const V3 &a) {
    return a * (1.0 / std::sqrt(dot(a, a)));
}

V3 v3(const Float3 &f) {
    return { f.x, f.y, f.z };
}

V3 vexp(const V3 &a) {
    return { std::exp(a.x), std::exp(a.y), std::exp(a.z) };
}

// The GLSL march (atmosphere.cpp: GLSL_MARCH, the passes) in double over the uniforms of one case.
struct Ref {
    BaseSurface::Render base;
    AtmosphereMedium m;
    Atmosphere::Frame f;

    bool sphere() const { return base.kind == BaseSurface::Kind::SPHERE; }
    V3 centre() const { return v3(base.point_m); }

    double altitude(const V3 &p) const {
        return sphere() ? std::sqrt(dot(p - centre(), p - centre())) - double(base.radius_m) : p.y - double(base.point_m.y);
    }
    V3 up(const V3 &p) const { return sphere() ? normalize(p - centre()) : V3{ 0.0, 1.0, 0.0 }; }
    bool level_span(const V3 &p, const V3 &d, double h, double t[2]) const {
        if (sphere()) {
            const V3 o = p - centre();
            const double r = double(base.radius_m) + h;
            const double b = dot(o, d);
            const double disc = b * b - (dot(o, o) - r * r);
            const double s = std::sqrt(ov::max(disc, 0.0));
            t[0] = -b - s;
            t[1] = -b + s;
            return disc >= 0.0;
        }
        const double below = h - altitude(p);
        if (d.y > 0.0) {
            t[0] = -1e30;
            t[1] = below / d.y;
        } else if (d.y < 0.0) {
            t[0] = below / d.y;
            t[1] = 1e30;
        } else {
            t[0] = -1e30;
            t[1] = 1e30;
        }
        return d.y != 0.0 || below > 0.0;
    }
    void medium(double h, V3 &r, double &mie, V3 &e) const {
        const double a = ov::max(h, 0.0);
        r = v3(m.rayleigh) * std::exp(-a / double(m.rayleigh_height_m));
        const double k = std::exp(-a / double(m.mie_height_m));
        mie = double(m.mie) * k;
        const double oz = ov::max(0.0, 1.0 - std::fabs(a - double(m.ozone_centre_m)) / (0.5 * double(m.ozone_width_m)));
        const double me = (double(m.mie) + double(m.mie_absorption)) * k;
        e = r + V3{ me, me, me } + v3(m.ozone) * oz;
    }
    double rayleigh_phase(double c) const { return 3.0 / (16.0 * PI) * (1.0 + c * c); }
    double mie_phase(double c) const {
        const double g = double(m.mie_g);
        const double k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
        return k * (1.0 + c * c) / std::pow(ov::max(1.0 + g * g - 2.0 * g * c, 1e-4), 1.5);
    }
    bool segment(const V3 &p, const V3 &d, double seg[2], bool &ground) const {
        ground = false;
        double top[2];
        if (!level_span(p, d, double(m.top_m), top) || top[1] <= 0.0) {
            return false;
        }
        seg[0] = ov::max(top[0], 0.0);
        seg[1] = top[1];
        double below[2];
        if (level_span(p, d, 0.0, below) && below[0] > seg[0] && below[0] < seg[1]) {
            seg[1] = below[0];
            ground = true;
        }
        seg[1] = ov::min(seg[1], seg[0] + double(f.max_path_m));
        return seg[1] > seg[0];
    }
    V3 transmittance(const V3 &p, const V3 &d, int n) const {
        double seg[2];
        bool ground;
        if (!segment(p, d, seg, ground)) {
            return { 1.0, 1.0, 1.0 };
        }
        if (ground) {
            return {};
        }
        V3 depth;
        const double len = seg[1] - seg[0];
        for (int i = 0; i < n; i++) {
            const double a = double(i) / n, b = double(i + 1) / n;
            V3 r, e;
            double mie;
            medium(altitude(p + d * (seg[0] + 0.25 * (a + b) * (a + b) * len)), r, mie, e);
            depth = depth + e * ((b * b - a * a) * len);
        }
        return vexp(depth * -1.0);
    }
    V3 ms(const float *p_table, double h, double sun_height) const {
        const int n = Atmosphere::MS_SIZE;
        const double fx = ov::clamp((0.5 + 0.5 * sun_height) * n - 0.5, 0.0, double(n - 1));
        const double fy = ov::clamp(h / double(m.top_m) * n - 0.5, 0.0, double(n - 1));
        const int ix = int(std::floor(fx)), iy = int(std::floor(fy));
        const int jx = ov::min(ix + 1, n - 1), jy = ov::min(iy + 1, n - 1);
        const double wx = fx - ix, wy = fy - iy;
        auto at = [&](int x, int y) {
            const float *t = p_table + (size_t(y) * n + x) * 4;
            return V3{ t[0], t[1], t[2] };
        };
        const V3 a = at(ix, iy) * (1.0 - wx) + at(jx, iy) * wx;
        const V3 b = at(ix, jy) * (1.0 - wx) + at(jx, jy) * wx;
        return a * (1.0 - wy) + b * wy;
    }
    void step(const float *p_table, const V3 &p, const V3 &d, const V3 &sun, double cos_sun, double t, double dt, V3 &light, V3 &trans) const {
        const V3 x = p + d * t;
        const double h = altitude(x);
        V3 r, e;
        double mie;
        medium(h, r, mie, e);
        const double pm = mie * mie_phase(cos_sun);
        V3 s = transmittance(x, sun, SUN_STEPS) * (r * rayleigh_phase(cos_sun) + V3{ pm, pm, pm });
        s = s + ms(p_table, h, dot(up(x), sun)) * (r + V3{ mie, mie, mie });
        const V3 td = vexp(e * -dt);
        const V3 one{ 1.0, 1.0, 1.0 };
        const V3 inv{ 1.0 / ov::max(e.x, 1e-20), 1.0 / ov::max(e.y, 1e-20), 1.0 / ov::max(e.z, 1e-20) };
        light = light + trans * (s * (one - td)) * inv;
        trans = trans * td;
    }
    V3 ground_light(const V3 &x, const V3 &sun) const {
        const V3 u = up(x);
        return transmittance(x + u * (1e-3 * double(m.top_m)), sun, SUN_STEPS) * (ov::max(dot(u, sun), 0.0) / PI) * v3(m.ground_albedo);
    }

    // The multiple-scattering table's texel (x, y) (GLSL_PASS_MULTISCATTER).
    V3 ms_texel(int p_x, int p_y) const {
        const int n = Atmosphere::MS_SIZE;
        const double radius = double(f.ms_radius_m);
        const double mu = (p_x + 0.5) / n * 2.0 - 1.0;
        const double h = (p_y + 0.5) / n * double(m.top_m);
        const V3 x{ 0.0, radius + h, 0.0 };
        const V3 sun{ std::sqrt(ov::max(1.0 - mu * mu, 0.0)), mu, 0.0 };
        auto sphere_far = [](const V3 &o, const V3 &d, double r) {
            const double b = dot(o, d);
            return -b + std::sqrt(ov::max(b * b - (dot(o, o) - r * r), 0.0));
        };
        auto sphere_near = [](const V3 &o, const V3 &d, double r, double &t) {
            const double b = dot(o, d);
            const double disc = b * b - (dot(o, o) - r * r);
            t = -b - std::sqrt(ov::max(disc, 0.0));
            return disc >= 0.0 && t > 0.0;
        };
        auto sun_t = [&](const V3 &y) {
            double tg;
            if (sphere_near(y, sun, radius, tg)) {
                return V3{};
            }
            const double len = sphere_far(y, sun, radius + double(m.top_m));
            V3 depth;
            for (int i = 0; i < SUN_STEPS; i++) {
                const double a = double(i) / SUN_STEPS, b = double(i + 1) / SUN_STEPS;
                V3 r, e;
                double mie;
                medium(std::sqrt(dot(y + sun * (0.25 * (a + b) * (a + b) * len), y + sun * (0.25 * (a + b) * (a + b) * len))) - radius, r, mie, e);
                depth = depth + e * ((b * b - a * a) * len);
            }
            return vexp(depth * -1.0);
        };
        V3 l2, fms;
        for (int i = 0; i < MS_DIRS; i++) {
            for (int j = 0; j < MS_DIRS; j++) {
                const double c = 1.0 - 2.0 * (j + 0.5) / MS_DIRS;
                const double phi = 2.0 * PI * (i + 0.5) / MS_DIRS;
                const double s = std::sqrt(ov::max(1.0 - c * c, 0.0));
                const V3 d{ s * std::cos(phi), c, s * std::sin(phi) };
                double t_end;
                const bool ground = sphere_near(x, d, radius, t_end);
                if (!ground) {
                    t_end = sphere_far(x, d, radius + double(m.top_m));
                }
                const double dt = t_end / MS_STEPS;
                V3 trans{ 1.0, 1.0, 1.0 };
                for (int k = 0; k < MS_STEPS; k++) {
                    const V3 y = x + d * ((k + 0.5) * dt);
                    V3 r, e;
                    double mie;
                    medium(std::sqrt(dot(y, y)) - radius, r, mie, e);
                    const V3 scat = r + V3{ mie, mie, mie };
                    const V3 td = vexp(e * -dt);
                    const V3 integral{ (1.0 - td.x) / ov::max(e.x, 1e-20), (1.0 - td.y) / ov::max(e.y, 1e-20), (1.0 - td.z) / ov::max(e.z, 1e-20) };
                    l2 = l2 + trans * sun_t(y) * scat * integral * (1.0 / (4.0 * PI));
                    fms = fms + trans * scat * integral;
                    trans = trans * td;
                }
                if (ground) {
                    const V3 g = x + d * t_end;
                    const V3 u = normalize(g);
                    l2 = l2 + trans * sun_t(g + u * (1e-3 * double(m.top_m))) * v3(m.ground_albedo) * (ov::max(dot(u, sun), 0.0) / PI);
                }
            }
        }
        const double n2 = double(MS_DIRS * MS_DIRS);
        l2 = l2 * (1.0 / n2);
        fms = fms * (1.0 / n2);
        return { l2.x / ov::max(1.0 - fms.x, 1e-3), l2.y / ov::max(1.0 - fms.y, 1e-3), l2.z / ov::max(1.0 - fms.z, 1e-3) };
    }

    // The camera volume's column (x, y): every slice's (light, mean transmittance) (GLSL_PASS_VOLUME).
    void column(const float *p_table, int p_x, int p_y, double r_out[Atmosphere::SLICES][4]) const {
        const V3 cam = v3(f.camera_m);
        const V3 sun = v3(f.sun);
        const V3 u = up(cam);
        // up_frame (shade_frame_glsl.hpp).
        const double sg = u.y < 0.0 ? -1.0 : 1.0;
        const V3 k = V3{ u.z, 0.0, -u.x } * sg;
        const double a = 1.0 / (1.0 + sg * u.y);
        const V3 ex{ 1.0, 0.0, 0.0 }, ez{ 0.0, 0.0, sg };
        const V3 t1 = ex + cross(k, ex) + cross(k, cross(k, ex)) * a;
        const V3 t2 = ez + cross(k, ez) + cross(k, cross(k, ez)) * a;
        const double uvx = (p_x + 0.5) / Atmosphere::VOLUME_W, uvy = (p_y + 0.5) / Atmosphere::VOLUME_H;
        const double phi = 2.0 * PI * uvx;
        const double xx = 2.0 * uvy - 1.0;
        const double theta = (xx < 0.0 ? -1.0 : (xx > 0.0 ? 1.0 : 0.0)) * xx * xx * (0.5 * PI);
        const V3 d = (t1 * std::cos(phi) + t2 * std::sin(phi)) * std::cos(theta) + u * std::sin(theta);
        const double cos_sun = dot(d, sun);
        double seg[2] = { 0.0, 0.0 };
        bool ground = false;
        const bool inside = segment(cam, d, seg, ground);
        V3 light;
        V3 trans{ 1.0, 1.0, 1.0 };
        double from = 0.0;
        auto put = [&](int s) {
            r_out[s][0] = light.x;
            r_out[s][1] = light.y;
            r_out[s][2] = light.z;
            r_out[s][3] = (trans.x + trans.y + trans.z) / 3.0;
        };
        for (int s = 0; s < Atmosphere::SLICES - 1; s++) {
            const double fr = double(s + 1) / (Atmosphere::SLICES - 1);
            const double to = double(f.range_m) * fr * fr;
            const double lo = ov::max(from, seg[0]), hi = ov::min(to, seg[1]);
            if (inside && hi > lo) {
                step(p_table, cam, d, sun, cos_sun, 0.5 * (lo + hi), hi - lo, light, trans);
            }
            put(s);
            from = to;
        }
        const double lo = ov::max(from, seg[0]);
        if (inside && seg[1] > lo) {
            const double len = seg[1] - lo;
            for (int i = 0; i < FAR_STEPS; i++) {
                const double q = double(i) / FAR_STEPS, w = double(i + 1) / FAR_STEPS;
                step(p_table, cam, d, sun, cos_sun, lo + 0.25 * (q + w) * (q + w) * len, (w * w - q * q) * len, light, trans);
            }
        }
        if (inside && ground) {
            light = light + trans * ground_light(cam + d * seg[1], sun);
        }
        put(Atmosphere::SLICES - 1);
    }
};

float half_to_float(uint16_t p_h) {
    const uint32_t sign = uint32_t(p_h >> 15) << 31;
    const uint32_t exp = (p_h >> 10) & 31u;
    const uint32_t man = p_h & 1023u;
    uint32_t bits;
    if (exp == 0) {
        const float v = std::ldexp(float(man), -24);
        std::memcpy(&bits, &v, 4);
        bits |= sign;
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

std::vector<float> texels(const ov::bytes &p_data) {
    std::vector<float> out(p_data.size() / 2);
    for (size_t i = 0; i < out.size(); i++) {
        uint16_t h;
        std::memcpy(&h, p_data.data() + i * 2, 2);
        out[i] = half_to_float(h);
    }
    return out;
}

double rel_error(double p_gpu, double p_ref, double p_floor) {
    return std::fabs(p_gpu - p_ref) / ov::max(std::fabs(p_ref), p_floor);
}

ov::bytes read(ov::gpu::Device &p_device, ov::gpu::Texture p_texture) {
    ov::bytes out;
    p_device.read_texture_async(p_texture, [&out](const ov::bytes &p_data) { out = p_data; });
    p_device.sync();
    return out;
}

struct Case {
    const char *name;
    BaseSurface base;
    double voxel_m;
    VoxelPos camera_voxel; // the render origin: the camera at its render position 0
    double camera_lift_m; // above the base along its up
};

} // namespace

namespace AtmosphereCheck {

ov::Value run(ov::gpu::Device &p_device) {
    Value report = Value::object();
    bool ok = true;
    double down_min_100km = -1.0; // the earth_100km case's, which the running object must give again
    const double vs = 0.05;
    const double earth = AtmosphereMedium::EARTH_RADIUS_M / vs;
    const BaseSurface earth_base = BaseSurface::sphere(VoxelPos(0, 0, 0), earth, 12);
    const BaseSurface tiny = BaseSurface::sphere(VoxelPos(0, 0, 0), 2000.0 / vs, 1);
    const Case cases[] = {
        { "flat", BaseSurface::flat(6000), vs, VoxelPos(123456789, 6040, -987654), 2.0 },
        { "earth_pole", earth_base, vs, earth_base.tangent_point(0, 0), 2.0 },
        { "earth_100km", earth_base, vs, earth_base.tangent_point(0, 0), 100500.0 },
        { "tiny_pole", tiny, vs, tiny.tangent_point(0, 0), 2.0 },
    };
    const Float3 sun = Float3{ 0.8f, 0.2f, 0.3f }.normalized();
    for (const Case &c : cases) {
        Value r = Value::object();
        Ref ref;
        ref.base = c.base.render(c.camera_voxel, c.voxel_m);
        ref.m = AtmosphereMedium::for_base(c.base.curvature_radius_m(c.voxel_m), 0.0);
        // The camera c.camera_lift_m above the base at the render origin.
        const V3 cam = ref.up(V3{}) * (c.camera_lift_m - ref.altitude(V3{}));
        const Atmosphere::Frame frame = Atmosphere::make_frame(ref.m, Atmosphere::ms_radius_m(c.base, c.voxel_m),
                Float3{ float(cam.x), float(cam.y), float(cam.z) }, sun, ref.base);
        ref.f = frame;

        const ov::bytes block_data = Atmosphere::block_bytes(ref.base, ref.m, frame);
        const ov::gpu::Pipeline pipe_ms = ov::gpu::compile(p_device, "atmosphere_check_ms", Atmosphere::glsl_pass_multiscatter());
        const ov::gpu::Pipeline pipe_vol = ov::gpu::compile(p_device, "atmosphere_check_volume", Atmosphere::glsl_pass_volume());
        if (!pipe_ms.is_valid() || !pipe_vol.is_valid()) {
            report["error"] = "a pass did not compile";
            report["ok"] = false;
            return report;
        }
        const uint32_t usage = ov::gpu::TEXTURE_STORAGE | ov::gpu::TEXTURE_SAMPLED | ov::gpu::TEXTURE_COPY_SRC;
        const ov::gpu::Texture vol = p_device.create_texture(ov::gpu::TextureFormat::RGBA16_FLOAT, Atmosphere::VOLUME_W, Atmosphere::VOLUME_H * Atmosphere::SLICES, usage);
        const ov::gpu::Texture ms = p_device.create_texture(ov::gpu::TextureFormat::RGBA16_FLOAT, Atmosphere::MS_SIZE, Atmosphere::MS_SIZE, usage);
        const ov::gpu::Buffer block = p_device.create_buffer(ov::gpu::BufferKind::UNIFORM, block_data.size(), block_data.data());
        using ov::gpu::Binding;
        const ov::gpu::BindSet set_ms = p_device.create_bind_set(pipe_ms, { Binding(0, ov::gpu::BufferKind::UNIFORM, block), Binding::storage_image(1, ms) });
        const ov::gpu::BindSet set_vol = p_device.create_bind_set(pipe_vol, { Binding(0, ov::gpu::BufferKind::UNIFORM, block), Binding::storage_image(1, vol), Binding::storage_image(2, ms) });
        p_device.dispatch(pipe_ms, set_ms, nullptr, 0, (Atmosphere::MS_SIZE + 7) / 8, (Atmosphere::MS_SIZE + 7) / 8, 1);
        p_device.dispatch(pipe_vol, set_vol, nullptr, 0, (Atmosphere::VOLUME_W + 7) / 8, (Atmosphere::VOLUME_H + 7) / 8, 1);
        const std::vector<float> ms_gpu = texels(read(p_device, ms));
        const std::vector<float> vol_gpu = texels(read(p_device, vol));

        // The table, every texel.
        double ms_worst = 0.0;
        double ms_max = 0.0;
        for (int y = 0; y < Atmosphere::MS_SIZE && ms_gpu.size() >= size_t(Atmosphere::MS_SIZE * Atmosphere::MS_SIZE * 4); y++) {
            for (int x = 0; x < Atmosphere::MS_SIZE; x++) {
                const V3 v = ref.ms_texel(x, y);
                const float *g = &ms_gpu[(size_t(y) * Atmosphere::MS_SIZE + x) * 4];
                ms_max = ov::max(ms_max, v.z);
                ms_worst = ov::max(ms_worst, ov::max(rel_error(g[0], v.x, 1e-4), ov::max(rel_error(g[1], v.y, 1e-4), rel_error(g[2], v.z, 1e-4))));
            }
        }
        // Columns of the volume (every 9th direction, every 5th height), every slice, read with the GPU's table.
        double vol_worst = 0.0;
        double sky_mean = 0.0;
        double down_min_ref = 1.0;
        double down_min_gpu = 1.0;
        int64_t checked = 0;
        int64_t sky_n = 0;
        if (vol_gpu.size() >= size_t(Atmosphere::VOLUME_W * Atmosphere::VOLUME_H * Atmosphere::SLICES * 4)) {
            for (int x = 0; x < Atmosphere::VOLUME_W; x += 9) {
                for (int y = 0; y < Atmosphere::VOLUME_H; y += 5) {
                    double col[Atmosphere::SLICES][4];
                    ref.column(ms_gpu.data(), x, y, col);
                    for (int s = 0; s < Atmosphere::SLICES; s++) {
                        const float *g = &vol_gpu[((size_t(s) * Atmosphere::VOLUME_H + y) * Atmosphere::VOLUME_W + x) * 4];
                        for (int k = 0; k < 4; k++) {
                            vol_worst = ov::max(vol_worst, rel_error(g[k], col[s][k], k < 3 ? 1e-4 : 1e-3));
                        }
                        checked++;
                    }
                    if (y < Atmosphere::VOLUME_H / 2) {
                        down_min_ref = ov::min(down_min_ref, col[Atmosphere::SLICES - 2][3]);
                        down_min_gpu = ov::min(down_min_gpu,
                                double(vol_gpu[((size_t(Atmosphere::SLICES - 2) * Atmosphere::VOLUME_H + y) * Atmosphere::VOLUME_W + x) * 4 + 3]));
                    }
                    if (y > Atmosphere::VOLUME_H / 2) {
                        sky_mean += col[Atmosphere::SLICES - 1][2];
                        sky_n++;
                    }
                }
            }
        }
        r["ms_texels"] = int64_t(ms_gpu.size() / 4);
        r["ms_worst_rel"] = ms_worst;
        r["ms_max_blue"] = ms_max;
        r["volume_texels"] = checked;
        r["volume_worst_rel"] = vol_worst;
        r["sky_blue_mean"] = sky_n > 0 ? sky_mean / double(sky_n) : 0.0;
        r["down_min_transmittance_ref"] = down_min_ref;
        r["down_min_transmittance_gpu"] = down_min_gpu;
        r["range_m"] = double(frame.range_m);
        const bool case_ok = checked > 0 && !ms_gpu.empty() && ms_worst < TOLERANCE && vol_worst < TOLERANCE;
        r["ok"] = case_ok;
        ok = ok && case_ok;
        report[c.name] = r;
        if (std::string(c.name) == "earth_100km") {
            down_min_100km = down_min_gpu;
        }
        const ov::gpu::Handle handles[] = { set_vol, set_ms, pipe_vol, pipe_ms, block, vol, ms };
        for (const ov::gpu::Handle h : handles) {
            p_device.free(h);
        }
    }
    // The running object (Atmosphere::start / frame, the hosts' path) at Earth's pole 100 km up: its volume's lowest
    // downward transmittance against the passes' above.
    {
        Atmosphere *running = new Atmosphere();
        running->start(&p_device, earth_base, vs, 0.0, 0.0, nullptr);
        const BaseSurface::Render base = earth_base.render(earth_base.tangent_point(0, 0), vs);
        running->frame(Float3{ 0.0f, 100500.0f, 0.0f }, sun, base);
        p_device.sync();
        const std::vector<float> vol = texels(read(p_device, running->get_volume()));
        double down_min = 1.0;
        for (int x = 0; x < Atmosphere::VOLUME_W && !vol.empty(); x += 9) { // the cases' columns
            for (int y = 0; y < Atmosphere::VOLUME_H / 2; y += 5) {
                down_min = ov::min(down_min, double(vol[((size_t(Atmosphere::SLICES - 2) * Atmosphere::VOLUME_H + y) * Atmosphere::VOLUME_W + x) * 4 + 3]));
            }
        }
        Value r = Value::object();
        r["range_m"] = double(running->range_m());
        r["down_min_transmittance"] = down_min;
        const bool running_ok = down_min_100km > 0.0 && rel_error(down_min, down_min_100km, 1e-3) < TOLERANCE;
        r["ok"] = running_ok;
        ok = ok && running_ok;
        report["running"] = r;
        running->destroy();
    }
    report["ok"] = ok;
    return report;
}

} // namespace AtmosphereCheck
