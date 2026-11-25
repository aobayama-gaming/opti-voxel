#include "weather_fluid.hpp"

#include "parallel_for.hpp"
#include "weather_vec.hpp"

#include "ov/math.hpp"

#include <cmath>

using weather_vec::cross3;
using weather_vec::dot3;
using weather_vec::tangent;

namespace {

double length3(const double p_v[3]) {
    return std::sqrt(dot3(p_v, p_v));
}

double dotf(const float *p_a, const float *p_b) {
    return double(p_a[0]) * double(p_b[0]) + double(p_a[1]) * double(p_b[1]) + double(p_a[2]) * double(p_b[2]);
}

} // namespace

WeatherFluid::WeatherFluid(std::shared_ptr<const WeatherGrid> p_grid, const Params &p_params) :
        cells(std::move(p_grid)), params(p_params) {
    if (params.levels < 2 || params.levels > MAX_LEVELS) {
        OV_ERR_PRINT(ov::format("WeatherFluid: %d levels is outside [2, %d].", params.levels, MAX_LEVELS));
        params.levels = ov::clamp(params.levels, 2, MAX_LEVELS);
    }
    if (params.scalars < 0 || params.scalars > MAX_SCALARS) {
        OV_ERR_PRINT(ov::format("WeatherFluid: %d scalars is outside [0, %d].", params.scalars, MAX_SCALARS));
        params.scalars = ov::clamp(params.scalars, 0, MAX_SCALARS);
    }
    if (!(params.top_m > 0.0)) {
        OV_ERR_PRINT("WeatherFluid: the top must be above the ground.");
        params.top_m = 600.0;
    }
    if (!(params.sponge_m > 0.0)) {
        params.sponge_m = 0.2 * params.top_m;
    }
    dz = params.top_m / params.levels;
    const WeatherGrid &g = *cells;
    cur.assign(g.padded_cells() * size_t(stride()), 0.0f);
    next = cur;
    if (params.confinement > 0.0) {
        vort.assign(g.padded_cells() * size_t(params.levels) * 3, 0.0f);
        vort_mag.assign(g.padded_cells() * size_t(params.levels), 0.0f);
    }
    p.assign(g.padded_cells() * size_t(params.levels), 0.0f);
    rhs.assign(cell_count(), 0.0f);
    make_columns();
}

void WeatherFluid::make_columns() {
    const WeatherGrid &g = *cells;
    column.assign(g.cells(), Column{});
    min_spacing_m = 1e30;
    for (int face = 0; face < g.faces(); face++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                Column &c = column[g.interior(face, i, j)];
                double pos[3], up[3], ignored[3];
                g.cell_surface(face, i, j, pos, up);
                // The side neighbours' centres (+x, -x, +y, -y) and the corners (x, y) of the cell.
                double nb[SIDES][3];
                g.cell_surface(face, i + 1, j, nb[0], ignored);
                g.cell_surface(face, i - 1, j, nb[1], ignored);
                g.cell_surface(face, i, j + 1, nb[2], ignored);
                g.cell_surface(face, i, j - 1, nb[3], ignored);
                double c00[3], c10[3], c01[3], c11[3];
                g.surface(face, i, j, c00, ignored);
                g.surface(face, i + 1, j, c10, ignored);
                g.surface(face, i, j + 1, c01, ignored);
                g.surface(face, i + 1, j + 1, c11, ignored);
                const double *edge[SIDES][2] = { { c10, c11 }, { c00, c01 }, { c01, c11 }, { c00, c10 } };
                for (int s = 0; s < SIDES; s++) {
                    double n[3], e[3];
                    for (int a = 0; a < 3; a++) {
                        n[a] = nb[s][a] - pos[a];
                        e[a] = edge[s][1][a] - edge[s][0][a];
                    }
                    c.distance[s] = float(length3(n));
                    c.length[s] = float(length3(e));
                    min_spacing_m = ov::min(min_spacing_m, double(c.distance[s]));
                    tangent(up, n);
                    const double l = length3(n);
                    for (int a = 0; a < 3; a++) {
                        c.normal[s][a] = float(n[a] / l);
                    }
                }
                double d1[3], d2[3], across[3];
                for (int a = 0; a < 3; a++) {
                    d1[a] = c11[a] - c00[a];
                    d2[a] = c01[a] - c10[a];
                }
                cross3(d1, d2, across);
                c.area = float(0.5 * length3(across));
                // ax . ex = 1, ax . ey = 0, both in the tangent plane (ay likewise): ex, ey span two cells.
                double ex[3], ey[3];
                for (int a = 0; a < 3; a++) {
                    ex[a] = nb[0][a] - nb[1][a];
                    ey[a] = nb[2][a] - nb[3][a];
                }
                tangent(up, ex);
                tangent(up, ey);
                double exy[3], ax[3], ay[3];
                cross3(ex, ey, exy);
                const double det = dot3(exy, up);
                cross3(ey, up, ax);
                cross3(up, ex, ay);
                for (int a = 0; a < 3; a++) {
                    c.pos[a] = pos[a];
                    c.up[a] = float(up[a]);
                    c.ax[a] = float(ax[a] / det);
                    c.ay[a] = float(ay[a] / det);
                }
                c.f = float(2.0 * params.omega_rad_s * up[1]);
            }
        }
    }
}

void WeatherFluid::neighbours(int p_face, int p_i, int p_j, size_t r_nb[SIDES]) const {
    const WeatherGrid &g = *cells;
    r_nb[0] = g.index(p_face, p_i + 1, p_j);
    r_nb[1] = g.index(p_face, p_i - 1, p_j);
    r_nb[2] = g.index(p_face, p_i, p_j + 1);
    r_nb[3] = g.index(p_face, p_i, p_j - 1);
}

void WeatherFluid::fill_halo() {
    cells->fill_halo(cur.data(), stride());
}

void WeatherFluid::column_up(size_t p_interior, double r_up[3]) const {
    for (int a = 0; a < 3; a++) {
        r_up[a] = column[p_interior].up[a];
    }
}

void WeatherFluid::column_pos(size_t p_interior, double r_pos[3]) const {
    for (int a = 0; a < 3; a++) {
        r_pos[a] = column[p_interior].pos[a];
    }
}

double WeatherFluid::max_dt() const {
    const WeatherGrid &g = *cells;
    const int st = stride();
    double rate = 0.0;
    for (int face = 0; face < g.faces(); face++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                const Column &c = column[g.interior(face, i, j)];
                const float *here = &cur[g.index(face, i, j) * size_t(st)];
                for (int k = 0; k < params.levels; k++) {
                    const float *u = here + wind_offset(k);
                    const double w = dotf(u, c.up);
                    const double h = std::sqrt(ov::max(dotf(u, u) - w * w, 0.0));
                    rate = ov::max(rate, ov::max(h / min_spacing_m, std::fabs(w) / dz));
                }
            }
        }
    }
    return rate > 0.0 ? MAX_COURANT / rate : 1e30;
}

void WeatherFluid::transport(const Column &p_c, const double p_d[3], float *r_wind) {
    double from[3];
    for (int a = 0; a < 3; a++) {
        from[a] = p_c.pos[a] + p_d[a];
    }
    const double len = length3(from);
    if (!(len > 0.0)) {
        return;
    }
    const double up[3] = { p_c.up[0], p_c.up[1], p_c.up[2] };
    for (int a = 0; a < 3; a++) {
        from[a] /= len;
    }
    // Rodrigues with k = from x up (|k| = sin, cos = from . up): v' = v cos + k x v + k (k . v) / (1 + cos).
    double k[3];
    cross3(from, up, k);
    const double cosine = dot3(from, up);
    const double v[3] = { r_wind[0], r_wind[1], r_wind[2] };
    double kv[3];
    cross3(k, v, kv);
    const double kdv = dot3(k, v) / (1.0 + cosine);
    for (int a = 0; a < 3; a++) {
        r_wind[a] = float(v[a] * cosine + kv[a] + k[a] * kdv);
    }
}

void WeatherFluid::advect(double p_dt) {
    const WeatherGrid &g = *cells;
    const int st = stride();
    const int per = per_level();
    const int levels = params.levels;
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherFluid::advect", [&](uint32_t p_row) {
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            const Column &c = column[g.interior(face, i, j)];
            const size_t at = g.index(face, i, j) * size_t(st);
            for (int k = 0; k < levels; k++) {
                const float *u = &cur[at + size_t(wind_offset(k))];
                const double w = dotf(u, c.up);
                double d[3];
                for (int a = 0; a < 3; a++) {
                    d[a] = -(double(u[a]) - w * c.up[a]) * p_dt;
                }
                // The departure point in the face's cells (the centres at integers) and levels.
                const double x = ov::clamp(i + 2.0 * (d[0] * c.ax[0] + d[1] * c.ax[1] + d[2] * c.ax[2]), -1.0, double(g.width()));
                const double y = ov::clamp(j + 2.0 * (d[0] * c.ay[0] + d[1] * c.ay[1] + d[2] * c.ay[2]), -1.0, double(g.height()));
                const double z = ov::clamp((level_m(k) - w * p_dt) / dz - 0.5, 0.0, double(levels - 1));
                const int i0 = ov::min(int(std::floor(x)), g.width() - 1);
                const int j0 = ov::min(int(std::floor(y)), g.height() - 1);
                const int k0 = ov::min(int(std::floor(z)), levels - 2);
                const double fx = x - i0;
                const double fy = y - j0;
                const double fz = z - k0;
                const size_t src[4] = { g.index(face, i0, j0) * size_t(st), g.index(face, i0 + 1, j0) * size_t(st),
                    g.index(face, i0, j0 + 1) * size_t(st), g.index(face, i0 + 1, j0 + 1) * size_t(st) };
                const double wxy[4] = { (1.0 - fx) * (1.0 - fy), fx * (1.0 - fy), (1.0 - fx) * fy, fx * fy };
                float *out = &next[at + size_t(wind_offset(k))];
                for (int f = 0; f < per; f++) {
                    double v = 0.0;
                    for (int n = 0; n < 4; n++) {
                        const float *s = &cur[src[n]];
                        v += wxy[n] * ((1.0 - fz) * s[k0 * per + f] + fz * s[(k0 + 1) * per + f]);
                    }
                    out[f] = float(v);
                }
                transport(c, d, out);
            }
        } }, ov::Priority::LOW);
    cur.swap(next);
    fill_halo();
}

void WeatherFluid::gradient(const Column &p_c, const float *p_f, size_t p_here, const size_t p_nb[SIDES], int p_stride, int p_per, int p_at,
        int p_level, double r_g[3]) const {
    auto value = [&](size_t p_cell, int p_k) { return double(p_f[p_cell * size_t(p_stride) + size_t(p_k * p_per + p_at)]); };
    const double di = value(p_nb[0], p_level) - value(p_nb[1], p_level);
    const double dj = value(p_nb[2], p_level) - value(p_nb[3], p_level);
    const int k0 = ov::max(p_level - 1, 0), k1 = ov::min(p_level + 1, params.levels - 1);
    const double dk = (value(p_here, k1) - value(p_here, k0)) / (double(k1 - k0) * dz);
    for (int a = 0; a < 3; a++) {
        r_g[a] = p_c.ax[a] * di + p_c.ay[a] * dj + p_c.up[a] * dk;
    }
}

void WeatherFluid::confine(double p_dt) {
    if (!(params.confinement > 0.0)) {
        return;
    }
    const WeatherGrid &g = *cells;
    const int st = stride();
    const int levels = params.levels;
    const int per = per_level();
    fill_halo(); // the physics wrote the interior
    // The curl: g_a the gradient of the wind's component a, (dz uy - dy uz, dx uz - dz ux, dy ux - dx uy).
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherFluid::curl", [&](uint32_t p_row) {
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            const Column &c = column[g.interior(face, i, j)];
            const size_t here = g.index(face, i, j);
            size_t nb[SIDES];
            neighbours(face, i, j, nb);
            for (int k = 0; k < levels; k++) {
                double gr[3][3];
                for (int a = 0; a < 3; a++) {
                    gradient(c, cur.data(), here, nb, st, per, a, k, gr[a]);
                }
                // Its part about the up only: the swirls seen from above. The whole curl's horizontal part is the
                // wind's shear over the ground, which the confinement sharpened into a mixing that cleared every cloud.
                const double curl[3] = { gr[2][1] - gr[1][2], gr[0][2] - gr[2][0], gr[1][0] - gr[0][1] };
                const double spin = curl[0] * c.up[0] + curl[1] * c.up[1] + curl[2] * c.up[2];
                float *w = &vort[(here * size_t(levels) + size_t(k)) * 3];
                for (int a = 0; a < 3; a++) {
                    w[a] = float(spin * c.up[a]);
                }
                vort_mag[here * size_t(levels) + size_t(k)] = float(std::fabs(spin));
            }
        } }, ov::Priority::LOW);
    g.fill_halo(vort_mag.data(), levels);
    const double push = params.confinement * std::cbrt(g.cell_m() * g.cell_m() * dz) * p_dt;
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherFluid::confine", [&](uint32_t p_row) {
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            const Column &c = column[g.interior(face, i, j)];
            const size_t here = g.index(face, i, j);
            size_t nb[SIDES];
            neighbours(face, i, j, nb);
            float *u = &cur[here * size_t(st)];
            for (int k = 0; k < levels; k++) {
                double n[3];
                gradient(c, vort_mag.data(), here, nb, levels, 1, 0, k, n);
                const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                if (len < 1e-12) {
                    continue;
                }
                const float *w = &vort[(here * size_t(levels) + size_t(k)) * 3];
                const double f[3] = { (n[1] * w[2] - n[2] * w[1]) / len, (n[2] * w[0] - n[0] * w[2]) / len, (n[0] * w[1] - n[1] * w[0]) / len };
                for (int a = 0; a < 3; a++) {
                    u[wind_offset(k) + a] += float(push * f[a]);
                }
            }
        } }, ov::Priority::LOW);
}

void WeatherFluid::force(double p_dt, const float *p_buoyancy) {
    confine(p_dt);
    const WeatherGrid &g = *cells;
    const int st = stride();
    const double sponge_base = params.top_m - params.sponge_m;
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherFluid::force", [&](uint32_t p_row) {
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            const size_t in = g.interior(face, i, j);
            const Column &c = column[in];
            const double up[3] = { c.up[0], c.up[1], c.up[2] };
            float *here = &cur[g.index(face, i, j) * size_t(st)];
            for (int k = 0; k < params.levels; k++) {
                float *u = here + wind_offset(k);
                double w = dotf(u, c.up);
                double h[3] = { u[0] - w * up[0], u[1] - w * up[1], u[2] - w * up[2] };
                weather_vec::coriolis(up, c.f, p_dt, h);
                w += double(p_buoyancy[cell_index(in, k)]) * p_dt;
                const double into = (level_m(k) - sponge_base) / params.sponge_m;
                if (into > 0.0) {
                    w /= 1.0 + p_dt / params.sponge_tau_s * into * into;
                }
                for (int a = 0; a < 3; a++) {
                    u[a] = float(h[a] + w * up[a]);
                }
            }
        } }, ov::Priority::LOW);
    fill_halo();
}

double WeatherFluid::divergence(size_t p_interior, int p_level) const {
    const WeatherGrid &g = *cells;
    const int st = stride();
    const int face = int(p_interior / (size_t(g.width()) * size_t(g.height())));
    const size_t rest = p_interior % (size_t(g.width()) * size_t(g.height()));
    const int j = int(rest / size_t(g.width()));
    const int i = int(rest % size_t(g.width()));
    const Column &c = column[p_interior];
    size_t nb[SIDES];
    neighbours(face, i, j, nb);
    const float *here = &cur[g.index(face, i, j) * size_t(st)];
    const float *u = here + wind_offset(p_level);
    double flux = 0.0;
    for (int s = 0; s < SIDES; s++) {
        const float *v = &cur[nb[s] * size_t(st) + size_t(wind_offset(p_level))];
        flux += 0.5 * (dotf(u, c.normal[s]) + dotf(v, c.normal[s])) * c.length[s] * dz;
    }
    const double w = dotf(u, c.up);
    if (p_level + 1 < params.levels) {
        flux += 0.5 * (w + dotf(here + wind_offset(p_level + 1), c.up)) * c.area;
    }
    if (p_level > 0) {
        flux -= 0.5 * (w + dotf(here + wind_offset(p_level - 1), c.up)) * c.area;
    }
    return flux / (double(c.area) * dz);
}

double WeatherFluid::max_divergence() const {
    double m = 0.0;
    for (size_t in = 0; in < cells->cells(); in++) {
        for (int k = 0; k < params.levels; k++) {
            m = ov::max(m, std::fabs(divergence(in, k)));
        }
    }
    return m;
}

void WeatherFluid::sweep(int p_colour) {
    const WeatherGrid &g = *cells;
    const int levels = params.levels;
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherFluid::sweep", [&](uint32_t p_row) {
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            const size_t in = g.interior(face, i, j);
            const Column &c = column[in];
            size_t nb[SIDES];
            neighbours(face, i, j, nb);
            const size_t at = g.index(face, i, j) * size_t(levels);
            const double vertical = double(c.area) / dz;
            for (int k = (i + j + p_colour) & 1; k < levels; k += 2) {
                double sum = 0.0;
                double diag = 0.0;
                for (int s = 0; s < SIDES; s++) {
                    const double a = double(c.length[s]) * dz / double(c.distance[s]);
                    sum += a * p[nb[s] * size_t(levels) + size_t(k)];
                    diag += a;
                }
                if (k + 1 < levels) {
                    sum += vertical * p[at + size_t(k + 1)];
                    diag += vertical;
                }
                if (k > 0) {
                    sum += vertical * p[at + size_t(k - 1)];
                    diag += vertical;
                }
                const double gs = (sum - rhs[cell_index(in, k)]) / diag;
                p[at + size_t(k)] = float(p[at + size_t(k)] + SOR * (gs - p[at + size_t(k)]));
            }
        } }, ov::Priority::LOW);
    g.fill_halo(p.data(), levels);
}

void WeatherFluid::project(double p_dt) {
    const WeatherGrid &g = *cells;
    const int st = stride();
    const int levels = params.levels;
    // The divergence's fluxes over dt: what the pressure takes out.
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherFluid::divergence", [&](uint32_t p_row) {
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            const size_t in = g.interior(face, i, j);
            for (int k = 0; k < levels; k++) {
                rhs[cell_index(in, k)] = float(divergence(in, k) * double(column[in].area) * dz / p_dt);
            }
        } }, ov::Priority::LOW);
    for (int it = 0; it < params.pressure_sweeps; it++) {
        sweep(0);
        sweep(1);
    }
    // The pressure's mean is free (rigid walls): kept at 0, so its floats keep their precision.
    double mean = 0.0;
    for (int face = 0; face < g.faces(); face++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                for (int k = 0; k < levels; k++) {
                    mean += p[g.index(face, i, j) * size_t(levels) + size_t(k)];
                }
            }
        }
    }
    mean /= double(cell_count());
    for (float &v : p) {
        v = float(v - mean);
    }
    // The wind loses dt times the pressure's central gradient (rigid ground and top: the pressure mirrored past them).
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherFluid::correct", [&](uint32_t p_row) {
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            const Column &c = column[g.interior(face, i, j)];
            size_t nb[SIDES];
            neighbours(face, i, j, nb);
            const size_t at = g.index(face, i, j) * size_t(levels);
            float *here = &cur[g.index(face, i, j) * size_t(st)];
            for (int k = 0; k < levels; k++) {
                const double di = double(p[nb[0] * size_t(levels) + size_t(k)]) - p[nb[1] * size_t(levels) + size_t(k)];
                const double dj = double(p[nb[2] * size_t(levels) + size_t(k)]) - p[nb[3] * size_t(levels) + size_t(k)];
                const double dk = double(p[at + size_t(ov::min(k + 1, levels - 1))]) - p[at + size_t(ov::max(k - 1, 0))];
                float *u = here + wind_offset(k);
                for (int a = 0; a < 3; a++) {
                    const double grad = c.ax[a] * di + c.ay[a] * dj + c.up[a] * dk / (2.0 * dz);
                    u[a] = float(u[a] - p_dt * grad);
                }
            }
        } }, ov::Priority::LOW);
    fill_halo();
}

void WeatherFluid::dry_buoyancy(int p_scalar, const std::vector<double> &p_theta_ref, float *r_buoyancy) const {
    const WeatherGrid &g = *cells;
    const int st = stride();
    for (int face = 0; face < g.faces(); face++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                const size_t in = g.interior(face, i, j);
                const float *here = &cur[g.index(face, i, j) * size_t(st)];
                for (int k = 0; k < params.levels; k++) {
                    const double theta = here[scalar_offset(k, p_scalar)];
                    // The reference in the state's precision: air at it is exactly neutral.
                    r_buoyancy[cell_index(in, k)] = float(GRAVITY * (theta - double(float(p_theta_ref[size_t(k)]))) / THETA_0);
                }
            }
        }
    }
}
