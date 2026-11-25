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
    corr = cur;
    wf.assign(g.padded_cells() * size_t(params.levels + 1), 0.0f);
    wf_next = wf;
    wf_corr = wf;
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
    halo();
}

void WeatherFluid::halo() {
    cells->fill_halo(cur.data(), stride());
    cells->fill_halo(wf.data(), params.levels + 1);
}

void WeatherFluid::vertical_from_centres() {
    const WeatherGrid &g = *cells;
    const int st = stride();
    for (int face = 0; face < g.faces(); face++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                const Column &c = column[g.interior(face, i, j)];
                const size_t at = g.index(face, i, j);
                const float *here = &cur[at * size_t(st)];
                float *w = face_w(at);
                w[0] = 0.0f;
                w[params.levels] = 0.0f;
                for (int m = 1; m < params.levels; m++) {
                    w[m] = float(0.5 * (dotf(here + wind_offset(m - 1), c.up) + dotf(here + wind_offset(m), c.up)));
                }
            }
        }
    }
    centre_vertical();
}

void WeatherFluid::centre_vertical() {
    const WeatherGrid &g = *cells;
    const int st = stride();
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherFluid::centre_vertical", [&](uint32_t p_row) {
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            const Column &c = column[g.interior(face, i, j)];
            const size_t at = g.index(face, i, j);
            float *here = &cur[at * size_t(st)];
            const float *w = face_w(at);
            for (int k = 0; k < params.levels; k++) {
                float *u = here + wind_offset(k);
                const double change = 0.5 * (double(w[k]) + w[k + 1]) - dotf(u, c.up);
                for (int a = 0; a < 3; a++) {
                    u[a] = float(u[a] + change * c.up[a]);
                }
            }
        } }, ov::Priority::LOW);
    halo();
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
                    rate = ov::max(rate, std::fabs(double(face_w(g.index(face, i, j))[k])) / dz);
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

WeatherFluid::Stencil WeatherFluid::trace(int p_face, int p_i, int p_j, const double p_d[3], double p_z, int p_top) const {
    const WeatherGrid &g = *cells;
    const Column &c = column[g.interior(p_face, p_i, p_j)];
    // The point in the face's cells (the centres at integers) and levels (or interfaces, p_top the last).
    const double x = ov::clamp(p_i + 2.0 * (p_d[0] * c.ax[0] + p_d[1] * c.ax[1] + p_d[2] * c.ax[2]), -1.0, double(g.width()));
    const double y = ov::clamp(p_j + 2.0 * (p_d[0] * c.ay[0] + p_d[1] * c.ay[1] + p_d[2] * c.ay[2]), -1.0, double(g.height()));
    const double z = ov::clamp(p_z, 0.0, double(p_top));
    const int i0 = ov::min(int(std::floor(x)), g.width() - 1);
    const int j0 = ov::min(int(std::floor(y)), g.height() - 1);
    Stencil s;
    s.k0 = ov::min(int(std::floor(z)), p_top - 1);
    const double fx = x - i0, fy = y - j0;
    s.fz = z - s.k0;
    s.src[0] = g.index(p_face, i0, j0);
    s.src[1] = g.index(p_face, i0 + 1, j0);
    s.src[2] = g.index(p_face, i0, j0 + 1);
    s.src[3] = g.index(p_face, i0 + 1, j0 + 1);
    s.wxy[0] = (1.0 - fx) * (1.0 - fy);
    s.wxy[1] = fx * (1.0 - fy);
    s.wxy[2] = (1.0 - fx) * fy;
    s.wxy[3] = fx * fy;
    return s;
}

double WeatherFluid::sample(const float *p_buf, size_t p_rec, int p_per, int p_f, const Stencil &p_s) {
    double v = 0.0;
    for (int n = 0; n < 4; n++) {
        const float *b = &p_buf[p_s.src[n] * p_rec];
        v += p_s.wxy[n] * ((1.0 - p_s.fz) * b[p_s.k0 * p_per + p_f] + p_s.fz * b[(p_s.k0 + 1) * p_per + p_f]);
    }
    return v;
}

void WeatherFluid::corner(const float *p_buf, size_t p_rec, int p_per, int p_f, const Stencil &p_s, int p_n, float *r_v, int p_count) {
    const float *b = &p_buf[p_s.src[p_n % 4] * p_rec + size_t((p_s.k0 + p_n / 4) * p_per + p_f)];
    for (int a = 0; a < p_count; a++) {
        r_v[a] = b[a];
    }
}

double WeatherFluid::level_step(const Column &p_c, const float *p_u, double p_dt, double r_d[3]) {
    const double w = dotf(p_u, p_c.up);
    for (int a = 0; a < 3; a++) {
        r_d[a] = (double(p_u[a]) - w * p_c.up[a]) * p_dt;
    }
    return w;
}

void WeatherFluid::interface_step(const Column &p_c, const float *p_lo, const float *p_hi, double p_dt, double r_d[3]) {
    double dl[3], dh[3];
    level_step(p_c, p_lo, p_dt, dl);
    level_step(p_c, p_hi, p_dt, dh);
    for (int a = 0; a < 3; a++) {
        r_d[a] = 0.5 * (dl[a] + dh[a]);
    }
}

void WeatherFluid::advect(double p_dt) {
    const WeatherGrid &g = *cells;
    const int st = stride();
    const int per = per_level();
    const int levels = params.levels;
    const size_t fst = size_t(levels + 1);
    // The air's displacement over -p_dt (p_sign -1: where it comes from) or p_dt (+1: where it goes) at a level's
    // centre or an interface, in the column's frame: d tangent (metres) and the stencil there.
    auto at_level = [&](int p_face, int p_i, int p_j, int p_k, double p_sign, double r_d[3]) {
        const Column &c = column[g.interior(p_face, p_i, p_j)];
        const double w = level_step(c, &cur[g.index(p_face, p_i, p_j) * size_t(st) + size_t(wind_offset(p_k))], p_sign * p_dt, r_d);
        return trace(p_face, p_i, p_j, r_d, (level_m(p_k) + p_sign * w * p_dt) / dz - 0.5, levels - 1);
    };
    auto at_interface = [&](int p_face, int p_i, int p_j, int p_m, double p_sign, double r_d[3]) {
        const Column &c = column[g.interior(p_face, p_i, p_j)];
        const size_t at = g.index(p_face, p_i, p_j);
        interface_step(c, &cur[at * size_t(st) + size_t(wind_offset(p_m - 1))], &cur[at * size_t(st) + size_t(wind_offset(p_m))], p_sign * p_dt, r_d);
        return trace(p_face, p_i, p_j, r_d, double(p_m) + p_sign * double(wf[at * fst + size_t(p_m)]) * p_dt / dz, levels);
    };
    // The forward step: each field read where the air comes from (the wind turned along), into next and wf_next.
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherFluid::advect", [&](uint32_t p_row) {
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            const Column &c = column[g.interior(face, i, j)];
            const size_t at = g.index(face, i, j);
            for (int k = 0; k < levels; k++) {
                double d[3];
                const Stencil s = at_level(face, i, j, k, -1.0, d);
                float *out = &next[at * size_t(st) + size_t(wind_offset(k))];
                for (int f = 0; f < per; f++) {
                    out[f] = float(sample(cur.data(), size_t(st), per, f, s));
                }
                transport(c, d, out);
            }
            float *out = &wf_next[at * fst];
            out[0] = 0.0f;
            out[levels] = 0.0f;
            for (int m = 1; m < levels; m++) {
                double d[3];
                out[m] = float(sample(wf.data(), fst, 1, 0, at_interface(face, i, j, m, -1.0, d)));
            }
        } }, ov::Priority::LOW);
    // MacCormack (Selle et al. 2008, "An unconditionally stable MacCormack method"): the forward result traced back
    // (where the air goes, read in it) misses the start by twice the forward step's error to first order; half the miss
    // added back makes the step second order. Where that leaves the forward stencil's range (a new extreme: the
    // correction's overshoot at a sharp front) the forward result stays.
    g.fill_halo(next.data(), st);
    g.fill_halo(wf_next.data(), levels + 1);
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherFluid::maccormack", [&](uint32_t p_row) {
        constexpr int MAX_PER = WIND + MAX_SCALARS;
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            const Column &c = column[g.interior(face, i, j)];
            const size_t at = g.index(face, i, j);
            for (int k = 0; k < levels; k++) {
                const size_t o = at * size_t(st) + size_t(wind_offset(k));
                double d[3], e[3];
                const Stencil fwd = at_level(face, i, j, k, -1.0, d);
                const Stencil bwd = at_level(face, i, j, k, 1.0, e);
                float back[MAX_PER];
                for (int f = 0; f < per; f++) {
                    back[f] = float(sample(next.data(), size_t(st), per, f, bwd));
                }
                transport(c, e, back);
                float lo[MAX_PER], hi[MAX_PER];
                for (int n = 0; n < 8; n++) {
                    float v[MAX_PER];
                    corner(cur.data(), size_t(st), per, 0, fwd, n, v, per);
                    transport(c, d, v);
                    for (int f = 0; f < per; f++) {
                        lo[f] = n == 0 ? v[f] : ov::min(lo[f], v[f]);
                        hi[f] = n == 0 ? v[f] : ov::max(hi[f], v[f]);
                    }
                }
                for (int f = 0; f < per; f++) {
                    const double v = double(next[o + size_t(f)]) + 0.5 * (double(cur[o + size_t(f)]) - back[f]);
                    corr[o + size_t(f)] = v >= lo[f] && v <= hi[f] ? float(v) : next[o + size_t(f)];
                }
            }
            float *out = &wf_corr[at * fst];
            out[0] = 0.0f;
            out[levels] = 0.0f;
            for (int m = 1; m < levels; m++) {
                double d[3], e[3];
                const Stencil fwd = at_interface(face, i, j, m, -1.0, d);
                const double back = sample(wf_next.data(), fst, 1, 0, at_interface(face, i, j, m, 1.0, e));
                float lo = 0.0f, hi = 0.0f;
                for (int n = 0; n < 8; n++) {
                    float v;
                    corner(wf.data(), fst, 1, 0, fwd, n, &v, 1);
                    lo = n == 0 ? v : ov::min(lo, v);
                    hi = n == 0 ? v : ov::max(hi, v);
                }
                const double v = double(wf_next[at * fst + size_t(m)]) + 0.5 * (double(wf[at * fst + size_t(m)]) - back);
                out[m] = v >= lo && v <= hi ? float(v) : wf_next[at * fst + size_t(m)];
            }
        } }, ov::Priority::LOW);
    cur.swap(corr);
    wf.swap(wf_corr);
    centre_vertical();
}

void WeatherFluid::force(double p_dt, const float *p_buoyancy) {
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
            // The buoyancy and the sponge on the interfaces (the buoyancy the mean of the two levels').
            float *wi = face_w(g.index(face, i, j));
            for (int m = 1; m < params.levels; m++) {
                double w = wi[m] + 0.5 * (double(p_buoyancy[cell_index(in, m - 1)]) + p_buoyancy[cell_index(in, m)]) * p_dt;
                const double into = (double(m) * dz - sponge_base) / params.sponge_m;
                if (into > 0.0) {
                    w /= 1.0 + p_dt / params.sponge_tau_s * into * into;
                }
                wi[m] = float(w);
            }
            for (int k = 0; k < params.levels; k++) {
                float *u = here + wind_offset(k);
                const double w = 0.5 * (double(wi[k]) + wi[k + 1]);
                double h[3] = { u[0] - dotf(u, c.up) * up[0], u[1] - dotf(u, c.up) * up[1], u[2] - dotf(u, c.up) * up[2] };
                weather_vec::coriolis(up, c.f, p_dt, h);
                for (int a = 0; a < 3; a++) {
                    u[a] = float(h[a] + w * up[a]);
                }
            }
        } }, ov::Priority::LOW);
    halo();
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
    const float *w = face_w(g.index(face, i, j));
    flux += (double(w[p_level + 1]) - w[p_level]) * c.area;
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
    // The wind loses dt times the pressure's gradient: the horizontal part at the centres (central differences), the
    // vertical on the interfaces (the difference of the two levels round each: exact; the ground and the top rigid).
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherFluid::correct", [&](uint32_t p_row) {
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            const Column &c = column[g.interior(face, i, j)];
            size_t nb[SIDES];
            neighbours(face, i, j, nb);
            const size_t at = g.index(face, i, j) * size_t(levels);
            float *here = &cur[g.index(face, i, j) * size_t(st)];
            float *wi = face_w(g.index(face, i, j));
            for (int m = 1; m < levels; m++) {
                wi[m] = float(wi[m] - p_dt * (double(p[at + size_t(m)]) - p[at + size_t(m - 1)]) / dz);
            }
            for (int k = 0; k < levels; k++) {
                const double di = double(p[nb[0] * size_t(levels) + size_t(k)]) - p[nb[1] * size_t(levels) + size_t(k)];
                const double dj = double(p[nb[2] * size_t(levels) + size_t(k)]) - p[nb[3] * size_t(levels) + size_t(k)];
                float *u = here + wind_offset(k);
                for (int a = 0; a < 3; a++) {
                    u[a] = float(u[a] - p_dt * (c.ax[a] * di + c.ay[a] * dj));
                }
            }
        } }, ov::Priority::LOW);
    centre_vertical();
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
