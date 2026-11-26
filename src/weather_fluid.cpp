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

WeatherLevels WeatherLevels::for_top(double p_top_m, double p_bottom_m, double p_stretch, int p_max) {
    WeatherLevels l;
    l.top_m = p_top_m;
    l.stretch = p_stretch;
    const double r = p_stretch;
    const double n = r > 1.0 ? std::log(1.0 + p_top_m * (r - 1.0) / p_bottom_m) / std::log(r) : p_top_m / p_bottom_m;
    l.count = ov::clamp(int(std::lround(n)), 2, p_max);
    return l;
}

double WeatherLevels::face_m(int p_m) const {
    if (stretch <= 1.0) {
        return top_m * double(p_m) / double(count);
    }
    return top_m * (std::pow(stretch, double(p_m)) - 1.0) / (std::pow(stretch, double(count)) - 1.0);
}

double WeatherLevels::index_at(double p_z_m) const {
    if (stretch <= 1.0) {
        return p_z_m / top_m * double(count);
    }
    return std::log(1.0 + p_z_m / top_m * (std::pow(stretch, double(count)) - 1.0)) / std::log(stretch);
}

ov::str WeatherLevels::glsl_fraction() const {
    if (stretch <= 1.0) {
        return "float weather_level_fraction(float hf) {\n    return hf;\n}\n";
    }
    const double a = std::pow(stretch, double(count)) - 1.0;
    return ov::format("float weather_level_fraction(float hf) {\n    return log(1.0 + hf * %.9g) / %.9g;\n}\n", a, std::log(1.0 + a));
}

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
    if (!(params.stretch >= 1.0) || params.stretch > 2.0) {
        OV_ERR_PRINT(ov::format("WeatherFluid: a stretch of %g is outside [1, 2].", params.stretch));
        params.stretch = ov::clamp(std::isfinite(params.stretch) ? params.stretch : 1.0, 1.0, 2.0);
    }
    vert = WeatherLevels{ params.levels, params.top_m, params.stretch };
    for (int k = 0; k < params.levels; k++) {
        lv_bottom.push_back(vert.face_m(k));
        lv_top.push_back(vert.face_m(k + 1));
        lv_centre.push_back(vert.centre_m(k));
    }
    for (int m = 0; m <= params.levels; m++) {
        iv_centre.push_back(vert.face_m(m));
        iv_bottom.push_back(m > 0 ? lv_centre[size_t(m - 1)] : 0.0);
        iv_top.push_back(m < params.levels ? lv_centre[size_t(m)] : params.top_m);
    }
    const WeatherGrid &g = *cells;
    cur.assign(g.padded_cells() * size_t(stride()), 0.0f);
    next = cur;
    corr = cur;
    side_speed.assign(g.cells() * size_t(params.levels) * SIDES, 0.0f);
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
                    double out[3], e[3], n[3];
                    for (int a = 0; a < 3; a++) {
                        out[a] = nb[s][a] - pos[a];
                        e[a] = edge[s][1][a] - edge[s][0][a];
                    }
                    c.length[s] = float(length3(e));
                    // Perpendicular to the edge in the tangent plane, towards the side's neighbour.
                    cross3(e, up, n);
                    tangent(up, n);
                    const double l = length3(n) * (dot3(n, out) < 0.0 ? -1.0 : 1.0);
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
    make_faces();
}

void WeatherFluid::make_faces() {
    const WeatherGrid &g = *cells;
    const size_t per_face = size_t(g.width()) * size_t(g.height());
    // The interior cell holding each side's neighbour centre: the next cell of the face, or across a seam the
    // neighbour face's cell under the halo centre (the faces' cells meet one to one along their edges).
    auto across = [&](int p_face, int p_i, int p_j) -> uint32_t {
        if (p_i >= 0 && p_i < g.width() && p_j >= 0 && p_j < g.height()) {
            return uint32_t(g.interior(p_face, p_i, p_j));
        }
        double pos[3], up[3];
        g.cell_surface(p_face, p_i, p_j, pos, up);
        const WeatherGrid::Point at = g.locate(pos);
        const int i = ov::clamp(int(std::floor(at.x)), 0, g.width() - 1);
        const int j = ov::clamp(int(std::floor(at.y)), 0, g.height() - 1);
        return uint32_t(g.interior(at.face, i, j));
    };
    for (int face = 0; face < g.faces(); face++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                Column &c = column[g.interior(face, i, j)];
                c.across[0] = across(face, i + 1, j);
                c.across[1] = across(face, i - 1, j);
                c.across[2] = across(face, i, j + 1);
                c.across[3] = across(face, i, j - 1);
            }
        }
    }
    for (size_t in = 0; in < g.cells(); in++) {
        Column &c = column[in];
        for (int s = 0; s < SIDES; s++) {
            const Column &o = column[c.across[s]];
            c.back[s] = 0xff;
            for (int t = 0; t < SIDES; t++) {
                if (o.across[t] == in) {
                    c.back[s] = uint8_t(t);
                }
            }
            OV_ERR_FAIL_COND_MSG(c.back[s] == 0xff, "WeatherFluid: a cell's neighbour across a side doesn't face it back.");
            const size_t n = c.across[s];
            c.across_pad[s] = uint32_t(g.index(int(n / per_face), int(n % size_t(g.width())), int((n % per_face) / size_t(g.width()))));
        }
        // The offsets to the true neighbours' centres (across a seam the neighbour face's cell, not the halo's point),
        // in the tangent plane: the distances along the normals (the two-point pressure flux's), and the least squares
        // of g . r_s = p_s - p over the sides: g = M^-1 sum_s (p_s - p) r_s, M = sum_s r_s r_s^T + |r|^2 up up^T (the
        // up's term only makes M invertible: the offsets are tangent). Exact for a linear pressure on any cell.
        double r[SIDES][3], scale = 0.0;
        for (int s = 0; s < SIDES; s++) {
            const Column &o = column[c.across[s]];
            for (int a = 0; a < 3; a++) {
                r[s][a] = o.pos[a] - c.pos[a];
            }
            const double up[3] = { c.up[0], c.up[1], c.up[2] };
            tangent(up, r[s]);
            c.distance[s] = float(r[s][0] * c.normal[s][0] + r[s][1] * c.normal[s][1] + r[s][2] * c.normal[s][2]);
            OV_ERR_FAIL_COND_MSG(!(c.distance[s] > 0.0f), "WeatherFluid: a cell's neighbour is behind its side.");
            min_spacing_m = ov::min(min_spacing_m, double(c.distance[s]));
            scale += dot3(r[s], r[s]) / SIDES;
        }
        double m[3][3];
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
                m[a][b] = scale * c.up[a] * c.up[b];
                for (int s = 0; s < SIDES; s++) {
                    m[a][b] += r[s][a] * r[s][b];
                }
            }
        }
        const double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        double inv[3][3];
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
                // The adjugate's (a, b): the cofactor of (b, a).
                const int r0 = (b + 1) % 3, r1 = (b + 2) % 3, c0 = (a + 1) % 3, c1 = (a + 2) % 3;
                inv[a][b] = (m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]) / det;
            }
        }
        for (int s = 0; s < SIDES; s++) {
            for (int a = 0; a < 3; a++) {
                c.gradient[s][a] = float(inv[a][0] * r[s][0] + inv[a][1] * r[s][1] + inv[a][2] * r[s][2]);
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
                    const double below = dotf(here + wind_offset(m - 1), c.up);
                    w[m] = float(below + up_share(m) * (dotf(here + wind_offset(m), c.up) - below));
                }
            }
        }
    }
    centre_vertical();
    face_speeds();
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
                    // The volumes' sum over the directions (the unsplit fluxes' bound): the tangent wind's two components at
                    // most sqrt(2) of its speed, w at the centre or either interface.
                    const float *wi = face_w(g.index(face, i, j));
                    const double wv = ov::max(std::fabs(w), ov::max(std::fabs(double(wi[k])), std::fabs(double(wi[k + 1]))));
                    // The thinnest volume w crosses here: the level or the one under it (stretch >= 1), whose
                    // thickness bounds the interfaces' round level k.
                    rate = ov::max(rate, std::sqrt(2.0) * h / min_spacing_m + wv / thick_m(ov::max(k - 1, 0)));
                }
            }
        }
    }
    return rate > 0.0 ? MAX_COURANT / rate : 1e30;
}

void WeatherFluid::face_speeds() {
    const WeatherGrid &g = *cells;
    const int st = stride();
    const int levels = params.levels;
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherFluid::face_speeds", [&](uint32_t p_row) {
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            const size_t in = g.interior(face, i, j);
            const Column &c = column[in];
            const float *here = &cur[g.index(face, i, j) * size_t(st)];
            for (int s = 0; s < SIDES; s++) {
                if (!owns(in, s)) {
                    continue;
                }
                for (int k = 0; k < levels; k++) {
                    const float *v = &cur[size_t(c.across_pad[s]) * size_t(st) + size_t(wind_offset(k))];
                    side_speed[(in * size_t(levels) + size_t(k)) * SIDES + size_t(s)] = float(0.5 * (dotf(here + wind_offset(k), c.normal[s]) + dotf(v, c.normal[s])));
                }
            }
        } }, ov::Priority::LOW);
}

double WeatherFluid::outward(size_t p_interior, int p_level, int p_side) const {
    const Column &c = column[p_interior];
    const size_t levels = size_t(params.levels);
    if (owns(p_interior, p_side)) {
        return side_speed[(p_interior * levels + size_t(p_level)) * SIDES + size_t(p_side)];
    }
    return -double(side_speed[(size_t(c.across[p_side]) * levels + size_t(p_level)) * SIDES + c.back[p_side]]);
}
template <typename Vertical, typename Side>
void WeatherFluid::flux_stage(const std::vector<float> &p_in, std::vector<float> &r_out, int p_count, int p_per, int p_first, int p_last, double p_dt,
        const double *p_bottom, const double *p_centre, const double *p_top, Vertical p_vertical, Side p_side) {
    const WeatherGrid &g = *cells;
    const size_t rec = size_t(p_count) * size_t(p_per);
    std::vector<float> &grad = p_count == params.levels ? grad_c : grad_w;
    grad.assign(g.padded_cells() * rec * 3, 0.0f);
    auto at = [&](size_t p_cell, int p_v, int p_f) { return double(p_in[p_cell * rec + size_t(p_v * p_per + p_f)]); };
    // The limited gradient of each field in each active volume (Barth & Jespersen 1989): the central differences', scaled
    // so that the value it gives at each of the 6 faces stays within the volume's and its neighbours' range.
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherFluid::gradients", [&](uint32_t p_row) {
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            const Column &c = column[g.interior(face, i, j)];
            const size_t here = g.index(face, i, j);
            size_t nb[SIDES];
            neighbours(face, i, j, nb);
            for (int v = p_first; v <= p_last; v++) {
                const int v0 = ov::max(v - 1, 0), v1 = ov::min(v + 1, p_count - 1);
                for (int f = 0; f < p_per; f++) {
                    const double x = at(here, v, f);
                    const double di = at(nb[0], v, f) - at(nb[1], v, f);
                    const double dj = at(nb[2], v, f) - at(nb[3], v, f);
                    const double dk = (at(here, v1, f) - at(here, v0, f)) / (p_centre[v1] - p_centre[v0]);
                    double gr[3];
                    for (int a = 0; a < 3; a++) {
                        gr[a] = c.ax[a] * di + c.ay[a] * dj + c.up[a] * dk;
                    }
                    double lo = ov::min(at(here, v0, f), at(here, v1, f)), hi = ov::max(at(here, v0, f), at(here, v1, f));
                    for (int s = 0; s < SIDES; s++) {
                        lo = ov::min(lo, at(nb[s], v, f));
                        hi = ov::max(hi, at(nb[s], v, f));
                    }
                    lo = ov::min(lo, x);
                    hi = ov::max(hi, x);
                    double alpha = 1.0;
                    auto limit = [&](double p_delta) {
                        if (p_delta > 1e-30) {
                            alpha = ov::min(alpha, (hi - x) / p_delta);
                        } else if (p_delta < -1e-30) {
                            alpha = ov::min(alpha, (lo - x) / p_delta);
                        }
                    };
                    for (int s = 0; s < SIDES; s++) {
                        limit(0.5 * c.distance[s] * (gr[0] * c.normal[s][0] + gr[1] * c.normal[s][1] + gr[2] * c.normal[s][2]));
                    }
                    const double gk = gr[0] * c.up[0] + gr[1] * c.up[1] + gr[2] * c.up[2];
                    limit((p_top[v] - p_centre[v]) * gk);
                    limit((p_bottom[v] - p_centre[v]) * gk);
                    float *o = &grad[(here * rec + size_t(v * p_per + f)) * 3];
                    for (int a = 0; a < 3; a++) {
                        o[a] = float(alpha * gr[a]);
                    }
                }
            }
        } }, ov::Priority::LOW);
    // A face's outward flux from its owner o through side p_s (volume p_v, field p_f), per second: the owner's speed,
    // length and normal, the value the upwind volume's reconstruction at the face. Both cells call it for the owner, so
    // what one loses the other gains to the bit.
    auto face_flux = [&](size_t p_o, int p_s, int p_v, int p_f) {
        const Column &c = column[p_o];
        const double u = p_side(p_o, p_v, p_s) * double(c.length[p_s]) * (p_top[p_v] - p_bottom[p_v]);
        const double h = 0.5 * double(c.distance[p_s]);
        const double r[3] = { h * c.normal[p_s][0], h * c.normal[p_s][1], h * c.normal[p_s][2] };
        const size_t cell = u >= 0.0 ? padded(p_o) : size_t(c.across_pad[p_s]);
        const double sign = u >= 0.0 ? 1.0 : -1.0;
        const float *gr = &grad[(cell * rec + size_t(p_v * p_per + p_f)) * 3];
        return u * (at(cell, p_v, p_f) + sign * (gr[0] * r[0] + gr[1] * r[1] + gr[2] * r[2]));
    };
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherFluid::fluxes", [&](uint32_t p_row) {
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            const size_t in = g.interior(face, i, j);
            const Column &c = column[in];
            const size_t here = g.index(face, i, j);
            auto value = [&](int p_v, int p_f, double p_up) {
                const float *gr = &grad[(here * rec + size_t(p_v * p_per + p_f)) * 3];
                return at(here, p_v, p_f) + p_up * (gr[0] * c.up[0] + gr[1] * c.up[1] + gr[2] * c.up[2]);
            };
            for (int v = 0; v < p_count; v++) {
                for (int f = 0; f < p_per; f++) {
                    double out = at(here, v, f);
                    if (v >= p_first && v <= p_last) {
                        double flux = 0.0; // outward, per second
                        for (int s = 0; s < SIDES; s++) {
                            flux += owns(in, s) ? face_flux(in, s, v, f) : -face_flux(c.across[s], c.back[s], v, f);
                        }
                        if (v > 0) {
                            const double w = p_vertical(here, v) * double(c.area);
                            flux -= w * (w >= 0.0 ? value(v - 1, f, p_top[v - 1] - p_centre[v - 1]) : value(v, f, p_bottom[v] - p_centre[v]));
                        }
                        if (v + 1 < p_count) {
                            const double w = p_vertical(here, v + 1) * double(c.area);
                            flux += w * (w >= 0.0 ? value(v, f, p_top[v] - p_centre[v]) : value(v + 1, f, p_bottom[v + 1] - p_centre[v + 1]));
                        }
                        out -= p_dt * flux / (double(c.area) * (p_top[v] - p_bottom[v]));
                    }
                    r_out[here * rec + size_t(v * p_per + f)] = float(out);
                }
            }
        } }, ov::Priority::LOW);
    g.fill_halo(r_out.data(), int(rec));
}

void WeatherFluid::advect(double p_dt) {
    const int levels = params.levels;
    const size_t fst = size_t(levels + 1);
    halo();
    // The levels' volumes: through their sides the centres' mean wind, through their tops and bottoms the interfaces'.
    // The interfaces' volumes (a level's centre to the next's): the two levels' side speeds weighted by their halves in
    // it, at their tops and bottoms the levels' centres' w (the interfaces' means). The ground's and the top's interface
    // stay 0.
    auto level_vertical = [&](size_t p_cell, int p_v) { return double(wf_from[p_cell * fst + size_t(p_v)]); };
    auto level_side = [&](size_t p_in, int p_v, int p_s) { return outward(p_in, p_v, p_s); };
    auto interface_vertical = [&](size_t p_cell, int p_v) {
        return 0.5 * (double(wf_from[p_cell * fst + size_t(p_v - 1)]) + wf_from[p_cell * fst + size_t(p_v)]);
    };
    auto interface_side = [&](size_t p_in, int p_v, int p_s) {
        const double below = thick_m(p_v - 1), above = thick_m(p_v);
        return (level_side(p_in, p_v - 1, p_s) * below + level_side(p_in, p_v, p_s) * above) / (below + above);
    };
    wf_from = wf; // the speeds stay the step's start's through both stages
    // Heun (SSP-RK2, Shu & Osher 1988): two stages, their mean with the start.
    const double *lb = lv_bottom.data(), *lc = lv_centre.data(), *lt = lv_top.data();
    const double *ib = iv_bottom.data(), *ic = iv_centre.data(), *it = iv_top.data();
    flux_stage(cur, next, levels, per_level(), 0, levels - 1, p_dt, lb, lc, lt, level_vertical, level_side);
    flux_stage(wf, wf_next, levels + 1, 1, 1, levels - 1, p_dt, ib, ic, it, interface_vertical, interface_side);
    flux_stage(next, corr, levels, per_level(), 0, levels - 1, p_dt, lb, lc, lt, level_vertical, level_side);
    flux_stage(wf_next, wf_corr, levels + 1, 1, 1, levels - 1, p_dt, ib, ic, it, interface_vertical, interface_side);
    for (size_t n = 0; n < cur.size(); n++) {
        cur[n] = 0.5f * (cur[n] + corr[n]);
    }
    for (size_t n = 0; n < wf.size(); n++) {
        wf[n] = 0.5f * (wf[n] + wf_corr[n]);
    }
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
            // The buoyancy and the sponge on the interfaces (the buoyancy linear between the two levels').
            float *wi = face_w(g.index(face, i, j));
            for (int m = 1; m < params.levels; m++) {
                const double b0 = p_buoyancy[cell_index(in, m - 1)];
                double w = wi[m] + (b0 + up_share(m) * (double(p_buoyancy[cell_index(in, m)]) - b0)) * p_dt;
                const double into = (face_m(m) - sponge_base) / params.sponge_m;
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
    const Column &c = column[p_interior];
    double flux = 0.0;
    for (int s = 0; s < SIDES; s++) {
        flux += outward(p_interior, p_level, s) * face_length(p_interior, s) * thick_m(p_level);
    }
    const float *w = face_w(padded(p_interior));
    flux += (double(w[p_level + 1]) - w[p_level]) * c.area;
    return flux / (double(c.area) * thick_m(p_level));
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
    // Red-black within a face; across a seam, where a neighbour may be the same colour, the half sweep's start values
    // (p_from), so the result doesn't hang on the threads' order.
    p_from = p;
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherFluid::sweep", [&](uint32_t p_row) {
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            const size_t in = g.interior(face, i, j);
            const Column &c = column[in];
            const size_t at = g.index(face, i, j) * size_t(levels);
            for (int k = (i + j + p_colour) & 1; k < levels; k += 2) {
                double sum = 0.0;
                double diag = 0.0;
                for (int s = 0; s < SIDES; s++) {
                    const double a = face_length(in, s) * thick_m(k) / face_distance(in, s);
                    sum += a * p_from[size_t(c.across_pad[s]) * size_t(levels) + size_t(k)];
                    diag += a;
                }
                if (k + 1 < levels) {
                    const double vertical = double(c.area) / (level_m(k + 1) - level_m(k));
                    sum += vertical * p_from[at + size_t(k + 1)];
                    diag += vertical;
                }
                if (k > 0) {
                    const double vertical = double(c.area) / (level_m(k) - level_m(k - 1));
                    sum += vertical * p_from[at + size_t(k - 1)];
                    diag += vertical;
                }
                const double gs = (sum - rhs[cell_index(in, k)]) / diag;
                p[at + size_t(k)] = float(p[at + size_t(k)] + SOR * (gs - p[at + size_t(k)]));
            }
        } }, ov::Priority::LOW);
}
void WeatherFluid::project(double p_dt) {
    const WeatherGrid &g = *cells;
    const int st = stride();
    const int levels = params.levels;
    // The divergence of the faces' speeds (the centres' means) over dt: what the pressure takes out.
    face_speeds();
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherFluid::divergence", [&](uint32_t p_row) {
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            const size_t in = g.interior(face, i, j);
            for (int k = 0; k < levels; k++) {
                rhs[cell_index(in, k)] = float(divergence(in, k) * double(column[in].area) * thick_m(k) / p_dt);
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
    g.fill_halo(p.data(), levels);
    // The faces' speeds (the advection's) lose dt times the difference of the two pressures across them, the solve's
    // own: the MAC projection of a cell-centred grid (Bell, Colella & Glaz 1989; Almgren et al. 1998), so the fluxes
    // that move the fields keep only the solver's residual divergence.
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherFluid::side_speeds", [&](uint32_t p_row) {
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            const size_t in = g.interior(face, i, j);
            const Column &c = column[in];
            const size_t here = g.index(face, i, j);
            for (int s = 0; s < SIDES; s++) {
                if (!owns(in, s)) {
                    continue;
                }
                for (int k = 0; k < levels; k++) {
                    const double dp = double(p[size_t(c.across_pad[s]) * size_t(levels) + size_t(k)]) - p[here * size_t(levels) + size_t(k)];
                    float &u = side_speed[(in * size_t(levels) + size_t(k)) * SIDES + size_t(s)];
                    u = float(u - p_dt * dp / c.distance[s]);
                }
            }
        } }, ov::Priority::LOW);
    // The wind loses dt times the pressure's gradient: the horizontal part at the centres (central differences), the
    // vertical on the interfaces (the difference of the two levels round each: exact; the ground and the top rigid).
    parallel_for(uint32_t(g.faces() * g.height()), "WeatherFluid::correct", [&](uint32_t p_row) {
        const int face = int(p_row) / g.height();
        const int j = int(p_row) % g.height();
        for (int i = 0; i < g.width(); i++) {
            const size_t in = g.interior(face, i, j);
            const Column &c = column[in];
            const size_t at = g.index(face, i, j) * size_t(levels);
            float *here = &cur[g.index(face, i, j) * size_t(st)];
            float *wi = face_w(g.index(face, i, j));
            for (int m = 1; m < levels; m++) {
                wi[m] = float(wi[m] - p_dt * (double(p[at + size_t(m)]) - p[at + size_t(m - 1)]) / (level_m(m) - level_m(m - 1)));
            }
            for (int k = 0; k < levels; k++) {
                double grad[3] = { 0.0, 0.0, 0.0 };
                for (int s = 0; s < SIDES; s++) {
                    const double gs = double(p[size_t(c.across_pad[s]) * size_t(levels) + size_t(k)]) - p[at + size_t(k)];
                    for (int a = 0; a < 3; a++) {
                        grad[a] += gs * c.gradient[s][a];
                    }
                }
                float *u = here + wind_offset(k);
                for (int a = 0; a < 3; a++) {
                    u[a] = float(u[a] - p_dt * grad[a]);
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
