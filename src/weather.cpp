#include "weather.hpp"

#include "parallel_for.hpp"
#include "trace.hpp"

#include "ov/math.hpp"

#include <cmath>

Weather::Weather(const WorldFile::Weather &p_weather, const BaseSurface &p_base, double p_voxel_m, uint32_t p_seed, ForcingFn p_forcing) :
        settings(p_weather), base(p_base), voxel_m(p_voxel_m), forcing(std::move(p_forcing)) {
    ov::str why;
    cells = WeatherGrid::for_base(p_base, p_voxel_m, p_weather.cell_m, &why);
    if (!cells) {
        // The world file refuses such a weather: a grid of the fewest cells keeps the worker's loops valid.
        OV_ERR_PRINT("Weather: " + why);
        cells = std::make_shared<CubeSphereGrid>(WeatherGrid::MIN_N, p_base.get_radius() * p_voxel_m);
    }
    params.radius_m = p_base.get_radius() * p_voxel_m;
    params.top_m = p_weather.top_m;
    params.levels = ov::clamp(int(std::lround(p_weather.top_m / cells->cell_m())), 1, WeatherSim::MAX_LEVELS);
    params.prevailing_mps = p_weather.prevailing_mps;
    params.storm_m = p_weather.storm_m;
    params.seed = p_seed;
    // The climate read at the level whose cells are about the grid's.
    const double cell_vox = cells->cell_m() / p_voxel_m;
    forcing_level = ov::max(0, int(std::floor(std::log2(ov::max(cell_vox, 1.0)))));
    worker = ov::executor().start_worker("weather", *this);
    worker->wake(); // the forcing and the spin-up start now
}

Weather::~Weather() {
    worker->stop(); // the tick running finishes, before the members it uses go
}

Climate::BiomeWeather Weather::forcing_at(const Climate::Table &p_table, const Terrain::LevelProgram &p_prog, const VoxelPos &p_voxel, int p_m) {
    const int m = ov::max(0, ov::min(p_m, p_prog.top));
    Climate::Float4 w[Climate::MAX_BIOMES] = {};
    const int count = ov::min(int(p_prog.weights.size()), int(p_table.biomes.size()));
    Terrain::point_nodes(p_prog, WorldCoord::floor_div(p_voxel, int64_t(1) << m), m, p_prog.weights.data(), count, w);
    double sum = 0.0;
    double v[6] = {};
    for (int b = 0; b < count; b++) {
        const double wb = ov::max(double(w[b].w), 0.0);
        const Climate::BiomeWeather &bw = p_table.biomes[size_t(b)].weather;
        sum += wb;
        v[0] += wb * bw.wind;
        v[1] += wb * bw.drag;
        v[2] += wb * bw.heat;
        v[3] += wb * bw.moisture;
        v[4] += wb * bw.gust;
        v[5] += wb * bw.storms;
    }
    if (!(sum > 0.0)) {
        return Climate::BiomeWeather();
    }
    return Climate::BiomeWeather{ float(v[0] / sum), float(v[1] / sum), float(v[2] / sum), float(v[3] / sum), float(v[4] / sum), float(v[5] / sum) };
}

// The worker's first step: each cell's forcing from the climate under its centre, the simulation, its spin-up.
void Weather::build() {
    OV_TRACE("weather", "forcing");
    const uint64_t t0 = ov::now_usec();
    const WeatherGrid &g = *cells;
    std::vector<Climate::BiomeWeather> cell_weather(g.cells());
    if (forcing) {
        parallel_for(uint32_t(g.faces() * g.height()), "Weather::forcing", [&](uint32_t p_row) {
            const int face = int(p_row) / g.height();
            const int j = int(p_row) % g.height();
            for (int i = 0; i < g.width(); i++) {
                double pos[3], up[3];
                g.cell_surface(face, i, j, pos, up);
                const VoxelPos ground = base.centre() + VoxelPos(std::llround(pos[0] / voxel_m), std::llround(pos[1] / voxel_m), std::llround(pos[2] / voxel_m));
                cell_weather[g.interior(face, i, j)] = forcing(ground, forcing_level);
            } }, ov::Priority::LOW);
    }
    sim = std::make_unique<WeatherSim>(cells, params, cell_weather);
    if (settings.spin_up_s > 0.0) {
        sim->step(settings.spin_up_s);
    }
    std::lock_guard<std::mutex> lock(mutex);
    forcing_stat.add(ov::now_usec() - t0);
    substep_s = sim->max_substep_s();
    omega_rad_s = sim->omega();
}

std::vector<float> Weather::look() const {
    const int st = stride();
    const int sim_st = sim->stride();
    const int levels_n = sim->levels();
    std::vector<float> out(cells->padded_cells() * size_t(st));
    const std::vector<float> &s = sim->state();
    const std::vector<float> &d = sim->diagnostics();
    for (size_t c = 0; c < cells->padded_cells(); c++) {
        const float *sc = &s[c * size_t(sim_st)];
        const float *dc = &d[c * WeatherSim::DIAGS];
        float *o = &out[c * size_t(st)];
        o[WIND_X] = sc[WeatherSim::WIND_X];
        o[WIND_Y] = sc[WeatherSim::WIND_Y];
        o[WIND_Z] = sc[WeatherSim::WIND_Z];
        o[COVER] = dc[WeatherSim::COVER];
        o[TYPE] = dc[WeatherSim::TYPE];
        o[RAIN] = dc[WeatherSim::RAIN];
        o[TEMP] = sc[WeatherSim::TEMP];
        o[VAPOUR] = sc[WeatherSim::VAPOUR];
        float aloft[3] = { 0.0f, 0.0f, 0.0f };
        for (int k = 0; k < levels_n; k++) {
            const float *l = sc + WeatherSim::level_offset(k);
            for (int a = 0; a < 3; a++) {
                aloft[a] += l[WeatherSim::WIND_X + a] / float(levels_n);
            }
            o[LOOK + k] = l[WeatherSim::CLOUD] / float(WeatherSim::FULL_CLOUD);
        }
        o[ALOFT_X] = aloft[0];
        o[ALOFT_Y] = aloft[1];
        o[ALOFT_Z] = aloft[2];
    }
    return out;
}

bool Weather::step() {
    if (!sim) {
        build();
    } else {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (done >= due) {
                return false;
            }
        }
        OV_TRACE("weather", "tick");
        const uint64_t t0 = ov::now_usec();
        sim->step(settings.time_scale / settings.tick_hz);
        std::lock_guard<std::mutex> lock(mutex);
        tick_stat.add(ov::now_usec() - t0);
        done++;
    }
    std::vector<float> state = look();
    bool more;
    {
        std::lock_guard<std::mutex> lock(mutex);
        next_look = std::move(state);
        next_tick = sim->ticks();
        next_done = done;
        next_time_s = sim->time_s();
        next_storms = sim->storms_started();
        ready_state = true;
        started = true;
        more = done < due;
    }
    published.notify_all();
    return more;
}

void Weather::finish() {
    if (!ov::executor().threaded()) {
        ov::executor().pump(); // steps the worker
        return;
    }
    std::unique_lock<std::mutex> lock(mutex);
    published.wait(lock, [this]() { return started && done >= due; });
}

bool Weather::update(uint64_t p_now_usec) {
    OV_TRACE("weather", "update");
    const double dt = last_usec == 0 ? 0.0 : ov::min(double(p_now_usec - last_usec) * 1e-6, MAX_FRAME_S);
    last_usec = p_now_usec;
    bool landed = false;
    bool wake = false;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (ready_state) {
            ready_state = false;
            landed = true;
            prev_look = has_live ? std::move(live_look) : next_look;
            live_look = std::move(next_look);
            live_tick = next_tick;
            live_done = next_done;
            live_time_s = next_time_s;
            live_storms = next_storms;
            has_live = true;
        }
        // The clock runs from the first state (the forcing and the spin-up take what they take).
        if (started) {
            clock_ticks += dt * settings.tick_hz;
            due = ov::max(due, uint64_t(clock_ticks + 1e-6)); // a whole tick summed from frames lands a hair under
            if (due > done + MAX_BEHIND) {
                dropped += due - done - MAX_BEHIND;
                done = due - MAX_BEHIND;
            }
            wake = done < due;
        }
    }
    if (wake) {
        worker->wake();
    }
    return landed;
}

float Weather::blend() const {
    // The state of clock tick k is due at k: the frame shows the time one tick behind the clock, between the previous
    // state and the latest.
    return has_live ? float(ov::clamp(clock_ticks - double(live_done), 0.0, 1.0)) : 1.0f;
}

bool Weather::at(const double p_pos[3], Sample &r_sample) const {
    if (!has_live) {
        return false;
    }
    float v[LOOK];
    cells->sample(live_look.data(), stride(), LOOK, p_pos, v);
    r_sample.wind_mps = Float3{ v[WIND_X], v[WIND_Y], v[WIND_Z] };
    r_sample.aloft_mps = Float3{ v[ALOFT_X], v[ALOFT_Y], v[ALOFT_Z] };
    r_sample.cover = v[COVER];
    r_sample.type = v[TYPE];
    r_sample.rain = v[RAIN];
    r_sample.temp = v[TEMP];
    r_sample.vapour = v[VAPOUR];
    return true;
}

bool Weather::clouds_at(const double p_pos[3], float *r_levels) const {
    if (!has_live) {
        return false;
    }
    cells->sample(live_look.data() + LOOK, stride(), params.levels, p_pos, r_levels);
    return true;
}

ov::Value Weather::get_stats() const {
    ov::Value d = ov::Value::object();
    std::lock_guard<std::mutex> lock(mutex);
    d["grid"] = ov::format("%d faces of %d x %d", cells->faces(), cells->width(), cells->height());
    d["cell_m"] = cells->cell_m();
    d["cells"] = int64_t(cells->cells());
    d["levels"] = int64_t(params.levels);
    d["top_m"] = params.top_m;
    d["ready"] = has_live;
    d["ticks"] = int64_t(live_tick);
    d["game_s"] = live_time_s;
    d["storms"] = int64_t(live_storms);
    d["dropped_ticks"] = int64_t(dropped);
    d["behind_ticks"] = int64_t(due > done ? due - done : 0);
    d["tick"] = tick_stat.to_value();
    d["forcing"] = forcing_stat.to_value();
    d["substep_s"] = substep_s;
    d["omega_rad_s"] = omega_rad_s;
    return d;
}
