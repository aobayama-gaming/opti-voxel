#include "weather.hpp"

#include "weather_debug.hpp"

#include "parallel_for.hpp"
#include "trace.hpp"

#include "ov/math.hpp"

#include <cmath>

Weather::Weather(const WorldFile::Weather &p_weather, const BaseSurface &p_base, double p_voxel_m, uint32_t p_seed, ForcingFn p_forcing,
        bool p_probing) :
        settings(p_weather), base(p_base), voxel_m(p_voxel_m), forcing(std::move(p_forcing)), probing(p_probing) {
    ov::str why;
    cells = WeatherGrid::for_base(p_base, p_voxel_m, p_weather.cell_m, &why);
    if (!cells) {
        // The world file refuses such a weather: a grid of the fewest cells keeps the worker's loops valid.
        OV_ERR_PRINT("Weather: " + why);
        cells = std::make_shared<CubeSphereGrid>(WeatherGrid::MIN_N, p_base.get_radius() * p_voxel_m);
    }
    radius = p_base.get_radius() * p_voxel_m;
    level_count = ov::clamp(int(std::lround(p_weather.top_m / cells->cell_m())), 2, WeatherFluid::MAX_LEVELS);
    seed = p_seed;
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
    constexpr int KEYS = int(sizeof(Climate::WEATHER_KEYS) / sizeof(Climate::WEATHER_KEYS[0]));
    double sum = 0.0;
    double v[KEYS] = {};
    for (int b = 0; b < count; b++) {
        const double wb = ov::max(double(w[b].w), 0.0);
        const Climate::BiomeWeather &bw = p_table.biomes[size_t(b)].weather;
        sum += wb;
        for (int k = 0; k < KEYS; k++) {
            v[k] += wb * double(bw.*Climate::WEATHER_KEYS[k].member);
        }
    }
    Climate::BiomeWeather out;
    if (sum > 0.0) {
        for (int k = 0; k < KEYS; k++) {
            out.*Climate::WEATHER_KEYS[k].member = float(v[k] / sum);
        }
    }
    return out;
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
    WeatherFluid::Params fp;
    fp.levels = level_count;
    fp.top_m = settings.top_m;
    fp.scalars = WeatherMoist::SCALARS;
    WeatherForcing::Params dp;
    dp.day_s = settings.day_s;
    dp.prevailing_mps = settings.prevailing_mps;
    dp.storm_m = settings.storm_m;
    dp.inversion_m = settings.inversion_m;
    dp.seed = seed;
    fp.omega_rad_s = 2.0 * 3.14159265358979323846 / settings.day_s;
    fluid = std::make_unique<WeatherFluid>(cells, fp);
    drive = std::make_unique<WeatherForcing>(*fluid, dp, std::move(cell_weather));
    WeatherMoist::Params mp;
    mp.exchange = settings.exchange;
    mp.spread = settings.cloud_spread;
    const WeatherMoist::Sounding sounding = drive->sounding(*fluid);
    moist = std::make_unique<WeatherMoist>(*fluid, sounding, mp);
    convection = std::make_unique<WeatherConvection>(*fluid, sounding, settings.inversion_m);
    drive->start(*fluid, sounding);
    const ov::Value summary = drive->summary(*fluid);
    buoyancy.assign(fluid->cell_count(), 0.0f);
    drive->grounds(ground);
    if (settings.spin_up_s > 0.0) {
        advance(settings.spin_up_s);
    }
    std::lock_guard<std::mutex> lock(mutex);
    forcing_stat.add(ov::now_usec() - t0);
    omega_rad_s = fp.omega_rad_s;
    forcing_summary = summary;
}

void Weather::set_sun(const Float3 &p_towards_sun) {
    std::lock_guard<std::mutex> lock(mutex);
    sun_dir = p_towards_sun;
}

double Weather::day_angle(double p_game_s) const {
    return -2.0 * 3.14159265358979323846 / settings.day_s * (p_game_s - settings.spin_up_s);
}

Float3 Weather::sun_at(double p_game_s) const {
    std::lock_guard<std::mutex> lock(mutex);
    return WeatherForcing::turn_sun(sun_dir, day_angle(p_game_s));
}

double Weather::shown_s() const {
    if (!has_live) {
        return settings.spin_up_s;
    }
    return live_time_s - (1.0 - double(blend())) * settings.time_scale / settings.tick_hz;
}

void Weather::advance(double p_dt) {
    double left = p_dt;
    while (left > 1e-9) {
        const double dt = ov::min(ov::min(left, MAX_STEP_S), fluid->max_dt());
        drive->warm(sun_at(sim_time_s + 0.5 * dt), dt, *fluid, moist->reference().rho, nullptr, ground);
        fluid->advect(dt);
        drive->apply(*fluid, dt, steps);
        if (settings.convection) {
            convection->apply(*fluid, moist->surface(), dt);
        }
        moist->apply(*fluid, ground, dt, buoyancy.data(), settings.convection ? convection->fluxes() : WeatherMoist::Convective{});
        fluid->force(dt, buoyancy.data());
        fluid->project(dt);
        left -= dt;
        sim_time_s += dt;
        steps++;
        substep_s = dt;
    }
}

std::vector<float> Weather::look() const {
    const WeatherGrid &g = *cells;
    const int st = stride();
    const int fst = fluid->stride();
    const int levels_n = level_count;
    std::vector<float> out(g.padded_cells() * size_t(st), 0.0f);
    const std::vector<float> &s = fluid->state();
    const std::vector<double> &exner = moist->reference().exner;
    for (int f = 0; f < g.faces(); f++) {
        for (int j = 0; j < g.height(); j++) {
            for (int i = 0; i < g.width(); i++) {
                const size_t c = g.index(f, i, j);
                const size_t in = g.interior(f, i, j);
                const float *sc = &s[c * size_t(fst)];
                float *o = &out[c * size_t(st)];
                for (int a = 0; a < 3; a++) {
                    o[WIND_X + a] = sc[fluid->wind_offset(0) + a];
                }
                float aloft[3] = { 0.0f, 0.0f, 0.0f };
                float cover = 0.0f;
                int cloudy = 0;
                for (int k = 0; k < levels_n; k++) {
                    for (int a = 0; a < 3; a++) {
                        aloft[a] += sc[fluid->wind_offset(k) + a] / float(levels_n);
                    }
                    const double water = sc[fluid->scalar_offset(k, WeatherMoist::QC)]; // the partial cloud's mean (the updrafts' in it)
                    const float cloud = float(water / FULL_CLOUD_QC);
                    o[LOOK + k] = cloud;
                    cover = ov::max(cover, cloud);
                    cloudy += cloud > 0.03f;
                }
                o[COVER] = ov::min(cover, 1.0f);
                o[TYPE] = ov::clamp(float(cloudy) / ov::max(float(levels_n) / 3.0f, 1.0f), 0.0f, 1.0f);
                o[RAIN] = float(double(moist->rain()[in]) * 3.6e6); // m/s of water to mm/h
                o[TEMP] = float(double(sc[fluid->scalar_offset(0, WeatherMoist::THETA)]) * exner[0] - 273.15);
                o[VAPOUR] = sc[fluid->scalar_offset(0, WeatherMoist::QV)] * 1000.0f;
                for (int a = 0; a < 3; a++) {
                    o[ALOFT_X + a] = aloft[a];
                }
            }
        }
    }
    g.fill_halo(out.data(), st);
    return out;
}

bool Weather::step() {
    if (!fluid) {
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
        advance(settings.time_scale / settings.tick_hz);
        std::lock_guard<std::mutex> lock(mutex);
        tick_stat.add(ov::now_usec() - t0);
        done++;
    }
    std::vector<float> state = look();
    std::shared_ptr<const WeatherDebug::Snapshot> snapshot =
            probing ? WeatherDebug::Snapshot::capture(*fluid, *moist, *convection, *drive, ground, buoyancy, sim_time_s) : nullptr;
    double day_k = 0.0, night_k = 0.0;
    drive->day_night(*fluid, sun_at(sim_time_s), ground, day_k, night_k);
    double rising = 0.0, top_sum = 0.0;
    for (const float t : convection->tops()) {
        if (!settings.convection) {
            break; // its tops are the start's
        }
        rising += t > 0.0f ? 1.0 : 0.0;
        top_sum += double(t);
    }
    bool more;
    {
        std::lock_guard<std::mutex> lock(mutex);
        ground_day_k = day_k;
        ground_night_k = night_k;
        convecting = rising / double(ov::max(convection->tops().size(), size_t(1)));
        convection_top_m = rising > 0.0 ? top_sum / rising : 0.0;
        next_look = std::move(state);
        next_tick = steps;
        next_done = done;
        next_time_s = sim_time_s;
        next_storms = drive->storms();
        next_probe = std::move(snapshot);
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
            live_probe = std::move(next_probe);
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
    cells->sample(live_look.data() + LOOK, stride(), level_count, p_pos, r_levels);
    return true;
}

ov::Value Weather::get_stats() const {
    ov::Value d = ov::Value::object();
    std::lock_guard<std::mutex> lock(mutex);
    d["grid"] = ov::format("%d faces of %d x %d", cells->faces(), cells->width(), cells->height());
    d["cell_m"] = cells->cell_m();
    d["cells"] = int64_t(cells->cells());
    d["levels"] = int64_t(level_count);
    d["top_m"] = settings.top_m;
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
    d["forcing_summary"] = forcing_summary;
    d["ground_day_k"] = ground_day_k;
    d["ground_night_k"] = ground_night_k;
    d["convecting"] = convecting;
    d["convection_top_m"] = convection_top_m;
    return d;
}
