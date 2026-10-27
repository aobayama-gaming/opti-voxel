#pragma once

#include "base_surface.hpp"
#include "gpu_device.hpp"

#include "ov/bytes.hpp"
#include "ov/str.hpp"

#include <functional>

// The atmosphere (docs/look.md): Hillaire 2020, "A Scalable and Production Ready Sky and Atmosphere Rendering
// Technique" (UE's SkyAtmosphere; reference code sebh/UnrealEngineSkyAtmosphere, MIT), made from the base's shape: its
// medium is a function of the altitude above the base (BaseSurface::glsl_render's base_altitude), every ray's extent
// comes from base_level_span and the sun's transmittance from a march through that altitude, so a base of any kind gets
// an atmosphere that follows it. Hillaire's tables indexed by a sphere's (r, mu) are gone; the one local approximation
// left is his own: the multiple scattering of a point is read from a table built on a sphere of the base's curvature
// radius (Earth's for the flat world), by its altitude and the sun's height over its up.
//
// Each frame two compute passes run on the host's main device: the multiple-scattering table (32 x 32, only when the
// medium or the base changes) and the camera volume, a camera-centred grid (128 directions round the base's up at the
// camera x 64 heights, Hillaire's non-linear latitude sharpening the horizon) of 32 distance slices, quadratic out to
// range_m(), the last one the whole ray (the sky). A texel holds the in-scattered light from the camera to its slice
// (RGB, for a sun of illuminance 1) and the mean transmittance (A), like UE's aerial perspective. The host's sky and
// page shaders read it through glsl_lookup.
struct AtmosphereMedium {
    // Earth's (Hillaire's reference values): coefficients in 1/m at the base, heights in metres.
    Float3 rayleigh{ 5.802e-6f, 13.558e-6f, 33.1e-6f }; // scattering
    float rayleigh_height_m = 8000.0f;
    float mie = 3.996e-6f; // scattering
    float mie_absorption = 4.40e-6f;
    float mie_height_m = 1200.0f;
    float mie_g = 0.8f; // Cornette-Shanks
    Float3 ozone{ 0.650e-6f, 1.881e-6f, 0.085e-6f }; // absorption at its peak
    float ozone_centre_m = 25000.0f;
    float ozone_width_m = 30000.0f; // a tent, zero past centre +- width / 2
    float top_m = 100000.0f;
    Float3 ground_albedo{ 0.3f, 0.3f, 0.3f };

    static constexpr double EARTH_RADIUS_M = 6360000.0;
    // The top stays at least this many times the ground's relief above the base, so the terrain is inside the air.
    static constexpr double RELIEF_TOPS = 4.0;

    // Earth's medium for a base of curvature radius p_radius_m (BaseSurface::curvature_radius_m) under a ground of relief
    // p_relief_m: under Earth's radius every height times R / R_earth (at least RELIEF_TOPS reliefs at the top) and
    // every coefficient over it, the same optical depths (a small planet keeps a blue sky and a limb); p_scale > 0 sets
    // that factor instead.
    static AtmosphereMedium for_base(double p_radius_m, double p_relief_m, double p_scale = 0.0);

    // The fields as the shaders' uniforms (GLSLUniforms).
    template <class F>
    void visit(F &&p_f) const {
        p_f("atmo_rayleigh", rayleigh);
        p_f("atmo_rayleigh_height_m", rayleigh_height_m);
        p_f("atmo_mie", mie);
        p_f("atmo_mie_absorption", mie_absorption);
        p_f("atmo_mie_height_m", mie_height_m);
        p_f("atmo_mie_g", mie_g);
        p_f("atmo_ozone", ozone);
        p_f("atmo_ozone_centre_m", ozone_centre_m);
        p_f("atmo_ozone_width_m", ozone_width_m);
        p_f("atmo_top_m", top_m);
        p_f("atmo_ground_albedo", ground_albedo);
    }
};

class Atmosphere {
public:
    static constexpr int VOLUME_W = 128; // directions round the up
    static constexpr int VOLUME_H = 64; // heights, the horizon in the middle
    static constexpr int SLICES = 32; // the last: the whole ray
    static constexpr int MS_SIZE = 32; // the multiple-scattering table: sun height x altitude
    static constexpr double AERIAL_RANGE_M = 128000.0; // the slices' reach past the camera's horizon

    // What changes per frame, as the passes' uniforms: the camera (render metres), the sun (unit, towards it), the
    // slices' range, the longest path through the atmosphere (a ray along a plane is cut there) and the radius of the
    // sphere the multiple scattering is built on.
    struct Frame {
        Float3 camera_m;
        float range_m = 0.0f;
        Float3 sun;
        float max_path_m = 0.0f;
        float ms_radius_m = 0.0f;

        template <class F>
        void visit(F &&p_f) const {
            p_f("atmo_camera_m", camera_m);
            p_f("atmo_range_m", range_m);
            p_f("atmo_sun", sun);
            p_f("atmo_max_path_m", max_path_m);
            p_f("atmo_ms_radius_m", ms_radius_m);
        }
    };

    // Called on the device's thread with the camera volume and the multiple-scattering table once made, with invalid
    // textures before they are freed: the host binds them to its shaders.
    using TexturesFn = std::function<void(ov::gpu::Texture p_volume, ov::gpu::Texture p_multiscatter)>;

    // p_base's curvature radius and the ground's relief set the medium's scale (AtmosphereMedium::for_base, p_scale), the
    // curvature the multiple scattering's sphere.
    void start(ov::gpu::Device *p_device, const BaseSurface &p_base, double p_voxel_m, double p_relief_m, double p_scale, TexturesFn p_textures);
    // This frame's passes, the camera at p_camera_m with the sun towards p_sun (unit) over p_base (World::base_render).
    void frame(const Float3 &p_camera_m, const Float3 &p_sun, const BaseSurface::Render &p_base);
    // Frees its textures and pipelines and deletes itself on the device's thread (after the work posted before).
    void destroy();

    const AtmosphereMedium &get_medium() const { return medium; }
    // The last frame's slice range (the hosts' lookup reads it as atmo_range_m).
    float range_m() const { return last.range_m; }
    // The Frame of a camera at p_camera_m over p_base (frame() uploads it): its range and longest path from the camera's
    // altitude and the medium's top over a sphere of p_ms_radius_m (ms_radius_m).
    static Frame make_frame(const AtmosphereMedium &p_medium, double p_ms_radius_m, const Float3 &p_camera_m, const Float3 &p_sun,
            const BaseSurface::Render &p_base);
    // The sphere the multiple scattering and the ranges are taken on: the base's curvature radius, Earth's on the flat world.
    static double ms_radius_m(const BaseSurface &p_base, double p_voxel_m);
    double get_ms_radius_m() const { return curvature_m; }
    // The camera volume (device thread; AtmosphereCheck reads the running object's).
    ov::gpu::Texture get_volume() const { return volume; }

    // The medium's uniforms (AtmosphereMedium::visit), for a host that declares loose uniforms (Godot).
    static ov::str glsl_medium_uniforms();
    // The medium and the march (atmo_medium, the phases, atmo_segment, atmo_transmittance, atmo_integrate), after
    // BaseSurface::glsl_render, ShadeFrame's up_frame and the medium's uniforms. atmo_integrate reads the multiple
    // scattering through `vec3 atmo_ms(float altitude, float sun_height)`, which the includer defines.
    static ov::str glsl_march();
    // The host's lookups (after glsl_march): `vec4 atmo_at(vec3 camera, vec3 dir, float dist)` (in-scattering, mean
    // transmittance from the camera to dist along dir), `vec4 atmo_sky(vec3 camera, vec3 dir)` (the whole ray) and
    // atmo_ms from the table. The host declares `sampler2D atmo_volume`, `sampler2D atmo_multiscatter` and
    // `float atmo_range_m`.
    static ov::str glsl_lookup();
    // The two passes' compute shaders (the check compiles them too).
    static ov::str glsl_pass_multiscatter();
    static ov::str glsl_pass_volume();
    // A Frame's and the base's and the medium's uniform block, as the passes take it (binding 0).
    static ov::bytes block_bytes(const BaseSurface::Render &p_base, const AtmosphereMedium &p_medium, const Frame &p_frame);

private:
    void _rt_start();
    void _rt_frame(const ov::bytes &p_block, bool p_multiscatter);
    void _rt_destroy();

    ov::gpu::Device *device = nullptr;
    TexturesFn textures_fn;
    AtmosphereMedium medium;
    double curvature_m = 0.0; // the multiple scattering's sphere (finite)
    Frame last;
    bool multiscatter_dirty = true;

    // Device-thread objects.
    ov::gpu::Pipeline pipe_multiscatter;
    ov::gpu::Pipeline pipe_volume;
    ov::gpu::Buffer block;
    ov::gpu::Texture volume;
    ov::gpu::Texture multiscatter;
    ov::gpu::BindSet set_multiscatter;
    ov::gpu::BindSet set_volume;
};
