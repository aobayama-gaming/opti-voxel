#pragma once

#include "atmosphere.hpp"
#include "base_surface.hpp"
#include "gpu_device.hpp"
#include "weather.hpp"

#include "ov/bytes.hpp"
#include "ov/str.hpp"

#include <functional>

// Volumetric clouds (docs/look.md), made from the base's shape like the atmosphere: a layer between two altitudes above
// the base (BaseSurface::glsl_render's base_altitude; its extent along a ray from base_level_span), so a base of any
// kind gets a cloud layer that follows it.
//
// The density is Schneider's (Horizon Zero Dawn's Nubis, SIGGRAPH 2015 / 2017; ported from clayjohn's
// godot-volumetric-cloud-demo, MIT): a 128^3 Perlin-Worley shape noise and a 32^3 Worley detail noise (3D textures made
// once by a compute pass), shaped by height gradients per cloud type, the cover and the type read from the shape noise at
// a weather scale. Every noise is read at the 3D point (no 2D weather map, which a curved base would tear), in the
// shading frame (shade_frame_glsl.hpp), so the clouds are the same from every render origin. The light is Hillaire's
// (Frostbite, "Physically Based Sky, Atmosphere and Cloud Rendering", SIGGRAPH 2016): the energy-conserving step, a
// dual-lobe Henyey-Greenstein phase, Wrenninge's octaves for the multiple scattering, the sun through the atmosphere
// (Atmosphere's transmittance march) and through the cloud (a march towards it), the sky's light from the atmosphere's
// multiple-scattering table (Hillaire 2020's isotropic sky radiance at an altitude and sun height).
//
// Each frame a compute pass marches a camera-centred grid of directions round the base's up at the camera (GRID_W
// azimuths x GRID_H elevations, the elevation linear), one invocation per 2 x 2 block marching one of its texels in turn
// and reprojecting the others from the last frame by the cloud's distance. A texel holds the clouds' light towards the
// camera (RGB, for a sun of illuminance 1, the air in front of them included) and their transmittance (A), and in a
// second texture the cloud's front and mean distances times its opacity (kilometres: a half float keeps 1e-3 of any of
// them; times the opacity, filtering weighs them by it). Hosts read it by direction and distance (glsl_lookup's
// cloud_at), like the atmosphere's camera volume. A second pass marches a shadow map: the sun's transmittance through the
// layer along lines parallel to the sun round the camera (SHADOW_SIZE^2 texels, its side the layer's shadow_min_extent_m
// or 4 camera altitudes, its origin snapped to them), each from where its line leaves the base towards the sun, so it
// holds for any receiver under the layer; the hosts' terrain reads it once per fragment (cloud_sun_shadow).
//
// The clouds are the weather's (Weather, docs/weather.md): the layer runs from the ground to its levels' top, and each
// level's cloud of its last two states is uploaded when a tick lands as a 3D texture, the grid's atlas a slice per level
// (R the previous state's, G the latest's), blended by where the frame stands between them, read by the grid's lookup
// and the altitude (linear between the levels); the noises shape it and drift with the wind at the camera. Planets
// only: no weather, no clouds.
struct CloudLayer {
    // Earth's: a cumulus layer from 1.5 to 4 km, the spread noise repeating every ~12 km, the weather every ~48 km, the
    // wind's offset wrapping every ~800 km; of_weather scales it.
    float bottom_m = 1500.0f;
    float top_m = 4000.0f;
    float extinction = 0.04f; // 1/m at density 1
    float albedo = 0.99f;
    float shadow_reach_m = 80000.0f; // the longest path of a shadow ray through the layer (a low sun over a plane)
    float shadow_min_extent_m = 64000.0f; // the shadow map's side round the camera, at least (Frame::shadow_extent_m; no uniform)
    // The noises' periods in LOD0 voxels (powers of two: the shading frame), set by of_weather.
    int spread_period_vox = 0;
    int weather_period_vox = 0;
    int warp_period_vox = 0;

    static constexpr double EARTH_SPREAD_PERIOD_M = 12000.0;
    static constexpr double EARTH_WEATHER_PERIOD_M = 48000.0;
    static constexpr double EARTH_WARP_PERIOD_M = 800000.0; // 2^23 voxels at 5 cm: the frame's longest period
    // The periods in the weather's cells (of_weather): the spread noise's (Clouds::SPREAD_*: its features a cell to
    // a quarter, where the simulation has its spread), the weather's (a multiple of it).
    static constexpr double WEATHER_CELLS = 16.0;
    static constexpr double SPREAD_CELLS = 4.0;

    // The weather's layer (Weather: its levels' top p_top_m over the ground, its cells about p_cell_m wide) at LOD0
    // voxels of p_voxel_m: from the ground to the top, Earth's optical depths at that thickness (the extinction over s =
    // top / 2.5 km, the shadow's reach times s), the periods the cells' (WEATHER_CELLS, SPREAD_CELLS): the clouds' water
    // and share are the simulation's, the noise only places them inside a cell. The march knows nothing
    // of the terrain: a hill reaching into a cloud is hidden behind the cloud's part in front of it.
    static CloudLayer of_weather(double p_top_m, double p_cell_m, double p_voxel_m);

    // The fields as the shaders' uniforms (GLSLUniforms).
    template <class F>
    void visit(F &&p_f) const {
        p_f("cloud_bottom_m", bottom_m);
        p_f("cloud_top_m", top_m);
        p_f("cloud_extinction", extinction);
        p_f("cloud_albedo", albedo);
        p_f("cloud_shadow_reach_m", shadow_reach_m);
        p_f("cloud_spread_period_vox", spread_period_vox);
        p_f("cloud_weather_period_vox", weather_period_vox);
        p_f("cloud_warp_period_vox", warp_period_vox);
    }
};

class Clouds {
public:
    static constexpr int SPREAD_SIZE = 128; // the spread noise, texels per side
    // Its coarsest lattice cells per side: one per weather cell, so the octaves are a cell, a half and a quarter (half a
    // cell to an eighth looked like cotton, 2026-11-26).
    static constexpr int SPREAD_LATTICE = int(CloudLayer::SPREAD_CELLS);
    static constexpr int SPREAD_OCTAVES = 3;
    static constexpr int GRID_W = 1536; // the direction grid: azimuths
    static constexpr int GRID_H = 768; // elevations, from straight down to straight up
    static constexpr int MAX_STEPS = 64; // a view ray's samples (fewer on a short one)
    static constexpr int MIN_STEPS = 16;
    static constexpr int STEPS_PER_THICKNESS = 32; // a view step: the layer's thickness over this, at most
    static constexpr int LIGHT_STEPS = 6; // towards the sun, growing
    static constexpr int SHADOW_SIZE = 512; // the shadow map's texels per side
    static constexpr int SHADOW_STEPS = 16; // a shadow map column's samples through the layer
    // Wrenninge's multiple-scattering octaves (Frostbite's a, b, c): octave o scatters MS_SCATTERING^o of the light,
    // reaches it through MS_EXTINCTION^o of the extinction, with a phase MS_ECCENTRICITY^o as eccentric. a > b is
    // Frostbite's "exaggerated" multiple scattering (not energy-conserving): with a <= b a cloud's inside, hundreds of
    // optical depths from the sun, stays black (the octaves miss the diffusion that whitens real clouds).
    static constexpr int OCTAVES = 3;
    static constexpr float MS_SCATTERING = 0.75f;
    static constexpr float MS_EXTINCTION = 0.12f;
    static constexpr float MS_ECCENTRICITY = 0.5f;
    // A marched texel's weight against its reprojected history.
    static constexpr float BLEND = 0.5f;

    // What changes per frame, as the pass's uniforms: the camera and the last frame's (render metres, the last moved
    // with the render origin), the sun (unit, towards it), the wind's offset of the noise (metres, wrapped to the warp
    // period), the march's reach (the horizon from the camera plus the layer top's, on the curvature sphere), the
    // weather texture's cells a side and blend, the frame's number (which texel of each block it marches), reset (no history: every texel marched),
    // the render origin's LOD0 voxel (low words: the shading frame) and the voxel size.
    struct Frame {
        Float3 camera_m;
        Float3 prev_camera_m;
        Float3 sun;
        Float3 wind_offset_m;
        // The shadow map: its corner (render metres), its axes (unit, perpendicular to the sun) and its side.
        Float3 shadow_origin_m;
        Float3 shadow_u;
        Float3 shadow_v;
        float shadow_extent_m = 0.0f;
        float reach_m = 0.0f;
        float cover_max = 0.0f; // the largest cover in the weather's two states (0: no cloud, the passes skip)
        float weather_blend = 0.0f; // between its previous state (0) and its latest (1)
        int frame = 0;
        int reset = 1;
        int origin_x = 0;
        int origin_y = 0;
        int origin_z = 0;
        float voxel_m = 0.5f;

        template <class F>
        void visit(F &&p_f) const {
            p_f("cloud_camera_m", camera_m);
            p_f("cloud_prev_camera_m", prev_camera_m);
            p_f("cloud_sun", sun);
            p_f("cloud_wind_offset_m", wind_offset_m);
            p_f("cloud_shadow_origin_m", shadow_origin_m);
            p_f("cloud_shadow_u", shadow_u);
            p_f("cloud_shadow_v", shadow_v);
            p_f("cloud_shadow_extent_m", shadow_extent_m);
            p_f("cloud_reach_m", reach_m);
            p_f("cloud_cover_max", cover_max);
            p_f("cloud_weather_blend", weather_blend);
            p_f("cloud_frame", frame);
            p_f("cloud_reset", reset);
            p_f("cloud_origin_x", origin_x);
            p_f("cloud_origin_y", origin_y);
            p_f("cloud_origin_z", origin_z);
            p_f("cloud_voxel_m", voxel_m);
        }
    };

    // Called on the device's thread with the grid's two textures and the shadow map once made (the hosts' samplers
    // HOST_TEXTURES, in that order; not when a pass failed to compile), with invalid textures before they are freed: the
    // host binds them to its shaders and turns its clouds on.
    using TexturesFn = std::function<void(ov::gpu::Texture p_light, ov::gpu::Texture p_depth, ov::gpu::Texture p_shadow)>;
    static constexpr const char *HOST_TEXTURES[3] = { "cloud_light", "cloud_depth", "cloud_shadow_map" };

    // What the hosts take per frame, as global uniforms: the shadow map's frame (Frame's fields of the same names).
    template <class F>
    static void visit_host_frame(const Frame &p_frame, F &&p_f) {
        p_f("cloud_shadow_origin_m", p_frame.shadow_origin_m);
        p_f("cloud_shadow_u", p_frame.shadow_u);
        p_f("cloud_shadow_v", p_frame.shadow_v);
        p_f("cloud_shadow_extent_m", p_frame.shadow_extent_m);
    }

    // p_atmosphere (started, outliving this) lights the clouds and hazes them in p_layer (CloudLayer::of_weather); the
    // weather's texture is p_weather's grid and levels.
    void start(ov::gpu::Device *p_device, const Atmosphere *p_atmosphere, const BaseSurface &p_base, double p_voxel_m, const CloudLayer &p_layer,
            const Weather &p_weather, TexturesFn p_textures);
    // Each frame before frame(): the weather (its blend; with p_landed, a new state: its last two uploaded) and the
    // wind the noises drift with (render metres per second).
    void set_weather(const Weather &p_weather, const Float3 &p_wind_mps, bool p_landed);
    // This frame's pass, the camera at p_camera_m with the sun towards p_sun over p_base (World::base_render), the render
    // origin at LOD0 voxel p_origin_voxel (after the atmosphere's frame: the pass reads its volume).
    void frame(const Float3 &p_camera_m, const Float3 &p_sun, const BaseSurface::Render &p_base, const VoxelPos &p_origin_voxel);
    // The render origin moved: render positions minus p_shift_m (World::rebase's shift).
    void shift(const Float3 &p_shift_m);
    // Frees its textures and pipelines and deletes itself on the device's thread (after the work posted before; before
    // the atmosphere's destroy).
    void destroy();

    const CloudLayer &layer() const { return layer_params; }
    // The last frame's (the hosts' globals: visit_host_frame).
    const Frame &last_frame() const { return frame_params; }

    // The host's lookups, after the atmosphere's (Atmosphere::glsl_lookup: ATMO_PI), the shading frame's up_frame and
    // BaseSurface::glsl_render: `vec4 cloud_at(vec3 camera, vec3 dir, float dist)` (the clouds' light, for a sun of
    // illuminance 1, and transmittance from the camera to dist along dir) and `float cloud_sun_shadow(vec3 p)` (the sun
    // through the layer at p, under it). The host declares the samplers HOST_TEXTURES and visit_host_frame's uniforms.
    static ov::str glsl_lookup();
    // The passes' compute shaders.
    // The passes over a weather grid (its lookup, WeatherGrid::glsl_lookup) of p_levels levels.
    static ov::str glsl_pass_march(const WeatherGrid &p_weather, int p_levels);
    static ov::str glsl_pass_shadow(const WeatherGrid &p_weather, int p_levels);
    // A camera at p_camera_m's Frame over p_base with the sun towards p_sun (frame() adds the history, the wind, the
    // weather and the origin): the march's reach over a sphere of p_curvature_m (Atmosphere::ms_radius_m) and the shadow map.
    static Frame make_frame(const CloudLayer &p_layer, double p_curvature_m, const Float3 &p_camera_m, const Float3 &p_sun,
            const BaseSurface::Render &p_base);
    // The pass's uniform block: the base's, the atmosphere's host constants and range, the layer's and the frame's.
    static ov::bytes block_bytes(const BaseSurface::Render &p_base, const Atmosphere::HostConstants &p_atmosphere, float p_atmo_range_m,
            const CloudLayer &p_layer, const Frame &p_frame);

private:
    // The weather's atlas (WeatherGrid's padded layout as a texture, a slice per level) and the passes' text over its grid.
    struct Atlas {
        uint32_t width;
        uint32_t height;
        uint32_t levels;
        ov::str march;
        ov::str shadow;
    };
    void _rt_start(const Atlas &p_atlas, const std::vector<uint16_t> &p_spread);
    void _rt_frame(const ov::bytes &p_block);
    void _rt_destroy();

    ov::gpu::Device *device = nullptr;
    const Atmosphere *atmosphere = nullptr;
    TexturesFn textures_fn;
    CloudLayer layer_params;
    Frame frame_params;
    double curvature_m = 0.0; // the reach's sphere (Atmosphere::ms_radius_m)
    Float3 wind_mps;
    float weather_blend = 0.0f;
    float cover_max = 0.0f;
    double wind_offset[3] = { 0.0, 0.0, 0.0 }; // metres, wrapped to the warp period
    uint64_t last_usec = 0;
    bool has_camera = false;

    // Device-thread objects.
    ov::gpu::Pipeline pipe_march;
    ov::gpu::Pipeline pipe_shadow;
    ov::gpu::Buffer block;
    ov::gpu::Texture spread_noise; // WeatherCloud::uniform_noise in R
    ov::gpu::Texture light;
    ov::gpu::Texture depth;
    ov::gpu::Texture history_light;
    ov::gpu::Texture history_depth;
    ov::gpu::Texture shadow;
    ov::gpu::Texture weather_map; // the weather grid's padded texels x FACES
    ov::gpu::BindSet set_march;
    ov::gpu::BindSet set_shadow;
};
