#pragma once

#include "exact_noise.hpp"
#include "world_coord.hpp"

#include "ov/format.hpp"

// The planet base (design note planet-base-curvature-2026-10-22): a reference surface, not a constraint on the
// terrain. At every point it gives an altitude d_base (positive above, in voxels of the chunk's LOD), an up
// direction n (d_base's gradient, unit or nearly) and a sea level (d_base = 0). The terrain on top stays 3D: the
// ground reads (n, d_base) (CascadeTerrain::height_sdf), as a Float4 (n.x, n.y, n.z, d_base) like the cascade's
// registers.
//
// FLAT: the plane y = datum, up +y. The only place that knows about y. Its altitude is the int64 difference
// saturated, then converted (the flat world's bits before the base existed, pinned by a unit test).
//
// SPHERE: |p - centre| - radius in two forms, picked by the LOD (switch_lod, the same on CPU and GPU):
// - below it, the curvature form d(c) + n.δ + 1/2 δ^T H δ around the centre c of the point's region (the cell of
//   2^(L + REGION_SHIFT) voxels holding it, chosen from the point and the LOD, never from the chunk, so two chunks
//   of one LOD give a shared sample the same bits). d(c), n and H come from the CPU in double, converted to float32
//   (Region); δ is a small integer vector, exact in float32. The neglected terms grow with the region, not the planet.
// - from it up, the radius form |p - centre| - radius in float32 (det_rsqrt), whose error (a few float steps of
//   the radius) is far below 1% of a cell there.
// Any smooth base the CPU can evaluate in double with its gradient and Hessian fits the curvature form
// (make_region, from shape()). On the GPU the regions an item touches are uploaded with its batch (base_regions.hpp,
// whose base_at picks the form like at()).
//
// TORUS: a ring round the y axis through the centre, |(|v.xz| - ring, v.y)| - tube with v = p - centre: the sphere's
// two forms with the torus's own shape (the radius form's lengths through det_rsqrt too).
//
// Every float op here is op for op the GLSL of glsl() (FrameSpike's base probes check it bit for bit).
class BaseSurface {
public:
    // The flat world's datum, metres of absolute y: far origins only show terrain at y = 0.
    static constexpr double DATUM_M = 300.0;
    // Saturation of the flat altitude (LOD0 voxels). A saturated |SDF| only makes chunk discovery keep more chunks,
    // never drop one, but it must stay several cells of the largest LOD.
    static constexpr int32_t ALT_LIMIT = 1 << 30;
    // A region is the cell of 2^(L + REGION_SHIFT) voxels holding the point: δ in [-32, 32) voxels of LOD L.
    static constexpr int REGION_SHIFT = 6;
    // The radius form's |p - centre| per axis in voxels of LOD L, saturated here: exact in float32.
    static constexpr int32_t RADIUS_LIMIT = 1 << 24;
    // The default switch to the radius form: Earth's radius at 5 cm voxels, with a wide margin both ways (the
    // curvature form holds to LOD 17, the radius form from LOD 10, design note).
    static constexpr int SWITCH_LOD = 12;
    // The curvature form reads δ from the point's low word: L + REGION_SHIFT <= 31.
    static constexpr int MAX_SWITCH_LOD = 32 - REGION_SHIFT;
    // A sphere's centre is a multiple of 2^CENTRE_SHIFT voxels, the largest LOD's cell: (p - centre) >> L is then exact
    // on every lattice (an unsnapped centre floors up to a cell). sphere() snaps it down like the world origin, which
    // moves it by up to 2^31 voxels per axis (107,000 km at 5 cm): place planets on that grid.
    static constexpr int CENTRE_SHIFT = 31;
    // The radius form's largest radius in cells of the switch LOD: its float32 |v| is then within ~2^-24 * 2^17 = 0.008
    // cells (Earth at 5 cm, switch 12: 31,000 cells).
    static constexpr int32_t MAX_RADIUS_CELLS = 1 << 17;

    enum class Kind {
        FLAT,
        SPHERE,
        TORUS
    };

    // The curvature form's data of one region at one LOD, in voxels of that LOD: d at its centre, n the gradient
    // there, H the Hessian (h[0..2] = xx, yy, zz; h[3..5] = xy, xz, yz).
    struct Region {
        float d = 0.0f;
        float n[3] = {};
        float h[6] = {};
    };

    static BaseSurface flat(int64_t p_datum_y);
    // p_centre is snapped down to a multiple of 2^CENTRE_SHIFT voxels. Refused (ERR, the default flat base): a switch
    // LOD outside [0, MAX_SWITCH_LOD], a radius outside (0, 2^60], or over MAX_RADIUS_CELLS cells of the switch LOD.
    static BaseSurface sphere(const VoxelPos &p_centre, double p_radius, int p_switch_lod = SWITCH_LOD);
    // A torus's smallest ring in tubes: a hole of half a tube, far wider than the relief and the air over it.
    static constexpr double MIN_RING_TUBES = 1.5;
    // A ring of p_ring voxels round the y axis, its tube p_tube: sphere()'s limits on the outer radius p_ring + p_tube, and
    // p_ring >= MIN_RING_TUBES * p_tube (ERR, the default flat base).
    static BaseSurface torus(const VoxelPos &p_centre, double p_ring, double p_tube, int p_switch_lod = SWITCH_LOD);
    // SPHERE, TORUS: the (snapped) centre.
    const VoxelPos &centre() const { return anchor; }
    // SPHERE, TORUS: the first LOD of the radius form (below it, the curvature form's regions).
    int get_switch_lod() const { return switch_lod; }
    // SPHERE: the radius; TORUS: the tube's (LOD0 voxels).
    double get_radius() const { return radius; }
    // TORUS: the ring's radius (LOD0 voxels).
    double get_ring() const { return ring; }

    Kind kind() const { return base_kind; }
    // Not FLAT: a centre, regions below the switch LOD, a relief from the radius (SDFCascade).
    bool curved() const { return base_kind != Kind::FLAT; }
    // FLAT: the datum's height (LOD0 voxels). ERR and 0 on another base.
    int64_t datum_y() const;

    // (n, d_base) at LOD0 voxel p_voxel, in voxels of LOD p_lod. Only the voxel counts (the cascade's lattice
    // points), and a SPHERE below the switch computes its region on the CPU.
    ExactNoise::Float4 at(const VoxelPos &p_voxel, int p_lod) const;
    // The unit up direction at a LOD0 voxel, in double (placement: SDFBase::up_at).
    UpAxis up_dir(const VoxelPos &p_voxel) const;
    // The altitude of a LOD0 voxel above the base, LOD0 voxels in double (reports, placement; not the SDF's float path).
    double altitude(const VoxelPos &p_voxel) const;
    // The point (p_u, p_v) LOD0 voxels along x and z of the base's reference tangent plane: FLAT the datum plane through
    // the world's (0, 0) column, SPHERE the plane touching the sphere at its +y pole. Reports sample it and trace along -up
    // (SDFCascade::ground_along); the player's --spawn takes these coordinates.
    VoxelPos tangent_point(int64_t p_u, int64_t p_v) const;
    // The base's point under (or over) a LOD0 voxel along its up, rounded to the voxel: FLAT the datum in its column,
    // SPHERE the radial projection (of the centre itself: the +y pole). Where biome sites stand (Biome::sites).
    VoxelPos on_base(const VoxelPos &p_voxel) const;

    // The region holding a point at a LOD: its cell id (p >> (L + REGION_SHIFT)).
    static Vector3i64 region_cell(const VoxelPos &p_voxel, int p_lod);
    // A SPHERE's region data at a cell of LOD p_lod, from double.
    Region region(const Vector3i64 &p_cell, int p_lod) const;
    // The Region of any base from its exact value at the region centre, LOD0 voxels: p_d, gradient p_g, Hessian p_h
    // (Region::h's order).
    static Region make_region(double p_d, const double p_g[3], const double p_h[6], int p_lod);
    // The centre of a region cell (LOD0 voxels).
    static VoxelPos region_centre(const Vector3i64 &p_cell, int p_lod);

    // The sphere's two forms (GLSL base_curvature / base_radius).
    static ExactNoise::Float4 curvature_form(const VoxelPos &p_voxel, int p_lod, const Region &p_region);
    ExactNoise::Float4 radius_form(const VoxelPos &p_voxel, int p_lod) const;

    // A stretched octave (the climate's "stretch_y", Band::stretch) reads its noise at p' = p - (1 - 2^-j) d_base(p) n(p):
    // its wavelength along the base's up 2^j times the tangential ones, whatever the base. FLAT: datum + (y - datum) / 2^j,
    // exact (the floor to the voxel, the remainder plus the fraction times 2^-j to the fraction, carried at 1). SPHERE: the
    // float offset of the base sample p_b (stretch_base) split into whole voxels and a fraction, each axis clamped to
    // STRETCH_LIMIT voxels. GLSL: base_stretched (FLAT), base_stretched_at (SPHERE).
    static constexpr float STRETCH_LIMIT = 1073741824.0f; // 2^30 voxels: whole floats, an int32 offset
    // The base sample a stretch reads at LOD0 voxel p_voxel, LOD p_m (the band's insertion lattice): at(); nothing on FLAT.
    ExactNoise::Float4 stretch_base(const VoxelPos &p_voxel, int p_m) const;
    WorldPos stretched(const WorldPos &p_pos, int p_j, const ExactNoise::Float4 &p_b, int p_m) const;
    // The noise's gradient back through the stretch: its up component times 2^-j. FLAT: g.y * 2^-j exactly; SPHERE:
    // g - (1 - 2^-j) (g.n) n, n = p_b's up. GLSL: base_stretch_gradient (FLAT), base_stretch_gradient_at (SPHERE).
    ExactNoise::Float4 stretch_gradient(const ExactNoise::Float4 &p_g, int p_j, const ExactNoise::Float4 &p_b) const;
    // The stretch's GLSL (the cascade's band functions splice it for a stretched spec; after the int64 helpers).
    ov::str glsl_stretch() const;

    // FLAT: `vec4 base_at(WorldPos p, int lod)`. SPHERE: `struct BaseRegion`, `vec4 base_curvature(WorldPos p,
    // int lod, BaseRegion r)` and `vec4 base_radius(WorldPos p, int lod)`; its base_at reads the batch's regions
    // (BaseRegions::glsl). After the frame functions, the int64 helpers (SPHERE) and GLSL_EXACT_NOISE (det_rsqrt).
    ov::str glsl() const;

    // The base in render space (metres from the render origin), what the hosts' shaders know of it: every render-space
    // user (the pages' heights and up, the sky's frame, the cloud layer) goes through glsl_render's
    // functions, so a new kind only adds its case there and here. From double: SPHERE's altitude
    // (|p|^2 - 2 p.c + K) / (|p - c| + R) keeps ~1e-4 m in float32 at Earth's radius, where |p - c| - R keeps half a metre.
    // base_level_span on a TORUS (sphere-traced, base_trace_cross): the smallest step as a fraction of the bounding
    // radius, the steps per crossing, the over-relaxation (a grazing ray's distance shrinks slowly; the level's solid
    // is thick, so a 1.5x step only skips a grazing chord) and the halvings refining each crossing.
    static constexpr float BASE_TRACE_EPS = 1e-5f;
    static constexpr int BASE_TRACE_STEPS = 128;
    static constexpr float BASE_TRACE_RELAX = 1.5f;
    static constexpr int BASE_TRACE_BISECT = 10;
    struct Render {
        Kind kind = Kind::FLAT;
        Float3 point_m; // FLAT: a point of the datum plane; SPHERE, TORUS: the centre
        float radius_m = 0.0f; // SPHERE; TORUS: the tube's
        float k_m2 = 0.0f; // SPHERE: |centre|^2 - radius^2 (m^2)
        float ring_m = 0.0f; // TORUS

        // Each field as the shaders' uniform (glsl_render_uniforms): p_f(name, value), value an int, a float or a Float3.
        // The one list of the uniforms' names and order: a host fills its shader parameters from it.
        template <class F>
        void visit(F &&p_f) const {
            p_f("base_kind", int(kind));
            p_f("base_point_m", point_m);
            p_f("base_radius_m", radius_m);
            p_f("base_k_m2", k_m2);
            p_f("base_ring_m", ring_m);
        }
        // base_altitude's formula (glsl_render) in float32 on the CPU: its precision (tests), not the GPU's bits.
        float altitude(const Float3 &p_m) const;
    };
    // The render-space base with the render origin at LOD0 voxel p_origin (it changes with the origin).
    Render render(const VoxelPos &p_origin, double p_voxel_m) const;
    // The base's radius of curvature in metres (its smallest; FLAT: infinite): what local approximations of its shape
    // (the atmosphere's multiple scattering, its scale) take.
    double curvature_radius_m(double p_voxel_m) const;
    // Render's fields as `uniform` declarations (Render::visit), for a host whose shaders splice glsl_render.
    static ov::str glsl_render_uniforms();
    // `float base_altitude(vec3 p)` (metres above the base, signed), `vec3 base_up(vec3 p)` (its unit gradient) and
    // `bool base_level_span(vec3 p, vec3 dir, float h, out vec2 t)` (where a ray is below altitude h), plain GLSL that
    // Godot's shading language takes too; after glsl_render_uniforms (or the host's own declarations of Render's fields).
    static ov::str glsl_render();

private:
    // A curved base's altitude at v = p - centre (LOD0 voxels, double), its gradient r_n and, unless null, its Hessian r_h
    // (Region::h's order): what region() and the double helpers read (never on FLAT).
    double shape(const double p_v[3], double r_n[3], double r_h[6] = nullptr) const;
    // shape() at a LOD0 voxel.
    double shape_at(const VoxelPos &p_voxel, double r_n[3], double r_h[6] = nullptr) const;
    // A FLAT base's at() (GLSL base_at).
    ExactNoise::Float4 flat_form(const VoxelPos &p_voxel, int p_lod) const;

    Kind base_kind = Kind::FLAT;
    VoxelPos anchor; // FLAT: the datum (only y counts); SPHERE, TORUS: the centre
    double radius = 0.0; // SPHERE; TORUS: the tube's (LOD0 voxels)
    double ring = 0.0; // TORUS, LOD0 voxels
    int switch_lod = SWITCH_LOD;
};
