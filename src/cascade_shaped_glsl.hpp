#pragma once

#include "cascade.hpp"

// The GLSL of shaped bands (Cascade::BandMod, cascade.cpp's modulated_channels and warped), op for op; spliced by
// CascadeGLSL's level shader (cascade_glsl.cpp).
namespace CascadeShapedGLSL {

// Whether p_spec's modulation shapes any band (its GLSL is then cascade_mods and the shaped inserts).
bool shaped(const Cascade::Spec &p_spec);
// `vec4 g = <noise>(<p_pos>, band...)`, through base_stretched and base_stretch_gradient (the base's up) when the group
// stretches a band (Cascade::band_noise).
ov::str noise_code(const Cascade::Spec &p_spec, int p_group, const ov::str &p_pos);
// The spec's base stretch in GLSL (BaseSurface::glsl_stretch; band_functions splices it when Spec::stretched()).
ov::str stretch_helper(const Cascade::Spec &p_spec);
// The GLSL BandMod, cl_smooth_abs and cascade_warped: after the climate's shared helpers (cl_flush, cl_mul).
ov::str shaped_helpers();
// `Ch insert_band_shaped_<g>(Ch ch, WorldPos p, CascadeBand band, BandMod md, int m)` for group p_group.
ov::str shaped_band_function(const Cascade::Spec &p_spec, int p_group);

} // namespace CascadeShapedGLSL
