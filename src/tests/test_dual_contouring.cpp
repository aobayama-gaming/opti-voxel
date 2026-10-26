#include "test_common.hpp"

#include "godot_conv.hpp"

#include "gpu_direct_glsl.hpp"
#include "gpu_glsl.hpp"

#include <godot_cpp/variant/string.hpp>

// Both vertex passes sum the crossings into local_vertex and turn the sum into the mass point on the
// line GLSL_MASS_POINT_VERTEX, which dual contouring rewrites into GLSL_QEF_VERTEX_CALL. The division
// has to survive the rewrite: qef_vertex's prior and its move cap are relative to the mass point, and
// handed the sum (up to 12 cells from the cell) it clamped the vertex onto the cell's far faces.
TEST_CASE("DualContouring: the vertex pass rewrite still divides the crossings by their count") {
    const String pass(GLSL_DIRECT_PASS_VERTICES);
    // A pass that no longer spells the lines would mesh Surface Nets (dual_contouring_pass errors).
    CHECK(pass.count(GLSL_MASS_POINT_VERTEX) == 1);
    CHECK(pass.count(GLSL_SURFACE_NORMAL) == 1);
    const String dc = godot_str(dual_contouring_pass(ov_str(pass)));
    CHECK(dc.contains("qef_vertex("));
    CHECK((dc.contains("/ float(active_edge_count)") || dc.contains("/= float(active_edge_count)")));
    // The normal at the placed vertex falls back to the Hermite normals instead of the mass point.
    CHECK(dc.contains("vertex_normal(qef_fallback_normal, cascade_vertex_gradient(gl_GlobalInvocationID, local_vertex))"));
}

// The Hermite normals are the lerp of the corner gradients the sample pass already stored: dual
// contouring costs no SDF evaluation over Surface Nets, and a cascade (lattice points only) needs no
// variant of it.
TEST_CASE("DualContouring: the QEF evaluates no SDF") {
    const String qef(GLSL_QEF_VERTEX);
    CHECK_FALSE(qef.contains("cascade_terrain_sdf("));
    CHECK_FALSE(qef.contains("cascade_sample_sdf("));
}

// The gradient at a vertex vanishes exactly on a medial axis (a capsule thinner than the cell, on a lattice
// line: the scatter's pillars put ~0.3% of their vertices there at LODs 1-4), and so do the nearby corner
// gradients, by symmetry. normalize() of it is NaN and the vertex was dropped: broken quads.
TEST_CASE("Vertex normal: the passes never normalize a raw gradient, a vanishing one points up") {
    const String pass(GLSL_DIRECT_PASS_VERTICES);
    CHECK_FALSE(pass.contains("normalize(cascade_vertex_gradient("));
    CHECK(pass.contains("vertex_normal(vec3(0.0), cascade_vertex_gradient(gl_GlobalInvocationID, local_vertex))"));
    CHECK(String(GLSL_VERTEX_NORMAL).contains(": vec3(0.0, 1.0, 0.0)"));
}
