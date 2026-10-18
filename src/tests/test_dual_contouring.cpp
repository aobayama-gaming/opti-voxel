#include "test_common.hpp"

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
    const String dc = dual_contouring_pass(pass);
    CHECK(dc.contains("qef_vertex("));
    CHECK((dc.contains("/ float(active_edge_count)") || dc.contains("/= float(active_edge_count)")));
    // The normal at the placed vertex falls back to the Hermite normals instead of the mass point.
    CHECK(dc.contains("qef_normalize(qef_fallback_normal, skirt_sdf(chunk, chunk_vertex, chunk_param).xyz)"));
}

// The Hermite normals are the lerp of the corner gradients the sample pass already stored: dual
// contouring costs no SDF evaluation over Surface Nets, and a cascade (lattice points only) needs no
// variant of it.
TEST_CASE("DualContouring: the QEF evaluates no SDF") {
    const String qef(GLSL_QEF_VERTEX);
    CHECK_FALSE(qef.contains("skirt_sdf("));
    CHECK_FALSE(qef.contains("evaluate_sdf("));
    CHECK_FALSE(qef.contains("terrain_sdf("));
}
