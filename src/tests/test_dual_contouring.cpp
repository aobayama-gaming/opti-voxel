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
    // A pass that no longer spells the line would silently mesh with Surface Nets.
    CHECK(pass.count(GLSL_MASS_POINT_VERTEX) == 1);
    const String dc = pass.replace(GLSL_MASS_POINT_VERTEX, GLSL_QEF_VERTEX_CALL);
    CHECK(dc.contains("qef_vertex("));
    CHECK((dc.contains("/ float(active_edge_count)") || dc.contains("/= float(active_edge_count)")));
}
