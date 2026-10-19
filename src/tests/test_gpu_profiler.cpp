#include "test_common.hpp"

#include "gpu_profiler.hpp"

namespace {

std::vector<std::string> labels(const std::vector<const char *> &p_names) {
    std::vector<String> names;
    for (const char *n : p_names) {
        names.push_back(n);
    }
    std::vector<std::string> out;
    for (const String &l : gpu_interval_labels(names)) {
        out.push_back(l.utf8().get_data());
    }
    return out;
}

} // namespace

TEST_CASE("gpu_interval_labels: renderer names start a section") {
    CHECK(labels({ "Render Shadows", "Pre Opaque Render", "Render Opaque Pass", "Tonemap" }) ==
            std::vector<std::string>{ "Render Shadows", "Pre Opaque Render", "Render Opaque Pass" });
    CHECK(labels({ "Frame Begin" }).empty());
}

TEST_CASE("gpu_interval_labels: mesher names end a step, _begin starts the batch") {
    // What runs before a batch's _begin belongs to the renderer section before it; _p0 ends the sdf pass.
    CHECK(labels({ "Frame Begin", "vx12_begin", "vx12_c0", "vx12_p0", "vx12_p1", "vx12_copy", "Prepare Render Frame", "Cull" }) ==
            std::vector<std::string>{ "Frame Begin", "vx_c0", "vx_p0", "vx_p1", "vx_copy", "(untracked)", "Prepare Render Frame" });
    // Two batches in a row: the gap between them is nobody's.
    CHECK(labels({ "vx1_begin", "vx1_copy", "vx2_begin", "vx2_copy" }) ==
            std::vector<std::string>{ "vx_copy", "(untracked)", "vx_copy" });
}
