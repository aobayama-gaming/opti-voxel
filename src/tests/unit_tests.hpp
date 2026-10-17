#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/dictionary.hpp>

using namespace godot;

// Runs the doctest unit tests of src/tests/ inside the engine, so tests can use Godot types
// (String, HashSet, Vector3i...) freely. Only compiled into non-release builds (OPTI_VOXEL_TESTS).
// Pure CPU, works with --headless. Run: tools/unit_test.sh [filter]
class UnitTests : public Node {
    GDCLASS(UnitTests, Node)

protected:
    static void _bind_methods();

public:
    // p_filter: doctest wildcard on test case names ("" = all). Returns
    // { ok, test_cases, test_cases_failed, asserts, asserts_failed }; doctest prints the details.
    Dictionary run(const String &p_filter);
};
