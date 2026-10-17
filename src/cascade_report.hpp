#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/dictionary.hpp>

using namespace godot;

// Report of the Hermite cascade (cascade.hpp), pure CPU (works with --headless): fidelity of a cascaded
// octave vs analytic perlin3 for 4 / 8 samples per wavelength and 8 / 4 channels (cascade_measure.hpp),
// points per chunk per LOD on an Earth (top lattice Λ_26) with the CPU time per chunk, and PNG slices
// (analytic octave | cascaded | error, value and shaded relief) for the look decision.
// Writes <out_dir>/report.md and <out_dir>/slice_*.png. Run: tools/cascade_report.sh
class CascadeReport : public Node {
    GDCLASS(CascadeReport, Node)

protected:
    static void _bind_methods();

public:
    // p_scale: sample size multiplier (1 = the default report, ~1 minute).
    Dictionary run(const String &p_out_dir, int p_scale);
};
