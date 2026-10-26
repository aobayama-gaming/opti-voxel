#pragma once

#include "ov/value.hpp"

// Report of the Hermite cascade (cascade.hpp), pure CPU: fidelity of a cascaded octave vs analytic perlin3 for 4 / 8
// samples per wavelength and 8 / 4 channels (cascade_measure.hpp), points per chunk per LOD on an Earth (top lattice
// Λ_26) with the CPU time per chunk, and PNG slices (analytic octave | cascaded | error, value and shaded relief) for
// the look decision. Writes <out_dir>/report.md and <out_dir>/slice_*.png. Godot: CascadeReportNode
// (tools/cascade_report.sh).
namespace CascadeReport {

// p_scale: sample size multiplier (1 = the default report, ~1 minute).
ov::Value run(const ov::str &p_out_dir, int p_scale);

} // namespace CascadeReport
