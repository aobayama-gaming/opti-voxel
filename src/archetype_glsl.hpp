#pragma once

#include "archetype_ir.hpp"
#include "ov/str.hpp"

namespace Archetype {

// GLSL "vec4 <p_name>(WorldPos p, int lod, DefRecord rec)" of a program (a Table Def, archetype_table.hpp):
// straight-line `precise` code, one statement per instruction in the interpreter's order (archetype_ir.cpp)
// and with its formulas, constants as raw bits, the T_* ops reading the record `rec` (its anchor, rotation,
// parameters, salt). Needs GLSL_DEF_RECORD_STRUCT, the frame functions (gpu_frame.hpp: sat_diff), the int64
// helpers (i64_sub) and GLSL_EXACT_NOISE (perlin3, det_rsqrt, exact_pow2) before it. With p_trace the
// signature gets "out uint trace": FNV-1a of every register's components (trace_hash), so a probe sees every
// instruction, not only the output.
ov::str generate_def_glsl(const Program &p_program, const ov::str &p_name, bool p_trace = false);

} // namespace Archetype
