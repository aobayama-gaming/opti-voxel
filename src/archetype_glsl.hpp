#pragma once

#include "archetype_ir.hpp"
#include "ov/str.hpp"

namespace Archetype {

// GLSL "vec4 <p_name>(WorldPos p, int lod)" of a program: straight-line `precise` code, one statement per
// instruction in the interpreter's order (archetype_ir.cpp) and with its formulas, constants as raw bits.
// Needs the frame functions (gpu_frame.hpp: sat_diff) and GLSL_EXACT_NOISE (perlin3, perlin2, det_rsqrt,
// exact_pow2) before it. With p_trace the signature gets "out uint trace": FNV-1a of every register's
// components (trace_hash), so a probe sees every instruction, not only the output.
ov::str generate_glsl(const Program &p_program, const ov::str &p_name, bool p_trace = false);

// GLSL "vec4 <p_name>(WorldPos p, int lod, DefRecord rec)" of a Table Def (archetype_table.hpp): the
// same code, with the T_* ops reading the record `rec` (its anchor, rotation, parameters, salt) instead
// of baked constants. Needs GLSL_DEF_RECORD, the frame functions and the int64 helpers (i64_sub) before
// it. A program without T_* ops generates the same body as generate_glsl.
ov::str generate_def_glsl(const Program &p_program, const ov::str &p_name, bool p_trace = false);

} // namespace Archetype
