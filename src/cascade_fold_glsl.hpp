#pragma once

#include "cascade.hpp"

#include <godot_cpp/variant/string.hpp>

// The fold on the GPU (Cascade::node_interp, Spec::node_level), generated into CascadeGLSL's level shader. Without a fold
// (Spec::folds false: no field group carried below its node level) none of it is emitted and the level shader's text is
// the one before it.
//
// Scratch: a fold group's nodes are read in place, in the scratch plane its node level wrote. A level writes only the
// planes of the groups it carries (its own ground, the state, the groups at or above their node level), so a group's
// plane at its node level is never overwritten by the chunk's levels below. Across chunks, a fold spec's passes give
// item c a fixed region of each buffer (c times the buffer's points per item, cascade_stride) instead of c n_j^3: in a
// mixed-LOD batch a coarser chunk's level j would otherwise land on a finer chunk's node plane, written levels earlier.
//
// Per item, every field group g at level m is one of: not carried (g_fields false), carried (m >= its node level: loaded
// from its parents, subdivided, stored: g_carry_<node> true) or folded (m below it: read from its nodes before the level's
// insertions, cascade_fold_<node>). The branches follow the item's own LOD, uniform per workgroup for chunk items.
namespace CascadeFoldGLSL {

// The node levels of p_spec's groups that p_range folds (Spec::folds per group), finest first.
std::vector<int> node_levels(const Cascade::Spec &p_spec, const Cascade::FieldRange &p_range);
// The flag a fold group's code is guarded by ("g_carry_<node>"), empty for a group carried wherever g_fields is (the
// ground, a band-less group, a group p_range never carries below its node level).
String carry_flag(const Cascade::Spec &p_spec, const Cascade::FieldRange &p_range, int p_group);

// The level template's parent scratch base and store slot base for item c (np / n: the parent / child box bound): c n^3
// without a fold (the text before it), else item c's fixed region (cascade_stride).
String parent_base(bool p_fold);
String store_base(bool p_fold);
// cascade_level_bound and cascade_stride (Cascade::level_bound and the scratch's points per item of buffer A / B) for items
// of p_item_size points per axis (a GLSL expression: the chunk size, or 1u for point items).
String stride_functions(const String &p_item_size);
// The carry flags' declarations (globals, next to g_fields) and their per-item values (after g_fields is set).
String flag_declarations(const Cascade::Spec &p_spec, const Cascade::FieldRange &p_range);
String flag_values(const Cascade::Spec &p_spec, const Cascade::FieldRange &p_range);
// cascade_fold_<node> for each node level (after the storage functions: p_stride is their vec4s per point).
String fold_functions(const Cascade::Spec &p_spec, const Cascade::FieldRange &p_range, int p_stride);
// The calls, before the level's insertions (the template's main: ch, m, lod, m_lo / m_hi, i, lo_lo / lo_hi, c in scope).
String fold_calls(const Cascade::Spec &p_spec, const Cascade::FieldRange &p_range);

} // namespace CascadeFoldGLSL
