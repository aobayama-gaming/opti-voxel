#pragma once

#include "archetype_ir.hpp"
#include "frame_probe.hpp"

#include <vector>

// FrameSpike's probes of the archetype IR: the generated GLSL (archetype_glsl.hpp) against the interpreter
// (Archetype::run), bit for bit, at PROBE_LODS.
namespace ArchetypeProbes {

// Program p_program as GLSL "arch_<p_name>": per LOD of PROBE_LODS, the output (4 words) and with p_trace
// the trace hash of every register (1 word), zeros read as +0 (Archetype::zero_canonical). The CPU side flags
// points outside the exact domain.
Probe probe(const ov::str &p_name, const Archetype::Program &p_program, bool p_trace);

// Each shape builder and the fBms (a union of one copy per p_anchors, sizes p_size voxels); every shape at
// p_frame_anchors (origin_relative: they sit at the same offset from each origin); then p_fuzz fuzzed programs
// (Archetype::random_program, 8 to 40 instructions) over p_anchors. The table's operators and rotations are the
// table probe's (frame_spike.cpp).
std::vector<Probe> make(const std::vector<VoxelPos> &p_anchors, const std::vector<VoxelPos> &p_frame_anchors, float p_size, int p_fuzz);

} // namespace ArchetypeProbes
