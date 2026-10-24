#pragma once

#include "ov/format.hpp"
#include "world_coord.hpp"

#include <functional>

// One entry of FrameSpike's probe table (frame_spike.cpp; the archetype IR's in archetype_probes.cpp): a
// value computed from a WorldPos on the CPU and on the GPU, `words` uint32 (floats as their bits), compared
// bit for bit.
struct Probe {
    ov::str name;
    int words = 1;
    bool is_float = false;
    ov::str glsl; // defines "void probe_<name>(WorldPos p, uint b)", writing results[b + 0 .. words - 1]
    // Writes the words; returns false where the evaluation left the exact domain (the archetype IR's
    // data-dependent exactness: a denormal or a huge intermediate, which the GPU could flush or round
    // differently). Those evaluations are counted apart, but they are compared and gated like the others.
    std::function<bool(const WorldPos &, uint32_t *)> cpu;
    // >= 0: word `continuity_word` may change by at most this much per LOD0 voxel along the
    // continuity lines (a wrong corner hash at a low-word carry shows up as a jump).
    double continuity_bound = -1.0;
    int continuity_word = 0;
    // The probe is a function of the position relative to the origin: its words at the hash-random points
    // (the same offsets from every origin) must match the first origin's.
    bool origin_relative = false;
};

// LODs the LOD-dependent probes are evaluated at. The GLSL gets them as lod_bias + L, lod_bias a push
// constant 0: a runtime value like the meshers' chunk LOD, not a literal the compiler could fold.
inline constexpr int PROBE_LODS[] = { 0, 1, 5, 14, 24 };
inline constexpr int PROBE_LOD_COUNT = int(sizeof(PROBE_LODS) / sizeof(PROBE_LODS[0]));
