#pragma once

// The murmur3 chain bits the core uses, verbatim from godot-cpp's templates/hashfuncs.hpp (which is
// Godot's own copy; MurmurHash3 is public domain). Vendored so the core's hashers stop including
// godot and keep producing the same hashes: identical constants, identical operation order, so
// Vector3i64Hasher and every container built on it hash exactly as before the de-Godot sweep.

#include <cstdint>

#ifndef OV_HASH_MURMUR3_SEED
#define OV_HASH_MURMUR3_SEED 0x7F07C65
#endif

namespace ov {

inline uint32_t murmur3_one_32(uint32_t p_in, uint32_t p_seed = OV_HASH_MURMUR3_SEED) {
    p_in *= 0xcc9e2d51;
    p_in = (p_in << 15) | (p_in >> 17);
    p_in *= 0x1b873593;

    p_seed ^= p_in;
    p_seed = (p_seed << 13) | (p_seed >> 19);
    p_seed = p_seed * 5 + 0xe6546b64;

    return p_seed;
}

inline uint32_t fmix32(uint32_t p_h) {
    p_h ^= p_h >> 16;
    p_h *= 0x85ebca6b;
    p_h ^= p_h >> 13;
    p_h *= 0xc2b2ae35;
    p_h ^= p_h >> 16;
    return p_h;
}

} // namespace ov
