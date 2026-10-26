#pragma once

// Engine-side conversions of the core's engine-free types: explicit, at call sites, never back into
// the core.

#include "gpu_frame.hpp"
#include "latency_hist.hpp"
#include "ov/bytes.hpp"
#include "ov/color.hpp"
#include "ov/value.hpp"
#include "world_coord.hpp"

#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector2i.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/vector3i.hpp>

#include <cstring>

inline godot::String godot_str(const ov::str &p_s) {
    return godot::String::utf8(p_s.c_str(), int64_t(p_s.size()));
}

inline ov::str ov_str(const godot::String &p_s) {
    const godot::CharString utf8 = p_s.utf8();
    return ov::str(utf8.get_data(), size_t(utf8.length()));
}

// An int64 id or voxel as report text (WorldCoord::to_string).
inline godot::String godot_str(const Vector3i64 &p_v) {
    return godot_str(WorldCoord::to_string(p_v));
}

inline godot::Vector2i godot_v2i(const ov::ivec2 &p_v) {
    return godot::Vector2i(p_v.x, p_v.y);
}

inline godot::Vector3i godot_v3i(const ov::ivec3 &p_v) {
    return godot::Vector3i(p_v.x, p_v.y, p_v.z);
}

inline godot::Vector3 godot_v3(const Float3 &p_v) {
    return godot::Vector3(p_v.x, p_v.y, p_v.z);
}

inline Float3 ov_f3(const godot::Vector3 &p_v) {
    return Float3{ float(p_v.x), float(p_v.y), float(p_v.z) };
}

inline godot::Color godot_color(const ov::Color &p_c) {
    return godot::Color(p_c.r, p_c.g, p_c.b, p_c.a);
}

// The core's reports and stats as godot's containers (and back: a Dictionary handed to the core).
inline godot::Variant godot_variant(const ov::Value &p_v) {
    switch (p_v.get_type()) {
        case ov::Value::BOOL:
            return p_v.as_bool();
        case ov::Value::INT:
            return p_v.as_int();
        case ov::Value::FLOAT:
            return p_v.as_double();
        case ov::Value::STRING:
            return godot_str(p_v.as_str());
        case ov::Value::ARRAY: {
            godot::Array a;
            for (const ov::Value &e : p_v.elements()) {
                a.push_back(godot_variant(e));
            }
            return a;
        }
        case ov::Value::OBJECT: {
            godot::Dictionary d;
            for (const std::pair<ov::str, ov::Value> &e : p_v.entries()) {
                d[godot_str(e.first)] = godot_variant(e.second);
            }
            return d;
        }
        default:
            return godot::Variant();
    }
}

inline godot::Dictionary godot_dict(const ov::Value &p_v) {
    return p_v.is_object() ? godot::Dictionary(godot_variant(p_v)) : godot::Dictionary();
}

inline godot::Array godot_array(const ov::Value &p_v) {
    return p_v.is_array() ? godot::Array(godot_variant(p_v)) : godot::Array();
}

inline ov::Value ov_value(const godot::Variant &p_v) {
    switch (p_v.get_type()) {
        case godot::Variant::BOOL:
            return ov::Value(bool(p_v));
        case godot::Variant::INT:
            return ov::Value(int64_t(p_v));
        case godot::Variant::FLOAT:
            return ov::Value(double(p_v));
        case godot::Variant::STRING:
        case godot::Variant::STRING_NAME:
            return ov::Value(ov_str(godot::String(p_v)));
        case godot::Variant::ARRAY: {
            const godot::Array a = p_v;
            ov::Value out = ov::Value::array();
            for (int64_t i = 0; i < a.size(); i++) {
                out.push_back(ov_value(a[i]));
            }
            return out;
        }
        case godot::Variant::DICTIONARY: {
            const godot::Dictionary d = p_v;
            const godot::Array keys = d.keys();
            ov::Value out = ov::Value::object();
            for (int64_t i = 0; i < keys.size(); i++) {
                out[ov_str(godot::String(keys[i]))] = ov_value(d[keys[i]]);
            }
            return out;
        }
        default:
            return ov::Value();
    }
}

inline godot::PackedByteArray godot_bytes(const ov::bytes &p_bytes) {
    godot::PackedByteArray packed;
    packed.resize(int64_t(p_bytes.size()));
    std::memcpy(packed.ptrw(), p_bytes.data(), p_bytes.size());
    return packed;
}

namespace GPUFrame {
// Chunk p_index of a batch's PackedByteArray (the slot encoder at its slot).
inline void encode_chunk_input(godot::PackedByteArray &r_bytes, int64_t p_index, const ChunkId &p_chunk, const ChunkParam &p_param) {
    ERR_FAIL_COND_MSG(p_index < 0 || (p_index + 1) * CHUNK_INPUT_BYTES > r_bytes.size(), "encode_chunk_input: chunk past the batch buffer.");
    encode_chunk_input(r_bytes.ptrw() + p_index * CHUNK_INPUT_BYTES, p_chunk, p_param);
}
} // namespace GPUFrame
