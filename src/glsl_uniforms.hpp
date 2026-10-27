#pragma once

#include "world_coord.hpp"

#include "ov/bytes.hpp"
#include "ov/str.hpp"

#include <cstring>

// Visitors over a struct's uniforms, for structs that list them once as visit(p_f) calling p_f(name, value) with an
// int, a float or a Float3 (BaseSurface::Render, AtmosphereMedium): the declarations and the bytes come from that one
// list, so no name or offset is typed twice.
namespace GLSLUniforms {

// `uniform T name;` per field (a host shader's loose uniforms: Godot's shading language).
struct Declarations {
    ov::str &out;
    void decl(const char *p_type, const char *p_name) { out += ov::str("uniform ") + p_type + " " + p_name + ";\n"; }
    void operator()(const char *p_name, int) { decl("int", p_name); }
    void operator()(const char *p_name, float) { decl("float", p_name); }
    void operator()(const char *p_name, const Float3 &) { decl("vec3", p_name); }
};

// `T name;` per field: the members of a std140 uniform block (a compute shader's).
struct BlockMembers {
    ov::str &out;
    void member(const char *p_type, const char *p_name) { out += ov::str(p_type) + " " + p_name + ";\n"; }
    void operator()(const char *p_name, int) { member("int", p_name); }
    void operator()(const char *p_name, float) { member("float", p_name); }
    void operator()(const char *p_name, const Float3 &) { member("vec3", p_name); }
};

// The fields' bytes in std140 (an int or a float on 4 bytes, a vec3 aligned to 16), in BlockMembers' order. The block's
// size is rounded up to 16 by pad().
struct Std140 {
    ov::bytes &out;
    void put(const void *p_data, size_t p_size, size_t p_align) {
        out.resize((out.size() + p_align - 1) / p_align * p_align);
        const size_t at = out.size();
        out.resize(at + p_size);
        std::memcpy(out.data() + at, p_data, p_size);
    }
    void operator()(const char *, int p_v) { put(&p_v, 4, 4); }
    void operator()(const char *, float p_v) { put(&p_v, 4, 4); }
    void operator()(const char *, const Float3 &p_v) {
        const float v[3] = { p_v.x, p_v.y, p_v.z };
        put(v, 12, 16);
    }
    void pad() { out.resize((out.size() + 15) / 16 * 16); }
};

} // namespace GLSLUniforms
