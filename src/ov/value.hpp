#pragma once

// A JSON-shaped value for the engine-free core: the stats and reports the engine hands its host (godot's Dictionary /
// Array before; the Godot adapter converts with godot_variant, godot_conv.hpp) and the JSON the core reads (ov/json.hpp).
// Objects keep their keys in insertion order, like godot's Dictionary; setting a key again keeps its place.
// Unlike godot's containers, a Value is a value: a copy written to never changes the original (fill a sub-object, then
// set it into its parent).

#include "ov/format.hpp"

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace ov {

class Value {
public:
    enum Type : uint8_t {
        NIL,
        BOOL,
        INT,
        FLOAT,
        STRING,
        ARRAY,
        OBJECT,
    };

    using Array = std::vector<Value>;
    using Object = std::vector<std::pair<str, Value>>;

    Value() = default;
    Value(bool p_v) :
            type(BOOL), b(p_v) {}
    template <typename T, std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, int> = 0>
    Value(T p_v) :
            type(INT), i(int64_t(p_v)) {}
    template <typename T, std::enable_if_t<std::is_enum_v<T>, int> = 0>
    Value(T p_v) :
            type(INT), i(int64_t(p_v)) {}
    Value(double p_v) :
            type(FLOAT), f(p_v) {}
    Value(float p_v) :
            type(FLOAT), f(double(p_v)) {}
    Value(const char *p_v) :
            type(STRING), s(p_v) {}
    Value(str p_v) :
            type(STRING), s(std::move(p_v)) {}
    Value(Array p_v) :
            type(ARRAY), a(std::make_shared<Array>(std::move(p_v))) {}
    Value(Object p_v) :
            type(OBJECT), o(std::make_shared<Object>(std::move(p_v))) {}

    static Value array() { return Value(Array()); }
    static Value object() { return Value(Object()); }

    Type get_type() const { return type; }
    bool is_nil() const { return type == NIL; }
    bool is_number() const { return type == INT || type == FLOAT; }
    bool is_string() const { return type == STRING; }
    bool is_array() const { return type == ARRAY; }
    bool is_object() const { return type == OBJECT; }

    // The value as that type (a number as either; false / 0 / empty otherwise).
    bool as_bool() const { return type == BOOL ? b : (type == INT ? i != 0 : (type == FLOAT ? f != 0.0 : false)); }
    int64_t as_int() const { return type == INT ? i : (type == FLOAT ? int64_t(f) : (type == BOOL ? int64_t(b) : 0)); }
    double as_double() const { return type == FLOAT ? f : (type == INT ? double(i) : (type == BOOL ? double(b) : 0.0)); }
    const str &as_str() const { return type == STRING ? s : empty_str(); }

    // Arrays and objects: their element count (0 for the other types).
    size_t size() const { return type == ARRAY ? a->size() : (type == OBJECT ? o->size() : 0); }
    bool is_empty() const { return size() == 0; }

    // Arrays. A nil value becomes an empty array on push_back. Elements are shared until written (copy on write).
    void push_back(Value p_v) {
        _own(ARRAY);
        a->push_back(std::move(p_v));
    }
    const Value &operator[](size_t p_index) const { return type == ARRAY && p_index < a->size() ? (*a)[p_index] : nil(); }
    const Value &operator[](int p_index) const { return p_index < 0 ? nil() : (*this)[size_t(p_index)]; }
    const Value &operator[](int64_t p_index) const { return p_index < 0 ? nil() : (*this)[size_t(p_index)]; }
    Value &at(size_t p_index) {
        _own(ARRAY);
        return (*a)[p_index];
    }
    // Removes an element (godot's remove_at).
    void remove_at(size_t p_index) {
        _own(ARRAY);
        if (p_index < a->size()) {
            a->erase(a->begin() + std::ptrdiff_t(p_index));
        }
    }
    const Array &elements() const { return type == ARRAY ? *a : empty_array(); }

    // Objects. A nil value becomes an empty object on a write; a missing key reads as nil.
    bool has(const str &p_key) const { return _find(p_key) != nullptr; }
    const Value &operator[](const str &p_key) const {
        const Value *v = _find(p_key);
        return v ? *v : nil();
    }
    const Value &operator[](const char *p_key) const { return (*this)[str(p_key)]; }
    // The key's value, appended (nil) when missing: godot's d["key"] = v.
    Value &operator[](const str &p_key) {
        _own(OBJECT);
        for (std::pair<str, Value> &e : *o) {
            if (e.first == p_key) {
                return e.second;
            }
        }
        o->emplace_back(p_key, Value());
        return o->back().second;
    }
    Value &operator[](const char *p_key) { return (*this)[str(p_key)]; }
    const Object &entries() const { return type == OBJECT ? *o : empty_object(); }
    // Removes a key (godot's erase); false if it wasn't there.
    bool erase(const str &p_key) {
        if (!has(p_key)) {
            return false;
        }
        _own(OBJECT);
        for (auto it = o->begin(); it != o->end(); ++it) {
            if (it->first == p_key) {
                o->erase(it);
                return true;
            }
        }
        return false;
    }
    // The entries of p_other set into this object: those it lacks, or all with p_overwrite (godot's merge).
    void merge(const Value &p_other, bool p_overwrite = false) {
        for (const std::pair<str, Value> &e : p_other.entries()) {
            if (p_overwrite || !has(e.first)) {
                (*this)[e.first] = e.second;
            }
        }
    }

private:
    Type type = NIL;
    union {
        bool b;
        int64_t i = 0;
        double f;
    };
    str s;
    std::shared_ptr<Array> a;
    std::shared_ptr<Object> o;

    static const Value &nil() {
        static const Value s_nil;
        return s_nil;
    }
    static const str &empty_str() {
        static const str s_empty;
        return s_empty;
    }
    static const Array &empty_array() {
        static const Array s_empty;
        return s_empty;
    }
    static const Object &empty_object() {
        static const Object s_empty;
        return s_empty;
    }

    const Value *_find(const str &p_key) const {
        if (type != OBJECT) {
            return nullptr;
        }
        for (const std::pair<str, Value> &e : *o) {
            if (e.first == p_key) {
                return &e.second;
            }
        }
        return nullptr;
    }

    // Makes this a container of p_type held by this value alone (a nil becomes an empty one).
    void _own(Type p_type) {
        if (type == NIL) {
            *this = p_type == ARRAY ? array() : object();
        }
        if (p_type == ARRAY && type == ARRAY && a.use_count() > 1) {
            a = std::make_shared<Array>(*a);
        } else if (p_type == OBJECT && type == OBJECT && o.use_count() > 1) {
            o = std::make_shared<Object>(*o);
        }
    }
};

} // namespace ov
