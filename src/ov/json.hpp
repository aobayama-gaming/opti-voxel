#pragma once

// JSON text to an ov::Value, as godot's JSON::parse_string read it: every number is a FLOAT (a double), an object
// keeps its keys in order and a repeated key takes the last value. Out of line (json.cpp).

#include "ov/value.hpp"

namespace ov {

// The value of p_text; nil and r_error set ("line N: ...") if it isn't one JSON value.
Value parse_json(const str &p_text, str &r_error);

} // namespace ov
