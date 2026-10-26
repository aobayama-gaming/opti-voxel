#include "ov/json.hpp"

#include <cmath>
#include <cstdlib>
#include <string>

namespace ov {

namespace {

class JsonParser {
public:
    explicit JsonParser(const str &p_text) :
            text(p_text) {}

    str error;

    Value run() {
        Value v = value(0);
        skip_space();
        if (error.empty() && at < text.size()) {
            fail("text after the value.");
        }
        return error.empty() ? v : Value();
    }

private:
    // Nesting deeper than this is refused (godot's JSON stops at 512 too).
    static constexpr int MAX_DEPTH = 512;

    const str &text;
    size_t at = 0;

    void fail(const char *p_msg) {
        if (error.empty()) {
            int line = 1;
            for (size_t i = 0; i < at && i < text.size(); i++) {
                line += text[i] == '\n';
            }
            error = format("line %d: %s", line, p_msg);
        }
    }

    void skip_space() {
        while (at < text.size() && (text[at] == ' ' || text[at] == '\t' || text[at] == '\n' || text[at] == '\r')) {
            at++;
        }
    }

    bool literal(const char *p_word) {
        const str word(p_word);
        if (text.compare(at, word.size(), word) != 0) {
            return false;
        }
        at += word.size();
        return true;
    }

    Value value(int p_depth) {
        if (p_depth > MAX_DEPTH) {
            fail("nested too deep.");
            return Value();
        }
        skip_space();
        if (at >= text.size()) {
            fail("a value is expected.");
            return Value();
        }
        const char c = text[at];
        if (c == '{') {
            return object(p_depth);
        }
        if (c == '[') {
            return list(p_depth);
        }
        if (c == '"') {
            return Value(string());
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            return number();
        }
        if (literal("true")) {
            return Value(true);
        }
        if (literal("false")) {
            return Value(false);
        }
        if (literal("null")) {
            return Value();
        }
        fail("unexpected character.");
        return Value();
    }

    Value object(int p_depth) {
        Value out = Value::object();
        at++; // {
        skip_space();
        if (at < text.size() && text[at] == '}') {
            at++;
            return out;
        }
        while (error.empty()) {
            skip_space();
            if (at >= text.size() || text[at] != '"') {
                fail("a key (a string) is expected.");
                break;
            }
            const str key = string();
            skip_space();
            if (at >= text.size() || text[at] != ':') {
                fail("':' is expected after a key.");
                break;
            }
            at++;
            Value v = value(p_depth + 1);
            out[key] = std::move(v);
            skip_space();
            if (at < text.size() && text[at] == ',') {
                at++;
            } else if (at < text.size() && text[at] == '}') {
                at++;
                break;
            } else {
                fail("',' or '}' is expected.");
            }
        }
        return out;
    }

    Value list(int p_depth) {
        Value out = Value::array();
        at++; // [
        skip_space();
        if (at < text.size() && text[at] == ']') {
            at++;
            return out;
        }
        while (error.empty()) {
            out.push_back(value(p_depth + 1));
            skip_space();
            if (at < text.size() && text[at] == ',') {
                at++;
            } else if (at < text.size() && text[at] == ']') {
                at++;
                break;
            } else {
                fail("',' or ']' is expected.");
            }
        }
        return out;
    }

    static int hex_digit(char p_c) {
        if (p_c >= '0' && p_c <= '9') {
            return p_c - '0';
        }
        if (p_c >= 'a' && p_c <= 'f') {
            return p_c - 'a' + 10;
        }
        if (p_c >= 'A' && p_c <= 'F') {
            return p_c - 'A' + 10;
        }
        return -1;
    }

    // Four hex digits after "\u", or -1.
    int32_t hex4() {
        int32_t v = 0;
        for (int k = 0; k < 4; k++) {
            const int d = at < text.size() ? hex_digit(text[at]) : -1;
            if (d < 0) {
                return -1;
            }
            v = v * 16 + d;
            at++;
        }
        return v;
    }

    static void append_utf8(str &r_out, uint32_t p_cp) {
        if (p_cp < 0x80) {
            r_out += char(p_cp);
        } else if (p_cp < 0x800) {
            r_out += char(0xC0 | (p_cp >> 6));
            r_out += char(0x80 | (p_cp & 0x3F));
        } else if (p_cp < 0x10000) {
            r_out += char(0xE0 | (p_cp >> 12));
            r_out += char(0x80 | ((p_cp >> 6) & 0x3F));
            r_out += char(0x80 | (p_cp & 0x3F));
        } else {
            r_out += char(0xF0 | (p_cp >> 18));
            r_out += char(0x80 | ((p_cp >> 12) & 0x3F));
            r_out += char(0x80 | ((p_cp >> 6) & 0x3F));
            r_out += char(0x80 | (p_cp & 0x3F));
        }
    }

    str string() {
        str out;
        at++; // "
        while (error.empty()) {
            if (at >= text.size()) {
                fail("unterminated string.");
                break;
            }
            const char c = text[at++];
            if (c == '"') {
                break;
            }
            if (c != '\\') {
                out += c;
                continue;
            }
            if (at >= text.size()) {
                fail("unterminated string.");
                break;
            }
            const char e = text[at++];
            switch (e) {
                case '"':
                case '\\':
                case '/':
                    out += e;
                    break;
                case 'b':
                    out += '\b';
                    break;
                case 'f':
                    out += '\f';
                    break;
                case 'n':
                    out += '\n';
                    break;
                case 'r':
                    out += '\r';
                    break;
                case 't':
                    out += '\t';
                    break;
                case 'u': {
                    int32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && literal("\\u")) {
                        const int32_t low = hex4();
                        cp = low >= 0xDC00 && low < 0xE000 ? 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00) : -1;
                    }
                    if (cp < 0) {
                        fail("invalid \\u escape.");
                    } else {
                        append_utf8(out, uint32_t(cp));
                    }
                    break;
                }
                default:
                    fail("invalid escape.");
            }
        }
        return out;
    }

    Value number() {
        const size_t begin = at;
        if (text[at] == '-') {
            at++;
        }
        auto digits = [&]() {
            const size_t from = at;
            while (at < text.size() && text[at] >= '0' && text[at] <= '9') {
                at++;
            }
            return at > from;
        };
        bool ok = digits();
        if (ok && at < text.size() && text[at] == '.') {
            at++;
            ok = digits();
        }
        if (ok && at < text.size() && (text[at] == 'e' || text[at] == 'E')) {
            at++;
            if (at < text.size() && (text[at] == '+' || text[at] == '-')) {
                at++;
            }
            ok = digits();
        }
        if (!ok) {
            fail("malformed number.");
            return Value();
        }
        return Value(std::strtod(text.substr(begin, at - begin).c_str(), nullptr));
    }
};

} // namespace

namespace {

void write_string(const str &p_s, str &r_out) {
    r_out += '"';
    for (const char c : p_s) {
        switch (c) {
            case '"':
                r_out += "\\\"";
                break;
            case '\\':
                r_out += "\\\\";
                break;
            case '\n':
                r_out += "\\n";
                break;
            case '\r':
                r_out += "\\r";
                break;
            case '\t':
                r_out += "\\t";
                break;
            default:
                if (uint8_t(c) < 0x20) {
                    r_out += format("\\u%04x", int(c));
                } else {
                    r_out += c;
                }
        }
    }
    r_out += '"';
}

void write_value(const Value &p_v, str &r_out) {
    switch (p_v.get_type()) {
        case Value::BOOL:
            r_out += p_v.as_bool() ? "true" : "false";
            break;
        case Value::INT:
            r_out += std::to_string(p_v.as_int());
            break;
        case Value::FLOAT:
            r_out += std::isfinite(p_v.as_double()) ? detail::num_text(p_v.as_double()) : str("null");
            break;
        case Value::STRING:
            write_string(p_v.as_str(), r_out);
            break;
        case Value::ARRAY: {
            r_out += '[';
            bool first = true;
            for (const Value &e : p_v.elements()) {
                r_out += first ? "" : ",";
                first = false;
                write_value(e, r_out);
            }
            r_out += ']';
            break;
        }
        case Value::OBJECT: {
            r_out += '{';
            bool first = true;
            for (const std::pair<str, Value> &e : p_v.entries()) {
                r_out += first ? "" : ",";
                first = false;
                write_string(e.first, r_out);
                r_out += ':';
                write_value(e.second, r_out);
            }
            r_out += '}';
            break;
        }
        default:
            r_out += "null";
    }
}

} // namespace

str to_json(const Value &p_value) {
    str out;
    write_value(p_value, out);
    return out;
}

Value parse_json(const str &p_text, str &r_error) {
    JsonParser parser(p_text);
    Value v = parser.run();
    r_error = parser.error;
    return v;
}

} // namespace ov
