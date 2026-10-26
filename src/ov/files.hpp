#pragma once

// The host's files for the engine-free core: a climate table read, a debug dump written. Paths are the host's (godot:
// res:// and user://); the adapter installs its reader and writer (register_types: FileAccess). Default: the path on the
// local file system.

#include "ov/format.hpp"

#include <fstream>
#include <sstream>

namespace ov {

// The whole file at p_path into r_text; false if it can't be read.
using FileReader = bool (*)(const str &p_path, str &r_text);

inline bool local_file_reader(const str &p_path, str &r_text) {
    std::ifstream f(p_path, std::ios::binary);
    if (!f) {
        return false;
    }
    std::ostringstream text;
    text << f.rdbuf();
    r_text = text.str();
    return true;
}

inline FileReader &file_reader() {
    static FileReader s_reader = local_file_reader;
    return s_reader;
}

inline void set_file_reader(FileReader p_reader) {
    file_reader() = p_reader;
}

inline bool read_file(const str &p_path, str &r_text) {
    return file_reader() != nullptr && file_reader()(p_path, r_text);
}

// The whole of p_text to the file at p_path (replaced); false if it can't be written.
using FileWriter = bool (*)(const str &p_path, const str &p_text);

inline bool local_file_writer(const str &p_path, const str &p_text) {
    std::ofstream f(p_path, std::ios::binary | std::ios::trunc);
    f << p_text;
    return bool(f);
}

inline FileWriter &file_writer() {
    static FileWriter s_writer = local_file_writer;
    return s_writer;
}

inline void set_file_writer(FileWriter p_writer) {
    file_writer() = p_writer;
}

inline bool write_file(const str &p_path, const str &p_text) {
    return file_writer() != nullptr && file_writer()(p_path, p_text);
}

// p_dir / p_name (godot's path_join).
inline str path_join(const str &p_dir, const str &p_name) {
    return p_dir.empty() || p_dir.back() == '/' ? p_dir + p_name : p_dir + "/" + p_name;
}

} // namespace ov
