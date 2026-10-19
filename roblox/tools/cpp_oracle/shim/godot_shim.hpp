#pragma once
// Minimal stand-ins for the godot-cpp types used by the CPU mesher, so the real src/CPU_mesher.cpp,
// src/noise.cpp and src/sdf_dummy.hpp compile as a plain program (no engine) and serve as the
// reference for the Luau port. Only what those files touch is implemented.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

// OPTI_DOUBLE: build the reference with every `float` promoted to double. The Luau port computes
// in doubles, so this build isolates logic differences from float32 rounding differences.
#ifdef OPTI_DOUBLE
#define float double
#define float_t double
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

namespace godot {

using real_t = float;

struct Vector3;

struct Vector2 {
	real_t x = 0, y = 0;
	Vector2() = default;
	Vector2(real_t p_x, real_t p_y) : x(p_x), y(p_y) {}
};

struct Vector3i {
	int32_t x = 0, y = 0, z = 0;
	Vector3i() = default;
	Vector3i(int32_t p_x, int32_t p_y, int32_t p_z) : x(p_x), y(p_y), z(p_z) {}
	Vector3i operator+(const Vector3i &o) const { return Vector3i(x + o.x, y + o.y, z + o.z); }
	Vector3i operator-(const Vector3i &o) const { return Vector3i(x - o.x, y - o.y, z - o.z); }
	bool operator==(const Vector3i &o) const { return x == o.x && y == o.y && z == o.z; }
	bool operator<(const Vector3i &o) const { return x != o.x ? x < o.x : (y != o.y ? y < o.y : z < o.z); }
	operator Vector3() const;
};

struct Vector3 {
	real_t x = 0, y = 0, z = 0;
	Vector3() = default;
	Vector3(real_t p_x, real_t p_y, real_t p_z) : x(p_x), y(p_y), z(p_z) {}
	real_t &operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
	const real_t &operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
	Vector3 operator+(const Vector3 &o) const { return Vector3(x + o.x, y + o.y, z + o.z); }
	Vector3 operator-(const Vector3 &o) const { return Vector3(x - o.x, y - o.y, z - o.z); }
	Vector3 operator*(const Vector3 &o) const { return Vector3(x * o.x, y * o.y, z * o.z); }
	Vector3 operator*(real_t s) const { return Vector3(x * s, y * s, z * s); }
	Vector3 operator/(real_t s) const { return Vector3(x / s, y / s, z / s); }
	Vector3 operator-() const { return Vector3(-x, -y, -z); }
	Vector3 &operator+=(const Vector3 &o) { x += o.x; y += o.y; z += o.z; return *this; }
	Vector3 &operator-=(const Vector3 &o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
	Vector3 &operator*=(real_t s) { x *= s; y *= s; z *= s; return *this; }
	Vector3 &operator/=(real_t s) { x /= s; y /= s; z /= s; return *this; }
	real_t dot(const Vector3 &o) const { return x * o.x + y * o.y + z * o.z; }
	Vector3 floor() const { return Vector3(std::floor(x), std::floor(y), std::floor(z)); }
	Vector3 min(const Vector3 &o) const { return Vector3(MIN(x, o.x), MIN(y, o.y), MIN(z, o.z)); }
	Vector3 max(const Vector3 &o) const { return Vector3(MAX(x, o.x), MAX(y, o.y), MAX(z, o.z)); }
	real_t length_squared() const { return dot(*this); }
	real_t length() const { return std::sqrt(length_squared()); }
	// Same as Godot: zero vector stays zero.
	Vector3 normalized() const {
		real_t l = length_squared();
		if (l == 0) {
			return Vector3();
		}
		l = std::sqrt(l);
		return Vector3(x / l, y / l, z / l);
	}
};

inline Vector3 operator*(real_t s, const Vector3 &v) { return v * s; }
inline Vector3i::operator Vector3() const { return Vector3(x, y, z); }

struct Vector4 {
	real_t x = 0, y = 0, z = 0, w = 0;
	Vector4() = default;
	Vector4(real_t p_x, real_t p_y, real_t p_z, real_t p_w) : x(p_x), y(p_y), z(p_z), w(p_w) {}
	Vector4 operator+(const Vector4 &o) const { return Vector4(x + o.x, y + o.y, z + o.z, w + o.w); }
	Vector4 operator-(const Vector4 &o) const { return Vector4(x - o.x, y - o.y, z - o.z, w - o.w); }
	Vector4 operator*(const Vector4 &o) const { return Vector4(x * o.x, y * o.y, z * o.z, w * o.w); }
	Vector4 operator*(real_t s) const { return Vector4(x * s, y * s, z * s, w * s); }
	Vector4 operator-() const { return Vector4(-x, -y, -z, -w); }
	Vector4 &operator+=(const Vector4 &o) { x += o.x; y += o.y; z += o.z; w += o.w; return *this; }
	real_t dot(const Vector4 &o) const { return x * o.x + y * o.y + z * o.z + w * o.w; }
	Vector4 floor() const { return Vector4(std::floor(x), std::floor(y), std::floor(z), std::floor(w)); }
	Vector4 abs() const { return Vector4(std::fabs(x), std::fabs(y), std::fabs(z), std::fabs(w)); }
	Vector4 max(const Vector4 &o) const { return Vector4(MAX(x, o.x), MAX(y, o.y), MAX(z, o.z), MAX(w, o.w)); }
};

struct Basis {
	Vector3 rows[3];
	Basis() : rows{ Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1) } {}
	Basis(const Vector3 &a, const Vector3 &b, const Vector3 &c) : rows{ a, b, c } {}
	Basis(real_t xx, real_t xy, real_t xz, real_t yx, real_t yy, real_t yz, real_t zx, real_t zy, real_t zz) :
			rows{ Vector3(xx, xy, xz), Vector3(yx, yy, yz), Vector3(zx, zy, zz) } {}
	Vector3 &operator[](int i) { return rows[i]; }
	const Vector3 &operator[](int i) const { return rows[i]; }
	Basis &operator+=(const Basis &o) {
		rows[0] += o.rows[0];
		rows[1] += o.rows[1];
		rows[2] += o.rows[2];
		return *this;
	}
	Vector3 xform(const Vector3 &v) const { return Vector3(rows[0].dot(v), rows[1].dot(v), rows[2].dot(v)); }
	void scale(const Vector3 &) {}
};

struct Transform3D {
	Transform3D() = default;
	Transform3D(const Basis &, const Vector3 &) {}
};

struct RID {
	bool is_valid() const { return false; }
};

struct String {
	String() = default;
	String(const char *) {}
};

template <typename T>
struct PackedArray : std::vector<T> {
	bool is_empty() const { return this->empty(); }
	int64_t size() const { return (int64_t)std::vector<T>::size(); }
};
using PackedVector3Array = PackedArray<Vector3>;
using PackedInt32Array = PackedArray<int32_t>;

template <typename K, typename V>
struct HashMap : std::map<K, V> {};

template <typename T>
struct Ref {
	void instantiate() {}
	bool is_null() const { return true; }
	bool is_valid() const { return false; }
	void unref() {}
	T *operator->() const { return nullptr; }
};

struct Object {};
struct Thread {
	enum { PRIORITY_NORMAL };
	template <typename C>
	void start(C, int) {}
	void wait_to_finish() {}
};
struct Mutex {
	void lock() {}
	void unlock() {}
};
struct Semaphore {
	void wait() {}
	void post() {}
};

struct Array {
	void resize(int) {}
	struct Slot {
		template <typename T>
		Slot &operator=(const T &) { return *this; }
	};
	Slot operator[](int) { return Slot(); }
};

struct Mesh {
	enum { ARRAY_VERTEX, ARRAY_NORMAL, ARRAY_INDEX, ARRAY_MAX };
};

struct RenderingServer {
	enum { PRIMITIVE_TRIANGLES };
	static RenderingServer *get_singleton() { return nullptr; }
	RID mesh_create() { return RID(); }
	RID instance_create() { return RID(); }
	RID get_test_cube() { return RID(); }
	template <typename... A>
	void mesh_add_surface_from_arrays(A...) {}
	template <typename... A>
	void instance_set_scenario(A...) {}
	template <typename... A>
	void instance_set_base(A...) {}
	template <typename... A>
	void instance_set_transform(A...) {}
	template <typename... A>
	void free_rid(A...) {}
};

namespace Math {
inline double floor(double v) { return std::floor(v); }
} // namespace Math

template <typename... A>
inline void print_line(A...) {
	std::fprintf(stderr, "print_line\n");
}

template <typename O, typename M>
inline int callable_mp(O, M) { return 0; }

} // namespace godot
