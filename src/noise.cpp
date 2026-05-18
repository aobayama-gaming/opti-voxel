#include "noise.hpp"




// GLSL `step(edge, x)` returns 0.0 if x < edge, and 1.0 otherwise.
inline Vector3 step(const Vector3& edge, const Vector3& x) {
    return Vector3(
        x.x < edge.x ? 0.0f : 1.0f,
        x.y < edge.y ? 0.0f : 1.0f,
        x.z < edge.z ? 0.0f : 1.0f
    );
}

inline Vector4 step(const Vector4& edge, const Vector4& x) {
    return Vector4(
        x.x < edge.x ? 0.0f : 1.0f,
        x.y < edge.y ? 0.0f : 1.0f,
        x.z < edge.z ? 0.0f : 1.0f,
        x.w < edge.w ? 0.0f : 1.0f
    );
}

inline Vector3 mod289(const Vector3& x) {
    return x - (x * (1.0f / 289.0f)).floor() * 289.0f;
}

inline Vector4 mod289(const Vector4& x) {
    return x - (x * (1.0f / 289.0f)).floor() * 289.0f;
}

inline Vector4 permute(const Vector4& x) {
    return mod289((x * 34.0f + Vector4(10.0f, 10.0f, 10.0f, 10.0f)) * x);
}

inline Vector4 taylorInvSqrt(const Vector4& r) {
    return Vector4(1.79284291400159f, 1.79284291400159f, 1.79284291400159f, 1.79284291400159f) - r * 0.85373472095314f;
}

Vector4 snoise(const Vector3& v) {
    const Vector2 C(1.0f / 6.0f, 1.0f / 3.0f);
    const Vector4 D(0.0f, 0.5f, 1.0f, 2.0f);

    // First corner
    Vector3 C_yyy(C.y, C.y, C.y);
    Vector3 C_xxx(C.x, C.x, C.x);
    
    float dot_v_Cyyy = v.dot(C_yyy);
    Vector3 i = (v + Vector3(dot_v_Cyyy, dot_v_Cyyy, dot_v_Cyyy)).floor();
    
    float dot_i_Cxxx = i.dot(C_xxx);
    Vector3 x0 = v - i + Vector3(dot_i_Cxxx, dot_i_Cxxx, dot_i_Cxxx);

    // Other corners
    Vector3 x0_yzx(x0.y, x0.z, x0.x);
    Vector3 g = step(x0_yzx, x0);
    Vector3 l = Vector3(1.0f, 1.0f, 1.0f) - g;
    
    Vector3 l_zxy(l.z, l.x, l.y);
    Vector3 i1 = g.min(l_zxy);
    Vector3 i2 = g.max(l_zxy);

    Vector3 x1 = x0 - i1 + C_xxx;
    Vector3 C_yyy_vec(C.y, C.y, C.y); 
    Vector3 x2 = x0 - i2 + C_yyy_vec;
    
    Vector3 D_yyy(D.y, D.y, D.y);
    Vector3 x3 = x0 - D_yyy;

    // Permutations
    i = mod289(i); 
    
    Vector4 p = permute(permute(permute(
                Vector4(i.z, i.z, i.z, i.z) + Vector4(0.0f, i1.z, i2.z, 1.0f))
            + Vector4(i.y, i.y, i.y, i.y) + Vector4(0.0f, i1.y, i2.y, 1.0f)) 
            + Vector4(i.x, i.x, i.x, i.x) + Vector4(0.0f, i1.x, i2.x, 1.0f));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    float n_ = 0.142857142857f; // 1.0/7.0
    Vector3 D_wyz(D.w, D.y, D.z);
    Vector3 D_xzx(D.x, D.z, D.x);
    Vector3 ns = D_wyz * n_ - D_xzx;

    Vector4 j = p - (p * ns.z * ns.z).floor() * 49.0f;  // mod(p,7*7)

    Vector4 x_ = (j * ns.z).floor();
    Vector4 y_ = (j - x_ * 7.0f).floor();               // mod(j,N)

    Vector4 ns_yyyy(ns.y, ns.y, ns.y, ns.y);
    Vector4 x = x_ * ns.x + ns_yyyy;
    Vector4 y = y_ * ns.x + ns_yyyy;
    Vector4 h = Vector4(1.0f, 1.0f, 1.0f, 1.0f) - x.abs() - y.abs();

    Vector4 b0(x.x, x.y, y.x, y.y);
    Vector4 b1(x.z, x.w, y.z, y.w);

    Vector4 s0 = b0.floor() * 2.0f + Vector4(1.0f, 1.0f, 1.0f, 1.0f);
    Vector4 s1 = b1.floor() * 2.0f + Vector4(1.0f, 1.0f, 1.0f, 1.0f);
    Vector4 sh = -step(h, Vector4(0.0f, 0.0f, 0.0f, 0.0f));

    Vector4 b0_xzyw(b0.x, b0.z, b0.y, b0.w);
    Vector4 s0_xzyw(s0.x, s0.z, s0.y, s0.w);
    Vector4 sh_xxyy(sh.x, sh.x, sh.y, sh.y);
    Vector4 a0 = b0_xzyw + s0_xzyw * sh_xxyy;

    Vector4 b1_xzyw(b1.x, b1.z, b1.y, b1.w);
    Vector4 s1_xzyw(s1.x, s1.z, s1.y, s1.w);
    Vector4 sh_zzww(sh.z, sh.z, sh.w, sh.w);
    Vector4 a1 = b1_xzyw + s1_xzyw * sh_zzww;

    Vector3 p0(a0.x, a0.y, h.x);
    Vector3 p1(a0.z, a0.w, h.y);
    Vector3 p2(a1.x, a1.y, h.z);
    Vector3 p3(a1.z, a1.w, h.w);

    // Normalise gradients
    Vector4 norm = taylorInvSqrt(Vector4(p0.dot(p0), p1.dot(p1), p2.dot(p2), p3.dot(p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    Vector4 m_diff = Vector4(0.5f, 0.5f, 0.5f, 0.5f) - Vector4(x0.dot(x0), x1.dot(x1), x2.dot(x2), x3.dot(x3));
    Vector4 m = m_diff.max(Vector4(0.0f, 0.0f, 0.0f, 0.0f));
    
    Vector4 m2 = m * m;
    Vector4 m4 = m2 * m2;
    Vector4 pdotx(p0.dot(x0), p1.dot(x1), p2.dot(x2), p3.dot(x3));

    // Determine noise gradient
    Vector4 temp = m2 * m * pdotx;
    
    Vector3 gradient = (x0 * temp.x + x1 * temp.y + x2 * temp.z + x3 * temp.w) * -8.0f;
    gradient += p0 * m4.x + p1 * m4.y + p2 * m4.z + p3 * m4.w;

    // Return final values exactly identically mapped inside a Vector4
    float noise_val = m4.dot(pdotx);
    return Vector4(gradient.x, gradient.y, gradient.z, noise_val) * 105.0f;
}

Vector4 SimplexNoise::noise(Vector3 pos){
    return snoise(pos);

}

Vector4 SimplexNoise::fractal(size_t octaves,NoiseParam param, Vector3 pos){
    if (octaves == 0) {
        return Vector4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    Vector4 accumulated_noise(0.0f, 0.0f, 0.0f, 0.0f);
    Vector3 sample_pos = pos * param.frequency;
    float amplitude = param.amplitude;

    for (size_t octave = 0; octave < octaves; ++octave) {
        accumulated_noise += SimplexNoise::noise(sample_pos) * amplitude;
        sample_pos *= param.lacunarity;
        amplitude *= param.persistence;
    }

    return accumulated_noise;
}