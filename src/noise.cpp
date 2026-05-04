#include "noise.hpp"


static constexpr const char GLSL_NOISE[] = R"(
//
// Description : Array and textureless GLSL 2D/3D/4D simplex 
//               noise functions.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : stegu
//     Lastmod : 20201014 (stegu)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//               https://github.com/stegu/webgl-noise

vec3 mod289(vec3 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x) {
     return mod289(((x*34.0)+10.0)*x);
}

vec4 taylorInvSqrt(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

vec4 snoise(vec3 v)
{
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; 
  vec3 x3 = x0 - D.yyy;      
// Permutations
  i = mod289(i); 
  vec4 p = permute( permute( permute( 
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  vec4 m2 = m * m;
  vec4 m4 = m2 * m2;
  vec4 pdotx = vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3));

// Determine noise gradient
  vec4 temp = m2 * m * pdotx;
  
  vec3 gradient = -8.0 * (temp.x * x0 + temp.y * x1 + temp.z * x2 + temp.w * x3);
  gradient += m4.x * p0 + m4.y * p1 + m4.z * p2 + m4.w * p3;

  return 105.0 * vec4(gradient ,dot(m4, pdotx));
}
)";

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


float SimplexNoise::noise(float x) {
    return snoise(Vector3(x, 0.0f, 0.0f)).w;
}

float SimplexNoise::noise(float x, float y) {
    return snoise(Vector3(x, y, 0.0f)).w;
}

float SimplexNoise::noise(float x, float y, float z) {
    return snoise(Vector3(x, y, z)).w;
}

float SimplexNoise::fractal(size_t octaves, float x) const {
    float output = 0.0f;
    float freq = mFrequency;
    float amp = mAmplitude;
    for (size_t i = 0; i < octaves; i++) {
        output += amp * noise(x * freq);
        freq *= mLacunarity;
        amp *= mPersistence;
    }
    return output;
}

float SimplexNoise::fractal(size_t octaves, float x, float y) const {
    float output = 0.0f;
    float freq = mFrequency;
    float amp = mAmplitude;
    for (size_t i = 0; i < octaves; i++) {
        output += amp * noise(x * freq, y * freq);
        freq *= mLacunarity;
        amp *= mPersistence;
    }
    return output;
}

float SimplexNoise::fractal(size_t octaves, float x, float y, float z) const {
    float output = 0.0f;
    float freq = mFrequency;
    float amp = mAmplitude;
    for (size_t i = 0; i < octaves; i++) {
        output += amp * noise(x * freq, y * freq, z * freq);
        freq *= mLacunarity;
        amp *= mPersistence;
    }
    return output;
}