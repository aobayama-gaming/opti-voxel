// The weather debugger's colour maps: Turbo (Mikhailov 2019, Google's polynomial fit) for a magnitude, blue-white-red
// for a signed field (0 white at the range's middle).
vec3 weather_turbo(float x) {
	const vec4 kr4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
	const vec4 kg4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
	const vec4 kb4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
	const vec2 kr2 = vec2(-152.94239396, 59.28637943);
	const vec2 kg2 = vec2(4.27729857, 2.82956604);
	const vec2 kb2 = vec2(-89.90310912, 27.34824973);
	x = clamp(x, 0.0, 1.0);
	vec4 v4 = vec4(1.0, x, x * x, x * x * x);
	vec2 v2 = v4.zw * v4.z;
	return vec3(dot(v4, kr4) + dot(v2, kr2), dot(v4, kg4) + dot(v2, kg2), dot(v4, kb4) + dot(v2, kb2));
}

vec3 weather_diverging(float t) {
	vec3 blue = vec3(0.23, 0.30, 0.75);
	vec3 red = vec3(0.71, 0.02, 0.15);
	vec3 white = vec3(0.95);
	return t < 0.5 ? mix(blue, white, t * 2.0) : mix(white, red, (t - 0.5) * 2.0);
}

vec3 weather_color(float v, float vmin, float vmax, bool diverging) {
	if (v != v) {
		return vec3(1.0, 0.0, 1.0); // NaN
	}
	float t = clamp((v - vmin) / max(vmax - vmin, 1e-20), 0.0, 1.0);
	return diverging ? weather_diverging(t) : weather_turbo(t);
}
