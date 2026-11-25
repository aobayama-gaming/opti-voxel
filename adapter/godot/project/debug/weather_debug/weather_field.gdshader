// The weather debugger's planet: each cell the field's value at its texel (WeatherDebugNode.planet_mesh's UV), its
// edges darkened (UV2: the corner), the picked cell outlined white.
shader_type spatial;
render_mode unshaded, cull_back;

#include "res://debug/weather_debug/colormap.gdshaderinc"

uniform sampler2D field : filter_nearest;
uniform float vmin = 0.0;
uniform float vmax = 1.0;
uniform bool diverging = false;
uniform float edge = 0.03;
uniform vec2 picked_uv = vec2(-1.0);

void fragment() {
	float v = texture(field, UV).r;
	vec3 c = weather_color(v, vmin, vmax, diverging);
	vec2 e = min(UV2, 1.0 - UV2);
	float line = step(min(e.x, e.y), edge);
	ivec2 size = textureSize(field, 0);
	bool picked = all(lessThan(abs(UV - picked_uv), 0.25 / vec2(size)));
	ALBEDO = picked ? mix(c, vec3(1.0), step(min(e.x, e.y), 0.15)) : mix(c, c * 0.55, line);
}
