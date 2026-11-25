// The weather debugger's flat views: a section (the field's texture, the ground at the bottom) or, with legend, the
// colour scale along x.
shader_type canvas_item;

#include "res://debug/weather_debug/colormap.gdshaderinc"

uniform sampler2D field : filter_nearest;
uniform float vmin = 0.0;
uniform float vmax = 1.0;
uniform bool diverging = false;
uniform bool legend = false;
uniform float picked_x = -1.0;

void fragment() {
	if (legend) {
		COLOR = vec4(weather_color(UV.x, 0.0, 1.0, diverging), 1.0);
	} else {
		float v = texture(field, vec2(UV.x, 1.0 - UV.y)).r;
		vec3 c = weather_color(v, vmin, vmax, diverging);
		ivec2 size = textureSize(field, 0);
		float cx = floor(UV.x * float(size.x));
		COLOR = vec4(cx == picked_x ? mix(c, vec3(1.0), 0.35) : c, 1.0);
	}
}
