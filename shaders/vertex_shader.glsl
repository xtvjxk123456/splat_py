#version 330 core
precision highp float;

layout(location = 0) in vec2 a_position;
layout(location = 1) in float a_index;

uniform mat4 u_proj;
uniform mat4 u_view;
uniform vec2 u_focal;
uniform vec2 u_viewport;

out float v_index;

void main() {
    v_index = a_index;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
