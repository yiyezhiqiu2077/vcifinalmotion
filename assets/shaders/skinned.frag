#version 410 core

layout(location = 0) in vec3 v_Normal;
layout(location = 0) out vec4 f_Color;

uniform vec3 u_Color;

void main() {
    vec3 n = normalize(v_Normal);
    vec3 l = normalize(vec3(0.3, 0.8, 0.6));
    float ndotl = clamp(dot(n, l), 0.0, 1.0);
    float shade = 0.35 + 0.65 * ndotl;
    f_Color = vec4(u_Color * shade, 1.0);
}
