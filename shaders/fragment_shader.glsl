#version 330 core
precision highp float;

uniform sampler2D u_centers;
uniform sampler2D u_scales;
uniform sampler2D u_colors;
uniform sampler2D u_rotations;

uniform mat4 u_proj;
uniform mat4 u_view;
uniform vec2 u_focal;
uniform vec2 u_viewport;

in float v_index;
out vec4 FragColor;

void main() {
    ivec2 tex_size = textureSize(u_centers, 0);
    int tex_width = tex_size.x;
    ivec2 tex_coord = ivec2(int(v_index) % tex_width, int(v_index) / tex_width);

    vec3 center = texelFetch(u_centers, tex_coord, 0).xyz;
    vec3 scale = texelFetch(u_scales, tex_coord, 0).xyz;
    vec4 color = texelFetch(u_colors, tex_coord, 0);
    vec4 rot = texelFetch(u_rotations, tex_coord, 0);

    // Transform center
    vec4 p_view = u_view * vec4(center, 1.0);

    // 计算2D协方差矩阵
    // M = S * R
    mat3 S = mat3(
        scale.x, 0.0, 0.0,
        0.0, scale.y, 0.0,
        0.0, 0.0, scale.z
    );

    mat3 R = mat3(
        1.0 - 2.0 * (rot.z * rot.z + rot.w * rot.w), 2.0 * (rot.y * rot.z - rot.x * rot.w), 2.0 * (rot.y * rot.w + rot.x * rot.z),
        2.0 * (rot.y * rot.z + rot.x * rot.w), 1.0 - 2.0 * (rot.y * rot.y + rot.w * rot.w), 2.0 * (rot.z * rot.w - rot.x * rot.y),
        2.0 * (rot.y * rot.w - rot.x * rot.z), 2.0 * (rot.z * rot.w + rot.x * rot.y), 1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z)
    );

    mat3 M = S * R;
    mat3 V = transpose(mat3(u_view)) * M;
    
    // T = J * W * V
    // J 是雅可比矩阵
    float s = 1.0 / p_view.w;
    mat2 J = mat2(
        u_focal.x * s, 0.0,
        0.0, u_focal.y * s
    );

    mat3 W = mat3(u_view);

    mat3 T = transpose(J * W * M);
    mat2 cov2d = mat2(
        T[0][0], T[1][0],
        T[0][1], T[1][1]
    );

    float det = determinant(cov2d);
    if (det == 0.0) discard;

    mat2 inv_cov2d = inverse(cov2d);
    
    vec2 p_proj = (p_view.xy / p_view.w) * u_focal;
    vec2 p_ndc = p_proj;
    vec2 p_screen = (p_ndc + 1.0) * u_viewport / 2.0;

    vec2 v_screen = gl_FragCoord.xy;
    vec2 d = p_screen - v_screen;
    float power = -0.5 * (d.x * d.x * inv_cov2d[0][0] + d.y * d.y * inv_cov2d[1][1] + 2.0 * d.x * d.y * inv_cov2d[0][1]);
    
    if (power > 0.0) discard;

    float alpha = min(0.99, color.a * exp(power));
    if (alpha < 1.0/255.0) discard;
    
    FragColor = vec4(color.rgb * alpha, alpha);
}
