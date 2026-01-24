import pygame
from pygame.locals import *
import numpy as np
import requests
from pyrr import Matrix44, Vector3, matrix44
import math

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

# --- 着色器代码 (从 main.js 移植) ---
# GLSL 版本声明改为 #version 330 core，并做了一些小的语法调整
# (例如 gl_FragColor -> FragColor, texture2D -> texture)
# texelFetch 需要 GLSL 1.30+，所以 330 是一个安全的选择

VERTEX_SHADER = """
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
"""

FRAGMENT_SHADER = """
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
"""

def create_texture_from_data(data, width, height, internal_format, data_format, data_type):
    """辅助函数，用于从Numpy数组创建OpenGL纹理"""
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, data_format, data_type, data)
    return tex_id

def load_splat_data(url):
    """从URL加载并解析.splat文件"""
    print(f"Loading splat data from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 使用流式传输逐步读取数据
        buffer = bytearray()
        for chunk in response.iter_content(chunk_size=8192):
            buffer.extend(chunk)

        buffer = bytes(buffer)
        print(f"Loaded {len(buffer)} bytes.")

        # .splat 文件格式：每个高斯44字节
        # 3*float32: 位置 (12 bytes)
        # 3*float32: 缩放 (12 bytes)
        # 4*uint8:   颜色 (4 bytes)
        # 4*float32: 旋转 (16 bytes)
        row_length = 3 * 4 + 3 * 4 + 4 + 4 * 4
        num_gaussians = len(buffer) // row_length
        print(f"Found {num_gaussians} gaussians.")

        # 使用Numpy的结构化数组进行高效解析
        dtype = np.dtype([
            ('pos', np.float32, 3),
            ('scale', np.float32, 3),
            ('rgba', np.uint8, 4),
            ('rot', np.float32, 4),
        ])
        
        # 注意： frombuffer 需要一个连续的内存块
        data = np.frombuffer(buffer, dtype=dtype)

        # 提取数据并进行预处理
        centers = data['pos'].copy()
        
        # 缩放值是对数存储的，需要取指数
        scales = np.exp(data['scale'])

        # 颜色需要从 uint8 [0, 255] 归一化到 float32 [0.0, 1.0]
        colors = data['rgba'].astype(np.float32) / 255.0

        # 旋转是四元数，需要归一化
        rotations = data['rot']
        norms = np.linalg.norm(rotations, axis=1, keepdims=True)
        rotations /= norms

        return centers, scales, colors, rotations, num_gaussians

    except requests.exceptions.RequestException as e:
        print(f"Error loading splat file: {e}")
        return None, None, None, None, 0


def main():
    # --- 初始化 Pygame 和 OpenGL ---
    pygame.init()
    width, height = 1024, 768
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("PyOpenGL Gaussian Splatting")

    # --- OpenGL 设置 ---
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    # 原始JS代码是 (GL_ONE_MINUS_DST_ALPHA, GL_ONE) 用于 pre-multiplied alpha
    # 这里我们使用更常见的 alpha blending
    # glBlendFunc(GL_ONE_MINUS_DST_ALPHA, GL_ONE) 
    
    glClearColor(0.1, 0.1, 0.1, 1.0)

    # --- 加载并编译着色器 ---
    try:
        shader_program = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )
    except Exception as e:
        print("Shader compilation error:", e)
        return

    # --- 加载 Splat 数据 ---
    splat_url = "https://huggingface.co/datasets/antimatter15/splat/resolve/main/train.splat"
    centers, scales, colors, rotations, num_gaussians = load_splat_data(splat_url)
    
    if num_gaussians == 0:
        return

    # --- 创建数据纹理 ---
    # 计算一个合适的纹理尺寸，使其接近正方形
    tex_width = int(math.ceil(math.sqrt(num_gaussians)))
    tex_height = int(math.ceil(num_gaussians / tex_width))
    
    # 填充数据以匹配纹理尺寸
    padded_len = tex_width * tex_height
    
    centers_padded = np.zeros((padded_len, 3), dtype=np.float32)
    centers_padded[:num_gaussians] = centers
    
    scales_padded = np.zeros((padded_len, 3), dtype=np.float32)
    scales_padded[:num_gaussians] = scales

    colors_padded = np.zeros((padded_len, 4), dtype=np.float32)
    colors_padded[:num_gaussians] = colors
    
    rotations_padded = np.zeros((padded_len, 4), dtype=np.float32)
    rotations_padded[:num_gaussians] = rotations

    # 创建OpenGL纹理
    glUseProgram(shader_program) # 激活程序以设置uniforms
    
    tex_centers = create_texture_from_data(centers_padded, tex_width, tex_height, GL_RGB32F, GL_RGB, GL_FLOAT)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, tex_centers)
    glUniform1i(glGetUniformLocation(shader_program, "u_centers"), 0)

    tex_scales = create_texture_from_data(scales_padded, tex_width, tex_height, GL_RGB32F, GL_RGB, GL_FLOAT)
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, tex_scales)
    glUniform1i(glGetUniformLocation(shader_program, "u_scales"), 1)

    tex_colors = create_texture_from_data(colors_padded, tex_width, tex_height, GL_RGBA32F, GL_RGBA, GL_FLOAT)
    glActiveTexture(GL_TEXTURE2)
    glBindTexture(GL_TEXTURE_2D, tex_colors)
    glUniform1i(glGetUniformLocation(shader_program, "u_colors"), 2)

    tex_rotations = create_texture_from_data(rotations_padded, tex_width, tex_height, GL_RGBA32F, GL_RGBA, GL_FLOAT)
    glActiveTexture(GL_TEXTURE3)
    glBindTexture(GL_TEXTURE_2D, tex_rotations)
    glUniform1i(glGetUniformLocation(shader_program, "u_rotations"), 3)

    # --- 设置用于实例化的顶点数据 ---
    # 1. 一个简单的四边形
    quad_vertices = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
    
    # 2. 每个实例的索引
    instance_indices = np.arange(num_gaussians, dtype=np.float32)

    # VAO, VBO
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    # 四边形顶点 VBO
    VBO_quad = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO_quad)
    glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0) # a_position
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

    # 实例索引 VBO
    VBO_indices = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO_indices)
    glBufferData(GL_ARRAY_BUFFER, instance_indices.nbytes, instance_indices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(1) # a_index
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, None)
    glVertexAttribDivisor(1, 1) # 关键：告诉OpenGL这是一个实例化属性

    glBindVertexArray(0)

    # --- 获取Uniform位置 ---
    proj_loc = glGetUniformLocation(shader_program, "u_proj")
    view_loc = glGetUniformLocation(shader_program, "u_view")
    focal_loc = glGetUniformLocation(shader_program, "u_focal")
    viewport_loc = glGetUniformLocation(shader_program, "u_viewport")

    # --- 相机和投影设置 ---
    fov = 60
    near, far = 0.1, 100
    projection_matrix = Matrix44.perspective_projection(fov, width / height, near, far)
    
    focal_x = (width / 2.0) / math.tan(math.radians(fov) / 2.0)
    focal_y = (height / 2.0) / math.tan(math.radians(fov) / 2.0)

    # 相机状态
    yaw, pitch = 0.0, 0.0
    distance = 4.0
    
    # --- 主循环 ---
    running = True
    mouse_down = False
    last_mouse_pos = (0, 0)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # 左键
                    mouse_down = True
                    last_mouse_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_down = False
            elif event.type == pygame.MOUSEMOTION:
                if mouse_down:
                    dx, dy = event.rel
                    yaw += dx * 0.2
                    pitch -= dy * 0.2
                    pitch = max(-89, min(89, pitch)) # 限制pitch
            elif event.type == pygame.MOUSEWHEEL:
                distance -= event.y * 0.5
                distance = max(1.0, distance) # 限制缩放

        # --- 更新相机视图矩阵 ---
        rotation_y = Matrix44.from_y_rotation(math.radians(yaw))
        rotation_x = Matrix44.from_x_rotation(math.radians(pitch))
        rotation_matrix = rotation_y * rotation_x
        
        # 计算相机位置（轨道相机）
        camera_pos = Vector3([0, 0, distance])
        camera_pos = rotation_matrix * camera_pos
        
        # 视图矩阵
        view_matrix = Matrix44.look_at(
            camera_pos,          # eye
            Vector3([0, 0, 0]),  # target
            Vector3([0, 1, 0])   # up
        )
        
        # --- 渲染 ---
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(shader_program)

        # 更新 Uniforms
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_matrix)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_matrix)
        glUniform2f(focal_loc, focal_x, focal_y)
        glUniform2f(viewport_loc, width, height)
        
        # 绑定VAO并绘制
        glBindVertexArray(VAO)
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, num_gaussians)
        glBindVertexArray(0)

        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()

if __name__ == '__main__':
    main()