import numpy as np
from OpenGL.GL import *
import glfw
import glm
import struct
import ctypes

# --- 着色器代码 (基于 antimatter15 的实现) ---
VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 position; // 矩形的四个顶点 [-1, -1, 1, 1]
layout (location = 1) in vec3 quad_pos;
layout (location = 2) in vec4 quad_color;
layout (location = 3) in vec3 quad_scale;
layout (location = 4) in vec4 quad_rot;

uniform mat4 projection;
uniform mat4 view;
uniform vec2 focal;
uniform vec2 viewport;

out vec4 vColor;
out vec2 vPosition;

void main() {
    vColor = quad_color;
    vPosition = position;

    // 构建旋转矩阵
    float r = quad_rot.x;
    float x = quad_rot.y;
    float y = quad_rot.z;
    float z = quad_rot.w;
    mat3 R = mat3(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y),
        2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x),
        2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y)
    );

    // 构建缩放矩阵
    mat3 S = mat3(quad_scale.x, 0, 0, 0, quad_scale.y, 0, 0, 0, quad_scale.z);
    mat3 M = R * S;
    mat3 Vrk = M * transpose(M);

    // 将 3D 协方差投影到 2D
    vec4 cam_pos = view * vec4(quad_pos, 1.0);
    
    // 简单的透视投影近似 (EWA Splatting)
    float f = focal.x;
    mat3 J = mat3(
        f / cam_pos.z, 0, -(f * cam_pos.x) / (cam_pos.z * cam_pos.z),
        0, f / cam_pos.z, -(f * cam_pos.y) / (cam_pos.z * cam_pos.z),
        0, 0, 0
    );
    mat3 W = mat3(view);
    mat3 T = J * W;
    mat3 cov2d = T * Vrk * transpose(T);

    // 加上微小的膨胀以防止数值不稳定
    cov2d[0][0] += 0.3;
    cov2d[1][1] += 0.3;

    // 计算 2D 椭圆的特征向量和大小
    float det = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[0][1];
    float mid = 0.5 * (cov2d[0][0] + cov2d[1][1]);
    float lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    float radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));

    gl_Position = projection * cam_pos + vec4(position * radius / viewport * 2.0, 0.0, 0.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec4 vColor;
in vec2 vPosition;
out vec4 fragColor;

void main() {
    float d = dot(vPosition, vPosition);
    if (d > 1.0) discard;
    float opacity = exp(-0.5 * d * 16.0) * vColor.a;
    fragColor = vec4(vColor.rgb * opacity, opacity);
}
"""

class SplatRenderer:
    def __init__(self, splat_path):
        self.load_splat(splat_path)
        self.init_opengl()

    def load_splat(self, path):
        # 读取 .splat 文件 (每个点 32 字节: pos(3f), scale(3f), color(4b), rot(4b))
        data = open(path, "rb").read()
        self.count = len(data) // 32
        self.vertex_data = np.frombuffer(data, dtype=np.float32).reshape(-1, 8)
        
        # 提取数据用于排序和渲染
        self.pos = self.vertex_data[:, 0:3].copy()
        # 预先处理颜色和旋转 (在 Python 中转换，或者直接传原始字节到 GPU)
        self.raw_data = np.frombuffer(data, dtype=np.uint8).reshape(-1, 32)

    def init_opengl(self):
        # 编译着色器
        self.shader = self.compile_shader(VERTEX_SHADER, FRAGMENT_SHADER)
        glUseProgram(self.shader)

        # 准备一个标准的矩形 (Quad) 用于实例化渲染
        quad_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, quad_vbo)
        vertices = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, None)
        glEnableVertexAttribArray(0)

        # 创建用于存放 Splat 数据的 VBO
        self.splat_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.splat_vbo)
        # 这里动态更新，先分配空间
        glBufferData(GL_ARRAY_BUFFER, self.raw_data.nbytes, None, GL_DYNAMIC_DRAW)

        # 设置顶点属性指针 (实例化数据)
        # pos: 3f32, scale: 3f32, color: 4u8, rot: 4u8
        stride = 32
        glEnableVertexAttribArray(1) # pos
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glVertexAttribDivisor(1, 1)

        glEnableVertexAttribArray(2) # color
        glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, stride, ctypes.c_void_p(24))
        glVertexAttribDivisor(2, 1)

        glEnableVertexAttribArray(3) # scale
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glVertexAttribDivisor(3, 1)

        glEnableVertexAttribArray(4) # rot
        glVertexAttribPointer(4, 4, GL_BYTE, GL_TRUE, stride, ctypes.c_void_p(28))
        glVertexAttribDivisor(4, 1)

    def compile_shader(self, vs_src, fs_src):
        def _compile(src, type):
            s = glCreateShader(type)
            glShaderSource(s, src)
            glCompileShader(s)
            if not glGetShaderiv(s, GL_COMPILE_STATUS):
                print(glGetShaderInfoLog(s))
            return s
        prog = glCreateProgram()
        glAttachShader(prog, _compile(vs_src, GL_VERTEX_SHADER))
        glAttachShader(prog, _compile(fs_src, GL_FRAGMENT_SHADER))
        glLinkProgram(prog)
        return prog

    def render(self, view, proj, width, height):
        # 1. 深度排序 (3DGS 必须从后往前渲染以保证透明度正确)
        # 计算每个点在 view 空间的 Z 值
        view_mat = np.array(view).reshape(4,4)
        # 简单的 CPU 排序 (在大规模数据下会是瓶颈)
        positions = self.pos
        # 投影 Z
        z_values = positions @ view_mat[:3, 2] + view_mat[3, 2]
        indices = np.argsort(z_values) # 从远到近

        sorted_data = self.raw_data[indices]

        # 2. 更新 GPU 数据
        glBindBuffer(GL_ARRAY_BUFFER, self.splat_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, sorted_data.nbytes, sorted_data)

        # 3. 绘制
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE_MINUS_DST_ALPHA, GL_ONE) # 注意：3DGS 使用特殊的混合方式
        
        glUseProgram(self.shader)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_FALSE, glm.value_ptr(proj))
        glUniform2f(glGetUniformLocation(self.shader, "focal"), 1100, 1100) # 示例焦距
        glUniform2f(glGetUniformLocation(self.shader, "viewport"), width, height)

        glBindVertexArray(self.vao)
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, self.count)

def main():
    if not glfw.init(): return
    window = glfw.create_window(1280, 720, "Python Splat Viewer", None, None)
    glfw.make_context_current(window)

    # 替换为你自己的 .splat 文件路径
    renderer = SplatRenderer("model.splat")

    while not glfw.window_should_close(window):
        w, h = glfw.get_framebuffer_size(window)
        glViewport(0, 0, w, h)

        # 简单的相机矩阵
        time = glfw.get_time()
        view = glm.lookAt(glm.vec3(np.sin(time)*5, 2, np.cos(time)*5), glm.vec3(0,0,0), glm.vec3(0,1,0))
        proj = glm.perspective(glm.radians(45), w/h, 0.1, 100.0)

        renderer.render(view, proj, w, h)
        
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
