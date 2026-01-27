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
out vec3 vConic;
out float vRadius;

void main() {
    vColor = quad_color;
    vPosition = position;

    // 构建旋转矩阵
    // rot 由 uint8 传入（0..255），这里按 convert.py 的编码还原到 [-1, 1] 并归一化
    vec4 q = (quad_rot - 128.0) / 128.0;
    q = normalize(q);
    // 许多 3DGS PLY 使用 (x,y,z,w) 顺序存储，这里按 w 在最后解码
    float r = q.w;
    float x = q.x;
    float y = q.y;
    float z = q.z;
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
    // 相机前方是负 Z，过滤掉相机后方点避免投影异常
    if (cam_pos.z >= -1e-4) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        vColor = vec4(0.0);
        vConic = vec3(0.0);
        vRadius = 0.0;
        return;
    }
    
    // 简单的透视投影近似 (EWA Splatting)
    float f = focal.x;
    float zc = -cam_pos.z;
    mat3 J = mat3(
        f / zc, 0, -(f * cam_pos.x) / (zc * zc),
        0, f / zc, -(f * cam_pos.y) / (zc * zc),
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

    // 传给片元：椭圆二次型系数（对称矩阵）
    float inv_det = 1.0 / max(det, 1e-6);
    vConic = vec3(
        cov2d[1][1] * inv_det,
        -cov2d[0][1] * inv_det,
        cov2d[0][0] * inv_det
    );
    vRadius = radius;

    // 先得到裁剪空间坐标，再按 w 缩放做屏幕空间扩展
    vec4 clip_pos = projection * cam_pos;
    clip_pos.xy += position * radius / viewport * 2.0 * clip_pos.w;
    gl_Position = clip_pos;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec4 vColor;
in vec2 vPosition;
in vec3 vConic;
in float vRadius;
out vec4 fragColor;

void main() {
    // 用椭圆二次型计算屏幕空间距离
    vec2 p = vPosition * vRadius;
    float d = vConic.x * p.x * p.x + 2.0 * vConic.y * p.x * p.y + vConic.z * p.y * p.y;
    if (d > 9.0) discard; // 3-sigma 裁剪
    float opacity = exp(-0.5 * d) * vColor.a;
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
        # rot 在 convert.py 中以 uint8 保存（0..255），这里不做归一化，交给 shader 解码
        glVertexAttribPointer(4, 4, GL_UNSIGNED_BYTE, GL_FALSE, stride, ctypes.c_void_p(28))
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
        # glm 的矩阵是列主序，这里用 value_ptr 按列主序取出，再按数学矩阵行列索引
        view_ptr = glm.value_ptr(view)
        view_mat = np.ctypeslib.as_array(view_ptr, shape=(16,)).reshape(4, 4, order="F")
        # 简单的 CPU 排序 (在大规模数据下会是瓶颈)
        positions = self.pos
        # view-space Z：用齐次坐标做完整变换，避免行/列主序混淆
        ones = np.ones((positions.shape[0], 1), dtype=positions.dtype)
        positions_h = np.concatenate([positions, ones], axis=1)
        cam_pos = positions_h @ view_mat.T
        z_values = cam_pos[:, 2]
        # 过滤相机后方点，避免投影异常导致的乱透
        front_mask = z_values < -1e-4
        front_indices = np.nonzero(front_mask)[0]
        z_front = z_values[front_mask]
        # 临时切换：使用 back-to-front（远到近）排序
        # OpenGL 视图空间里，镜头前方通常是负 Z，越远数值越小（更负）
        order = np.argsort(z_front)
        indices = front_indices[order]

        sorted_data = self.raw_data[indices]
        draw_count = sorted_data.shape[0]

        # 2. 更新 GPU 数据
        glBindBuffer(GL_ARRAY_BUFFER, self.splat_vbo)
        if draw_count > 0:
            glBufferSubData(GL_ARRAY_BUFFER, 0, sorted_data.nbytes, sorted_data)

        # 3. 绘制
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_BLEND)
        # 临时切换：标准预乘 alpha 混合（back-to-front）
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
        
        glUseProgram(self.shader)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_FALSE, glm.value_ptr(proj))
        # 根据投影矩阵计算像素焦距
        focal_x = 0.5 * width * proj[0][0]
        focal_y = 0.5 * height * proj[1][1]
        glUniform2f(glGetUniformLocation(self.shader, "focal"), focal_x, focal_y)
        glUniform2f(glGetUniformLocation(self.shader, "viewport"), width, height)

        glBindVertexArray(self.vao)
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, draw_count)

def main():
    if not glfw.init(): return
    window = glfw.create_window(1280, 720, "Python Splat Viewer", None, None)
    glfw.make_context_current(window)

    # 替换为你自己的 .splat 文件路径
    renderer = SplatRenderer("model.splat")

    # Simple orbit camera state (mouse drag + keyboard zoom)
    cam = {
        "yaw": 0.0,
        "pitch": 0.0,
        "distance": 100,
        "target": glm.vec3(0, 0, 0),
        "dragging": False,
        "last_x": 0.0,
        "last_y": 0.0,
    }

    def on_mouse_button(win, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            cam["dragging"] = (action == glfw.PRESS)
            if cam["dragging"]:
                x, y = glfw.get_cursor_pos(win)
                cam["last_x"], cam["last_y"] = x, y

    def on_cursor_pos(win, x, y):
        if not cam["dragging"]:
            return
        dx = x - cam["last_x"]
        dy = y - cam["last_y"]
        cam["last_x"], cam["last_y"] = x, y
        cam["yaw"] += dx * 0.01
        cam["pitch"] += dy * 0.01
        cam["pitch"] = max(-1.5, min(1.5, cam["pitch"]))

    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_cursor_pos_callback(window, on_cursor_pos)

    while not glfw.window_should_close(window):
        w, h = glfw.get_framebuffer_size(window)
        glViewport(0, 0, w, h)

        # 简单的相机矩阵
        # Keyboard controls: WASD/arrow rotate, Q/E zoom
        if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            cam["yaw"] -= 0.02
        if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            cam["yaw"] += 0.02
        if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS or glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            cam["pitch"] -= 0.02
        if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS or glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            cam["pitch"] += 0.02
        cam["pitch"] = max(-1.5, min(1.5, cam["pitch"]))
        if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
            cam["distance"] *= 0.98
        if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
            cam["distance"] *= 1.02
        cam["distance"] = max(1.0, min(50.0, cam["distance"]))

        cam_pos = glm.vec3(
            cam["distance"] * np.cos(cam["pitch"]) * np.sin(cam["yaw"]),
            cam["distance"] * np.sin(cam["pitch"]),
            cam["distance"] * np.cos(cam["pitch"]) * np.cos(cam["yaw"]),
        ) + cam["target"]
        view = glm.lookAt(cam_pos, cam["target"], glm.vec3(0, 1, 0))
        proj = glm.perspective(glm.radians(45), w/h, 0.1, 100.0)

        renderer.render(view, proj, w, h)
        
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
