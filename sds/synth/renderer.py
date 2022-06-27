from enum import Enum
from typing import Optional, Dict, List, Any, Tuple

import cv2
import glfw
import numpy as np
import pymesh
from scipy.spatial.transform import Rotation as R

from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLUT import *
from OpenGL.GLU import *


class OutputType(Enum):
    Normals = 0
    Noc = 1
    Overlay = 2


Color4i = Tuple[int, int, int, int]
Color4f = Tuple[float, float, float, float]


def color_to_float(col: Color4i) -> Color4f:
    return col[0] / 255, col[1] / 255, col[2] / 255, col[3] / 255


class ProgContainer:
    def __init__(self):
        self.vertex_shader = shaders.compileShader("""
        #version 330 core
        const float M_PI = 3.1415926535897932384626433832795;
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;
        uniform mat4 obj_cam_mat;
        uniform mat4 proj_mat;
        uniform int draw_type; // 0 - normals, 1 - noc, 2 - overlay
        uniform float max_dist_to_center;
        uniform vec4 obj_color;

        out vec4 color_out;
        out vec4 pos_out;

        void main()
        {
            pos_out = obj_cam_mat * vec4(position, 1.0);

            gl_Position = proj_mat * pos_out;
            mat3 rot = mat3(obj_cam_mat);
            vec3 col;

            if (draw_type == 0) {
                vec3 norm_pos = rot * normal;
                col = norm_pos / 2.0 + 0.5;
                color_out = vec4(col, 1);
            } else if (draw_type == 1) {
                vec3 cam_pos = rot * position;
                col = -cam_pos / max_dist_to_center / 2.0 + 0.5;
                color_out = vec4(col, 1);
            } else if (draw_type == 2) {
                color_out = obj_color;
            }
        }
        """, GL_VERTEX_SHADER)

        self.geometry_shader = shaders.compileShader("""
        #version 330 core
        layout (triangles) in;
        layout (triangle_strip, max_vertices = 3) out;

        uniform mat4 obj_cam_mat;
        uniform int draw_type;
        in vec4 color_out[3];
        in vec4 pos_out[3];
        out vec4 v_color;

        void main()
        {
            if (draw_type == 0) {
                vec3 t1 = pos_out[1].xyz - pos_out[0].xyz;        
                vec3 t2 = pos_out[2].xyz - pos_out[0].xyz;
                if (length(t1) < 1e-7 || length(t2) < 1e-7) {
                    v_color = vec4(0.0, 0.0, 0.0, 1.0);
                } else {
                    vec3 n = normalize(cross(t1, t2));
                    vec3 center = (pos_out[0].xyz + pos_out[1].xyz + pos_out[2].xyz) / 3;
                    //vec3 to_center = normalize(obj_cam_mat[3].xyz - center);
                    vec3 to_center = normalize(-center);
                    if (dot(n, to_center) < 0) {
                        n = -n;
                    }
                    // v_color = vec4(n * 0.5 + 0.5, 1.0);
                    float r = min(max(n.x / 2 + 0.5, 0), 1.0);
                    float g = min(max(n.y / 2 + 0.5, 0), 1.0);
                    float b = min(max(n.z / 2 + 0.5, 0), 1.0);
                    v_color = vec4(r, g, b, 1.0);
                }
                for (int i = 0; i < 3; i++) {
                    gl_Position = gl_in[i].gl_Position;
                    EmitVertex();
                }
                EndPrimitive();
            } else if (draw_type == 1 || draw_type == 2) {
                for (int i = 0; i < 3; i++) {
                    gl_Position = gl_in[i].gl_Position;
                    v_color = color_out[i];
                    EmitVertex();
                }
                EndPrimitive();
            }
        }
        """, GL_GEOMETRY_SHADER)

        self.fragment_shader = shaders.compileShader("""
        #version 330 core
        precision highp float;
        in vec4 v_color;
        out vec4 outputColor;
        void main()
        {
            outputColor = v_color;
        }
        """, GL_FRAGMENT_SHADER)
        self.program = shaders.compileProgram(self.vertex_shader, self.geometry_shader, self.fragment_shader)
        glUseProgram(self.program)

        self.obj_cam_mat_loc = glGetUniformLocation(self.program, 'obj_cam_mat')
        self.proj_mat_loc = glGetUniformLocation(self.program, 'proj_mat')
        self.draw_type_loc = glGetUniformLocation(self.program, 'draw_type')
        self.obj_color_loc = glGetUniformLocation(self.program, 'obj_color')
        self.max_dist_to_center_loc = glGetUniformLocation(self.program, 'max_dist_to_center')

    def use(self):
        glUseProgram(self.program)


class MeshObj:
    def __init__(self, verts: np.ndarray, normals: np.ndarray, faces: np.ndarray, program: ProgContainer):
        self.program = program
        verts = verts.astype(np.float32)
        norms = normals.astype(np.float32)
        self.verts_norms = np.concatenate([verts, norms], axis=1).astype(np.float32)
        self.faces = faces.astype(np.uint32)
        self.max_dist_to_center = np.max(np.linalg.norm(verts, axis=1))

        self.program.use()

        self.buffers = glGenBuffers(2)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[0])
        glBufferData(GL_ARRAY_BUFFER, self.verts_norms.size * 4, self.verts_norms, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.buffers[1])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.faces, GL_STATIC_DRAW)

    def draw(self, obj_cam_mat: np.ndarray, out_type: OutputType, obj_color: Optional[Color4i] = None):
        self.program.use()

        proj_mat = glGetFloat(GL_PROJECTION_MATRIX)
        glUniformMatrix4fv(self.program.obj_cam_mat_loc, 1, True, obj_cam_mat.astype(np.float32))
        glUniformMatrix4fv(self.program.proj_mat_loc, 1, False, proj_mat.astype(np.float32))
        glUniform1i(self.program.draw_type_loc, out_type.value)
        glUniform1f(self.program.max_dist_to_center_loc, self.max_dist_to_center)
        if out_type == OutputType.Overlay and obj_color is not None:
            glUniform4f(self.program.obj_color_loc, *color_to_float(obj_color))

        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[0])
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.buffers[1])
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        glDrawElements(GL_TRIANGLES, self.faces.size, GL_UNSIGNED_INT, None)


class Renderer:
    def __init__(self, models: Dict[str, Dict], win_size: Tuple[int, int] = (640, 480), title: str = 'Renderer'):
        self.models = models
        self.title = title
        self.width, self.height = win_size
        self.cam_mat = None

        if not glfw.init():
            raise Exception(f'Error. Cannot init GLFW!')
        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            raise Exception(f'Error. Cannot create window of size: {self.width}x{self.height}')
        glfw.make_context_current(self.window)
        glViewport(0, 0, self.width, self.height)
        glClearColor(0, 0, 0, 0)
        glEnable(GL_DEPTH_TEST)
        glfw.set_window_size_callback(self.window, self._on_window_resize)
        glfw.set_framebuffer_size_callback(self.window, self._on_framebuffer_resize)

        self.prog = ProgContainer()

        self.mesh_objs = self.create_mesh_objs()
        self.cv_to_opengl_mat = np.eye(4, dtype=np.float32)
        self.cv_to_opengl_mat[:3, :3] = R.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix()

    def _on_window_resize(self, win, width, height):
        print(f'_on_window_resize: {width}x{height}')

    def _on_framebuffer_resize(self, win, width, height):
        print(f'_on_framebuffer_resize: {width}x{height}')

    def init(self):
        glfw.swap_buffers(self.window)

    def create_mesh_objs(self) -> Dict[str, MeshObj]:
        res = {}
        for glob_id, obj in self.models.items():
            mesh: pymesh.Mesh = obj['mesh']
            verts, faces = mesh.vertices, mesh.faces
            normals_attr_name = 'vertex_normal'
            if not mesh.has_attribute(normals_attr_name):
                mesh.add_attribute('vertex_normal')
            normals = mesh.get_vertex_attribute('vertex_normal')
            res[glob_id] = MeshObj(verts, normals, faces, self.prog)
        return res

    def set_window_size(self, win_size: Tuple[int, int]):
        width, height = win_size
        if (self.width, self.height) == (width, height):
            return
        self.width, self.height = width, height
        glfw.set_window_size(self.window, self.width, self.height)
        glViewport(0, 0, self.width, self.height)
        for _ in range(2):
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glfw.swap_buffers(self.window)

    def set_camera_matrix(self, cam_mat: np.ndarray):
        if self.cam_mat is not None and np.allclose(self.cam_mat, cam_mat):
            return
        self.cam_mat = cam_mat
        fovy_half_tan = (self.height / 2) / self.cam_mat[1, 1]
        fovy = np.arctan(fovy_half_tan) * 2 * (180 / np.pi)
        print(f'Fov y: {fovy:.2f}')
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(fovy, self.width / self.height, 0.01, 10)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def gen_colors(self, cam_mat: np.ndarray, objs: Dict[str, Any], out_type: OutputType, obj_color: Optional[Color4i] = None) -> np.ndarray:
        self.set_camera_matrix(cam_mat)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        for obj in objs.values():
            mesh_obj = self.mesh_objs[obj['glob_id']]
            obj_cam_mat = obj['H_m2c']
            obj_cam_mat = self.cv_to_opengl_mat @ obj_cam_mat

            mesh_obj.draw(obj_cam_mat, out_type, obj_color)

        glfw.swap_buffers(self.window)

        image_type = GL_RGBA if out_type == OutputType.Overlay else GL_RGB
        image_buffer = glReadPixels(0, 0, self.width, self.height, image_type, GL_UNSIGNED_BYTE)
        image = np.frombuffer(image_buffer, dtype=np.uint8).reshape((self.height, self.width, -1))
        image = cv2.flip(image, 0)

        # imgf_src = (image.astype(np.float32) - 127.5) / 127.5
        # imgf = np.linalg.norm(imgf_src, axis=-1)
        # print(f'!!! {imgf.min()}, {imgf.max()}, {imgf.mean()}')
        # imgf_src /= imgf[..., None]
        # imgf_src = (imgf_src * 127.5 + 127.5).astype(np.uint8)
        # imgf_src = cv2.cvtColor(imgf_src, cv2.COLOR_RGB2BGR)
        # cv2.imshow('test', imgf_src)

        return image
