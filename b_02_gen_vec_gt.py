import json
import sys
import time
from enum import Enum
from pathlib import Path
import shutil
from typing import Optional, Dict, List, Any, Tuple

import cv2
import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLUT import *
from OpenGL.GLU import *
import h5py
import yaml

import numpy as np
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
import pymesh
from scipy.spatial.transform import Rotation as R


from sds.utils import utils


class OutputType(Enum):
    Normals = 'normals'
    Noc = 'noc'


class Config(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    sds_root_path: Path = Field(
        ...,
        description='Path to SDS datasets (containing datasets: ITODD, TLESS, etc.)',
        cli=('--sds-root-path',),
    )
    target_dataset_name: str = Field(
        ...,
        description='Target dataset name. Has to be a subdirectory of SDS_ROOT_PATH, one of: "itodd", "tless", etc.',
        cli=('--target-dataset-name',),
    )
    distractor_dataset_name: str = Field(
        ...,
        description='Distractor dataset name. Has to be a subdirectory of SDS_ROOT_PATH, one of: "itodd", "tless", etc.',
        cli=('--distractor-dataset-name',),
    )
    models_subdir: str = Field(
        'models',
        description='Models subdirectory. Has to contain ply files (default: "models")',
        required=False,
        cli=('--models-subdir',),
    )
    output_type: OutputType = Field(
        ...,
        description=f'Output type. {OutputType.Normals.value} - normals in camera frame scaled in order to fit (0, 1) '
                    f'interval. {OutputType.Noc.value} - Normalized object coordinates in camera frame. Coordinates '
                    f'are taken relatively to object frame\'s center and scaled in order to fit (0, 1)',
        required=True,
        cli=('--output-type',),
    )
    debug: bool = Field(
        False,
        description='Debug mode. Renderings are calculated and visualized but not saved',
        required=False,
        cli=('--debug',),
    )


def load_models(models_path: Path) -> Dict[str, Dict]:
    ds_name = models_path.parent.name
    models = utils.read_yaml(models_path / 'models.yaml')
    res = {}
    for obj_id, obj in models.items():
        obj_fpath = models_path / f'{obj_id}.ply'
        glob_id = f'{ds_name}_{obj_id}'
        res[glob_id] = {
            **obj,
            'mesh': pymesh.load_mesh(obj_fpath.as_posix()),
        }
    return res


def list_dataset(data_path: Path) -> Dict[str, Dict]:
    scenes = {}
    n_items = 0
    for scene_path in data_path.iterdir():
        if not scene_path.is_dir():
            continue
        scene_fpaths = []
        for fpath in scene_path.iterdir():
            if not (fpath.is_file() and fpath.suffix == '.hdf5'):
                continue
            scene_fpaths.append(fpath)
        scene = {
            'id': scene_path.name,
            'path': scene_path,
            'items': scene_fpaths,
            'size': len(scene_fpaths)
        }
        scenes[scene_path.name] = scene
        n_items += len(scene_fpaths)

    res = {
        'path': data_path,
        'size': n_items,
        'scenes': scenes,
    }
    return res


def read_gt(hdf5_fpath: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], Dict[str, Dict]]:
    with h5py.File(hdf5_fpath.as_posix(), 'r') as f:
        gt_str = f['gt'][...].item().decode('utf-8')
        gt = json.loads(gt_str)
        img = f['colors'][...]

    cam_mat, img_size = np.array(gt['camera']['K']), tuple(gt['camera']['image_size'])
    objs = gt['objects']
    for obj in objs.values():
        obj['H_m2c'] = np.array(obj['H_m2c'])

    return img, cam_mat, img_size, objs


class ProgContainer:
    def __init__(self):
        self.vertex_shader = shaders.compileShader("""
        #version 330 core
        const float M_PI = 3.1415926535897932384626433832795;
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;
        uniform mat4 obj_cam_mat;
        uniform mat4 proj_mat;
        uniform bool draw_normals;
        uniform float max_dist_to_center;

        out vec4 color_out;
        out vec4 pos_out;

        void main()
        {
            pos_out = obj_cam_mat * vec4(position, 1.0);
            
            gl_Position = proj_mat * pos_out;
            mat3 rot = mat3(obj_cam_mat);
            vec3 col;

            if (draw_normals) {
                vec3 norm_pos = rot * normal;
                col = norm_pos / 2.0 + 0.5;
            } else {
                vec3 cam_pos = rot * position;
                col = -cam_pos / max_dist_to_center / 2.0 + 0.5;
            }

            color_out = vec4(col, 1);
        }
        """, GL_VERTEX_SHADER)

        self.geometry_shader = shaders.compileShader("""
        #version 330 core
        layout (triangles) in;
        layout (triangle_strip, max_vertices = 3) out;

        in vec4 color_out[3];
        in vec4 pos_out[3];
        uniform bool draw_normals;
        out vec4 v_color;

        void main()
        {
            if (draw_normals) {
                vec3 t1 = pos_out[1].xyz - pos_out[0].xyz;        
                vec3 t2 = pos_out[2].xyz - pos_out[0].xyz;
                if (length(t1) < 1e-6 || length(t2) < 1e-6) {
                    v_color = vec4(0.0, 0.0, 0.0, 1.0);
                } else {
                    vec3 n = normalize(cross(t1, t2));
                    if (n.z < 0) {
                        n = -n;
                    }
                    v_color = vec4(n * 0.5 + 0.5, 1.0);
                }
                for (int i = 0; i < 3; i++) {
                    gl_Position = gl_in[i].gl_Position;
                    EmitVertex();
                }
                EndPrimitive();
            } else {
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
        self.draw_normals_loc = glGetUniformLocation(self.program, 'draw_normals')
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

    def draw(self, obj_cam_mat: np.ndarray, out_type: OutputType):
        self.program.use()

        proj_mat = glGetFloat(GL_PROJECTION_MATRIX)
        glUniformMatrix4fv(self.program.obj_cam_mat_loc, 1, True, obj_cam_mat.astype(np.float32))
        glUniformMatrix4fv(self.program.proj_mat_loc, 1, False, proj_mat.astype(np.float32))
        glUniform1i(self.program.draw_normals_loc, out_type == OutputType.Normals)
        glUniform1f(self.program.max_dist_to_center_loc, self.max_dist_to_center)

        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[0])
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.buffers[1])
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        glDrawElements(GL_TRIANGLES, self.faces.size, GL_UNSIGNED_INT, None)


class Renderer:
    def __init__(self, models: Dict[str, Dict], win_size: Tuple[int, int] = (640, 480), title: str = 'Renderer', debug: bool = False):
        self.models = models
        self.title = title
        self.width, self.height = win_size
        self.debug = debug
        self.cam_mat = None

        if not glfw.init():
            raise Exception(f'Error. Cannot init GLFW!')
        # glfw.window_hint(glfw.VISIBLE, self.debug)
        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            raise Exception(f'Error. Cannot create window of size: {self.width}x{self.height}')
        glfw.make_context_current(self.window)
        glViewport(0, 0, self.width, self.height)
        glClearColor(0, 0, 0, 0)
        glEnable(GL_DEPTH_TEST)

        self.prog = ProgContainer()

        self.mesh_objs = self.create_mesh_objs()
        self.cv_to_opengl_mat = np.eye(4, dtype=np.float32)
        self.cv_to_opengl_mat[:3, :3] = R.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix()

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
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glfw.swap_buffers(self.window)
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

    def gen_colors(self, cam_mat: np.ndarray, objs: Dict[str, Any], out_type: OutputType) -> np.ndarray:
        self.set_camera_matrix(cam_mat)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        for obj in objs.values():
            mesh_obj = self.mesh_objs[obj['glob_id']]
            obj_cam_mat = obj['H_m2c']
            obj_cam_mat = self.cv_to_opengl_mat @ obj_cam_mat

            mesh_obj.draw(obj_cam_mat, out_type)

        glfw.swap_buffers(self.window)

        image_buffer = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(image_buffer, dtype=np.uint8).reshape((self.height, self.width, 3))
        image = cv2.flip(image, 0)
        return image


def main(cfg: Config) -> int:
    print(cfg)
    target_ds_path = cfg.sds_root_path / cfg.target_dataset_name
    dist_ds_path = cfg.sds_root_path / cfg.distractor_dataset_name
    target_models_path = target_ds_path / cfg.models_subdir
    dist_models_path = dist_ds_path / cfg.models_subdir

    target_models = load_models(target_models_path)
    dist_models = load_models(dist_models_path)
    models = {
        **target_models,
        **dist_models,
    }

    data_postifx = ''
    data_path = target_ds_path / f'data{data_postifx}'
    data = list_dataset(data_path)
    print(f'Number of scenes: {len(data["scenes"])}. Files total: {data["size"]}')

    renderer = Renderer(models=models, debug=cfg.debug)
    renderer.init()

    dst_root_path = data_path.parent / f'{data_path.name}_{cfg.output_type.value}'
    dst_root_path.mkdir(parents=True, exist_ok=True)

    for scene in data['scenes'].values():
        scene_path: Path = scene['path']
        dst_scene_path = dst_root_path / f'{scene_path.name}'
        dst_scene_path.mkdir(parents=True, exist_ok=True)
        for fpath in scene['items']:
            print(fpath)
            img, cam_mat, img_size, objs = read_gt(fpath)
            renderer.set_window_size(img_size)

            colors = renderer.gen_colors(cam_mat, objs, cfg.output_type)
            colors = cv2.cvtColor(colors, cv2.COLOR_RGB2BGR)

            if cfg.debug:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow('img', img)
                cv2.imshow('Colors', colors)
                if cv2.waitKey() in (27, ord('q')):
                    sys.exit(0)
            else:
                dst_fpath = dst_scene_path / fpath.with_suffix('.png').name
                print(f'Saving output to {dst_fpath}')
                cv2.imwrite(dst_fpath.as_posix(), colors)

    return 0


if __name__ == '__main__':
    run_and_exit(Config, main, 'Script adding files to the dataset containing GT vector data: normal maps, '
                               'vectors from surface to object\'s center')

