import blenderproc as bproc

import argparse
import itertools
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, Tuple, List, Optional, Union
import time

import h5py
from ilock import ILock
import numpy as np
from pydantic_yaml import YamlModel

import bpy
from mathutils import Matrix

from blenderproc.python.camera import CameraUtility
from blenderproc.python.renderer.SegMapRendererUtility import _colorize_objects_for_instance_segmentation
from blenderproc.python.types.MeshObjectUtility import MeshObject
from blenderproc.python.utility.BlenderUtility import get_all_blender_mesh_objects, load_image
from blenderproc.python.utility.Utility import Utility, UndoAfterExecution
from blenderproc.python.writer.WriterUtility import _WriterUtility
from blenderproc.python.renderer import RendererUtility
from blenderproc.python.material import MaterialLoaderUtility

print(os.environ)
desired_gpu_ids = os.environ.get('DESIRED_GPU_IDS')
if desired_gpu_ids is not None:
    desired_gpu_ids = [int(gpu_id) for gpu_id in desired_gpu_ids.split(',')]
bproc.init(desired_gpu_ids=desired_gpu_ids)


def render_segmap(obj_key: str,
                  output_dir: Optional[str] = None, temp_dir: Optional[str] = None, file_prefix: str = "segmap_",
                  output_key: str = "segmap", segcolormap_output_file_prefix: str = "instance_attribute_map_",
                  segcolormap_output_key: str = "segcolormap", use_alpha_channel: bool = False,
                  render_colorspace_size_per_dimension: int = 2048) -> \
        Dict[str, Union[np.ndarray, List[np.ndarray]]]:
    """ Renders segmentation maps for all frames

    :param output_dir: The directory to write images to.
    :param temp_dir: The directory to write intermediate data to.
    :param file_prefix: The prefix to use for writing the images.
    :param output_key: The key to use for registering the output.
    :param segcolormap_output_file_prefix: The prefix to use for writing the segmentation-color map csv.
    :param segcolormap_output_key: The key to use for registering the segmentation-color map output.
    :param use_alpha_channel: If true, the alpha channel stored in .png textures is used.
    :param render_colorspace_size_per_dimension: As we use float16 for storing the rendering, the interval of \
                                                 integers which can be precisely stored is [-2048, 2048]. As \
                                                 blender does not allow negative values for colors, we use \
                                                 [0, 2048] ** 3 as our color space which allows ~8 billion \
                                                 different colors/objects. This should be enough.
    :return: dict of lists of segmaps and (for instance segmentation) segcolormaps
    """

    if output_dir is None:
        output_dir = Utility.get_temporary_directory()
    if temp_dir is None:
        temp_dir = Utility.get_temporary_directory()

    with UndoAfterExecution():
        RendererUtility.render_init()
        # the amount of samples must be one and there can not be any noise threshold
        RendererUtility.set_max_amount_of_samples(1)
        RendererUtility.set_noise_threshold(0)
        RendererUtility.set_denoiser(None)
        RendererUtility.set_light_bounces(1, 0, 0, 1, 0, 8, 0)

        # Get objects with meshes (i.e. not lights or cameras)
        objs_with_mats = get_all_blender_mesh_objects()

        colors, num_splits_per_dimension, objects = _colorize_objects_for_instance_segmentation(
            objs_with_mats, use_alpha_channel, render_colorspace_size_per_dimension)

        bpy.context.scene.cycles.filter_width = 0.0

        if use_alpha_channel:
            MaterialLoaderUtility.add_alpha_channel_to_textures(blurry_edges=False)

        # Determine path for temporary and for final output
        temporary_segmentation_file_path = os.path.join(temp_dir, "seg_")
        final_segmentation_file_path = os.path.join(output_dir, file_prefix)

        RendererUtility.set_output_format("OPEN_EXR", 16)
        RendererUtility.render(temp_dir, "seg_", None, return_data=False)

        # Find optimal dtype of output based on max index
        for dtype in [np.uint8, np.uint16, np.uint32]:
            optimal_dtype = dtype
            if np.iinfo(optimal_dtype).max >= len(colors) - 1:
                break

        return_dict: Dict[str, Union[List[np.ndarray], List[Dict]]] = {}

        # After rendering
        for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):  # for each rendered frame
            file_path = temporary_segmentation_file_path + ("%04d" % frame) + ".exr"
            segmentation = load_image(file_path)
            print(file_path, segmentation.shape)

            segmap = Utility.map_back_from_equally_spaced_equidistant_values(
                segmentation, num_splits_per_dimension, render_colorspace_size_per_dimension)
            segmap = segmap.astype(optimal_dtype)

            key_to_num = {}
            segmap_vals = set(np.unique(segmap))
            for ind, obj in enumerate(objs_with_mats):
                obj_num = ind + 1
                if obj_num not in segmap_vals:
                    continue
                if obj_key in obj:
                    key_to_num[obj[obj_key]] = obj_num
                else:
                    segmap[segmap == obj_num] = 0
            return_dict.setdefault(f'segmap_key_to_num', []).append(key_to_num)

            return_dict.setdefault(f'segmap', []).append(segmap)
            fname = final_segmentation_file_path + ("%04d" % frame)
            np.save(fname, segmap)

        Utility.register_output(output_dir, file_prefix, output_key, ".npy", "2.0.0")

    return return_dict


def write_hdf5(output_dir_path: str, next_file_num: int, output_data_dict: Dict[str, List[Union[np.ndarray, list, dict]]],
               append_to_existing_output: bool = False, stereo_separate_keys: bool = False):
    """
    Saves the information provided inside of the output_data_dict into a .hdf5 container

    :param output_dir_path: The folder path in which the .hdf5 containers will be generated
    :param output_data_dict: The container, which keeps the different images, which should be saved to disc.
                             Each key will be saved as its own key in the .hdf5 container.
    :param append_to_existing_output: If this is True, the output_dir_path folder will be scanned for pre-existing
                                      .hdf5 containers and the numbering of the newly added containers, will start
                                      right where the last run left off.
    :param stereo_separate_keys: If this is True and the rendering was done in stereo mode, than the stereo images
                                 won't be saved in one tensor [2, img_x, img_y, channels], where the img[0] is the
                                 left image and img[1] the right. They will be saved in separate keys: for example
                                 for colors in colors_0 and colors_1.
    """

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    amount_of_frames = 0
    for data_block in output_data_dict.values():
        if isinstance(data_block, list):
            amount_of_frames = max([amount_of_frames, len(data_block)])

    # # if append to existing output is turned on the existing folder is searched for the highest occurring
    # # index, which is then used as starting point for this run
    # if append_to_existing_output:
    #     frame_offset = 0
    #     # Look for hdf5 file with highest index
    #     for path in os.listdir(output_dir_path):
    #         if path.endswith(".hdf5"):
    #             index = path[:-len(".hdf5")]
    #             if index.isnumeric():
    #                 frame_offset = max(frame_offset, int(index) + 1)
    # else:
    #     frame_offset = 0
    frame_offset = next_file_num

    if amount_of_frames != bpy.context.scene.frame_end - bpy.context.scene.frame_start:
        raise Exception("The amount of images stored in the output_data_dict does not correspond with the amount"
                        "of images specified by frame_start to frame_end.")

    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):
        # for each frame a new .hdf5 file is generated
        frame_no = frame - bpy.context.scene.frame_start + frame_offset
        hdf5_path = os.path.join(output_dir_path, f'{frame_no:06d}.hdf5')
        with h5py.File(hdf5_path, "w") as file:
            # Go through all the output types
            print(f"Merging data for frame {frame} into {hdf5_path}")

            adjusted_frame = frame - bpy.context.scene.frame_start
            for key, data_block in output_data_dict.items():
                if adjusted_frame < len(data_block):
                    # get the current data block for the current frame
                    used_data_block = data_block[adjusted_frame]
                    if stereo_separate_keys and (bpy.context.scene.render.use_multiview or
                                                 used_data_block.shape[0] == 2):
                        # stereo mode was activated
                        _WriterUtility.write_to_hdf_file(file, key + "_0", data_block[adjusted_frame][0])
                        _WriterUtility.write_to_hdf_file(file, key + "_1", data_block[adjusted_frame][1])
                    else:
                        _WriterUtility.write_to_hdf_file(file, key, data_block[adjusted_frame])
                else:
                    raise Exception(f"There are more frames {adjusted_frame} then there are blocks of information "
                                    f" {len(data_block)} in the given list for key {key}.")
            blender_proc_version = Utility.get_current_version()
            if blender_proc_version is not None:
                _WriterUtility.write_to_hdf_file(file, "blender_proc_version", np.string_(blender_proc_version))


class SceneConfig(YamlModel):
    images_num: int
    target_objects_num: int
    distractor_objects_num: int
    max_duplicate_objects: int


class GenConfig(YamlModel):
    room_size: int
    scenes_num: int
    depth_antialiasing: bool
    render_samples: int
    image_size: Tuple[int, int]
    scene: SceneConfig


def scene_name_from_no(scene_no: int) -> str:
    return f'{scene_no:06d}'


class Config(YamlModel):
    sds_root_path: Path
    target_dataset_name: str
    distractor_dataset_name: str
    src_path: Path
    cc_textures_path: Path
    output_path: Optional[Path]
    output_postfix: Optional[str]
    images_per_dir: int
    generation: GenConfig

    @staticmethod
    def load(config_path: Path) -> 'Config':
        print(f'Reading config from {config_path}')
        with open(config_path, 'r') as f:
            cfg = Config.parse_raw(f.read())
            if cfg.output_path is None:
                output_postfix = cfg.output_postfix or ''
                if output_postfix and not output_postfix.startswith('_'):
                    output_postfix = f'_{output_postfix}'
                cfg.output_path = cfg.sds_root_path / cfg.target_dataset_name / f'data{output_postfix}'
        return cfg


class GlobObj:
    meshes: List[MeshObject]
    active_meshes: List[MeshObject]

    def __init__(self, mesh: MeshObject, ds_name: str, obj_id: str):
        self.meshes = [mesh]
        self.active_meshes = []
        self.ds_name = ds_name
        self.obj_id = obj_id
        self.glob_id = f'{self.ds_name}_{self.obj_id}'

        self.mesh_instances_num = 0
        self.prepare_mesh(mesh)

        self.active_meshes = []

    def prepare_mesh(self, mesh: MeshObject):
        self.mesh_instances_num += 1
        full_id = f'{self.glob_id}_{self.mesh_instances_num:02d}'
        mesh.set_cp('full_id', full_id)
        mesh.set_shading_mode('auto')
        mesh.enable_rigidbody(True, mass=1.0, friction=100.0, linear_damping=0.99,
                              angular_damping=0.99, collision_margin=0.0005)

    def meshes_sampling_start(self):
        if self.active_meshes:
            self.meshes.extend(self.active_meshes)
            self.active_meshes.clear()

    def meshes_sampling_add(self):
        if self.meshes:
            self.active_meshes.append(self.meshes.pop())
        else:
            mesh = self.active_meshes[-1]
            bobj = mesh.blender_obj
            new_bobj = bobj.copy()
            new_bobj.data = bobj.data.copy()
            new_bobj.animation_data_clear()
            bpy.context.collection.objects.link(new_bobj)
            new_mesh = MeshObject(new_bobj)
            self.prepare_mesh(new_mesh)
            self.active_meshes.append(new_mesh)

    def meshes_sampling_stop(self):
        for mesh in self.active_meshes:
            mesh.hide(False)
            mesh.blender_obj.rigid_body.type = 'ACTIVE'

        for mesh in self.meshes:
            mesh.set_location((100, 100, 100))
            mesh.hide(True)
            mesh.blender_obj.rigid_body.type = 'PASSIVE'


GlobObjs = Dict[str, GlobObj]


def load_objs(ds_root_path: Path, ds_name: str) -> GlobObjs:
    models_path = ds_root_path / ds_name / 'models'
    models = read_yaml(models_path / 'models.yaml')
    glob_objs = {}
    for obj_id in models:
        obj_path = models_path / f'{obj_id}.ply'
        objs = bproc.loader.load_obj(obj_path.as_posix())
        assert len(objs) == 1
        obj = objs[0]
        glob_objs[obj_id] = GlobObj(obj, ds_name, obj_id)
    return glob_objs


def make_room(sz: int):
    sz_half = sz / 2
    room_planes = [bproc.object.create_primitive('PLANE', size=sz),
                   bproc.object.create_primitive('PLANE', size=sz, location=[0, -sz_half, sz_half],
                                                 rotation=[-1.570796, 0, 0]),
                   bproc.object.create_primitive('PLANE', size=sz, location=[0, sz_half, sz_half],
                                                 rotation=[1.570796, 0, 0]),
                   bproc.object.create_primitive('PLANE', size=sz, location=[sz_half, 0, sz_half],
                                                 rotation=[0, -1.570796, 0]),
                   bproc.object.create_primitive('PLANE', size=sz, location=[-sz_half, 0, sz_half],
                                                 rotation=[0, 1.570796, 0])]
    for plane in room_planes:
        plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction=100.0, linear_damping=0.99,
                               angular_damping=0.99)

    # sample light color and strenght from ceiling
    light_plane = bproc.object.create_primitive('PLANE', size=sz * 1.5, location=[0, 0, sz * 2])
    light_plane.set_name('light_plane')

    return room_planes, light_plane


def make_light():
    light_plane_material = bproc.material.create('light_material')

    # sample point light on shell
    light_point = bproc.types.Light()
    light_point.set_energy(20)

    return light_plane_material, light_point


class DirsIter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cfg.output_path.mkdir(parents=True, exist_ok=True)
        self.dirs_files_num, self.dirs_to_fill, self.last_dir_num = self.list_dirs()
        self.ind_dirs_to_fill = 0

    def list_dirs(self) -> Tuple[Dict[Path, int], List[Path], int]:
        dirs_files_num, dirs_to_fill, last_dir_num = {}, [], -1
        for subpath in self.cfg.output_path.iterdir():
            if not subpath.is_dir() or not subpath.name.isnumeric():
                continue
            _, _, files = next(os.walk(subpath))
            n_files = len(files)
            dirs_files_num[subpath] = n_files
            if n_files < self.cfg.images_per_dir:
                dirs_to_fill.append(subpath)

            last_dir_num = max(last_dir_num, int(subpath.name))

        return dirs_files_num, dirs_to_fill, last_dir_num

    def next_dir(self) -> Path:
        if self.ind_dirs_to_fill < len(self.dirs_to_fill):
            dir_to_fill = self.dirs_to_fill[self.ind_dirs_to_fill]
            self.dirs_files_num[dir_to_fill] += self.cfg.generation.scene.images_num
            if self.dirs_files_num[dir_to_fill] >= self.cfg.images_per_dir:
                self.ind_dirs_to_fill += 1
            return dir_to_fill

        self.last_dir_num += 1
        new_dir_path = self.cfg.output_path / f'{self.last_dir_num:06d}'
        new_dir_path.mkdir()
        self.dirs_to_fill.append(new_dir_path)
        self.dirs_files_num[new_dir_path] = self.cfg.generation.scene.images_num
        return new_dir_path


class DirsIterMp:
    def __init__(self, cfg: Config):
        self._lock = ILock(cfg.output_path.as_posix())
        self._cfg = cfg
        self._cfg.output_path.mkdir(parents=True, exist_ok=True)
        self._dir_no = 0
        self._dir_files_num = 0
        self._next_file_no = 0
        self._dir_files_check_needed = True
        self._iter_fpath = self._cfg.output_path / 'iter.json'

    def _read_iter_json(self):
        if self._iter_fpath.exists():
            try:
                iter_json = read_json(self._iter_fpath)
                self._dir_no = iter_json['dir_no']
                self._dir_files_num = iter_json['dir_files_num']
                self._next_file_no = iter_json['next_file_no']
                self._dir_files_check_needed = False
            except:
                print(f'Error reading iter json from {self._iter_fpath}')
                print(traceback.format_exc())
                self._dir_no, self._dir_files_num, self._next_file_no = 0, 0, 0

    def _write_iter_json(self):
        iter_json = {
            'dir_no': self._dir_no,
            'dir_files_num': self._dir_files_num,
            'next_file_no': self._next_file_no,
        }
        write_json(iter_json, self._iter_fpath)

    def _iter_dir(self):
        self._read_iter_json()
        while True:
            if self._dir_files_check_needed:
                dir_name = scene_name_from_no(self._dir_no)
                subpath = self._cfg.output_path / dir_name
                if subpath.exists():
                    _, _, fnames = next(os.walk(subpath))
                    fnums = [int(fname.split('.')[0]) for fname in fnames]
                    self._next_file_no = max(fnums, default=-1) + 1
                    self._dir_files_num = len(fnames)
                self._dir_files_check_needed = False
            if self._dir_files_num < self._cfg.images_per_dir:
                break
            self._dir_no += 1
            self._dir_files_num = 0
            self._next_file_no = 0
            self._dir_files_check_needed = True
        self._dir_files_num += self._cfg.generation.scene.images_num
        self._next_file_no += self._cfg.generation.scene.images_num
        self._write_iter_json()

    def next_dir(self) -> Tuple[Path, int]:
        with self._lock:
            self._iter_dir()
        dir_name = scene_name_from_no(self._dir_no)
        dir_path = self._cfg.output_path / dir_name
        return dir_path, self._next_file_no - self._cfg.generation.scene.images_num


def sample_add_objs(objs: List[GlobObj], max_objects: int, max_duplicates: int) -> Tuple[List[GlobObj], List[MeshObject]]:
    n_objs = len(objs)
    objs = objs.copy()
    sampled_objs = {}
    for obj in objs:
        obj.meshes_sampling_start()

    for i_sample in range(max_objects):
        if len(objs) == 0:
            print(f'Cannot sample {max_objects} objects with possible {max_duplicates} duplicates '
                  f'from {n_objs}. Sampled objects: {len(sampled_objs)}')
            break
        i_obj = np.random.randint(0, len(objs))
        obj = objs[i_obj]
        obj.meshes_sampling_add()
        sampled_objs[obj.glob_id] = obj
        if len(obj.active_meshes) >= max_duplicates:
            obj.meshes_sampling_stop()
            objs.pop(i_obj)

    for obj in objs:
        obj.meshes_sampling_stop()

    sampled_objs = list(sampled_objs.values())
    sampled_meshes = list(itertools.chain(*[obj.active_meshes for obj in sampled_objs]))
    return sampled_objs, sampled_meshes


class SceneGen:
    def __init__(self, cfg: SceneConfig, target_objs: GlobObjs, dist_objs: GlobObjs):
        self.cfg = cfg
        self.target_objs, self.dist_objs = list(target_objs.values()), list(dist_objs.values())
        self.objs = self.target_objs + self.dist_objs
        self.sampled_target_objs = None
        self.sampled_dist_objs = None
        self.sampled_target_meshes = None
        self.sampled_dist_meshes = None
        self.sampled_objs = None
        self.sampled_meshes = None
        self.bvh_tree = None

    def sample_objs(self):
        self.sampled_target_objs, self.sampled_target_meshes = sample_add_objs(self.target_objs, self.cfg.target_objects_num, self.cfg.max_duplicate_objects)
        self.sampled_dist_objs, self.sampled_dist_meshes = sample_add_objs(self.dist_objs, self.cfg.distractor_objects_num, self.cfg.max_duplicate_objects)
        self.sampled_objs = self.sampled_target_objs + self.sampled_dist_objs
        self.sampled_meshes = self.sampled_target_meshes + self.sampled_dist_meshes
        self.bvh_tree = bproc.object.create_bvh_tree_multi_objects(self.sampled_meshes)

    def _sample_pose(self, mesh: MeshObject):
        pose1 = np.random.uniform([-0.6, -0.6, 0.2], [-0.4, -0.4, 0.4])
        pose2 = np.random.uniform([0.4, 0.4, 1.2], [0.6, 0.6, 1.4])
        mesh.set_location(np.random.uniform(pose1, pose2))

    def sample_poses(self):
        for mesh in self.sampled_meshes:
            self._sample_pose(mesh)

    def sample_textures(self):
        for obj in self.sampled_objs:
            for mesh in obj.active_meshes:
                mat = mesh.get_materials()[0]
                col = np.random.uniform(0.1, 0.7)
                grey_col = (col, col, col, 1.)
                mat.set_principled_shader_value('Base Color', grey_col)

                mat.set_principled_shader_value("Roughness", np.random.uniform(0, 0.5))
                if obj.ds_name == 'itodd':
                    mat.set_principled_shader_value("Specular", np.random.uniform(0.3, 1.0))
                    mat.set_principled_shader_value("Metallic", np.random.uniform(0, 1.0))
                if obj.ds_name == 'tless':
                    mat.set_principled_shader_value("Metallic", np.random.uniform(0, 0.5))


class DsGen:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.target_objs = load_objs(self.cfg.sds_root_path, self.cfg.target_dataset_name)
        self.dist_objs = load_objs(self.cfg.sds_root_path, self.cfg.distractor_dataset_name)
        self.glob_objs = {obj.glob_id: obj for obj in itertools.chain(self.target_objs.values(), self.dist_objs.values())}
        self.room_planes, self.light_plane = make_room(sz=self.cfg.generation.room_size)
        self.light_plane_material, self.light_point = make_light()
        self.light_plane.replace_materials(self.light_plane_material)
        self.cc_textures = bproc.loader.load_ccmaterials(self.cfg.cc_textures_path.as_posix())
        self.scene_gen = SceneGen(cfg.generation.scene, self.target_objs, self.dist_objs)
        self.dirs_iter = DirsIterMp(self.cfg)
        self.normals_initialized = False

    def sample_light(self):
        emission_color = np.random.uniform([0.7, 0.7, 0.7, 1.0], [1.0, 1.0, 1.0, 1.0])
        self.light_plane_material.make_emissive(
            emission_strength=np.random.uniform(0.25, 0.5),
            emission_color=list(emission_color))
        self.light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
        self.light_point.set_energy(np.random.uniform(100, 350))
        location = bproc.sampler.shell(center=[0, 0, 0], radius_min=0.5, radius_max=1.5,
                                       elevation_min=5, elevation_max=89)
        self.light_point.set_location(location)

    def sample_walls_textures(self):
        # load cc_textures
        random_cc_texture = np.random.choice(self.cc_textures)
        for plane in self.room_planes:
            plane.replace_materials(random_cc_texture)

    def simulate_physics(self):
        bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                          max_simulation_time=10,
                                                          check_object_interval=1,
                                                          substeps_per_frame=50,
                                                          solver_iters=25)

    def sample_camera_poses(self):
        cam_poses = 0
        while cam_poses < self.cfg.generation.scene.images_num:
            # Sample location
            location = bproc.sampler.shell(center=[0, 0, 0],
                                           radius_min=0.5,
                                           radius_max=self.cfg.generation.room_size / 2 - 0.01,
                                           elevation_min=5,
                                           elevation_max=89)
            # Determine point of interest in scene as the object closest to the mean of a subset of objects
            sampled_meshes = self.scene_gen.sampled_meshes
            poi = bproc.object.compute_poi(np.random.choice(sampled_meshes, size=len(sampled_meshes) // 2, replace=False))
            # Compute rotation based on vector going from location towards poi
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location,
                                                                     inplane_rot=np.random.uniform(-3.14159, 3.14159))
            # print(f'Loc: {location}. Poi: {poi}. Mat:\n{rotation_matrix}')
            # Add homog cam pose based on location an rotation
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

            # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
            if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, self.scene_gen.bvh_tree):
                # Persist camera pose
                bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
                cam_poses += 1
                print(f'Persisting camera pose #{cam_poses}')
            else:
                print(f'Camera-objects collision (pose {cam_poses})')

    def render(self) -> Dict[str, Any]:
        try:
            data = bproc.renderer.render()
        except:
            print(traceback.format_exc())
            raise
        data.update(render_segmap(obj_key='full_id'))
        print('Rendered data keys:', data.keys())
        return data

    def get_frame_gt(self) -> Dict[str, Any]:
        H_c2w_opencv = Matrix(_WriterUtility.get_cam_attribute(
            bpy.context.scene.camera, 'cam2world_matrix', local_frame_change=['X', '-Y', '-Z']))
        H_w2c_opencv = H_c2w_opencv.inverted()
        res = {}
        for obj in self.scene_gen.sampled_objs:
            for mesh in obj.active_meshes:
                H_m2w = Matrix(_WriterUtility.get_common_attribute(mesh.blender_obj, 'matrix_world'))
                H_m2c = H_w2c_opencv @ H_m2w
                full_id = mesh.blender_obj['full_id']
                parts = full_id.split('_')
                instance_num = int(parts[-1])
                res[mesh.blender_obj['full_id']] = {
                    'ds_name': obj.ds_name,
                    'ds_obj_id': obj.obj_id,
                    'glob_id': obj.glob_id,
                    'instance_num': instance_num,
                    'H_m2c': [list(r) for r in H_m2c],
                }
        return res

    def get_gt(self) -> List[Dict[str, Any]]:
        camera = {
            'K': CameraUtility.get_intrinsics_as_K_matrix(),
            'image_size': self.cfg.generation.image_size,
        }
        res = []
        for frame_id in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):
            bpy.context.scene.frame_set(frame_id)
            objects = self.get_frame_gt()
            res.append({
                'camera': camera,
                'objects': objects,
            })
        return res

    def save_data(self, data: Dict[str, Any]):
        out_path, next_file_num = self.dirs_iter.next_dir()
        print(f'Writing result in hdf5 format to {out_path}')
        write_hdf5(out_path.as_posix(), next_file_num, data, append_to_existing_output=True)

    def gen_scene(self, i_scene: int):
        print(f'Generating scene #{i_scene:06d}')
        self.scene_gen.sample_objs()
        self.scene_gen.sample_poses()
        self.scene_gen.sample_textures()
        self.sample_light()
        self.sample_walls_textures()
        self.simulate_physics()
        self.sample_camera_poses()

        # if not self.normals_initialized:
        #     bproc.renderer.enable_normals_output()
        #     self.normals_initialized = True

        data = self.render()
        data['gt'] = self.get_gt()

        self.save_data(data)

    def run(self):
        # bproc.renderer.enable_depth_output(activate_antialiasing=self.cfg.generation.depth_antialiasing)
        bproc.renderer.set_max_amount_of_samples(self.cfg.generation.render_samples)

        bproc.camera.set_resolution(*self.cfg.generation.image_size)

        for i_scene in range(self.cfg.generation.scenes_num):
            self.gen_scene(i_scene)


def generate(cfg: Config):
    print(cfg)
    dsgen = DsGen(cfg)
    dsgen.run()


def init() -> Config:
    parser = argparse.ArgumentParser(description='BOP dataset generation')
    parser.add_argument('--config-path', type=Path, required=True,
                        help='Path to YAML configuration file with generation parameters')
    args = parser.parse_args()
    cfg = Config.load(args.config_path)

    print(f'Appending: {cfg.src_path}')
    sys.path.append(cfg.src_path.as_posix())

    return cfg


if __name__ == '__main__':
    CFG = init()

    # After init() we have our sources added to sys.path, so we can import all local modules
    from sds.utils.utils import read_yaml, read_json, write_json

    generate(CFG)

