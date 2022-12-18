import sys
from time import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
import signal
from tqdm import trange

from sds.data.ds_pose_gen_mp import DsPoseGenMp
from sds.model.model_pose_aae import build_aae_layers, AaeLoss
from sds.synth.renderer import Renderer, OutputType
from sds.utils.tf_utils import tf_set_gpu_incremental_memory_growth

tf_set_gpu_incremental_memory_growth()

from sds.data.ds_pose_gen import DsPoseGen
from sds.utils.utils import datetime_str, calc_3d_pts_uniform, Rot3dIter, canonical_cam_mat_from_img, make_transform
from sds.utils.ds_utils import load_mesh, load_objs
from train_utils import ds_pose_preproc, ds_pose_mp_preproc, find_last_aae_model_path


class Config(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    obj_id: str = Field(
        ...,
        description='Object unique identifier.',
        cli=('--obj-id',),
    )
    obj_path: Path = Field(
        ...,
        description='Path to 3d object model (stl, obj, ...) which will be used for training AAE.',
        cli=('--obj-path',),
    )
    train_root_path: Path = Field(
        ...,
        description='Training directory with subdirectories corresponding to separate learning process each.',
        required=True,
        cli=('--train-root-path',),
    )
    img_size: int = Field(
        256,
        description='Encoder input and Decoder output image size.',
        required=False,
        cli=('--img-size',),
    )
    weights: str = Field(
        'last',
        description='Model\'s weights source. Its value can be either subdirectory name from TRAIN_ROOT_PATH or the '
                    '"last" string constant. In the latter case latest training subdirectory for OBJ_ID and IMG_SIZE '
                    'will be looked for.',
        required=False,
        cli=('--weights',),
    )
    rot_vecs_num: int = Field(
        ...,
        description='Number of unit length rotation vectors. Vectors will be distributed on sphere uniformly.',
        required=True,
        cli=('--rot-vecs-num',),
    )
    rot_angs_num: int = Field(
        ...,
        description='Number of inplane rotation angles. Angles will be distributed evenly in interval (0, 180).',
        required=True,
        cli=('--rot-angs-num',),
    )
    batch_size: int = Field(
        ...,
        required=True,
        description='Training batch size (number of images for single forward/backward network iteration).',
        cli=('--batch-size',),
    )
    pose_gen_workers: int = Field(
        0,
        description='Number of processes in order to parallelize online rendering for Pose dataset generation. '
                    'If not set or set to value <= 0, rendering is a single separate thread of the main process.',
        required=False,
        cli=('--pose-gen-workers',),
    )


def make_dict(cfg: Config) -> int:
    print(cfg)

    if cfg.obj_path.is_dir():
        ds_path, models_subdir = cfg.obj_path.parent, cfg.obj_path.name
        ds_root_path, ds_name = ds_path.parent, ds_path.name
        obj_glob_id = f'{ds_name}_{cfg.obj_id}'
        objs = load_objs(ds_root_path, ds_name, models_subdir=models_subdir, load_meshes=True, load_target_glob_id=obj_glob_id)
        diam = objs[obj_glob_id]['diameter']
        objs = {obj_glob_id: objs[obj_glob_id]}
    else:
        mesh, mesh_dict = load_mesh(cfg.obj_path)
        verts_diff = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
        diam = np.abs(verts_diff).max()
        obj_glob_id = cfg.obj_id
        objs = {
            obj_glob_id: {
                'mesh': mesh_dict,
                'diameter': diam,
            }
        }

    if cfg.weights == 'last':
        weights_path = find_last_aae_model_path(cfg.train_root_path, cfg.obj_id, cfg.img_size)
        assert weights_path is not None, f'Cannot find last model weights for object {cfg.obj_id} and image ' \
                                         f'size {cfg.img_size} in {cfg.train_root_path}'
    else:
        weights_path = cfg.train_root_path / cfg.weights
    weights_path = weights_path / 'weights' / 'last.pb'

    print('Building model for inference')
    inp, out = build_aae_layers(img_size=cfg.img_size, batch_size=cfg.batch_size, train=False)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    print(f'Loading model from {weights_path}')
    model.load_weights(weights_path.as_posix())

    rot_iter = Rot3dIter(cfg.rot_vecs_num, cfg.rot_angs_num)
    hide_window = True
    img_size = (cfg.img_size, cfg.img_size)
    ren = Renderer(objs, img_size, hide_window=hide_window)
    cam_mat = canonical_cam_mat_from_img(img_size)
    ren.set_camera_matrix(cam_mat)
    f = cam_mat[0, 0]
    # z / f = diam / img_size
    z = f * diam / cfg.img_size
    pos = np.array((0, 0, z), float)

    n = len(rot_iter)
    img_batch, rot_batch = [], []
    pbar = trange(n, desc=f'AAE Dict', unit='image')
    aae_matches = []
    for i in pbar:
        rot_vec, rot_ang = rot_iter[i]
        obj_m2c = make_transform(rot_vec, rot_ang, pos)
        objs = {
            obj_glob_id: {
                'glob_id': obj_glob_id,
                'H_m2c': obj_m2c,
            }
        }
        img_noc = ren.gen_colors(cam_mat, objs, OutputType.Noc)
        img_norms = ren.gen_colors(cam_mat, objs, OutputType.Normals)
        img = np.concatenate([img_noc, img_norms], axis=-1)
        img_batch.append(tf.convert_to_tensor(img))
        rot_batch.append((*rot_vec, rot_ang))

        if len(img_batch) == cfg.batch_size or i == n - 1:
            inp = tf.stack(img_batch, axis=0)
            inp = tf.cast(inp, tf.float32) / 255.
            out = model(inp)
            match = tf.concat((rot_batch, out), axis=1)
            aae_matches.append(match)
            img_batch, rot_batch = [], []

    pbar.close()

    aae_matches = tf.concat(aae_matches, axis=0).numpy()
    aae_dict_fname = f'aae_dict_rvecs_{cfg.rot_vecs_num}_rangs_{cfg.rot_angs_num}'
    aae_dict_fpath = weights_path.parent.parent / aae_dict_fname
    print(f'Saving dictionary of shape {aae_matches.shape} and type {aae_matches.dtype} to {aae_dict_fpath}')
    np.save(aae_dict_fpath.as_posix(), aae_matches)

    return 0


if __name__ == '__main__':
    def exception_handler(ex: BaseException) -> int:
        raise ex
    run_and_exit(Config, make_dict, 'Create AAE dictionary', exception_handler=exception_handler)

