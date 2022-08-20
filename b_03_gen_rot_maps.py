import json
import sys
import time
from enum import Enum
from pathlib import Path
import shutil
from typing import Optional, Dict, List, Any, Tuple

import cv2
import h5py
import yaml

import numpy as np
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
import pymesh

from sds.synth.renderer import Renderer, OutputType
from sds.utils import utils
from sds.utils.common import int_to_tuple
from sds.utils.utils import canonical_cam_mat_from_img, make_transform, calc_ref_dist_from_camera
from sds.utils.ds_utils import load_objs


class Config(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    sds_root_path: Path = Field(
        ...,
        description='Path to SDS datasets (containing datasets: ITODD, TLESS, etc.)',
        cli=('--sds-root-path',),
    )
    dataset_name: str = Field(
        ...,
        description='Dataset name. Has to be a subdirectory of SDS_ROOT_PATH, one of: "itodd", "tless", etc.',
        cli=('--dataset-name',),
    )
    models_subdir: str = Field(
        'models',
        description='Models subdirectory. Has to contain ply files (default: "models")',
        required=False,
        cli=('--models-subdir',),
    )
    model_id_num: int = Field(
        ...,
        description='Number of object from current dataset. All objects numbered from 1 to len(objects) ones'
                    'in a global initialization step',
        required=True,
        cli=('--model-id-num',),
    )
    win_size: int = Field(
        128,
        description='Renderer window size in pixels',
        required=False,
        cli=('--win-size',),
    )
    rot_grid_size: int = Field(
        128,
        description='Rotation vector discretization dimension size. So vector will be chosen'
                    'the filled sphere fit in cube ROT_GRID_SIZE x ROT_GRID_SIZE x ROT_GRID_SIZE',
        required=False,
        cli=('--rot-grid-size',),
    )


def ravel(*arrs: np.ndarray) -> List[np.ndarray]:
    res = []
    for arr in arrs:
        res.append(arr.ravel())
    return res


def gen_full_sphere_coords_inds(n: int) -> Tuple[np.ndarray, np.ndarray]:
    inds = np.arange(0, n)
    coords = inds / inds.mean() - 1
    ix, iy, iz = np.meshgrid(inds, inds, inds, indexing='ij')
    x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
    inds = np.stack(ravel(ix, iy, iz)).T
    coords = np.stack(ravel(x, y, z)).T
    coords_norms = np.linalg.norm(coords, axis=1)
    fsphere_mask = coords_norms <= 1
    inds, coords = inds[fsphere_mask], coords[fsphere_mask]

    ii = np.arange(0, len(inds))
    np.random.shuffle(ii)
    inds, coords = inds[ii], coords[ii]

    return inds, coords * np.pi


def main(cfg: Config) -> int:
    ds_path = cfg.sds_root_path / cfg.dataset_name
    objs = load_objs(cfg.sds_root_path, cfg.dataset_name, models_subdir=cfg.models_subdir, load_meshes=True, load_target_id_num=cfg.model_id_num)
    id_num_to_glob = {obj['id_num']: oid for oid, obj in objs.items()}
    obj_glob_id = id_num_to_glob[cfg.model_id_num]
    print(f'Object glob id chosen: {obj_glob_id}')

    objs_ren = {obj_glob_id: objs[obj_glob_id]}
    img_size = int_to_tuple(cfg.win_size)
    renderer = Renderer(objs_ren, win_size=img_size)
    cam_mat = canonical_cam_mat_from_img(img_size)
    renderer.set_camera_matrix(cam_mat)

    inds, rot_vecs = gen_full_sphere_coords_inds(cfg.rot_grid_size)
    print(inds.shape, inds.dtype)
    print(rot_vecs.shape, rot_vecs.dtype)

    n = len(inds)
    obj_pose = {
        obj_glob_id: {
            'glob_id': obj_glob_id,
            'H_m2c': np.empty((4, 4)),
        }
    }
    pos_z = calc_ref_dist_from_camera(cam_mat, img_size, objs[obj_glob_id])
    pos = np.array((0, 0, pos_z))

    def make_tr(rv: np.ndarray) -> np.ndarray:
        al = np.linalg.norm(rv)
        rv = rv / al
        return make_transform(rv, al, pos)

    n2 = n // 2
    dists = np.zeros(n2, np.float64)
    for i in range(n2):
        rv1, rv2 = rot_vecs[2 * i], rot_vecs[2 * i + 1]
        t1 = time.time()
        obj_pose[obj_glob_id]['H_m2c'] = make_tr(rv1)
        img1 = renderer.gen_colors(cam_mat, obj_pose, OutputType.Noc)
        obj_pose[obj_glob_id]['H_m2c'] = make_tr(rv2)
        img2 = renderer.gen_colors(cam_mat, obj_pose, OutputType.Noc)
        diff = np.abs(img2 - img1) / 255.0
        mask = np.maximum(img1.max(axis=-1), img2.max(axis=-1)) > 0
        diff_mean = np.mean(diff[mask])
        dists[i] = diff_mean
        print(f'{i:07d}. diff mean: {diff_mean:.3f}')
        t2 = time.time()
        print(f'renderer: {t2 - t1:.3f}')
        # img = np.concatenate((img1, img2), axis=1)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imshow('img', img)
        # if cv2.waitKey() in (27, ord('q')):
        #     break
        # if i == 10:
        #     break

    out_path = ds_path / f'rot_maps_g{cfg.rot_grid_size}_w{cfg.win_size}'
    out_path.mkdir(exist_ok=True)
    out_path /= f'{obj_glob_id}.npz'
    np.savez(out_path.as_posix(), inds=inds, rot_vecs=rot_vecs, dists=dists)
    return 0


if __name__ == '__main__':
    run_and_exit(Config, main, 'Generate object pose similarity maps')

