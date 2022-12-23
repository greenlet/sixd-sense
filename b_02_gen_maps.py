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
from sds.utils.ds_utils import load_objs


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
    ds_path: Optional[Path] = Field(
        None,
        description='Dataset path which contains data subdirectory with list of folders with hdf5 files. '
                    'If not set, SDS_ROOT_PATH / TARGET_DATASET_NAME path will be used.',
        cli=('--ds-path',),
    )
    models_subdir: str = Field(
        'models',
        description='Models subdirectory. Has to contain ply files (default: "models")',
        required=False,
        cli=('--models-subdir',),
    )
    output_type: str = Field(
        ...,
        description=f'Output type. "normals" - normals in camera frame scaled in order to fit (0, 1) '
                    f'interval. "noc" - Normalized object coordinates in camera frame. Coordinates '
                    f'are taken relatively to object frame\'s center and scaled in order to fit (0, 1)',
        required=True,
        cli=('--output-type',),
    )
    rewrite: bool = Field(
        False,
        description='If set, existing maps will be rewritten. Otherwise only missing '
                    'ones will be generated',
        required=False,
        cli=('--rewrite', ),
    )
    debug: bool = Field(
        False,
        description='Debug mode. Renderings are calculated and visualized but not saved',
        required=False,
        cli=('--debug',),
    )


def list_dataset(data_path: Path, dst_root_path: Path, skip_if_dst_exists: bool) -> Dict[str, Dict]:
    scenes = {}
    n_items = 0
    for scene_path in data_path.iterdir():
        if not scene_path.is_dir():
            continue
        dst_scene_path = dst_root_path / f'{scene_path.name}'
        scene_fpaths = []
        for fpath in scene_path.iterdir():
            if not (fpath.is_file() and fpath.suffix == '.hdf5'):
                continue
            dst_fpath = dst_scene_path / fpath.with_suffix('.png').name
            if not (skip_if_dst_exists and dst_fpath.exists()):
                scene_fpaths.append((fpath, dst_fpath))
        if scene_fpaths:
            scene = {
                'id': scene_path.name,
                'path': scene_path,
                'dst_path': dst_scene_path,
                'items': scene_fpaths,
                'size': len(scene_fpaths)
            }
            scenes[scene_path.name] = scene
            n_items += len(scene_fpaths)

    res = {
        'path': data_path,
        'dst_path': dst_root_path,
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


def main(cfg: Config) -> int:
    print(cfg)
    if cfg.ds_path is not None:
        target_ds_path = cfg.ds_path
    else:
        target_ds_path = cfg.sds_root_path / cfg.target_dataset_name

    models = load_objs(cfg.sds_root_path, cfg.target_dataset_name, cfg.distractor_dataset_name, cfg.models_subdir, load_meshes=True)

    data_postfix = ''
    data_path = target_ds_path / f'data{data_postfix}'
    if cfg.output_type == 'normals':
        out_type = OutputType.Normals
    elif cfg.output_type == 'noc':
        out_type = OutputType.Noc
    else:
        raise f'Unknown output type "{cfg.output_type}". "normals", "noc" values expected'

    dst_root_path = data_path.parent / f'{data_path.name}_{cfg.output_type}'
    dst_root_path.mkdir(parents=True, exist_ok=True)

    data = list_dataset(data_path, dst_root_path, skip_if_dst_exists=not cfg.rewrite and not cfg.debug)
    print(f'Number of scenes: {len(data["scenes"])}. Files total: {data["size"]}')

    renderer = Renderer(models=models)

    for scene in data['scenes'].values():
        dst_scene_path: Path = scene['dst_path']
        dst_scene_path.mkdir(parents=True, exist_ok=True)
        for fpath, dst_fpath in scene['items']:
            print(fpath)
            img, cam_mat, img_size, objs = read_gt(fpath)
            renderer.set_window_size(img_size)

            colors = renderer.gen_colors(cam_mat, objs, out_type)
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

