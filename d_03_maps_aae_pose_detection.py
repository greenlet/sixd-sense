import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
import tensorflow as tf

from sds.data.index import load_cache_ds_index
from sds.data.ds_loader import DsLoader
from sds.model.utils import tf_img_to_float, tf_float_to_img, np_img_to_float, np_float_to_img
from sds.utils.utils import gen_colors
from sds.utils.ds_utils import load_objs
from sds.data.image_loader import ImageLoader
from sds.model.params import ScaledParams
from sds.utils.tf_utils import tf_set_use_device, CUDA_ENV_NAME, tf_set_gpu_incremental_memory_growth
from train_utils import find_train_weights_path, find_index_file, build_maps_model, color_segmentation, normalize, \
    find_last_maps_train_path, find_last_aae_train_path


class Config(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    sds_root_path: Path = Field(
        ...,
        description='Path to SDS datasets (containing datasets: ITODD, TLESS, etc.)',
        required=True,
        cli=('--sds-root-path',),
    )
    target_dataset_name: str = Field(
        ...,
        description='Target dataset name. Has to be a subdirectory of SDS_ROOT_PATH, one of: "itodd", "tless", etc.',
        required=True,
        cli=('--target-dataset-name',),
    )
    distractor_dataset_name: str = Field(
        ...,
        description='Distractor dataset name. Has to be a subdirectory of SDS_ROOT_PATH, one of: "itodd", "tless", etc.',
        required=True,
        cli=('--distractor-dataset-name',),
    )
    maps_train_root_path: Path = Field(
        ...,
        description='Path to a directory containing subdirectories corresponding to different '
                    'maps model training processes each.',
        required=True,
        cli=('--maps-train-root-path',),
    )
    aae_train_root_path: Path = Field(
        ...,
        description='Path to a directory containing subdirectories corresponding to different '
                    'aae model training processes each.',
        required=True,
        cli=('--aae-train-root-path',),
    )
    maps_phi: int = Field(
        ...,
        description=f'Maps model phi number. Must be: 0 <= MAPS_PHI <= {ScaledParams.phi_max()}',
        required=True,
        cli=('--maps-phi',),
    )
    maps_weights: str = Field(
        'last',
        description='Either the value "last" which means that latest subdirectory of MAPS_TRAIN_ROOT_PATH '
                    'or the name of the subdirectory. Best weights from this subdirectory will be loaded.',
        required=False,
        cli=('--maps-weights',)
    )
    aae_weights: str = Field(
        'last',
        description='Either the value "last" which means that latest subdirectory of AAE_TRAIN_ROOT_PATH '
                    'or the name of the subdirectory. Best weights from this subdirectory will be loaded.',
        required=False,
        cli=('--aae-weights',)
    )
    aae_img_size: int = Field(
        256,
        description='AAE Encoder input and Decoder output image size.',
        required=False,
        cli=('--aae-img-size',),
    )
    aae_rot_vecs_num: int = Field(
        ...,
        description='Number of unit length rotation vectors for AAE pose map. Vectors will be distributed on sphere uniformly.',
        required=True,
        cli=('--aae-rot-vecs-num',),
    )
    aae_rot_angs_num: int = Field(
        ...,
        description='Number of inplane rotation angles for AAE pose map. Angles will be distributed evenly in interval (0, 180).',
        required=True,
        cli=('--aae-rot-angs-num',),
    )
    device_id: str = Field(
        '-1',
        description='Device id. Can have one of the values: "-1", "0", "1", ..., "n". '
                    'If set to "-1", CPU will be used, "GPU" with DEVICE_ID number will be used otherwise. '
                    'This device will be used for both Maps and AAE models inference',
        required=False,
        cli=('--device-id',)
    )
    data_source: str = Field(
        'val',
        description='Images source which will be used to visualize maps predictions. Can be either '
                    'path to images directory or have value "val". In latter case validation dataset '
                    'will be used for predictions and GT will be shown',
        required=False,
        cli=('--data-source',),
    )
    obj_ids: List[str] = Field(
        [],
        description='Object ids to run. If empty, objects with AAE weights/maps will be used only. '
                    'When set to "all", weights for all dataset objects will be searched. In case of '
                    'OBJECT_IDS contains nonempty set or value "all", AAE weights for all corresponding object '
                    'are required.'
    )


def find_aae_weights_dicts(train_root_path: Path, glob_ids: List[str], img_sz: int, rvecs_num: int, rangs_num: int) -> Dict[str, Dict[str, Optional[Path]]]:
    res = {}
    for glob_id in glob_ids:
        obj_id = glob_id.split('_', 1)[1]
        weights_path = find_last_aae_train_path(train_root_path, obj_id, img_sz)
        dict_path = None
        if weights_path:
            dict_path = weights_path / f'aae_dict_rvecs_{rvecs_num}_rangs_{rangs_num}.npy'
            if not dict_path.exists():
                dict_path = None
        res[glob_id] = {'weights': weights_path, 'dict': dict_path}
    return res


class AaePoseMatcher:
    def __init__(self, weights_path: Path, dict_path: Path, ):
        self.weights_path = weights_path
        self.dict_path = dict_path
        m =


def predict_on_dataset(model: tf.keras.models.Model, ds_loader: DsLoader):
    w, h = ds_loader.img_size
    img_vis = np.zeros((2 * h, 4 * w, 3), dtype=np.uint8)
    colors = gen_colors()
    cv2.imshow('maps', img_vis)
    cv2.moveWindow('maps', 0, 0)

    for img, (noc_gt, norms_gt, seg_gt) in ds_loader.gen():
        img_in = tf.convert_to_tensor(img)[None, ...]
        img_in = tf_img_to_float(img_in)
        noc_pred, norms_pred, seg_pred = model(img_in)
        seg_pred = tf.math.argmax(seg_pred, -1)
        noc_pred, norms_pred, seg_pred = noc_pred[0].numpy(), norms_pred[0].numpy(), seg_pred[0].numpy()

        norms_gt1 = (norms_gt - 127.5) / 127.5
        norms_gt1 = np.linalg.norm(norms_gt1, axis=-1)
        norms_pred1 = np.linalg.norm(norms_pred, axis=-1)
        print(f'norms_gt. min: {norms_gt1.min()}. max: {norms_gt1.max()}. mean: {norms_gt1.mean()}')
        print(f'norms_pred. min: {norms_pred1.min()}. max: {norms_pred1.max()}. mean: {norms_pred1.mean()}')
        # norms_gt = np_float_to_img(normalize(np_img_to_float(norms_gt)))
        normalize(norms_pred, inplace=True)
        # norms_pred *= norms_gt1.max()

        img_vis[:h, :w] = img
        img_vis[:h, w:2 * w] = noc_gt
        img_vis[:h, 2 * w:3 * w] = norms_gt
        img_vis[:h, 3 * w:] = color_segmentation(colors, seg_gt)
        img_vis[h:, w:2 * w] = tf_float_to_img(noc_pred)
        img_vis[h:, 2 * w:3 * w] = tf_float_to_img(norms_pred)
        img_vis[h:, 3 * w:] = color_segmentation(colors, seg_pred)

        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        cv2.imshow('maps', img_vis)
        if cv2.waitKey() in (27, ord('q')):
            break


def predict_on_images(model: tf.keras.models.Model, img_loader: ImageLoader):
    pass


def main(cfg: Config) -> int:
    print(cfg)

    if cfg.maps_weights == 'last':
        _, maps_weights_path = find_last_maps_train_path(cfg.maps_train_root_path)
    else:
        _, maps_weights_path = find_last_maps_train_path(cfg.maps_train_root_path, cfg.maps_weights)

    tf_set_use_device(cfg.device_id)

    objs = load_objs(cfg.sds_root_path, cfg.target_dataset_name, load_meshes=True)
    n_classes = len(objs)
    aae_required = len(cfg.obj_ids) > 0
    if cfg.obj_ids and not 'all' in cfg.obj_ids:
        objs_new = {}
        for oid in cfg.obj_ids:
            gid = f'{cfg.target_dataset_name}_{oid}'
            objs_new[gid] = objs[gid]
        objs = objs_new

    print('Building model')
    maps_model = build_maps_model(n_classes, cfg.maps_phi, False)
    print(f'Loading model from {maps_weights_path}')
    maps_model.load_weights(maps_weights_path.as_posix())
    maps_model_params = ScaledParams(cfg.maps_phi)

    glob_ids = list(objs.keys())
    aae_paths = find_aae_weights_dicts(cfg.aae_train_root_path, glob_ids, cfg.aae_img_size, cfg.aae_rot_vecs_num, cfg.aae_rot_angs_num)
    found_glob_ids = []
    not_found = []
    for glob_id, paths in aae_paths.items():
        weights_path, dict_path = paths['weights'], paths['dict']
        if weights_path and dict_path:
            found_glob_ids.append(glob_id)
        elif not weights_path:
            not_found.append(f'AAE weights not found for object {glob_id} and image size {cfg.aae_img_size}')
        else:
            not_found.append(f'AAE dict file not found in {weights_path} for rot_vecs = {cfg.aae_rot_vecs_num} '
                             f'and rot_angs = {cfg.aae_rot_angs_num}')
    if not_found:
        tab = ' ' * 4
        not_found = [f'{tab}{s}' for s in not_found]
        print('\n'.join(not_found))
        if aae_required:
            sys.exit(1)

    ds_loader = None
    img_loader = None
    if cfg.data_source == 'val':
        index_fpath = find_index_file(cfg.maps_train_root_path)
        if index_fpath is None:
            print(f'Cannot find dataset index file in {cfg.maps_train_root_path}')
            return 1
        ds_index = load_cache_ds_index(index_fpath=index_fpath)
        ds_loader = DsLoader(ds_index, objs, is_training=False, img_size=maps_model_params.input_size)
    else:
        images_path = Path(cfg.data_source)
        img_loader = ImageLoader(images_path)

    # if ds_loader is not None:
    #     predict_on_dataset(model, ds_loader)
    # else:
    #     predict_on_images(model, img_loader)

    return 0


if __name__ == '__main__':
    run_and_exit(Config, main, 'Run maps then AAE networks with refinement afterwards')

