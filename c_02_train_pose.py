import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict

import numpy as np
import tensorflow as tf
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit

from sds.data.ds_pose_gen import DsPoseGen
from sds.data.ds_pose_loader import DsPoseLoader
from sds.data.index import load_cache_ds_index
from sds.data.ds_loader import DsLoader
from sds.model.losses import MseNZLoss, CosNZLoss
from sds.model.params import ScaledParams
from sds.model.processing import float_to_img, img_to_float
from sds.utils.tf_utils import tf_set_gpu_incremental_memory_growth
from sds.utils.utils import datetime_str, gen_colors, load_objs
from train_utils import build_maps_model, color_segmentation, normalize


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
        description='Target dataset name. Has to be a subdirectory of SDS_ROOT_PATH, one of: "itodd", "tless", etc.',
        cli=('--target-dataset-name',),
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
    train_root_path: Path = Field(
        ...,
        description='Training directory. For each new training new subdirectory will be created ',
        required=True,
        cli=('--train-root-path',),
    )
    learning_rate: float = Field(
        ...,
        required=True,
        description='Learning rate',
        cli=('--learning-rate',),
    )
    epochs: int = Field(
        ...,
        required=True,
        description='Total number of training epochs. When proceeding training, EPOCHS acts as a global limit'
    )
    batch_size: int = Field(
        ...,
        required=True,
        description='Training batch size (number of images for single forward/backward network iteration)',
        cli=('--batch-size',)
    )
    train_steps: int = Field(
        ...,
        required=True,
        description='Number of train steps per epoch',
        cli=('--train-steps',)
    )
    val_steps: int = Field(
        ...,
        required=True,
        description='Number of validation steps per epoch',
        cli=('--val-steps',)
    )
    n_img_vis: int = Field(
        4,
        description='Number of images for inference visualization on each train/val epoch',
        required=False,
        cli=('--n-img-vis', ),
    )


def preprocess(img: tf.Tensor, maps: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, Tuple[tf.Tensor, ...]]:
    return img_to_float(img), maps


def build_ds(ds_loader: DsLoader, batch_size: int) -> tf.data.Dataset:
    out_sig = (
        tf.TensorSpec(shape=(*ds_loader.img_size, 3), dtype=tf.float32),
        (
            tf.TensorSpec(shape=(*ds_loader.img_size, 3), dtype=tf.int32),
            tf.TensorSpec(shape=(*ds_loader.img_size, 3), dtype=tf.int32),
            tf.TensorSpec(shape=(*ds_loader.img_size, 1), dtype=tf.int32),
        )
    )
    ds = tf.data.Dataset \
        .from_generator(ds_loader.gen, output_signature=out_sig) \
        .batch(batch_size) \
        .map(preprocess) \
        .prefetch(tf.data.AUTOTUNE)
    return ds


def build_datasets(cfg: Config, objs: Dict, obj_glob_id: str, img_size: int = 128) -> Tuple[DsPoseGen, DsPoseGen, tf.data.Dataset, tf.data.Dataset]:
    ds_path = cfg.sds_root_path / cfg.dataset_name
    ds_index = load_cache_ds_index(ds_path, 1.0)
    ds_pose_train = DsPoseGen(objs, img_size, obj_glob_id, aug_enabled=True, hist_sz=cfg.n_img_vis)
    ds_pose_val = DsPoseLoader(ds_path, objs, img_size, obj_glob_id, hist_sz=cfg.n_img_vis)
    ds_train = build_ds(ds_loader_train, cfg.batch_size)
    ds_val = build_ds(ds_loader_val, cfg.batch_size)
    return ds_loader_train, ds_loader_val, ds_train, ds_val


def get_subdir_name(dt: datetime, ds_name: str, obj_glob_id: str):
    return f'{datetime_str(dt)}_{ds_name}_t{obj_glob_id}'


def main(cfg: Config) -> int:
    print(cfg)

    objs = load_objs(cfg.sds_root_path, cfg.dataset_name, load_meshes=True)
    dsl_train, dsl_val, ds_train, ds_val = build_datasets(cfg)
    
    id_num_to_glob = {obj['id_num']:oid for oid, obj in objs.items()}
    obj_glob_id = id_num_to_glob[cfg.model_id_num]
    out_subdir_name = get_subdir_name(datetime.now(), cfg.dataset_name, obj_glob_id)
    out_path = cfg.train_root_path / out_subdir_name
    weights_out_path = out_path / 'weights'
    weights_out_path.mkdir(parents=True, exist_ok=True)
    params_out_path = out_path / 'params'
    params_out_path.mkdir(parents=True, exist_ok=True)
    
    tf_set_gpu_incremental_memory_growth()

    return 0


if __name__ == '__main__':
    run_and_exit(Config, main, 'Convert ply object to meters, restructure objects\' metadata')
