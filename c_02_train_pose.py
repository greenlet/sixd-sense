import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict

import numpy as np
import tensorflow as tf
from numpy import ndarray
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit

from sds.data.ds_pose_gen import DsPoseGen
from sds.data.ds_pose_loader import DsPoseLoader
from sds.data.index import load_cache_ds_index
from sds.data.ds_loader import DsLoader
from sds.data.utils import DsPoseItem
from sds.model.losses import MseNZLoss, CosNZLoss, RotVecLoss
from sds.model.model_pose import build_pose_layers
from sds.model.params import ScaledParams
from sds.model.processing import tf_float_to_img, tf_img_to_float, np_img_to_float
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
        cli=('--n-img-vis',),
    )


def ds_item_to_numbers(item: DsPoseItem) -> Tuple[Tuple[ndarray, ndarray], Tuple[np.ndarray, np.ndarray]]:
    img = np.concatenate([item.img_noc_out, item.img_norms_out], axis=-1)
    img = np_img_to_float(img)
    params_in = np.array([item.resize_factor, item.bb_center_cam[0], item.bb_center_cam[1]])
    ang = np.linalg.norm(item.rot_vec)
    rv = item.rot_vec
    if ang > np.pi:
        rv = rv / ang * (np.pi - ang)
    return (img, params_in), (rv, item.pos)


def ds_pose_train(ds_pose: Union[DsPoseGen, DsPoseLoader]):
    def f():
        for item in ds_pose.gen():
            yield ds_item_to_numbers(item)

    return f


def build_ds(ds_loader: Union[DsPoseGen, DsPoseLoader], batch_size: int) -> tf.data.Dataset:
    out_sig = (
        (
            tf.TensorSpec(shape=(*ds_loader.img_out_size, 6), dtype=tf.float32),
            tf.TensorSpec(shape=(3,), dtype=tf.float32),
        ),
        (
            tf.TensorSpec(shape=(3,), dtype=tf.float32),
            tf.TensorSpec(shape=(3,), dtype=tf.float32),
        ),
    )

    ds = tf.data.Dataset \
        .from_generator(ds_pose_train(ds_loader), output_signature=out_sig) \
        .batch(batch_size)
    return ds


def build_datasets(cfg: Config, objs: Dict, obj_glob_id: str, img_size: int = 128) -> Tuple[
    DsPoseGen, DsPoseLoader, tf.data.Dataset, tf.data.Dataset]:
    ds_path = cfg.sds_root_path / cfg.dataset_name
    ds_pose_train = DsPoseGen(objs, obj_glob_id, img_size, aug_enabled=True, hist_sz=cfg.n_img_vis,
                              multi_threading=True)
    ds_pose_val = DsPoseLoader(ds_path, objs, img_size, obj_glob_id, hist_sz=cfg.n_img_vis)
    ds_train = build_ds(ds_pose_train, cfg.batch_size)
    ds_val = build_ds(ds_pose_val, cfg.batch_size)
    return ds_pose_train, ds_pose_val, ds_train, ds_val


def get_subdir_name(dt: datetime, obj_glob_id: str):
    return f'{datetime_str(dt)}_{obj_glob_id}'


def main(cfg: Config) -> int:
    print(cfg)

    objs = load_objs(cfg.sds_root_path, cfg.dataset_name, load_meshes=True)
    id_num_to_glob = {obj['id_num']: oid for oid, obj in objs.items()}
    obj_glob_id = id_num_to_glob[cfg.model_id_num]
    print(f'Object glob id chosen: {obj_glob_id}')
    dsp_train, dsp_val, ds_train, ds_val = build_datasets(cfg, objs, obj_glob_id)

    out_subdir_name = get_subdir_name(datetime.now(), obj_glob_id)
    out_path = cfg.train_root_path / out_subdir_name
    weights_out_path = out_path / 'weights'
    weights_out_path.mkdir(parents=True, exist_ok=True)
    params_out_path = out_path / 'params'
    params_out_path.mkdir(parents=True, exist_ok=True)

    tf_set_gpu_incremental_memory_growth()

    inp, out = build_pose_layers()
    model = tf.keras.models.Model(inputs=inp, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
        loss=[
            RotVecLoss(256),
            tf.keras.losses.MeanSquaredError(),
        ],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            (weights_out_path / 'last' / 'last.pb').as_posix(),
            save_best_only=False, save_weights_only=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            (weights_out_path / 'best' / 'best.pb').as_posix(),
            save_best_only=True, save_weights_only=True,
        ),
        tf.keras.callbacks.TensorBoard(
            out_path.as_posix(),
            histogram_freq=0,
            batch_size=cfg.batch_size,
            write_graph=False,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=8, min_lr=1e-8,
        ),
    ]
    model.fit(
        ds_train,
        epochs=cfg.epochs,
        steps_per_epoch=cfg.train_steps,
        validation_data=ds_val,
        validation_steps=cfg.val_steps,
        callbacks=callbacks,
    )

    return 0


if __name__ == '__main__':
    run_and_exit(Config, main, 'Convert ply object to meters, restructure objects\' metadata')
