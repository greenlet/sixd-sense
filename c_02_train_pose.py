import math
import os
import shutil
import sys
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
from sds.model.losses import MseNZLoss, CosNZLoss, RotVecLoss, TransLoss
from sds.model.model_pose import build_pose_layers, build_pose_layers_2d
from sds.model.params import ScaledParams
from sds.model.processing import tf_float_to_img, tf_img_to_float, np_img_to_float
from sds.utils.tf_utils import tf_set_gpu_incremental_memory_growth
from sds.utils.utils import datetime_str, gen_colors
from sds.utils.ds_utils import load_objs
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
    img_size: int = Field(
        256,
        description='Training input image size. Currently only value 256 is supported',
        required=True,
        cli=('--img-size',),
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
    cam_f, cam_cx, cam_cy = item.cam_mat[0, 0], item.cam_mat[0, 2], item.cam_mat[1, 2]
    bb_center_x, bb_center_y = item.bb_center
    params_in = np.array([cam_f, cam_cx, cam_cy, bb_center_x, bb_center_y, item.resize_factor])
    ang = np.linalg.norm(item.rot_vec)
    rot_vec = item.rot_vec
    if ang > np.pi:
        rot_vec = rot_vec / ang * (ang - 2 * np.pi)
    return (img, params_in), (rot_vec, item.pos)


def ds_pose_train(ds_pose: Union[DsPoseGen, DsPoseLoader]):
    def f():
        for item in ds_pose.gen():
            yield ds_item_to_numbers(item)

    return f


def ds_pose_target(nb: int):
    def f(inp: Tuple[tf.Tensor, tf.Tensor], out: Tuple[tf.Tensor, tf.Tensor]):
        img, params = inp
        img_noc = img[..., :3]
        rvec, pos = out
        # Batch size
        sim = []
        mask = tf.greater(tf.reduce_max(img_noc, axis=-1), 0)
        img_noc = tf.cast(img_noc, tf.float32)
        for i in range(nb):
            img1 = img_noc[i]
            mask1 = mask[i]
            sim.append(tf.concat([[i, i], rvec[i], [0]], 0))
            for j in range(i + 1, nb):
                img2 = img_noc[j]
                mask2 = mask[j]
                diff = tf.abs(img2 - img1) / 255.0
                diff = tf.reduce_mean(diff[mask1 | mask2])
                sim.append(tf.concat([[i, j], rvec[j], tf.reshape(diff, (1,))], 0))
        img = tf_img_to_float(img)
        return (img, params), (sim, pos)

    return f


def build_ds(ds_loader: Union[DsPoseGen, DsPoseLoader], batch_size: int) -> tf.data.Dataset:
    out_sig = (
        (
            tf.TensorSpec(shape=(*ds_loader.img_out_size, 6), dtype=tf.uint8),
            tf.TensorSpec(shape=(6,), dtype=tf.float32),
        ),
        (
            tf.TensorSpec(shape=(3,), dtype=tf.float32),
            tf.TensorSpec(shape=(3,), dtype=tf.float32),
        ),
    )

    ds = tf.data.Dataset \
        .from_generator(ds_pose_train(ds_loader), output_signature=out_sig) \
        .batch(batch_size) \
        .map(ds_pose_target(batch_size)) \
        .prefetch(tf.data.AUTOTUNE)
    return ds


def build_datasets(cfg: Config, objs: Dict, obj_glob_id: str) -> Tuple[
    DsPoseGen, DsPoseLoader, tf.data.Dataset, tf.data.Dataset]:
    ds_path = cfg.sds_root_path / cfg.dataset_name
    ds_pose_train = DsPoseGen(objs, obj_glob_id, cfg.img_size, aug_enabled=True, hist_sz=cfg.n_img_vis,
                              multi_threading=True)
    ds_pose_val = DsPoseLoader(ds_path, objs, cfg.img_size, obj_glob_id, hist_sz=cfg.n_img_vis)
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

    inp, out = build_pose_layers(cfg.img_size)
    model = tf.keras.models.Model(inputs=inp, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=cfg.learning_rate),
        loss=[
            RotVecLoss(100),
            TransLoss(),
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
            factor=0.5, patience=5, min_lr=1e-8,
        ),
    ]

    ds_train, ds_val = ds_val, ds_train
    model.fit(
        ds_train,
        epochs=cfg.epochs,
        steps_per_epoch=cfg.train_steps,
        validation_data=ds_val,
        validation_steps=cfg.val_steps,
        callbacks=callbacks,
    )

    dsp_train.stop()
    return 0


if __name__ == '__main__':
    run_and_exit(Config, main, 'Train pose network')
