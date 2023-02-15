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
from tqdm import tqdm, trange

from sds.data.ds_pose_gen_mp import DsPoseGenMp
from sds.utils.tf_utils import tf_set_gpu_incremental_memory_growth
tf_set_gpu_incremental_memory_growth()

from sds.data.ds_pose_gen import DsPoseGen
from sds.data.ds_pose_loader import DsPoseLoader
from sds.data.index import load_cache_ds_index
from sds.data.ds_loader import DsLoader
from sds.data.utils import DsPoseItem
from sds.model.losses import MapsMseLoss, MapsCosLoss, RotVecLoss, TransLoss
from sds.model.model_pose import build_pose_layers, RotHeadType, ROT_HEAD_TYPE_VALUES
from sds.model.params import ScaledParams
from sds.model.utils import tf_img_to_float, tf_float_to_img, np_img_to_float
from sds.utils.utils import datetime_str, gen_colors
from sds.utils.ds_utils import load_objs
from train_utils import build_maps_model, color_segmentation, normalize, ds_pose_preproc, ds_pose_mp_preproc


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
    obj_models_subdir: str = Field(
        'models',
        description='Objects\' models subdirectory. Has to contain ply files (default: "models")',
        required=False,
        cli=('--obj-models-subdir',),
    )
    obj_id_num: int = Field(
        ...,
        description='Unique numeric identifier of an object of choice. Object is from target set. '
                    'All objects numbered from 1 to len(objects) ones in a global initialization step',
        required=True,
        cli=('--obj-id-num',),
    )
    rot_head_type: str = Field(
        ...,
        description=f'Model architecture type. Can have values: {", ".join(ROT_HEAD_TYPE_VALUES)}',
        required=True,
        cli=('--rot-head-type',)
    )
    img_size: int = Field(
        256,
        description='Training input image size',
        required=False,
        cli=('--img-size',),
    )
    rot_grid_size: int = Field(
        128,
        description=f'Rotation parameters discretization number. For {RotHeadType.Conv2d} '
                    f'head architecture it is angle space discretization. For {RotHeadType.Conv3d} '
                    f'it is rotation vector discretization.',
        required=False,
        cli=('--rot-grid-size',),
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
    pose_gen_workers: int = Field(
        0,
        description='Number of processes in order to parallelize online rendering for Pose dataset generation. '
                    'If not set or set to value <= 0, rendering is a single separate thread of the main process.',
        required=False,
        cli=('--pose-gen-workers',),
    )


def create_negative_target_(nb: int):
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
            for j in range(i + 1, min(i + 2, nb)):
                img2 = img_noc[j]
                mask2 = mask[j]
                diff = tf.abs(img2 - img1) / 255.0
                diff = tf.reduce_mean(diff[mask1 | mask2])
                sim.append(tf.concat([[i, j], rvec[j], tf.reshape(diff, (1,))], 0))
        img = tf_img_to_float(img)
        return (img, params), (sim, pos)

    return f


def create_negative_target(nb: int):
    def f(inp: Tuple[tf.Tensor, tf.Tensor], out: Tuple[tf.Tensor, tf.Tensor]):
        img, params = inp
        img_noc = img[..., :3]
        rvec, pos = out
        # Batch size
        sim = []
        mask = tf.greater(tf.reduce_max(img_noc, axis=-1), 0)
        img_noc = tf.cast(img_noc, tf.float32)
        diff = tf.abs(img_noc[1:] - img_noc[:-1]) / 255.0
        for i in range(nb):
            sim.append(tf.concat([[i, i], rvec[i], [0]], 0))
        for i in range(nb - 1):
                d = tf.reduce_mean(diff[i][mask[i] | mask[i + 1]])
                sim.append(tf.concat([[i, i + 1], rvec[i + 1], tf.reshape(d, (1,))], 0))
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
        .from_generator(ds_pose_preproc(ds_loader), output_signature=out_sig) \
        .batch(batch_size) \
        .map(create_negative_target(batch_size)) \
        .prefetch(tf.data.AUTOTUNE)
    return ds


def build_ds_mp(ds_pose_mp: DsPoseGenMp):
    out_sig = (
        (
            tf.TensorSpec(shape=(ds_pose_mp.batch_size, *ds_pose_mp.img_out_size, 6), dtype=tf.uint8),
            tf.TensorSpec(shape=(ds_pose_mp.batch_size, 6,), dtype=tf.float32),
        ),
        (
            tf.TensorSpec(shape=(ds_pose_mp.batch_size, 3,), dtype=tf.float32),
            tf.TensorSpec(shape=(ds_pose_mp.batch_size, 3,), dtype=tf.float32),
        ),
    )
    ds = tf.data.Dataset \
        .from_generator(ds_pose_mp_preproc(ds_pose_mp), output_signature=out_sig) \
        .map(create_negative_target(ds_pose_mp.batch_size)) \
        .prefetch(tf.data.AUTOTUNE)
    return ds


def build_datasets(cfg: Config, objs: Dict, obj_glob_id: str) -> Tuple[
        Union[DsPoseGen, DsPoseGenMp], DsPoseLoader, tf.data.Dataset, tf.data.Dataset]:
    ds_path = cfg.sds_root_path / cfg.dataset_name
    render_base_size = (1280, 1024)
    if cfg.pose_gen_workers > 0:
        ds_pose_train = DsPoseGenMp(objs, obj_glob_id, cfg.img_size, render_base_size, aug_enabled=True, batch_size=cfg.batch_size, n_workers=cfg.pose_gen_workers)
        ds_train = build_ds_mp(ds_pose_train)
    else:
        ds_pose_train = DsPoseGen(objs, obj_glob_id, cfg.img_size, render_base_size, aug_enabled=True, multi_threading=True)
        ds_train = build_ds(ds_pose_train, cfg.batch_size)

    ds_pose_val = DsPoseLoader(ds_path, objs, cfg.img_size, obj_glob_id)
    ds_val = build_ds(ds_pose_val, cfg.batch_size)

    return ds_pose_train, ds_pose_val, ds_train, ds_val


def get_subdir_name(cfg: Config, obj_glob_id: str):
    dt = datetime.now()
    dt_str = datetime_str(dt)
    return f'ds_{cfg.dataset_name}--head_{cfg.rot_head_type}--imgsz_{cfg.img_size}--' \
           f'rotsz_{cfg.rot_grid_size}--oid_{obj_glob_id}--{dt_str}'


def main(cfg: Config) -> int:
    print(cfg)

    objs = load_objs(cfg.sds_root_path, cfg.dataset_name, load_meshes=True)
    id_num_to_glob = {obj['id_num']: oid for oid, obj in objs.items()}
    obj_glob_id = id_num_to_glob[cfg.obj_id_num]
    print(f'Object glob id chosen: {obj_glob_id}')
    dsp_train, dsp_val, ds_train, ds_val = build_datasets(cfg, objs, obj_glob_id)

    out_subdir_name = get_subdir_name(cfg, obj_glob_id)
    out_path = cfg.train_root_path / out_subdir_name
    weights_out_path = out_path / 'weights'
    weights_out_path.mkdir(parents=True, exist_ok=True)
    params_out_path = out_path / 'params'
    params_out_path.mkdir(parents=True, exist_ok=True)

    rot_head_type = RotHeadType(cfg.rot_head_type)
    assert rot_head_type == RotHeadType.Conv3d, f'{RotHeadType.Conv2d} is not supported yet'

    inp, out = build_pose_layers(rot_head_type, cfg.img_size)
    model = tf.keras.models.Model(inputs=inp, outputs=out)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=cfg.learning_rate)
    rot_vec_loss = RotVecLoss(cfg.rot_grid_size)
    trans_loss = TransLoss()

    callbacks_ = [
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
            write_graph=False,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
        ),
        # tf.keras.callbacks.ReduceLROnPlateau(
        #     factor=0.5, patience=5, min_lr=1e-8,
        # ),
    ]
    callbacks = tf.keras.callbacks.CallbackList(callbacks_, add_history=True, model=model)

    # @tf.function
    def train_step():
        x_train, (rv_true, tr_true) = next(ds_train_iter)
        with tf.GradientTape() as tape:
            rot_pred, tr_pred = model(x_train, training=True)
            rv_loss = rot_vec_loss.call(rv_true, rot_pred)
            # tr_loss = trans_loss.call(tr_true, tr_pred)
            tr_loss = tf.constant(0.0)
            loss = rv_loss + tr_loss
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return rv_loss, tr_loss, loss

    # @tf.function
    def val_step():
        x_val, (rv_true, tr_true) = next(ds_val_iter)
        rot_pred, tr_pred = model(x_val, training=False)
        rv_loss = rot_vec_loss.call(rv_true, rot_pred)
        # tr_loss = trans_loss.call(tr_true, tr_pred)
        tr_loss = tf.constant(0.0)
        loss = rv_loss + tr_loss
        return rv_loss, tr_loss, loss

    ds_train_iter, ds_val_iter = iter(ds_train), iter(ds_val)
    # ds_train_iter, ds_val_iter = ds_val_iter, ds_train_iter

    logs = {}
    callbacks.on_train_begin(logs)
    for epoch in range(cfg.epochs):
        callbacks.on_epoch_begin(epoch, logs=logs)

        pbar = trange(cfg.train_steps, desc=f'Epoch {epoch + 1}', unit='batch')
        for step in pbar:
            model.reset_states()

            callbacks.on_batch_begin(step, logs=logs)
            callbacks.on_train_batch_begin(step, logs=logs)

            rv_loss, tr_loss, loss = train_step()
            logs.update({
                'rot_vec_loss': rv_loss,
                'trans_loss': tr_loss,
                'loss': loss,
            })

            callbacks.on_train_batch_end(step, logs=logs)
            callbacks.on_batch_end(step, logs=logs)

            pbar.set_postfix_str(f'Train. rv_loss: {rv_loss.numpy():.3f}. '
                                 f'tr_loss: {tr_loss.numpy():.3f}. loss: {loss.numpy():.3f}')
        pbar.close()

        pbar = trange(cfg.val_steps, desc=f'Epoch {epoch + 1}. Val', unit='batch')
        for step in pbar:
            model.reset_states()

            callbacks.on_batch_begin(step, logs=logs)
            callbacks.on_test_batch_begin(step, logs=logs)

            rv_loss, tr_loss, loss = val_step()
            logs.update({
                'val_rot_vec_loss': rv_loss,
                'val_trans_loss': tr_loss,
                'val_loss': loss,
            })

            callbacks.on_test_batch_end(step, logs=logs)
            callbacks.on_batch_end(step, logs=logs)

            pbar.update(step)
            pbar.set_postfix_str(f'Val. rv_loss: {rv_loss.numpy():.3f}. '
                                 f'tr_loss: {tr_loss.numpy():.3f}. loss: {loss.numpy():.3f}')
        pbar.close()
        callbacks.on_epoch_end(epoch, logs=logs)

    callbacks.on_train_end(logs=logs)

    dsp_train.stop()

    return 0


if __name__ == '__main__':
    def exception_handler(ex: BaseException) -> int:
        raise ex

    run_and_exit(Config, main, 'Train pose network', exception_handler=exception_handler)
