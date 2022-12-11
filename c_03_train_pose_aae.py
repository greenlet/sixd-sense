import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict

import numpy as np
import pymesh
import tensorflow as tf
from numpy import ndarray
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
from tqdm import tqdm, trange

from sds.data.ds_pose_gen_mp import DsPoseGenMp
from sds.model.model_pose_aae import build_aae_layers, AaeLoss
from sds.utils.tf_utils import tf_set_gpu_incremental_memory_growth

tf_set_gpu_incremental_memory_growth()

from sds.data.ds_pose_gen import DsPoseGen
from sds.data.ds_pose_loader import DsPoseLoader
from sds.data.index import load_cache_ds_index
from sds.data.ds_loader import DsLoader
from sds.data.utils import DsPoseItem
from sds.model.losses import MseNZLoss, CosNZLoss, RotVecLoss, TransLoss
from sds.model.model_pose import build_pose_layers, RotHeadType, ROT_HEAD_TYPE_VALUES
from sds.model.model_pose_graph import build_hybrid_layers
from sds.model.params import ScaledParams
from sds.model.utils import tf_img_to_float, tf_float_to_img, np_img_to_float
from sds.utils.utils import datetime_str, gen_colors
from sds.utils.ds_utils import load_objs, load_mesh
from train_utils import build_maps_model, color_segmentation, normalize, ds_pose_preproc, ds_pose_mp_preproc


class Config(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    obj_path: Path = Field(
        ...,
        description='Path to 3d object model (stl, obj, ...) which will be used for training AAE.',
        cli=('--obj-path',),
    )
    obj_id: str = Field(
        ...,
        description='Object unique identifier.',
        cli=('--obj-id',),
    )
    train_root_path: Path = Field(
        ...,
        description='Training directory. For each new training new subdirectory will be created.',
        required=True,
        cli=('--train-root-path',),
    )
    img_size: int = Field(
        256,
        description='Encoder input and Decoder output image size.',
        required=False,
        cli=('--img-size',),
    )
    learning_rate: float = Field(
        ...,
        required=True,
        description='Learning rate.',
        cli=('--learning-rate',),
    )
    loss_bootstrap_ratio: int = Field(
        4,
        required=False,
        description='AAE loss bootstrap ratio. For each input and output image pair '
                    'loss will be calculated only for top IMG_SIZE * IMG_SIZE / LOSS_BOOTSTRAP_RATIO '
                    'quadratic differences.',
        cli=('--loss-bootstrap-ratio',),
    )
    iterations: int = Field(
        ...,
        required=True,
        description='Number of training iterations.',
        cli=('--iterations',),
    )
    batch_size: int = Field(
        ...,
        required=True,
        description='Training batch size (number of images for single forward/backward network iteration).',
        cli=('--batch-size',),
    )
    loss_acc_interval: int = Field(
        ...,
        required=True,
        description='Number of iterations for calculating mean loss value and writing it to tensorboard.',
        cli=('--loss-acc-interval',),
    )
    model_save_interval: int = Field(
        ...,
        required=True,
        description='Number of iterations between consecutive writing model weights to disk.',
        cli=('--model-save-interval',),
    )
    pose_gen_workers: int = Field(
        0,
        description='Number of processes in order to parallelize online rendering for Pose dataset generation. '
                    'If not set or set to value <= 0, rendering is a single separate thread of the main process.',
        required=False,
        cli=('--pose-gen-workers',),
    )


def get_subdir_name(cfg: Config, obj_glob_id: str):
    dt_str = datetime_str()
    return f'obj_{cfg.obj_id}--imgsz_{cfg.img_size}--{dt_str}'


def tile_images(imgs_true: tf.Tensor, imgs_pred: tf.Tensor, max_cols: int = 2) -> tf.Tensor:
    imgs_true, imgs_pred = imgs_true.numpy(), imgs_pred.numpy()
    batch_size, img_sz = imgs_true.shape[0], imgs_true.shape[1]
    if batch_size <= max_cols:
        rows, cols = 1, batch_size
    else:
        rows, cols = batch_size // max_cols + batch_size % max_cols, max_cols
    height, width = 2 * rows * img_sz, 2 * cols * img_sz
    img = np.zeros((height, width, 3), np.float32)
    for ib in range(batch_size):
        r, c = 2 * (ib // cols) * img_sz, 2 * (ib % cols) * img_sz
        img[r:r + img_sz, c:c + img_sz] = imgs_true[ib, :, :, :3]
        img[r:r + img_sz, c + img_sz:c + 2 * img_sz] = imgs_pred[ib, :, :, :3]
        img[r + img_sz:r + 2 * img_sz, c:c + img_sz] = imgs_true[ib, :, :, 3:]
        img[r + img_sz:r + 2 * img_sz, c + img_sz:c + 2 * img_sz] = imgs_pred[ib, :, :, 3:]
    return tf.convert_to_tensor(img)


def train_aae(cfg: Config) -> int:
    print(cfg)

    mesh, mesh_dict = load_mesh(cfg.obj_path)
    verts_diff = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    diameter = np.abs(verts_diff).max()
    objs = {
        cfg.obj_id: {
            'mesh': mesh_dict,
            'diameter': diameter,
        }
    }

    render_base_size = (1280, 1024)
    channels = 6

    if cfg.pose_gen_workers > 0:
        out_sig = (
            (
                tf.TensorSpec(shape=(cfg.batch_size, cfg.img_size, cfg.img_size, 6), dtype=tf.uint8),
                tf.TensorSpec(shape=(cfg.batch_size, 6,), dtype=tf.float32),
            ),
            (
                tf.TensorSpec(shape=(cfg.batch_size, 3,), dtype=tf.float32),
                tf.TensorSpec(shape=(cfg.batch_size, 3,), dtype=tf.float32),
            ),
        )
        ds_pose_gen = DsPoseGenMp(objs, cfg.obj_id, cfg.img_size, render_base_size, aug_enabled=True,
                                  batch_size=cfg.batch_size, n_workers=cfg.pose_gen_workers)
        ds_train = tf.data.Dataset \
            .from_generator(ds_pose_mp_preproc(ds_pose_gen), output_signature=out_sig) \
            .prefetch(tf.data.AUTOTUNE)
    else:
        out_sig = (
            (
                tf.TensorSpec(shape=(cfg.img_size, cfg.img_size, 6), dtype=tf.uint8),
                tf.TensorSpec(shape=(6,), dtype=tf.float32),
            ),
            (
                tf.TensorSpec(shape=(3,), dtype=tf.float32),
                tf.TensorSpec(shape=(3,), dtype=tf.float32),
            ),
        )
        ds_pose_gen = DsPoseGen(objs, cfg.obj_id, cfg.img_size, render_base_size, aug_enabled=True,
                                multi_threading=False)
        ds_train = tf.data.Dataset \
            .from_generator(ds_pose_preproc(ds_pose_gen), output_signature=out_sig) \
            .batch(cfg.batch_size) \
            .prefetch(tf.data.AUTOTUNE)

    # for item in ds_train:
    #     print(item)
    #     break

    out_subdir_name = get_subdir_name(cfg, cfg.obj_id)
    out_path = cfg.train_root_path / out_subdir_name
    weights_out_path = out_path / 'weights'
    weights_out_path.mkdir(parents=True, exist_ok=True)
    weights_fpath = weights_out_path / 'last.pb'
    logs_out_path = out_path / 'logs'
    logs_out_path.mkdir(parents=True, exist_ok=True)

    inp, out = build_aae_layers(img_size=cfg.img_size, batch_size=cfg.batch_size)
    model = tf.keras.models.Model(inputs=inp, outputs=out)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=cfg.learning_rate)
    aae_loss = AaeLoss(cfg.batch_size, cfg.img_size, channels, cfg.loss_bootstrap_ratio)

    @tf.function
    def train_step():
        x_train, y_train = next(ds_train_iter)
        y_true = tf.cast(x_train[0], tf.float32) / 255.
        with tf.GradientTape() as tape:
            y_pred = model(y_true, training=True)
            ls = aae_loss(y_true, y_pred)
        grads = tape.gradient(ls, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return ls, y_true, y_pred

    ds_train_iter = iter(ds_train)
    pbar = trange(cfg.iterations, desc=f'AAE Train', unit='batch')
    loss_acc = tf.constant(0, tf.float32)
    loss_steps = 0
    writer = tf.summary.create_file_writer(logs_out_path.as_posix())
    writer.set_as_default()
    if cfg.iterations // cfg.loss_acc_interval <= 100:
        imgs_write_interval = cfg.loss_acc_interval
    else:
        imgs_write_interval = cfg.iterations // 100
    written = False
    for i in pbar:
        loss, y_true, y_pred = train_step()
        loss_acc = loss_acc + loss

        loss_steps += 1
        step = i + 1
        if step % cfg.loss_acc_interval == 0 or step == cfg.iterations:
            loss_acc /= loss_steps
            tf.summary.scalar('aae_loss', loss_acc, step)
            written = True
            loss_acc, loss_steps = tf.constant(0, tf.float32), 0

        if step % imgs_write_interval == 0 or step == cfg.iterations:
            img = tile_images(y_true, y_pred)
            tf.summary.image('predict', img[None, ...], step)
            written = True

        if step % cfg.model_save_interval == 0 or step == cfg.iterations:
            model.save_weights(weights_fpath.as_posix())

        if written:
            writer.flush()
            written = False

        pbar.set_postfix_str(f'Train. loss: {loss.numpy():.3f}')

    pbar.close()
    ds_pose_gen.stop()

    return 0


if __name__ == '__main__':
    def exception_handler(ex: BaseException) -> int:
        raise ex
    run_and_exit(Config, train_aae, 'Train AAE network', exception_handler=exception_handler)
