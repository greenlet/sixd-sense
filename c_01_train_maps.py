import json
import sys
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
import shutil
from typing import Optional, Dict, List, Any, Tuple

import cv2
import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLUT import *
from OpenGL.GLU import *
import h5py
import yaml

import numpy as np
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
import pymesh
from scipy.spatial.transform import Rotation as R
import tensorflow as tf

from sds.data.index import load_cache_ds_index, DsIndex
from sds.data.loader import DsLoader, load_objs
from sds.model.model import bifpn_init, bifpn_layer, final_upscale
from sds.model.params import ScaledParams
from sds.utils import utils
from sds.utils.utils import datetime_str


def set_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


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
    models_subdir: str = Field(
        'models',
        description='Models subdirectory. Has to contain ply files (default: "models")',
        required=False,
        cli=('--models-subdir',),
    )
    phi: int = Field(
        ...,
        description=f'Model phi number. Must be: 0 <= PHI <= {ScaledParams.phi_max()}',
        required=True,
        cli=('--phi',),
    )
    freeze_bn: bool = Field(
        False,
        description='If set, BatchNormalization layers will be disabled during training',
        required=False,
        cli=('--freeze-bn',),
    )
    train_root_path: Path = Field(
        ...,
        description='Training directory. For each new training new subdirectory will be created '
                    'in LOGS_ROOT_PATH correspond',
        required=True,
        cli=('--train-root-path',)
    )
    weights_to_use: Optional[str] = Field(
        'none',
        description='Can be either full path to model weights, or "last" word meaning that last weights from '
                    'WEIGHTS_ROOT_PATH will be used, or WEIGHTS_ROOT_PATH subdirectory prefix (if multiple directories '
                    'match prefix last one will be chosen. If not set or set to "none", no weights will be loaded',
        cli=('--weights-to-use',),
    )
    new_learning_subdir: bool = Field(
        False,
        description='If True and there are weights resolved from WEIGHTS_TO_USE located in subdir of TRAIN_ROOT_PATH, '
                    'training will be proceeded from the last point of those weights and in a new subdir',
        cli=('--new-learning-subdir',),
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
    train_split_perc: float = Field(
        90.0,
        required=False,
        description='Train dataset size relatively to whole dataset, in percents',
        cli=('--train-split-perc',)
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
    debug: bool = Field(
        False,
        description='Debug mode',
        required=False,
        cli=('--debug',),
    )


class TripleLoss(tf.keras.losses.Loss):
    def __init__(self, n_classes: int, weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        super().__init__()
        self.n_classes = n_classes
        self.weights = tf.constant(weights)
        self.seg_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.noc_loss = tf.keras.losses.MeanSquaredError()
        self.norms_loss = tf.keras.losses.CosineSimilarity()

    def call(self, y_true, y_pred, *args):
        # print('y_true', type(y_true), type(y_true[0]))
        # print('y_pred', type(y_pred), type(y_pred[0]))
        print('y_true', type(y_true), y_true.shape)
        print('y_pred', type(y_pred), y_pred.shape)
        noc_true, noc_pred = y_true[..., :3], y_pred[..., :3]
        norms_true, norms_pred = y_true[..., 3:6], y_pred[..., 3:6]
        seg_true, seg_pred = y_true[..., -1:], y_pred[..., 6:]
        seg_true = tf.cast(seg_true, tf.int32)
        # print('noc', noc_true.shape, noc_pred.shape)
        # print('norms', norms_true.shape, norms_pred.shape)
        # print('seg', seg_true.shape, seg_pred.shape)
        noc_loss = self.noc_loss(noc_true, noc_pred)
        norms_loss = self.norms_loss(norms_true, norms_pred) + 1
        seg_loss = self.seg_loss(seg_true, seg_pred)
        return self.weights[0] * noc_loss + self.weights[1] * norms_loss + self.weights[2] * seg_loss


class MseNZLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__(name=self.__class__.__name__)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mask = tf.cast(y_true, tf.bool)
        y_true, y_pred = y_true[mask], y_pred[mask]
        y_true = (tf.cast(y_true, tf.float32) - 127.5) / 128.0
        diff = (y_true - y_pred) ** 2
        res = tf.reduce_mean(diff)
        return res


class CosNZLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__(name=self.__class__.__name__)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mask = tf.cast(y_true, tf.bool)
        y_true, y_pred = y_true[mask], y_pred[mask]
        y_true = (tf.cast(y_true, tf.float32) - 127.5) / 128.0
        y_true, y_pred = tf.reshape(y_true, (-1, 3)), tf.reshape(y_pred, (-1, 3))
        loss_cos = tf.reduce_sum(y_true * y_pred, axis=-1) + 1
        y_pred_norm = tf.norm(y_pred, axis=-1)
        loss_cos /= y_pred_norm
        loss_norm = 0.05 * (y_pred_norm - 1) ** 2
        res = tf.reduce_mean(loss_cos + loss_norm)
        return res


class ScceNZLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__(name=self.__class__.__name__)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mask = tf.cast(y_true, tf.bool)[..., 0]
        y_true, y_pred = y_true[mask], y_pred[mask]
        # y_true, y_pred = tf.boolean_mask(y_true, mask), tf.boolean_mask(y_pred, mask)
        return self.loss(y_true, y_pred)


def build_ds(ds_loader: DsLoader, batch_size: int) -> tf.data.Dataset:
    out_sig = (
        tf.TensorSpec(shape=(*ds_loader.img_size, 3), dtype=tf.float32),
        (
            tf.TensorSpec(shape=(*ds_loader.img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(*ds_loader.img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(*ds_loader.img_size, 1), dtype=tf.int32),
        )
    )
    ds = tf.data.Dataset \
        .from_generator(ds_loader.gen, output_signature=out_sig) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)
    return ds


def build_datasets(cfg: Config) -> Tuple[DsLoader, DsLoader, tf.data.Dataset, tf.data.Dataset]:
    ds_path = cfg.sds_root_path / cfg.target_dataset_name
    params = ScaledParams(cfg.phi)
    ds_index = load_cache_ds_index(
        ds_path, train_ratio=cfg.train_split_perc / 100,
        force_reload=False,
    )
    objs = load_objs(cfg.sds_root_path, cfg.target_dataset_name, cfg.distractor_dataset_name)
    ds_loader_train = DsLoader(
        ds_index, objs, is_training=True, img_size=params.input_size,
        shuffle_enabled=True, aug_enabled=True,
    )
    ds_loader_val = DsLoader(
        ds_index, objs, is_training=True, img_size=params.input_size,
        shuffle_enabled=True, aug_enabled=False,
    )
    ds_val = build_ds(ds_loader_train, cfg.batch_size)
    ds_train = build_ds(ds_loader_val, cfg.batch_size)
    return ds_loader_train, ds_loader_val, ds_train, ds_val


def build_model(n_classes: int, cfg: Config) -> tf.keras.models.Model:
    params = ScaledParams(cfg.phi)
    image_input = tf.keras.Input(params.input_shape, dtype=tf.float32)
    _, bb_feature_maps = params.backbone_class(input_tensor=image_input, freeze_bn=cfg.freeze_bn)

    fpn_init_feature_maps = bifpn_init(bb_feature_maps, params.bifpn_width, name='BifpnInit')
    fpn_feature_maps = bifpn_layer(fpn_init_feature_maps, params.bifpn_width, name='Bifpn1')

    noc_out = final_upscale(fpn_feature_maps, 3, name='Noc')
    norms_out = final_upscale(fpn_feature_maps, 3, name='Normals')
    seg_out = final_upscale(fpn_feature_maps, n_classes + 1, name='Segmentation')

    model = tf.keras.models.Model(inputs=[image_input], outputs=[noc_out, norms_out, seg_out])
    return model


def get_subdir_name(dt: datetime, ds_name: str, n_train: int, n_val: int):
    return f'{datetime_str(dt)}_{ds_name}_t{n_train}_v{n_val}'


def main(cfg: Config) -> int:
    print(cfg)

    dsl_train, dsl_val, ds_train, ds_val = build_datasets(cfg)
    # it = iter(ds_train)
    # item = next(it)
    # print(item[0].shape, item[0].dtype, len(item[1]))
    # for x in item[1]:
    #     print(x.shape, x.dtype, tf.reduce_min(x), tf.reduce_max(x), tf.reduce_mean(x))
    # it = iter(ds_val)
    # item = next(it)
    # print(item[0].shape, item[0].dtype, len(item[1]))
    # for x in item[1]:
    #     print(x.shape, x.dtype, tf.reduce_min(x), tf.reduce_max(x), tf.reduce_mean(x))

    out_subdir_name = get_subdir_name(
        datetime.now(), cfg.target_dataset_name, len(dsl_train), len(dsl_val))
    out_path = cfg.train_root_path / out_subdir_name
    weights_out_path = out_path / 'weights'
    weights_out_path.mkdir(parents=True, exist_ok=True)

    n_classes = len(dsl_train.objs)
    set_memory_growth()

    model = build_model(n_classes, cfg)
    # loss = TripleLoss(n_classes)
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
        loss=[
            MseNZLoss(),
            CosNZLoss(),
            ScceNZLoss(),
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
            factor=0.5, patience=7, min_lr=1e-8,
        )
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
