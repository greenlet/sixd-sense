import json
import math
import sys
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
import shutil
from typing import Optional, Dict, List, Any, Tuple


import numpy as np
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
import pymesh
from scipy.spatial.transform import Rotation as R
import tensorflow as tf

from sds.data.index import load_cache_ds_index, DsIndex
from sds.data.loader import DsLoader, load_objs
from sds.model.losses import MseNZLoss, CosNZLoss, SparseCategoricalCrossEntropyNZLoss
from sds.model.model import bifpn_init, bifpn_layer, final_upscale
from sds.model.params import ScaledParams
from sds.model.processing import float_to_img
from sds.utils import utils
from sds.utils.utils import datetime_str, gen_colors


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
        ds_index, objs, is_training=False, img_size=params.input_size,
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


class PredictionVisualizer(tf.keras.callbacks.Callback):
    log_path: Path
    writer: Optional[tf.summary.SummaryWriter] = None
    model: Optional[tf.keras.models.Model] = None

    def __init__(self, log_path: Path, n_samples: int, ds_loader: DsLoader):
        super().__init__()
        self.log_path = log_path
        self.n_samples = n_samples
        self.n_cols = math.ceil(math.sqrt(self.n_samples))
        self.n_rows = math.ceil(self.n_samples / self.n_cols)
        self.img_size = ds_loader.img_size
        self.img_w, self.img_h = self.img_size
        self.grid_shape = tf.TensorShape((self.img_h * self.n_rows, self.img_w * self.n_cols, 3))
        self.ds = build_ds(ds_loader, self.n_samples)
        self.ds_iter = iter(self.ds)
        self.colors = gen_colors()
        self.epoch = -1

    def images_to_grid(self, imgs: tf.Tensor) -> tf.Tensor:
        grid = np.zeros(self.grid_shape, np.uint8)
        for i in range(self.n_samples):
            if i == imgs.shape[0]:
                break
            ir, ic = i // self.n_cols, i % self.n_cols
            r1, r2 = ir * self.img_h, (ir + 1) * self.img_h
            c1, c2 = ic * self.img_w, (ic + 1) * self.img_w
            grid[r1:r2, c1:c2] = imgs[i].numpy()
        return tf.convert_to_tensor(grid)

    def color_seg(self, seg: tf.Tensor) -> tf.Tensor:
        seg = np.squeeze(seg.numpy())
        cls_nums = np.unique(seg)
        res = np.zeros((*seg.shape[:3], 3), np.uint8)
        for cls_num in cls_nums:
            cls_num = int(cls_num)
            # Skipping background
            if cls_num == 0:
                continue
            color = self.colors[cls_num]
            res[seg == cls_num] = color
        return tf.convert_to_tensor(res)

    def set_model(self, model: tf.keras.models.Model):
        self.model = model

    def on_train_begin(self, logs=None):
        self.writer = tf.summary.create_file_writer(
            self.log_path.as_posix(), filename_suffix='.image', name=self.__class__.__name__,
        )

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_test_end(self, logs=None):
        imgs, maps_gt = next(self.ds_iter)
        nocs_gt, norms_gt, segs_gt = maps_gt
        nocs_pred, norms_pred, segs_pred = self.model(imgs)

        segs_gt = self.color_seg(segs_gt)

        nocs_pred = float_to_img(nocs_pred)
        norms_pred = float_to_img(norms_pred)
        segs_pred = tf.math.argmax(segs_pred, -1)
        segs_pred = self.color_seg(segs_pred)

        nocs_gt_grid = self.images_to_grid(nocs_gt)
        norms_gt_grid = self.images_to_grid(norms_gt)
        segs_gt_grid = self.images_to_grid(segs_gt)

        nocs_pred_grid = self.images_to_grid(nocs_pred)
        norms_pred_grid = self.images_to_grid(norms_pred)
        segs_pred_grid = self.images_to_grid(segs_pred)

        self.writer.set_as_default()
        tf.summary.image('segs', [segs_gt_grid, segs_pred_grid], self.epoch)
        tf.summary.image('nocs', [nocs_gt_grid, nocs_pred_grid], self.epoch)
        tf.summary.image('norms', [norms_gt_grid, norms_pred_grid], self.epoch)
        self.writer.flush()


def main(cfg: Config) -> int:
    print(cfg)

    dsl_train, dsl_val, ds_train, ds_val = build_datasets(cfg)

    out_subdir_name = get_subdir_name(
        datetime.now(), cfg.target_dataset_name, len(dsl_train), len(dsl_val))
    out_path = cfg.train_root_path / out_subdir_name
    weights_out_path = out_path / 'weights'
    weights_out_path.mkdir(parents=True, exist_ok=True)

    n_classes = len(dsl_train.objs)
    set_memory_growth()

    model = build_model(n_classes, cfg)
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
        loss=[
            MseNZLoss(),
            CosNZLoss(),
            SparseCategoricalCrossEntropyNZLoss(),
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
        ),
        PredictionVisualizer(out_path, 9, dsl_val),
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
