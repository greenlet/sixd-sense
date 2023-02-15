import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np
import tensorflow as tf
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit

from sds.data.index import load_cache_ds_index
from sds.data.ds_loader import DsLoader
from sds.model.losses import MapsMseLoss, MapsCosLoss, SparseCategoricalCrossEntropyNZLoss
from sds.model.params import ScaledParams
from sds.model.utils import tf_img_to_float, tf_float_to_img
from sds.utils.tf_utils import tf_set_use_device
from sds.utils.utils import gen_colors
from sds.utils.ds_utils import load_objs
from train_utils import build_maps_model, color_segmentation, normalize, make_maps_train_subdir_name, \
    find_last_maps_train_path


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
        description='Training directory. For each new training new subdirectory will be created ',
        required=True,
        cli=('--train-root-path',)
    )
    weights: Optional[str] = Field(
        'none',
        description='Either full path to the model weights, or "last" word meaning that last weights from '
                    'WEIGHTS_ROOT_PATH will be used, or else WEIGHTS_ROOT_PATH subdirectory. '
                    'If WEIGHTS not set or set to "none", no weights will be loaded and new training subdirectory '
                    'in WEIGHTS_ROOT_PATH will be created.',
        cli=('--weights',),
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
    device_id: str = Field(
        '-1',
        description='Device id. Can have one of the values: "-1", "0", "1", ..., "n". '
                    'If set to "-1", CPU will be used, "GPU" with DEVICE_ID number will be used otherwise.',
        required=False,
        cli=('--device-id',)
    )
    debug: bool = Field(
        False,
        description='Debug mode',
        required=False,
        cli=('--debug',),
    )
    n_img_vis: int = Field(
        9,
        description='Number of images for inference visualization on each train/val epoch',
        required=False,
        cli=('--n-img-vis', ),
    )


def preprocess(img: tf.Tensor, maps: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, Tuple[tf.Tensor, ...]]:
    return tf_img_to_float(img), maps


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


def build_datasets(cfg: Config) -> Tuple[DsLoader, DsLoader, tf.data.Dataset, tf.data.Dataset]:
    if cfg.ds_path is not None:
        ds_path = cfg.ds_path
    else:
        ds_path = cfg.sds_root_path / cfg.target_dataset_name
    params = ScaledParams(cfg.phi)
    ds_index = load_cache_ds_index(
        ds_path, train_ratio=cfg.train_split_perc / 100,
        force_reload=False,
    )
    objs = load_objs(cfg.sds_root_path, cfg.target_dataset_name, cfg.distractor_dataset_name)
    ds_loader_train = DsLoader(
        ds_index, objs, is_training=True, img_size=params.input_size,
        shuffle_enabled=True, aug_enabled=True, hist_sz=cfg.n_img_vis,
    )
    ds_loader_val = DsLoader(
        ds_index, objs, is_training=False, img_size=params.input_size,
        shuffle_enabled=True, aug_enabled=False, hist_sz=cfg.n_img_vis,
    )
    ds_train = build_ds(ds_loader_train, cfg.batch_size)
    ds_val = build_ds(ds_loader_val, cfg.batch_size)
    return ds_loader_train, ds_loader_val, ds_train, ds_val


class PredictionVisualizer(tf.keras.callbacks.Callback):
    log_path: Path
    writer: Optional[tf.summary.SummaryWriter] = None
    model: Optional[tf.keras.models.Model] = None

    def __init__(self, log_path: Path, dsl_train: DsLoader, dsl_val: DsLoader):
        super().__init__()
        self.log_path = log_path
        self.dsl_train = dsl_train
        self.dsl_val = dsl_val
        self.n_samples = self.dsl_train.hist_sz
        self.n_cols = math.ceil(math.sqrt(self.n_samples))
        self.n_rows = math.ceil(self.n_samples / self.n_cols)
        self.img_size = dsl_train.img_size
        self.img_w, self.img_h = self.img_size
        self.grid_shape = tf.TensorShape((self.img_h * self.n_rows, self.img_w * self.n_cols, 3))
        self.colors = gen_colors()
        self.epoch = -1

    def images_to_grid(self, imgs: Union[tf.Tensor, List[np.ndarray]]) -> tf.Tensor:
        if isinstance(imgs, tf.Tensor):
            imgs = imgs.numpy()
        grid = np.zeros(self.grid_shape, np.uint8)
        for i in range(len(imgs)):
            if i == len(imgs):
                break
            ir, ic = i // self.n_cols, i % self.n_cols
            r1, r2 = ir * self.img_h, (ir + 1) * self.img_h
            c1, c2 = ic * self.img_w, (ic + 1) * self.img_w
            img = imgs[i]
            if isinstance(img, tf.Tensor):
                img = img.numpy()
            grid[r1:r2, c1:c2] = img
        return tf.convert_to_tensor(grid)

    def color_seg(self, seg: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        if isinstance(seg, tf.Tensor):
            seg = seg.numpy()
        res = color_segmentation(self.colors, seg)
        return tf.convert_to_tensor(res)

    def set_model(self, model: tf.keras.models.Model):
        self.model = model

    def on_train_begin(self, logs=None):
        self.writer = tf.summary.create_file_writer(
            self.log_path.as_posix(), filename_suffix='.image', name=self.__class__.__name__,
        )

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.dsl_train.on_epoch_begin()
        self.dsl_val.on_epoch_begin()

    def normalize(self, t: Union[tf.Tensor, np.ndarray], eps: float = 1e-5) -> tf.Tensor:
        a = t if isinstance(t, np.ndarray) else t.numpy().copy()
        a = normalize(a)
        return tf.convert_to_tensor(a)

    def vis_hist(self, dsl: DsLoader) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        imgs, nocs, norms, segs = [], [], [], []
        for img, (noc, norm, seg) in dsl.hist:
            imgs.append(img)
            nocs.append(noc)
            norms.append(norm)
            segs.append(seg)

        imgs_in = tf.stack(imgs)
        imgs_in = tf_img_to_float(imgs_in)
        nocs_pred, norms_pred, segs_pred = self.model(imgs_in)

        segs = self.color_seg(np.stack(segs))

        nocs_pred = tf_float_to_img(nocs_pred)
        norms_pred = self.normalize(norms_pred)
        norms_pred = tf_float_to_img(norms_pred)
        segs_pred = tf.math.argmax(segs_pred, -1)
        segs_pred = self.color_seg(segs_pred)

        imgs_grid = self.images_to_grid(imgs)

        nocs_gt_grid = self.images_to_grid(nocs)
        norms_gt_grid = self.images_to_grid(norms)
        segs_gt_grid = self.images_to_grid(segs)
        maps_gt_grid = nocs_gt_grid, norms_gt_grid, segs_gt_grid

        nocs_pred_grid = self.images_to_grid(nocs_pred)
        norms_pred_grid = self.images_to_grid(norms_pred)
        segs_pred_grid = self.images_to_grid(segs_pred)
        maps_pred_grid = nocs_pred_grid, norms_pred_grid, segs_pred_grid

        return imgs_grid, maps_gt_grid, maps_pred_grid

    def on_test_end(self, logs=None):
        if not self.dsl_train.hist or not self.dsl_val.hist:
            return
        imgs_train, maps_gt_train, maps_pred_train = self.vis_hist(self.dsl_train)
        imgs_val, maps_gt_val, maps_pred_val = self.vis_hist(self.dsl_val)
        nocs_gt_train, norms_gt_train, segs_gt_train = maps_gt_train
        nocs_pred_train, norms_pred_train, segs_pred_train = maps_pred_train
        nocs_gt_val, norms_gt_val, segs_gt_val = maps_gt_val
        nocs_pred_val, norms_pred_val, segs_pred_val = maps_pred_val

        self.writer.set_as_default()
        tf.summary.image('imgs_train_val', [imgs_train, imgs_val], self.epoch)
        tf.summary.image('nocs_train', [nocs_gt_train, nocs_pred_train], self.epoch)
        tf.summary.image('norms_train', [norms_gt_train, norms_pred_train], self.epoch)
        tf.summary.image('segs_train', [segs_gt_train, segs_pred_train], self.epoch)
        tf.summary.image('nocs_val', [nocs_gt_val, nocs_pred_val], self.epoch)
        tf.summary.image('norms_val', [norms_gt_val, norms_pred_val], self.epoch)
        tf.summary.image('segs_val', [segs_gt_val, segs_pred_val], self.epoch)
        self.writer.flush()


def main(cfg: Config) -> int:
    print(cfg)

    tf_set_use_device(cfg.device_id)

    dsl_train, dsl_val, ds_train, ds_val = build_datasets(cfg)
    index_fpath = dsl_train.ds_index.cache_file_path
    n_classes = len(dsl_train.objs)

    if cfg.weights == 'none':
        out_subdir_name = make_maps_train_subdir_name(cfg.target_dataset_name, cfg.phi)
        train_path = cfg.train_root_path / out_subdir_name
    elif cfg.weights == 'last':
        train_path = find_last_maps_train_path(cfg.train_root_path, cfg.target_dataset_name, cfg.phi)
        if train_path is None:
            print(f'Cannot find train_path for dataset = {cfg.target_dataset_name} and phi = cfg.phi in '
                  f'{cfg.train_root_path} directory')
            sys.exit(1)
    else:
        train_path = cfg.train_root_path / cfg.weights
        if not train_path.exists():
            train_path = Path(cfg.weights)
            if not train_path.exists():
                print(f'"{cfg.weights}" is neither subdirectory of {cfg.train_root_path} nor the path '
                      f'to a train directory')
                sys.exit(1)

    weights_path = train_path / 'weights'
    weights_last_fpath = weights_path / 'last' / 'last.pb'
    weights_best_fpath = weights_path / 'best' / 'best.pb'
    params_path = train_path / 'params'
    model = build_maps_model(n_classes, cfg.phi, cfg.freeze_bn)
    if cfg.weights == 'none':
        params_path.mkdir(parents=True, exist_ok=True)
        weights_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(index_fpath, params_path / index_fpath.name)
    else:
        model.load_weights(weights_last_fpath.as_posix())

    # losses = [
    #     ## -- NOC --
    #     MapsMseLoss(nonzero=True),
    #     ## -- Normals --
    #     MapsCosLoss(nonzero=True),
    #     ## -- Segmentation --
    #     tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0),
    # ]
    losses = [
        ## -- NOC --
        MapsMseLoss(nonzero=True),
        ## -- Normals --
        MapsMseLoss(nonzero=True),
        ## -- Segmentation --
        SparseCategoricalCrossEntropyNZLoss(),
    ]
    # losses = [
    #     ## -- NOC --
    #     MapsMseLoss(nonzero=False),
    #     ## -- Normals --
    #     MapsMseLoss(nonzero=False),
    #     ## -- Segmentation --
    #     tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0),
    # ]

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
        loss=losses,
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            weights_last_fpath.as_posix(),
            save_best_only=False, save_weights_only=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            weights_best_fpath.as_posix(),
            save_best_only=True, save_weights_only=True,
        ),
        tf.keras.callbacks.TensorBoard(
            train_path.as_posix(),
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
            factor=0.5, patience=6, min_lr=1e-9,
        ),
        PredictionVisualizer(train_path, dsl_train, dsl_val),
    ]
    model.fit(
        ds_train,
        epochs=cfg.epochs,
        steps_per_epoch=cfg.train_steps,
        validation_data=ds_val,
        validation_steps=cfg.val_steps,
        callbacks=callbacks,
        # initial_epoch=60,
    )

    return 0


if __name__ == '__main__':
    run_and_exit(Config, main, 'Train maps network')
