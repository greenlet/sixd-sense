import json
import sys
import time
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

from sds.data.loader import load_cache_ds_list, DsData
from sds.model.params import ScaledParams
from sds.utils import utils


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
        description=f'Model phi number. Must be: 0 <= PHI <= {len(ScaledParams.phi_max())}',
        required=True,
        cli=('--phi',),
    )
    freeze_bn: bool = Field(
        False,
        description='If set, BatchNormalization layers will be disabled during training',
        required=False,
        cli=('--freeze-bn',),
    )
    logs_root_path: Path = Field(
        ...,
        description='Tensorboard logs root directory. Different subdirectories in LOGS_ROOT_PATH correspond to '
                    'different learning processes',
        required=True,
        cli=('--logs-root-path',)
    )
    weights_root_path: Path = Field(
        ...,
        description='Weights root directory. Different subdirectories in LOGS_ROOT_PATH correspond to '
                    'different learning processes. Model weights and training state are stored in each directory',
        required=True,
        cli=('--weights-root-path',)
    )
    weights_to_use: str = Field(
        ...,
        description='Can be either full path to model weights, or "last" word meaning that last weights from '
                    'WEIGHTS_ROOT_PATH will be used, or WEIGHTS_ROOT_PATH subdirectory prefix (if multiple directories '
                    'match prefix last one will be chosen',
        cli=('--weights-to-use',),
    )
    new_learning_subdir: bool = Field(
        ...,
        description='If True and there are weights resolved from WEIGHTS_TO_USE located in subdir of WEIGHTS_ROOT_PATH, '
                    'training will be proceeded from the last point of those weights and in corresponding subdir',
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


def load_item_tf(ds_data: DsData, ind: tf.Tensor):
    ind = ind.numpy()
    subdir_ind, fname = ds_data.items.files[ind]
    subdir = ds_data.items.subdirs[subdir_ind]
    fpath = ds_data.root_path / subdir / fname



def build_ds_loader(ds_data: Path, inds: List[int]) -> tf.data.Dataset:
    pass


def build_ds_loaders(ds_data: DsData) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

    n_total = len(ds_data.items.files)
    ds = tf.data.Dataset.from_generator(range(n_total))

    ds = ds.map(lambda i: tf.py_function(func=load_item_tf, inp=[ds_data, i], Tout=[tf.float32]),
                num_parallel_calls=tf.data.AUTOTUNE)


def main(cfg: Config) -> int:
    print(cfg)

    ds_path = cfg.sds_root_path / cfg.target_dataset_name
    ds_data = load_cache_ds_list(ds_path)



    return 0


if __name__ == '__main__':
    run_and_exit(Config, main, 'Convert ply object to meters, restructure objects\' metadata')

