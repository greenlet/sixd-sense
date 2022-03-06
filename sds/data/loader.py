import math
import os
import pickle
import sys
import time
import traceback
from collections import namedtuple
from pathlib import Path
from typing import List, Tuple, Union, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
import tensorflow as tf

from sds.utils import utils


class DsFiles(BaseModel):
    root_path: Path
    subdirs: List[str]
    files: List[Tuple[int, str]]


class DsData(BaseModel):
    root_path: Path
    items: DsFiles
    noc: DsFiles
    norms: DsFiles
    train_ratio: float = 0
    inds_train: List[int] = []
    inds_val: List[int] = []

    def set_train_ratio(self, train_ratio: float):
        self.train_ratio = train_ratio
        inds = list(range(len(self.items.files)))
        np.shuffle(inds)
        n_total = len(inds)
        n_train = math.ceil(n_total * train_ratio)
        self.inds_train, self.inds_val = inds[:n_train], inds[n_train:]
        print(f'Train-val split. Train ratio: {train_ratio:0.2f}. n_train = {len(self.inds_train)}, n_val = {len(self.inds_val)}')

    @staticmethod
    def save_file_path(root_path: Path, train_ratio: float) -> Path:
        return root_path / f'files_list_{train_ratio:.2f}.pkl'

    def save(self):
        fpath = self.save_file_path(self.root_path, self.train_ratio)
        with open(fpath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, root_path: Path, train_ratio: float) -> Optional['DsData']:
        fpath = cls.save_file_path(root_path, train_ratio)
        if not fpath.exists():
            return None
        try:
            with open(fpath, 'rb') as f:
                return pickle.load(f)
        except:
            traceback.print_exc()


def list_paths(ds_path: Path, data_dir: str = 'data') -> DsData:
    ds_data = DsData(root_path=ds_path,
                     items=DsFiles(root_path=ds_path / data_dir, subdirs=[], files=[]),
                     noc=DsFiles(root_path=ds_path / f'{data_dir}_noc', subdirs=[], files=[]),
                     norms=DsFiles(root_path=ds_path / f'{data_dir}_normals', subdirs=[], files=[]),
                     )
    for scene_path in sorted(ds_data.items.root_path.iterdir()):
        scene_name = scene_path.name
        scene_is_empty = True
        for item_path in sorted(scene_path.iterdir()):
            img_fname = item_path.with_suffix('.png').name
            norm_path = ds_data.norms.root_path / scene_name / img_fname
            noc_path = ds_data.noc.root_path / scene_name / img_fname
            if not norm_path.exists() or not noc_path.exists():
                continue
            if scene_is_empty:
                ds_data.items.subdirs.append(scene_name)
                ds_data.noc.subdirs.append(scene_name)
                ds_data.norms.subdirs.append(scene_name)
                scene_is_empty = False
            subdir_ind = len(ds_data.items.subdirs) - 1
            ds_data.items.files.append((subdir_ind, item_path.name))
            ds_data.noc.files.append((subdir_ind, img_fname))
            ds_data.norms.files.append((subdir_ind, img_fname))
    return ds_data


def load_cache_ds_list(ds_path: Path, train_ratio: float = 0.9, force_reload: bool = False) -> DsData:
    ds_data = None
    if not force_reload:
        ds_data = DsData.load(ds_path, train_ratio)

    if ds_data is None:
        ds_data = list_paths(ds_path)
        ds_data.set_train_ratio(train_ratio)
        ds_data.save()

    return ds_data


def _test_list_files():
    sds_path = Path(os.path.expandvars('$HOME/data/sds'))
    ds_name = 'itodd'
    ds_path = sds_path / ds_name
    t = time.time()
    print('load_cache_ds_list start')
    reload = False
    # reload = True
    ds_data = load_cache_ds_list(ds_path, force_reload=reload)
    print(f'load_cache_ds_list stop: {time.time() - t:.3f}')
    print(len(ds_data.items.files), 'train:', len(ds_data.inds_train), 'val:', len(ds_data.inds_val))


if __name__ == '__main__':
    _test_list_files()

