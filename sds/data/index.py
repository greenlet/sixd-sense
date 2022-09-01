import math
import math
import os
import pickle
import re
import time
import traceback
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from pydantic import BaseModel


def index_fname_prefix(train_ratio: float):
    return f'files_index_{train_ratio:.2f}'


def get_save_file_path(root_path: Path, train_ratio: float) -> Optional[Path]:
    prefix = index_fname_prefix(train_ratio)
    fname_pat = re.compile('^' + prefix + r'_t(\d+)_v(\d+)\.pkl$')
    fpath_max = None
    sz_max = 0
    for fpath in root_path.iterdir():
        if not fpath.is_file():
            continue
        m = fname_pat.match(fpath.name)
        if m:
            train_sz, val_sz = int(m.group(1)), int(m.group(2))
            sz = train_sz + val_sz
            if sz > sz_max:
                sz_max = sz
                fpath_max = fpath
    return fpath_max


def save_file_name(train_ratio: float, n_train: int, n_val: int) -> str:
    return f'{index_fname_prefix(train_ratio)}_t{n_train}_v{n_val}.pkl'


class DsFiles(BaseModel):
    root_path: Path
    subdirs: List[str]
    files: List[Tuple[int, str]]

    def get_path(self, i: int) -> Path:
        subdir_ind, fname = self.files[i]
        return self.root_path / self.subdirs[subdir_ind] / fname


class DsIndex(BaseModel):
    root_path: Path
    items: DsFiles
    noc: DsFiles
    norms: DsFiles
    train_ratio: float = 0
    inds_train: List[int] = []
    inds_val: List[int] = []
    num_to_obj_id: Dict[int, str] = {}

    def set_train_ratio(self, train_ratio: float):
        self.train_ratio = train_ratio
        inds = list(range(len(self.items.files)))
        np.random.shuffle(inds)
        n_total = len(inds)
        n_train = math.ceil(n_total * train_ratio)
        self.inds_train, self.inds_val = inds[:n_train], inds[n_train:]
        print(
            f'Train-val split. Train ratio: {train_ratio:0.2f}. n_train = {len(self.inds_train)}, n_val = {len(self.inds_val)}')

    @property
    def cache_file_path(self):
        fname = save_file_name(self.train_ratio, len(self.inds_train), len(self.inds_val))
        return self.root_path / fname

    def save(self):
        with open(self.cache_file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, root_path: Optional[Path] = None, train_ratio: Optional[float] = None, index_fpath: Optional[Path] = None) -> Optional['DsIndex']:
        if index_fpath is not None:
            with open(index_fpath, 'rb') as f:
                return pickle.load(f)

        assert root_path is not None and train_ratio is not None

        fpath = get_save_file_path(root_path, train_ratio)
        if fpath is None:
            return None
        try:
            with open(fpath, 'rb') as f:
                return pickle.load(f)
        except:
            traceback.print_exc()

    def get_paths(self, i: int) -> Tuple[Path, Path, Path]:
        return self.items.get_path(i), self.noc.get_path(i), self.norms.get_path(i)


def list_paths(ds_path: Path, data_dir: str = 'data') -> DsIndex:
    ds_index = DsIndex(
        root_path=ds_path,
        items=DsFiles(root_path=ds_path / data_dir, subdirs=[], files=[]),
        noc=DsFiles(root_path=ds_path / f'{data_dir}_noc', subdirs=[], files=[]),
        norms=DsFiles(root_path=ds_path / f'{data_dir}_normals', subdirs=[], files=[]),
    )
    for scene_path in sorted(p for p in ds_index.items.root_path.iterdir() if p.is_dir()):
        scene_name = scene_path.name
        scene_is_empty = True
        for item_path in sorted(scene_path.iterdir()):
            img_fname = item_path.with_suffix('.png').name
            norm_path = ds_index.norms.root_path / scene_name / img_fname
            noc_path = ds_index.noc.root_path / scene_name / img_fname
            if not norm_path.exists() or not noc_path.exists():
                continue
            if scene_is_empty:
                ds_index.items.subdirs.append(scene_name)
                ds_index.noc.subdirs.append(scene_name)
                ds_index.norms.subdirs.append(scene_name)
                scene_is_empty = False
            subdir_ind = len(ds_index.items.subdirs) - 1
            ds_index.items.files.append((subdir_ind, item_path.name))
            ds_index.noc.files.append((subdir_ind, img_fname))
            ds_index.norms.files.append((subdir_ind, img_fname))

    return ds_index


def load_cache_ds_index(ds_path: Optional[Path] = None, train_ratio: float = 0.9, force_reload: bool = False, index_fpath: Optional[Path] = None) -> DsIndex:
    ds_index = None
    if index_fpath is not None:
        return DsIndex.load(index_fpath=index_fpath)

    if not force_reload:
        ds_index = DsIndex.load(root_path=ds_path, train_ratio=train_ratio)

    if ds_index is None:
        ds_index = list_paths(ds_path)
        ds_index.set_train_ratio(train_ratio)
        ds_index.save()

    return ds_index


def _test_list_files():
    root_path = '/ws'
    sds_path = Path(os.path.expandvars(f'{root_path}/data/sds'))
    ds_name = 'itodd'
    ds_path = sds_path / ds_name
    t = time.time()
    print('load_cache_ds_list start')
    reload = False
    # reload = True
    ds_index = load_cache_ds_index(ds_path, force_reload=reload)
    index_fpath = ds_path / 'files_index_0.90_t96120_v10680.pkl'
    print(f'Loading index from {index_fpath}')
    ds_index = load_cache_ds_index(index_fpath=index_fpath)
    print(f'load_cache_ds_list stop: {time.time() - t:.3f}')
    print(len(ds_index.items.files), 'train:', len(ds_index.inds_train), 'val:', len(ds_index.inds_val))


if __name__ == '__main__':
    _test_list_files()

