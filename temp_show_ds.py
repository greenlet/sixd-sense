import json
import sys
import time
from enum import Enum
from pathlib import Path
import shutil
from typing import Optional, Dict, List, Any, Tuple

import cv2
import json
from pathlib import Path

import h5py
import numpy as np
import pymesh

from sds.utils.utils import gen_colors
from train_utils import color_segmentation

ds_path = Path('/ws/data/sds/itodd')
models_path = ds_path / 'models'


colors = gen_colors()
# dir_path = ds_path / 'data_debug/000000'
# dir_path = ds_path / 'data/000000'
dir_path = ds_path / 'data/000106'
fpaths = list(dir_path.iterdir())
fpaths.sort()
for fpath in fpaths:
    print(fpath)
    with h5py.File(fpath.as_posix(), 'r') as f:
        img = f['colors'][...]
        segmap = f['segmap'][...]

    seg_img = color_segmentation(colors, segmap)
    cv2.imshow('img', img)
    cv2.imshow('segmap', seg_img)

    if cv2.waitKey() in (27, ord('q')):
        break

