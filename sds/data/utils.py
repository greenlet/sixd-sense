import dataclasses
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import h5py
import numpy as np


def glob_inst_to_glob_id(glob_inst_id: str) -> str:
    return glob_inst_id[:glob_inst_id.rfind('_')]


@dataclasses.dataclass
class GtObj:
    # Lowercase name of dataset: itodd, tless, ...
    ds_name: str
    # Object id unique inside dataset: obj_000001, obj_000002, ...
    ds_obj_id: str
    # Has the form '{ds_name}_{ds_obj_id}'
    glob_id: str
    # Ordinal number of the object copy on the scene starting from 1
    instance_num: int
    # Model to camera 4x4 transform matrix
    H_m2c: np.ndarray


@dataclasses.dataclass
class GtItem:
    # RGB rendered image
    img: Optional[np.ndarray]
    # Camera intrinsic 3x3 matrix
    cam_mat: np.ndarray
    # Image size of the form (width, height)
    img_size: Tuple[int, int]
    # Keys are <glob-id>_<instance-num>
    # Values are object instances with their ids and positions with respect to camera
    gt_objs: Dict[str, GtObj]
    # 2D integer matrix with values corresponding to object instances (or 0 for background)
    # Object to segmap number correspondence can be looked up in segmap_key_to_num dictionary
    segmap: Optional[np.ndarray]
    # Keys are object identifiers in the form <glob-id>_<instance-num>
    # Values are numbers from segmap starting from 1 (0 - background)
    # Only objects present on img listed
    segmap_key_to_num: Dict[str, int]

    # Convert segmap with arbitrary locally unique segmentation numbers
    # defined in seg_map_key_to_num to segmap with globally unique class numbers: glob_num
    def recode_segmap_to_glob_num(self, objs: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
        glob_segmap = np.zeros_like(self.segmap)
        glob_id_to_num = {}
        glob_num_to_id = {}
        for glob_inst_id, seg_num in self.segmap_key_to_num.items():
            glob_id = glob_inst_to_glob_id(glob_inst_id)
            glob_num = objs[glob_id]['glob_num']
            glob_segmap[self.segmap == seg_num] = glob_num
            glob_id_to_num[glob_id] = glob_num
            glob_num_to_id[glob_num] = glob_id
        return glob_segmap, glob_id_to_num, glob_num_to_id


def read_gt_item(hdf5_fpath: Path, load_img: bool = True, load_segmap: bool = True) -> GtItem:
    with h5py.File(hdf5_fpath.as_posix(), 'r') as f:
        gt_str = f['gt'][...].item().decode('utf-8')
        gt = json.loads(gt_str)
        img = None if not load_img else f['colors'][...]
        segmap = None if not load_segmap else f['segmap'][...]
        segmap_key_to_num_str = f['segmap_key_to_num'][...].item().decode('utf-8')
        segmap_key_to_num = json.loads(segmap_key_to_num_str)

    cam_mat, img_size = np.array(gt['camera']['K']), tuple(gt['camera']['image_size'])
    objs = gt['objects']
    gt_objs = {}
    for oid, obj in objs.items():
        obj['H_m2c'] = np.array(obj['H_m2c'])
        gt_objs[oid] = GtObj(**obj)

    return GtItem(img, cam_mat, img_size, gt_objs, segmap, segmap_key_to_num)



