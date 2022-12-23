import dataclasses
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

import cv2
import h5py
import numpy as np

from sds.utils.utils import IntOrTuple, int_to_tuple


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


BBox = Tuple[int, int, int, int]
Crop = List[Tuple[int, int]]
Pad = List[Tuple[int, int]]


def bbox_from_mask(mask: np.ndarray) -> Optional[BBox]:
    rwise_max = np.max(mask, axis=1)
    rinds = np.where(rwise_max > 0)[0]
    if not len(rinds):
        return
    y0, y1 = rinds.min(), rinds.max()
    cwise_max = np.max(mask, axis=0)
    cinds = np.where(cwise_max > 0)[0]
    x0, x1 = cinds.min(), cinds.max()
    return x0, y0, x1, y1


def project_point_f(cam_mat: np.ndarray, p3: np.ndarray) -> np.ndarray:
    p2 = cam_mat @ p3
    p2[:2] /= p2[2]
    return p2[:2]


def crop_bbox(imgs: Tuple[np.ndarray, ...], bb_p1: np.ndarray, bb_sz: np.ndarray) -> Tuple[np.ndarray, ...]:
    img_sz = np.array([imgs[0].shape[1], imgs[0].shape[0]])
    bb_p2 = bb_p1 + bb_sz
    pad1, pad2 = np.maximum(-bb_p1, 0), np.maximum(bb_p2 - img_sz + 1, 0)
    bb_p1_in, bb_p2_in = bb_p1 + pad1, bb_p2 - pad2
    ix, iy = slice(bb_p1_in[0], bb_p2_in[0] + 1), slice(bb_p1_in[1], bb_p2_in[1] + 1)
    pad_2ch, pad_3ch = None, None
    if pad1.any() or pad2.any():
        pad_2ch = [(pad1[1], pad2[1]), (pad1[0], pad2[0])]
        pad_3ch = [*pad_2ch, (0, 0)]
    imgs_out = []
    for img in imgs:
        img_out = img[iy, ix]
        pad = pad_2ch if img.ndim == 2 else pad_3ch
        if pad is not None:
            img_out = np.pad(img_out, pad)
        imgs_out.append(img_out)
    return tuple(imgs_out)


def extract_pose(imgs_with_seg: Tuple[np.ndarray, ...], obj_m2c: np.ndarray, cam_mat: np.ndarray) -> Optional[Tuple[np.ndarray, int, Tuple[np.ndarray, ...]]]:
    bbox = bbox_from_mask(imgs_with_seg[-1])
    if bbox is None:
        return
    bb_p1, bb_p2 = np.array(bbox[:2]), np.array(bbox[2:])
    bb_c = project_point_f(cam_mat, obj_m2c[:3, 3])
    bb_sz_half = np.maximum(np.abs(bb_c - bb_p1), np.abs(bb_p2 - bb_c)).max()
    bb_sz_ext_half = bb_sz_half * 1.1
    bb_sz_ext = 2 * bb_sz_ext_half
    bb_p1_ext = bb_c - bb_sz_ext_half

    crop_size = bb_sz_ext.astype(np.int32)
    imgs_cropped = crop_bbox(imgs_with_seg, bb_p1_ext.astype(np.int32), crop_size)

    return bb_c, crop_size, imgs_cropped


def resize_img(img: np.ndarray, size_out: IntOrTuple, fit_in: bool = True, copy_if_intact: bool = False) -> Tuple[np.ndarray, Optional[Pad], Optional[Crop]]:
    size_in = img.shape[1], img.shape[0]
    size_out = int_to_tuple(size_out)
    pad, crop = None, None
    if size_in == size_out:
        if copy_if_intact:
            img = img.copy()
        return img, pad, crop

    diff = size_out[0] / size_in[0] - size_out[1] / size_in[1]

    if fit_in and diff < 0 or not fit_in and diff > 0:
        size_mid = size_out[0], int(size_in[1] * size_out[0] / size_in[0])
    else:
        size_mid = int(size_in[0] * size_out[1] / size_in[1]), size_out[1]

    is_mask = img.dtype == bool
    resized = False
    if size_mid != size_in:
        inter = cv2.INTER_AREA
        if size_mid[0] <= size_in[0]:
            inter = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        if is_mask:
            img = img.astype(float)

        img = cv2.resize(img, size_mid, interpolation=inter)
        resized = True

    size_mid, size_out = np.array(size_mid), np.array(size_out)
    if fit_in:
        assert np.all(size_out >= size_mid), f'size_mid = {size_mid} must be <= size_out = {size_out}'
        delta = size_out - size_mid
        if np.any(delta):
            off1 = delta // 2
            off2 = delta - off1
            pad = [(off1[1], off2[1]), (off1[0], off2[0])]
            if img.ndim == 3:
                pad.append((0, 0))
            img = np.pad(img, pad)
    else:
        assert np.all(size_out <= size_mid), f'size_mid = {size_mid} must be >= size_out = {size_out}'
        delta = size_mid - size_out
        if np.any(delta):
            off = delta // 2
            crop = [(off[1], off[1] + size_out[1]), (off[0], off[0] + size_out[0])]
            img = img[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]

    if resized and is_mask:
        img = img.astype(bool)

    return np.ascontiguousarray(img), pad, crop


def resize_imgs(imgs: Tuple[np.ndarray, ...], size_out: int, fit_in: bool = True, copy_if_intact: bool = False) -> Tuple[np.ndarray, ...]:
    imgs_out = []
    for img in imgs:
        img, _, _ = resize_img(img, size_out, fit_in, copy_if_intact)
        imgs_out.append(img)

    return tuple(imgs_out)


@dataclasses.dataclass
class DsPoseItem:
    img_noc_src: np.ndarray
    img_norms_src: np.ndarray
    img_noc_out: np.ndarray
    img_norms_out: np.ndarray
    cam_mat: np.ndarray
    bb_center: np.ndarray
    resize_factor: float
    rot_vec: np.ndarray
    pos: np.ndarray
    img_src: Optional[np.ndarray] = None
    img_out: Optional[np.ndarray] = None

    # def __str__(self):
    #     return f'{self.__class__.__name__}. Source image: '


def ds_pose_item_to_numbers(item: DsPoseItem) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    img = np.concatenate([item.img_noc_out, item.img_norms_out], axis=-1)
    cam_f, cam_cx, cam_cy = item.cam_mat[0, 0], item.cam_mat[0, 2], item.cam_mat[1, 2]
    bb_center_x, bb_center_y = item.bb_center
    params_in = np.array([cam_f, cam_cx, cam_cy, bb_center_x, bb_center_y, item.resize_factor])
    ang = np.linalg.norm(item.rot_vec)
    rot_vec = item.rot_vec
    if ang > np.pi:
        rot_vec = rot_vec / ang * (ang - 2 * np.pi)
    return (img, params_in), (rot_vec, item.pos)


