import time
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as o3d_rend
import pymesh
from skimage import io

from sds.data.ds_index_gt import load_index_gt
from sds.data.utils import read_gt_item, GtObj, GtItem
from sds.synth.renderer import Renderer, OutputType
from sds.utils.common import IntOrTuple, int_to_tuple
from sds.utils.utils import load_objs

BBox = Tuple[int, int, int, int]


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


def crop(imgs: Tuple[np.ndarray, ...], bb_p1: np.ndarray, bb_sz: np.ndarray) -> Tuple[np.ndarray, ...]:
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


def resize(imgs: Tuple[np.ndarray, ...], size_out: int) -> Tuple[np.ndarray, ...]:
    size_cur = imgs[0].shape[0]
    if size_cur == size_out:
        return imgs

    imgs_out = []
    for img in imgs:
        is_mask = img.dtype == bool
        inter = cv2.INTER_AREA
        if size_cur < size_out:
            inter = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        if is_mask:
            img = img.astype(np.float64)
        img_out = cv2.resize(img, (size_out, size_out), interpolation=inter)
        if is_mask:
            img_out = img_out.astype(bool)
        imgs_out.append(img_out)

    return tuple(imgs_out)


def extract_obj_gt(gt_item: GtItem, gt_obj: GtObj, imgs_with_seg: Tuple[np.ndarray, ...]) -> Optional[Tuple[np.ndarray, int, Tuple[np.ndarray, ...]]]:
    bbox = bbox_from_mask(imgs_with_seg[-1])
    if bbox is None:
        return
    cam_mat = gt_item.cam_mat
    bb_p1, bb_p2 = np.array(bbox[:2]), np.array(bbox[2:])
    bb_c = project_point_f(cam_mat, gt_obj.H_m2c[:3, 3])
    bb_sz_half = np.maximum(np.abs(bb_c - bb_p1), np.abs(bb_p2 - bb_c)).max()
    bb_sz_ext_half = bb_sz_half * 1.1
    bb_sz_ext = 2 * bb_sz_ext_half
    bb_p1_ext = bb_c - bb_sz_ext_half

    size_new = bb_sz_ext.astype(np.int32)
    imgs_cropped = crop(imgs_with_seg, bb_p1_ext.astype(np.int32), size_new)

    return bb_c, size_new, imgs_cropped


class DsPoseLoader:
    def __init__(self, ds_path: Path, objs: Dict[str, Any], img_size: IntOrTuple, obj_glob_id: str, hist_sz: int = 0):
        self.ds_path = ds_path
        self.objs = objs
        self.img_size = int_to_tuple(img_size)
        assert(self.img_size[0] == self.img_size[1])
        self.obj_glob_id = obj_glob_id
        df = load_index_gt(self.ds_path, self.objs)
        self.df_obj = df.loc[df[self.obj_glob_id] > 0, ('scene_num', 'file_num')]
        self.hist_sz = hist_sz
        # imgs, res = self.get_item(0)
        # self.src_size = imgs[0].shape[1], imgs[0].shape[0]

    def __len__(self):
        return len(self.df_obj)

    def get_item(self, i: int) -> Tuple[GtItem, Tuple[np.ndarray, ...], List[Tuple[str, GtObj, np.ndarray, np.ndarray, float, Tuple[np.ndarray, ...]]]]:
        row = self.df_obj.iloc[i]
        scene_name = f'{row.scene_num:06d}'
        fname_base = f'{row.file_num:06d}'
        hdf5_fpath = self.ds_path / 'data' / scene_name / f'{fname_base}.hdf5'
        noc_fpath = self.ds_path / 'data_noc' / scene_name / f'{fname_base}.png'
        norms_fpath = self.ds_path / 'data_normals' / scene_name / f'{fname_base}.png'
        gt_item = read_gt_item(hdf5_fpath)
        noc_img = io.imread(noc_fpath.as_posix())
        norms_img = io.imread(norms_fpath.as_posix())
        imgs: Tuple[np.ndarray, ...] = gt_item.img, noc_img, norms_img
        crops = []
        for glob_inst_id, gt_obj in gt_item.gt_objs.items():
            if gt_obj.glob_id != self.obj_glob_id or glob_inst_id not in gt_item.segmap_key_to_num:
                continue
            seg_num = gt_item.segmap_key_to_num[glob_inst_id]
            obj_seg = gt_item.segmap == seg_num
            imgs_with_seg = (*imgs, obj_seg)
            center, size_new, imgs1 = extract_obj_gt(gt_item, gt_obj, imgs_with_seg)

            imgs2 = resize(imgs1, self.img_size[0])
            resize_factor = self.img_size[0] / size_new

            crops.append((glob_inst_id, gt_obj, obj_seg, center, resize_factor, imgs2))

        return gt_item, imgs, crops


def _test_ds_pose_loader():
    sds_root_path = Path('/data/data/sds')
    target_ds_name, dist_ds_name = 'itodd', 'tless'
    ds_path = sds_root_path / target_ds_name
    objs = load_objs(sds_root_path, target_ds_name, dist_ds_name)
    obj_num_to_id = {obj['glob_num']: glob_id for glob_id, obj in objs.items()}
    img_size = 256
    obj_num = 1
    obj_glob_id = obj_num_to_id[obj_num]
    dsp_loader = DsPoseLoader(ds_path, objs, img_size, obj_glob_id)
    obj = objs[obj_glob_id]
    obj_id = obj['id']
    mesh_path = ds_path / 'models' / f'{obj_id}.ply'
    mesh = pymesh.load_mesh(mesh_path.as_posix())
    models = {obj_glob_id: {'mesh': mesh}}
    ren = Renderer(models)
    cam_mat_new = np.array([[img_size, 0, img_size / 2], [0, img_size, img_size / 2], [0, 0, 1]])
    img_out = np.zeros((img_size * 2, img_size * 2, 3), np.uint8)

    for i in range(len(dsp_loader)):
        print(f'Loading item {i}')
        t = time.time()
        gt_item, imgs, crops = dsp_loader.get_item(i)
        print(f'Item {i} loaded in {time.time() - t:.3f} sec')
        cv2.imshow('img', imgs[0])

        robjs_src = {}
        obj_color = (255, 0, 0, 100)
        for glob_inst_id, gt_obj, _, _, _, _ in crops:
            robjs_src[glob_inst_id] = {'glob_id': obj_glob_id, 'H_m2c': gt_obj.H_m2c}
        print(f'Number of {obj_glob_id}: {len(robjs_src)}')
        ren.set_window_size(gt_item.img_size)
        ren.gen_colors(gt_item.cam_mat, robjs_src, OutputType.Noc, obj_color)

        for crop in crops:
            glob_inst_id, gt_obj, obj_seg, center, resize_factor, imgs_new = crop
            img_crop, noc_crop, norms_crop, seg_crop = imgs_new
            img_out[:img_size, :img_size] = img_crop
            img_out[:img_size, img_size:2 * img_size] = noc_crop
            img_out[img_size:2 * img_size, :img_size] = norms_crop
            img_out[img_size:2 * img_size, img_size:2 * img_size] = seg_crop[..., None] * 255
            seg_frac = np.mean(seg_crop) * 100
            cv2.putText(img_out, f'{seg_frac:3.02f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('cropped', img_out)

            if cv2.waitKey() in (27, ord('q')):
                sys.exit(0)


if __name__ == '__main__':
    _test_ds_pose_loader()

