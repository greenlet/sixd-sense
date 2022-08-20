import time
from pathlib import Path
import sys
import threading as thr
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as o3d_rend
import pymesh
from skimage import io
from scipy.spatial.transform import Rotation as R


from sds.data.ds_index_gt import load_index_gt
from sds.data.utils import read_gt_item, GtObj, GtItem, extract_pose, resize_imgs, DsPoseItem
from sds.synth.renderer import Renderer, OutputType
from sds.utils.common import IntOrTuple, int_to_tuple
from sds.utils.ds_utils import load_objs


def pose_mat_to_parts(T: np.ndarray) -> Tuple[float, float, float]:
    rot_mat = T[:3, :3]
    pos_z = T[2, 3]
    rot_vec = R.from_matrix(rot_mat).as_rotvec()
    rot_ang = np.linalg.norm(rot_vec)
    if rot_ang < 1e-6:
        return .0, .0, pos_z
    rotx, roty, rotz = rot_vec
    ang_xy = np.arctan2(rotx, roty) + np.pi
    ang_z = np.arctan(rotz) + np.pi / 2
    return ang_xy, ang_z, pos_z


class DsPoseLoader:
    def __init__(self, ds_path: Path, objs: Dict[str, Any], img_size: IntOrTuple, obj_glob_id: str, hist_sz: int = 0):
        print(self.__class__.__name__, '__init__. thread:', thr.get_ident())
        self.ds_path = ds_path
        self.objs = objs
        self.img_out_size = int_to_tuple(img_size)
        assert(self.img_out_size[0] == self.img_out_size[1])
        self.obj_glob_id = obj_glob_id
        df = load_index_gt(self.ds_path, self.objs)
        self.df_obj = df.loc[df[self.obj_glob_id] > 0, ('scene_num', 'file_num')]
        self.hist_sz = hist_sz
        self.hist = []

    def __len__(self):
        return len(self.df_obj)

    def get_item(self, i: int) -> List[DsPoseItem]:
        row = self.df_obj.iloc[i]
        scene_name = f'{row.scene_num:06d}'
        fname_base = f'{row.file_num:06d}'
        hdf5_fpath = self.ds_path / 'data' / scene_name / f'{fname_base}.hdf5'
        noc_fpath = self.ds_path / 'data_noc' / scene_name / f'{fname_base}.png'
        norms_fpath = self.ds_path / 'data_normals' / scene_name / f'{fname_base}.png'
        gt_item = read_gt_item(hdf5_fpath)
        img_noc = io.imread(noc_fpath.as_posix())
        img_norms = io.imread(norms_fpath.as_posix())
        imgs: Tuple[np.ndarray, ...] = (gt_item.img, img_noc, img_norms)
        ds_items: List[DsPoseItem] = []
        for glob_inst_id, gt_obj in gt_item.gt_objs.items():
            if gt_obj.glob_id != self.obj_glob_id or glob_inst_id not in gt_item.segmap_key_to_num:
                continue
            seg_num = gt_item.segmap_key_to_num[glob_inst_id]
            obj_seg = gt_item.segmap == seg_num
            imgs_with_seg = (*imgs, obj_seg)
            bb_center, size_new, imgs1 = extract_pose(imgs_with_seg, gt_obj.H_m2c, gt_item.cam_mat)
            img1, img1_noc, img1_norms, img1_seg = imgs1
            if img1_seg.mean() < 0.05:
                continue
            img1_seg_neg = ~img1_seg
            img1_noc, img1_norms = img1_noc.copy(), img1_norms.copy()
            img1_noc[img1_seg_neg] = 0
            img1_norms[img1_seg_neg] = 0
            imgs2 = resize_imgs((img1, img1_noc, img1_norms, img1_seg), self.img_out_size[0])
            resize_factor = self.img_out_size[0] / size_new

            img_out, img_noc_out, img_norms_out, seg_out = imgs2
            rot_vec = R.from_matrix(gt_obj.H_m2c[:3, :3]).as_rotvec()
            pos = gt_obj.H_m2c[:3, 3]

            ds_item = DsPoseItem(img_noc, img_norms, img_noc_out, img_norms_out,
                                 gt_item.cam_mat, bb_center, resize_factor,
                                 rot_vec, pos, gt_item.img, img_out)
            ds_items.append(ds_item)
        return ds_items

    def add_to_hist(self, item: DsPoseItem):
        if len(self.hist) < self.hist_sz:
            self.hist.append(item)

    def gen(self):
        n = len(self)
        i = 0
        while True:
            for item in self.get_item(i):
                self.add_to_hist(item)
                t1 = time.time()
                yield item
                # print(f'item {i:06d} time: {time.time() - t1:.3f}')
            i = (i + 1) % n

    def on_epoch_begin(self):
        self.hist.clear()


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

    cv2.namedWindow('pose')
    cv2.moveWindow('pose', 200, 100)

    for item in dsp_loader.gen():
        cv2.imshow('img_src', item.img_src)
        img = np.zeros((2 * img_size, 2 * img_size, 3), np.uint8)
        img[:img_size, :img_size] = item.img_noc_out
        img[:img_size, img_size:2 * img_size] = item.img_norms_out
        img[img_size:2 * img_size, :img_size] = item.img_noc_out
        img[img_size:2 * img_size, img_size:2 * img_size] = item.img_norms_out
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('pose', img)

        if cv2.waitKey() in (27, ord('q')):
            break


if __name__ == '__main__':
    _test_ds_pose_loader()

