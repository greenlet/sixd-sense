from pathlib import Path
from typing import Tuple, Dict, Any

import cv2
import imgaug as ia
from imgaug import augmenters as iaa, SegmentationMapsOnImage
import numpy as np
from scipy.spatial.transform import Rotation as R

from sds.synth.renderer import Renderer, OutputType
from sds.utils.common import IntOrTuple, int_to_tuple
from sds.utils.utils import load_objs


def calc_ref_dist_from_camera(cam_mat: np.ndarray, img_size: Tuple[int, int], obj: Dict[str, Any], hist_sz: int = 0) -> float:
    img_sz = np.array(img_size, dtype=np.float64)
    f, c = cam_mat[[0, 1], [0, 1]], cam_mat[:2, 2]
    sz = np.minimum(c, img_sz - c)
    diam_px = sz * 2 * 0.9
    diam = obj['diameter']
    dist = diam * f / diam_px
    return np.max(dist)
    # diam / dist * f <= diam_px
    # (dist, dist) >= diam * f / diam_px


def canonical_cam_mat_from_img(img_size: Tuple[int, int]) -> np.ndarray:
    w, h = img_size
    f = max(img_size)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


DsItem = Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


class DsPoseGen:
    def __init__(self, objs: Dict, img_size: IntOrTuple, obj_glob_id: str,
                 aug_enabled: bool = False, hist_sz: int = 0):
        self.objs = objs
        self.img_size = int_to_tuple(img_size)
        self.obj_glob_id = obj_glob_id
        self.aug_enabled = aug_enabled
        self.aug = None
        self.create_aug()
        self.hist_sz = hist_sz
        self.renderer = Renderer(self.objs, self.img_size)
        self.obj = self.objs[self.obj_glob_id]
        self.cam_mat = canonical_cam_mat_from_img(self.img_size)
        self.ref_dist = calc_ref_dist_from_camera(self.cam_mat, self.img_size, self.obj)
        self.dist_max_delta = 0.25
        self.renderer.set_window_size(self.img_size)
        self.renderer.set_camera_matrix(self.cam_mat)

        self.hist_sz = hist_sz
        self.hist = []

    def create_aug(self):
        if not self.aug_enabled:
            return
        self.aug = iaa.Sometimes(0.5, iaa.OneOf([
            iaa.CoarseDropout(p=(0.2, 0.5), size_percent=0.02),
            iaa.Cutout(nb_iterations=(2, 5), size=(0.1, 0.4), squared=False, fill_mode='constant', cval=0),
        ]))

    @staticmethod
    def gen_rot_vec() -> Tuple[np.ndarray, float]:
        while True:
            r = np.random.random((3,))
            rl = np.linalg.norm(r)
            if rl > 1e-6:
                return r / rl, np.random.random() * np.pi

    def gen_pos(self) -> np.ndarray:
        r = np.random.random(3)
        delta = (r - 0.5) * self.dist_max_delta * self.ref_dist
        dist = np.array([0, 0, self.ref_dist]) + delta
        return dist

    @staticmethod
    def make_transform(rot_vec: np.ndarray, rot_alpha: float, pos: np.ndarray) -> np.ndarray:
        rot = R.from_rotvec(rot_vec * rot_alpha)
        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = pos
        return T

    def augment(self, img_noc: np.ndarray, img_norms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.aug_enabled:
            return img_noc, img_norms
        img = np.concatenate([img_noc, img_norms], axis=-1)
        img_aug = self.aug(image=img)
        mask = img > 0
        mask_aug = img_aug > 0
        if np.sum(mask_aug) / np.sum(mask) < 0.3:
            img_aug = img

        img_noc, img_norms = img_aug[..., :3], img_aug[..., 3:]
        return img_noc, img_norms

    def gen_item(self) -> DsItem:
        rot_vec, rot_alpha = self.gen_rot_vec()
        pos = self.gen_pos()
        T = self.make_transform(rot_vec, rot_alpha, pos)
        objs = {
            self.obj_glob_id: {
                'glob_id': self.obj_glob_id,
                'H_m2c': T,
            }
        }
        img_noc = self.renderer.gen_colors(self.cam_mat, objs, OutputType.Noc)
        img_norms = self.renderer.gen_colors(self.cam_mat, objs, OutputType.Normals)
        img_noc_aug, img_norms_aug = self.augment(img_noc, img_norms)
        return T, (img_noc, img_norms), (img_noc_aug, img_norms_aug)

    def add_to_hist(self, item: DsItem):
        if len(self.hist) < self.hist_sz:
            self.hist.append(item)

    def gen(self):
        while True:
            item = self.gen_item()
            self.add_to_hist(item)
            yield item

    def on_epoch_begin(self):
        self.hist.clear()


def _test_ds_pose_gen():
    ds_name = 'itodd'
    ds_path = Path('/data/data/sds') / ds_name
    objs = load_objs(ds_path / 'models', ds_name)
    # img_size = 128
    img_size = 400
    obj_num = 1
    num_to_obj_id = {obj['id_num']: obj_id for obj_id, obj in objs.items()}
    print(num_to_obj_id)
    obj_id = num_to_obj_id[obj_num]
    dsgen = DsPoseGen(objs, img_size, obj_id, aug_enabled=True)
    cv2.namedWindow('pose')
    cv2.moveWindow('pose', 200, 100)
    while True:
        T, (img_noc, img_norms), (img_noc_aug, img_norms_aug) = dsgen.gen_item()

        img = np.zeros((2 * img_size, 2 * img_size, 3), np.uint8)
        img[:img_size, :img_size] = img_noc
        img[:img_size, img_size:2 * img_size] = img_norms
        img[img_size:2 * img_size, :img_size] = img_noc_aug
        img[img_size:2 * img_size, img_size:2 * img_size] = img_norms_aug
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('pose', img)

        if cv2.waitKey() in (27, ord('q')):
            break


if __name__ == '__main__':
    _test_ds_pose_gen()
