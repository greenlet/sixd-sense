from pathlib import Path
import threading as thr
from typing import Tuple, Dict, Optional

import cv2
from imgaug import augmenters as iaa
import numpy as np

from sds.data.utils import extract_pose, resize_imgs, DsPoseItem
from sds.synth.renderer import Renderer, OutputType
from sds.utils.common import IntOrTuple, int_to_tuple
from sds.utils.utils import canonical_cam_mat_from_img, gen_rot_vec, make_transform
from sds.utils.ds_utils import load_objs


class ConeFrustum:
    """
    :param z0: Absolute value of the nearest z-coordinate in meters
    :param z1: Absolute value of the farthest z-coordinate in meters
    :param fovx: Horizontal FOV angle in radians
    :param fovy: Vertical FOV angle in radians
    """
    def __init__(self, z0: float, z1: float, fovx: float, fovy: float):
        assert 0 <= z0 < z1
        assert 0 < fovx < np.pi
        assert 0 < fovy < np.pi
        self.z0 = z0
        self.z1 = z1
        self.fovx = fovx
        self.fovy = fovy
        self.low = -self.fovx / 2, -self.fovy / 2, self.z0
        self.high = self.fovx / 2, self.fovy / 2, self.z1

    def get_random_pose(self) -> np.ndarray:
        ax, ay, z = np.random.uniform(self.low, self.high)
        x, y = z * np.tan(ax), z * np.tan(ay)
        return np.array((x, y, z))

    def get_frustum_from_hor_offset(self, offset: float) -> Tuple[Optional['ConeFrustum'], float]:
        ang = max(self.fovx, self.fovy) / 2
        z_off = offset / np.tan(ang)
        if z_off >= self.z1:
            return None, 0
        return ConeFrustum(max(self.z0 - z_off, 0), self.z1 - z_off, self.fovx, self.fovy), z_off


def calc_frustum(obj_diam: float, img_size: Tuple[int, int]) -> Tuple[ConeFrustum, ConeFrustum, float]:
    w, h = img_size
    f = max(img_size)
    fovy = np.arctan(h / 2 / f) * 2
    fovx = np.arctan(w / 2 / f) * 2
    z0 = obj_diam / 2 / np.tan(max(fovx, fovy) / 2)
    z1 = z0 / 0.07
    cam_frustum = ConeFrustum(z0, z1, fovx, fovy)
    obj_frustum, z_offset = cam_frustum.get_frustum_from_hor_offset(obj_diam / 2)
    assert obj_frustum is not None
    return cam_frustum, obj_frustum, z_offset


class DsPoseGen:
    def __init__(self, objs: Dict, obj_glob_id: str, img_out_size: int,
                 img_base_size: IntOrTuple = 1024, aug_enabled: bool = False, hist_sz: int = 0,
                 multi_threading: bool = False):
        self.objs = objs
        self.obj_glob_id = obj_glob_id
        self.img_base_size = int_to_tuple(img_base_size)
        self.img_out_size = int_to_tuple(img_out_size)
        diam = self.objs[self.obj_glob_id]['diameter']
        self.cam_frustum, self.obj_frustum, self.obj_z_offset = calc_frustum(diam, self.img_base_size)
        self.aug_enabled = aug_enabled
        self.aug = None
        self.create_aug()
        self.hist_sz = hist_sz
        self.obj = self.objs[self.obj_glob_id]
        self.cam_mat = canonical_cam_mat_from_img(self.img_base_size)

        self.multi_threading = multi_threading
        if self.multi_threading:
            self.renderer_thread = thr.Thread(target=self.thread_func)
            self.task_counter = 0
            self.task_event = thr.Event()
            self.result_event = thr.Event()
            self.results = []
            self.stopped = False
            self.renderer_thread.start()
        else:
            self.renderer = Renderer(self.objs, self.img_base_size)
            self.renderer.set_camera_matrix(self.cam_mat)

        self.hist_sz = hist_sz
        self.hist = []

    def thread_func(self):
        self.renderer = Renderer(self.objs, self.img_base_size)
        self.renderer.set_camera_matrix(self.cam_mat)
        while not self.stopped:
            self.task_event.wait()
            while self.task_counter and not self.stopped:
                item = self.gen_item()
                self.results.append(item)
                self.task_counter -= 1
                self.result_event.set()

    def create_aug(self):
        if not self.aug_enabled:
            return
        self.aug = iaa.Sometimes(0.5, iaa.OneOf([
            iaa.CoarseDropout(p=(0.2, 0.5), size_percent=0.02),
            iaa.Cutout(nb_iterations=(2, 5), size=(0.1, 0.4), squared=False, fill_mode='constant', cval=0),
        ]))

    def gen_pos(self) -> np.ndarray:
        obj_pose = self.obj_frustum.get_random_pose()
        obj_pose[-1] += self.obj_z_offset
        return obj_pose

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

    def gen_item(self) -> DsPoseItem:
        rot_vec, rot_alpha = gen_rot_vec()
        pos = self.gen_pos()
        obj_m2c = make_transform(rot_vec, rot_alpha, pos)
        objs = {
            self.obj_glob_id: {
                'glob_id': self.obj_glob_id,
                'H_m2c': obj_m2c,
            }
        }
        img_noc = self.renderer.gen_colors(self.cam_mat, objs, OutputType.Noc)
        img_norms = self.renderer.gen_colors(self.cam_mat, objs, OutputType.Normals)

        seg = img_noc > 0
        imgs_with_seg = img_noc, img_norms, seg
        pose = extract_pose(imgs_with_seg, obj_m2c, self.cam_mat)
        assert pose is not None
        bb_center, crop_size, imgs_with_seg = pose
        img_noc_new, img_norms_new = resize_imgs(imgs_with_seg[:-1], self.img_out_size[0])
        img_noc_new, img_norms_new = self.augment(img_noc_new, img_norms_new)

        rot_vec *= rot_alpha
        resize_factor = self.img_out_size[0] / crop_size
        pos = obj_m2c[:3, 3]

        item = DsPoseItem(img_noc, img_norms, img_noc_new, img_norms_new,
                          self.cam_mat, bb_center, resize_factor,
                          rot_vec, pos)
        return item

    def add_to_hist(self, item: DsPoseItem):
        if len(self.hist) < self.hist_sz:
            self.hist.append(item)

    def gen(self):
        while True:
            if self.multi_threading:
                self.task_counter += 1
                self.task_event.set()
                while not self.results:
                    self.result_event.wait()
                if self.stopped:
                    return
                item = self.results.pop(0)
            else:
                item = self.gen_item()

            self.add_to_hist(item)
            yield item

    def on_epoch_begin(self):
        self.hist.clear()

    def stop(self):
        if not self.multi_threading:
            return
        self.stopped = True
        self.task_event.set()
        self.result_event.set()
        self.renderer_thread.join()


def _test_ds_pose_gen():
    ds_name = 'itodd'
    ds_path = Path('/data/data/sds') / ds_name
    objs = load_objs(ds_path.parent, ds_name, load_meshes=True)
    # img_size = 128
    img_size = 400
    obj_num = 1
    num_to_obj_id = {obj['id_num']: obj_id for obj_id, obj in objs.items()}
    print(num_to_obj_id)
    obj_id = num_to_obj_id[obj_num]
    dsgen = DsPoseGen(objs, obj_id, img_size, aug_enabled=True)
    cv2.namedWindow('pose')
    cv2.moveWindow('pose', 200, 100)
    while True:
        item = dsgen.gen_item()

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
    _test_ds_pose_gen()
