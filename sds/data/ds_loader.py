import dataclasses
import json
from pathlib import Path
from typing import Dict, Tuple, Union, Optional

import cv2
import h5py
import imgaug as ia
from imgaug import augmenters as iaa, SegmentationMapsOnImage
import numpy as np

from sds.data.index import DsIndex, load_cache_ds_index

# Loads glob_id -> obj dictionary
from sds.data.utils import read_gt_item, GtItem, glob_inst_to_glob_id
from sds.utils.common import IntOrTuple, int_to_tuple
from sds.utils.utils import gen_colors, load_objs


DsItem = Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]


class DsLoader:
    def __init__(self, ds_index: DsIndex, objs: Dict[str, Dict], is_training: bool, img_size: IntOrTuple,
                 shuffle_enabled: bool = False, aug_enabled: bool = False, hist_sz: int = 0):
        self.ds_index = ds_index
        self.ds_path = self.ds_index.root_path
        self.objs = objs
        self.num_to_obj_id = {obj['id_num']: glob_id for glob_id, obj in self.objs.items()}
        self.is_training = is_training
        self.img_size = int_to_tuple(img_size)
        self.img_width, self.img_height = self.img_size
        if self.is_training:
            self.ds_inds = self.ds_index.inds_train
        else:
            self.ds_inds = self.ds_index.inds_val
        self.shaffle_enabled = shuffle_enabled
        self.shuffle_if_enabled()

        self.aug_enabled = aug_enabled
        self.aug_geom = None
        self.aug_img = None
        self.aug_resize = None
        self.create_aug_resize()

        self.hist_sz = hist_sz
        self.hist = []
        self.gen_ind = 0

    def shuffle(self):
        np.random.shuffle(self.ds_inds)

    def shuffle_if_enabled(self):
        if self.shaffle_enabled:
            self.shuffle()

    def __len__(self) -> int:
        return len(self.ds_inds)

    def create_aug_resize(self):
        aspect_ratio = self.img_width / self.img_height
        resize_dims = {'width': self.img_width, 'height': self.img_height}

        if self.aug_enabled:
            self.aug_geom = iaa.Affine(
                scale=(0.5, 2.0),
            )

            self.aug_img = iaa.Sequential([
                iaa.Sometimes(0.5, iaa.OneOf([
                    iaa.MultiplyAndAddToBrightness(
                        mul=(0.9, 1.5),
                        add=(-10, 40),
                    ),
                    iaa.MultiplyHueAndSaturation(mul=(0.7, 1.3)),
                    iaa.Grayscale(),
                    iaa.GammaContrast(),
                    iaa.HistogramEqualization(),
                ])),
                iaa.Sometimes(0.5, iaa.OneOf([
                    iaa.AdditiveGaussianNoise(scale=(2, 12)),
                    iaa.Dropout(p=(0.01, 0.04)),
                    iaa.SaltAndPepper(p=(0.01, 0.03)),
                ])),
                iaa.Sometimes(0.5, iaa.OneOf([
                    iaa.Sharpen(),
                    iaa.Emboss(),
                    iaa.pillike.Autocontrast(),
                ])),
                iaa.Sometimes(0.5, iaa.OneOf([
                    iaa.GaussianBlur(sigma=(0.05, 0.5)),
                    iaa.AverageBlur(k=(1, 5)),
                    iaa.MotionBlur(),
                ]))
            ])

            self.aug_resize = iaa.Sequential([
                iaa.OneOf([
                    iaa.CenterCropToAspectRatio(aspect_ratio),
                    iaa.CenterPadToAspectRatio(aspect_ratio),
                ]),
                iaa.Resize(resize_dims),
            ])

        else:
            self.aug_resize = iaa.Sequential([
                iaa.CenterPadToAspectRatio(aspect_ratio),
                iaa.Resize(resize_dims),
            ])

    def augment(self, img: np.ndarray, noc: np.ndarray, norms: np.ndarray, segmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(segmap.shape) == 2:
            segmap = segmap[..., None]

        img_join = np.concatenate([img, noc, norms], axis=2)
        if self.aug_geom is not None:
            img_join, segmap = self.aug_geom(images=(img_join, ), segmentation_maps=(segmap, ))
            img_join, segmap = img_join[0], segmap[0]

        if self.aug_img is not None:
            img_join[..., :3] = self.aug_img(image=img_join[..., :3])

        img_join, segmap = self.aug_resize(images=(img_join, ), segmentation_maps=(segmap, ))
        img_join, segmap = img_join[0], segmap[0]
        img, noc, norms = img_join[..., :3], img_join[..., 3:6], img_join[..., 6:]
        return img, noc, norms, segmap

    def load_item(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        item_path, noc_path, norms_path = self.ds_index.get_paths(self.ds_inds[i])
        item = read_gt_item(item_path)
        noc_img = cv2.imread(noc_path.as_posix())
        norms_img = cv2.imread(norms_path.as_posix())
        noc_img = cv2.cvtColor(noc_img, cv2.COLOR_BGR2RGB)
        norms_img = cv2.cvtColor(norms_img, cv2.COLOR_BGR2RGB)
        glob_segmap, _, _ = item.recode_segmap_to_glob_num(self.objs)
        return item.img, noc_img, norms_img, glob_segmap

    def load_resize_augment(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        item = self.load_item(i)
        return self.augment(*item)

    def preprocess(self, img: np.ndarray, noc: np.ndarray, norms: np.ndarray, seg: np.ndarray) -> DsItem:
        seg = seg.astype(np.int32)
        if len(seg.shape) == 2:
            seg = seg[..., None]
        return img, (noc, norms, seg)

    def add_to_hist(self, item: DsItem):
        if len(self.hist) < self.hist_sz:
            self.hist.append(item)

    def gen(self):
        n = len(self.ds_inds)
        while True:
            item = self.load_resize_augment(self.gen_ind)
            item = self.preprocess(*item)
            self.add_to_hist(item)
            yield item
            self.gen_ind += 1
            if self.gen_ind == n:
                self.shuffle_if_enabled()
                self.gen_ind = 0

    def on_epoch_begin(self):
        self.hist.clear()


def _test_loader():
    root_path = Path('/data/data/sds')
    ds_path = root_path / 'itodd'
    ds_index = load_cache_ds_index(ds_path)
    objs = load_objs(root_path, 'itodd', 'tless')
    ds_loader = DsLoader(ds_index, objs, True, 600, shuffle_enabled=True, aug_enabled=True)
    colors = gen_colors()
    cv2.namedWindow('grid')
    cv2.moveWindow('grid', 100, 10)
    for i in range(len(ds_loader)):
        img, noc, norms, segmap = ds_loader.load_resize_augment(i)
        seg_img = SegmentationMapsOnImage(segmap, shape=img.shape)
        seg_color = seg_img.draw_on_image(img, colors=colors)[0]
        grid = ia.draw_grid([img, seg_color, noc, norms], cols=2)
        cv2.imshow('grid', grid[:, :, ::-1])
        if cv2.waitKey() in (ord('q'), 27):
            break


if __name__ == '__main__':
    _test_loader()

