import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Union

import numpy as np
import tensorflow as tf

from sds.data.ds_pose_gen import DsPoseGen
from sds.data.ds_pose_gen_mp import DsPoseGenMp
from sds.data.ds_pose_loader import DsPoseLoader
from sds.data.utils import ds_pose_item_to_numbers
from sds.model.model import bifpn_init, bifpn_layer, final_upscale
from sds.model.params import ScaledParams

TRAIN_SUBDIR_PAT = re.compile(r'^(\d{8}_\d{6})_.+$')


def find_latest_train_path(train_root_path: Path, prefix: str = '') -> Optional[Path]:
    max_path = None
    max_dt = None
    for subdir in train_root_path.iterdir():
        m = TRAIN_SUBDIR_PAT.match(subdir.name)
        if not m or not subdir.name.startswith(prefix):
            continue
        dt = datetime.strptime(m.group(1), '%Y%m%d_%H%M%S')
        if max_dt is None or max_dt < dt:
            max_dt = dt
            max_path = subdir
    return max_path


def find_train_weights_path(weights_path: Path, best: bool = True) -> Tuple[Optional[Path], Optional[Path]]:
    if weights_path.is_file():
        # <train-subdir>/weights/best/<checkpoint-files>
        return weights_path.parent.parent.parent, weights_path
    train_path = None
    if weights_path.is_dir():
        train_path = find_latest_train_path(weights_path)
    if train_path is None and weights_path.parent.is_dir():
        train_path = find_latest_train_path(weights_path.parent, weights_path.name)
    if train_path is not None:
        last_part = 'best/best.pb' if best else 'last/last.pb'
        return train_path, train_path / 'weights' / last_part
    return None, None


def find_index_file(train_path: Path) -> Optional[Path]:
    params_path = train_path / 'params'
    for fpath in params_path.iterdir():
        if fpath.name.startswith('files_index'):
            return fpath


def build_maps_model(n_classes: int, phi: int, freeze_bn: bool) -> tf.keras.models.Model:
    params = ScaledParams(phi)
    image_input = tf.keras.Input(params.input_shape, dtype=tf.float32)
    _, bb_feature_maps = params.backbone_class(input_tensor=image_input, freeze_bn=freeze_bn)

    fpn_init_feature_maps = bifpn_init(bb_feature_maps, params.bifpn_width, name='BifpnInit')
    fpn_feature_maps = bifpn_layer(fpn_init_feature_maps, params.bifpn_width, name='Bifpn1')

    noc_out = final_upscale(fpn_feature_maps, 3, name='Noc')
    norms_out = final_upscale(fpn_feature_maps, 3, name='Normals')
    seg_out = final_upscale(fpn_feature_maps, n_classes + 1, name='Segmentation')

    model = tf.keras.models.Model(inputs=[image_input], outputs=[noc_out, norms_out, seg_out])
    return model


def color_segmentation(colors: List[Tuple[int, int, int]], seg: np.ndarray) -> np.ndarray:
    seg = np.squeeze(seg)
    cls_nums = np.unique(seg)
    res = np.zeros((*seg.shape[:3], 3), np.uint8)
    for cls_num in cls_nums:
        cls_num = int(cls_num)
        # Skipping background
        if cls_num == 0:
            continue
        color = colors[cls_num]
        res[seg == cls_num] = color
    return res


def normalize(a: np.ndarray, eps: float = 1e-5, inplace=False):
    if not inplace:
        a = a.copy()
    an = np.linalg.norm(a, axis=-1)
    am = an >= eps
    a[am] /= an[am][..., None]
    return a


def ds_pose_preproc(ds_pose: Union[DsPoseGen, DsPoseLoader]):
    def f():
        for item in ds_pose.gen():
            # yield ds_pose_item_to_numbers(item)
            inp, out = ds_pose_item_to_numbers(item)
            inp = tf.convert_to_tensor(inp[0], tf.uint8), tf.convert_to_tensor(inp[1], tf.float32)
            out = tf.convert_to_tensor(out[0], tf.float32), tf.convert_to_tensor(out[1], tf.float32)
            yield inp, out

    return f


def ds_pose_mp_preproc(ds_pose_mp: DsPoseGenMp):
    def f():
        while not ds_pose_mp.stopped:
            batch = ds_pose_mp.get_batch()
            inp, out = zip(*batch)
            inp_img, inp_params = zip(*inp)
            out_rv, out_tr = zip(*out)
            inp_img = tf.stack([i for i in inp_img], axis=0)
            inp_params = tf.stack([i for i in inp_params], axis=0)
            out_rv = tf.stack([o for o in out_rv], axis=0)
            out_tr = tf.stack([o for o in out_tr], axis=0)
            inp_img, inp_params = tf.cast(inp_img, tf.uint8), tf.cast(inp_params, tf.float32)
            out_rv, out_tr = tf.cast(out_rv, tf.float32), tf.cast(out_tr, tf.float32)
            yield (inp_img, inp_params), (out_rv, out_tr)

    return f


def find_last_aae_model_path(train_root_path: Path, obj_id: str, img_sz: int) -> Optional[Path]:
    prefix = f'oid_{obj_id}--imgsz_{img_sz}--'
    paths = [p for p in train_root_path.iterdir() if p.name.startswith(prefix)]
    paths.sort()
    return paths[-1] if paths else None


