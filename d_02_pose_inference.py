import sys
from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional, Any

import cv2
import numpy as np
import tensorflow as tf
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
from scipy.spatial.transform import Rotation as R

from sds.data.ds_pose_gen import DsPoseGen
from sds.data.ds_pose_loader import DsPoseLoader
from sds.data.utils import ds_pose_item_to_numbers
from sds.model.model_pose import build_pose_layers, build_pose_layers, RotHeadType
from sds.model.utils import tf_img_to_float
from sds.synth.renderer import Renderer, OutputType
from sds.utils.tf_utils import tf_set_use_device
from sds.utils.ds_utils import load_objs
from train_utils import find_train_weights_path


class Config(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    sds_root_path: Path = Field(
        ...,
        description='Path to SDS datasets (containing datasets: ITODD, TLESS, etc.)',
        cli=('--sds-root-path',),
    )
    dataset_name: str = Field(
        ...,
        description='Dataset name. Has to be a subdirectory of SDS_ROOT_PATH, one of: "itodd", "tless", etc.',
        cli=('--dataset-name',),
    )
    weights_path: Path = Field(
        ...,
        description='Either path to weights pb-file or path to a directory. In case WEIGHTS_PATH '
                    'is a directory and contains different training results, path to the latest '
                    'best weights found will be returned. WEIGHTS_PATH can also be a prefix path of training '
                    'subdirectory, weights path from this subdirectory will be returned in that case',
        required=True,
        cli=('--weights-path',)
    )
    data_source: str = Field(
        'loader',
        description='Images source which will be used to visualize maps predictions. Can have either '
                    'value "loader" or "generator". In former case synthetic rendered dataset will be loaded. '
                    'In latter case, 3D maps generator will be used for inference.',
        required=False,
        cli=('--data-source',),
    )
    img_size: int = Field(
        256,
        description='Inference image size. Currently only value 256 is supported',
        required=True,
        cli=('--img-size',),
    )
    use_gpu: bool = Field(
        False,
        description='If set, GPU will be used. CPU is used by default',
        required=False,
        cli=('--use-gpu',),
    )


def get_rot_vec_from_pred(prob: tf.Tensor) -> np.array:
    prob_max = tf.reduce_max(prob)
    ind = tf.where(prob == prob_max)
    print(tf.reduce_max(prob), ind[:5])
    ind = ind[0].numpy()
    N = prob.shape[-1]
    rv = ind / N * 2 * np.pi - np.pi
    return rv


def predict_on_dataset(model: tf.keras.models.Model, ds_loader: Union[DsPoseLoader, DsPoseGen], objs: Dict[str, Any]):
    w, h = ds_loader.img_out_size
    img_vis = np.zeros((2 * h, 2 * w, 3), dtype=np.uint8)
    ren = Renderer(objs, (1280, 1024))
    robjs = {
        ds_loader.obj_glob_id: {
            'glob_id': ds_loader.obj_glob_id,
            'H_m2c': np.identity(4),
        }
    }

    def set_pose(rot_vec: np.ndarray, tr: np.ndarray):
        m2c = np.identity(4)
        m2c[:3, :3] = R.from_rotvec(rot_vec).as_matrix()
        m2c[:3, 3] = tr
        robjs[ds_loader.obj_glob_id]['H_m2c'] = m2c

    cv2.imshow('maps', img_vis)
    cv2.moveWindow('maps', 0, 0)
    for item in ds_loader.gen():
        ren.set_window_size((item.img_noc_src.shape[1], item.img_noc_src.shape[0]))
        (img, params_in), (rot_vec, pos) = ds_pose_item_to_numbers(item)
        img, params_in = tf.convert_to_tensor(img)[None], tf.convert_to_tensor(params_in)[None]
        img = tf_img_to_float(img)
        rv_prob_pred, pos_pred = model((img, params_in))
        print('GT:', rot_vec, pos)
        print('Pred:', rv_prob_pred.shape, pos_pred)
        rv_pred = get_rot_vec_from_pred(rv_prob_pred[0])
        print(rv_pred)

        set_pose(rv_pred, pos_pred[0])
        img_pred = ren.gen_colors(item.cam_mat, robjs, OutputType.Normals)
        set_pose(rot_vec, pos)
        img_gt = ren.gen_colors(item.cam_mat, robjs, OutputType.Normals)
        cv2.imshow('img_pred', cv2.cvtColor(img_pred, cv2.COLOR_RGB2BGR))
        cv2.imshow('img_gt', cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR))

        img_vis[:h, :w] = item.img_noc_out
        img_vis[:h, w:2 * w] = item.img_norms_out
        if item.img_out is not None:
            img_vis[h:2 * h, w:2 * w] = item.img_out

        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        cv2.imshow('maps', img_vis)
        if item.img_src is not None:
            cv2.imshow('src', item.img_src)
        if cv2.waitKey() in (27, ord('q')):
            break


def main(cfg: Config) -> int:
    print(cfg)
    ds_path = cfg.sds_root_path / cfg.dataset_name

    # best = True
    # train_path, weights_path = find_train_weights_path(cfg.weights_path, best=best)
    # if train_path is None or not train_path.exists():
    #     print(f'Cannot find weights in {cfg.weights_path}')
    #     return 1
    #
    # obj_glob_id = '_'.join(weights_path.parent.parent.parent.name.split('_')[2:])

    weights_path = cfg.weights_path / 'weights/best/best.pb'
    obj_glob_id = cfg.weights_path.name.split('--')[-2][4:]

    tf_set_use_device(cpu=not cfg.use_gpu)

    objs = load_objs(cfg.sds_root_path, cfg.dataset_name, load_meshes=True, load_target_glob_id=obj_glob_id)
    objs = {obj_glob_id: objs[obj_glob_id]}

    print('Building model')
    inp, out = build_pose_layers(RotHeadType.Conv3d, cfg.img_size)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    print(f'Loading model from {weights_path}')
    model.load_weights(weights_path.as_posix())

    if cfg.data_source == 'loader':
        ds_loader = DsPoseLoader(ds_path, objs, cfg.img_size, obj_glob_id)
    elif cfg.data_source == 'generator':
        ds_loader = DsPoseGen(objs, obj_glob_id, 256)
    else:
        print(f'Unknown data source "{cfg.data_source}". Either "loader" or "generator" expected')
        sys.exit(1)

    predict_on_dataset(model, ds_loader, objs)

    return 0


if __name__ == '__main__':
    run_and_exit(Config, main, 'Run pose network inference')

