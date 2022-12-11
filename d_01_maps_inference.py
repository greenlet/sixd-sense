from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
import tensorflow as tf

from sds.data.index import load_cache_ds_index
from sds.data.ds_loader import DsLoader
from sds.model.utils import tf_img_to_float, tf_float_to_img, np_img_to_float, np_float_to_img
from sds.utils.utils import gen_colors
from sds.utils.ds_utils import load_objs
from sds.data.image_loader import ImageLoader
from sds.model.params import ScaledParams
from sds.utils.tf_utils import tf_set_use_device
from train_utils import find_train_weights_path, find_index_file, build_maps_model, color_segmentation, normalize


class Config(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    sds_root_path: Path = Field(
        ...,
        description='Path to SDS datasets (containing datasets: ITODD, TLESS, etc.)',
        cli=('--sds-root-path',),
    )
    target_dataset_name: str = Field(
        ...,
        description='Target dataset name. Has to be a subdirectory of SDS_ROOT_PATH, one of: "itodd", "tless", etc.',
        cli=('--target-dataset-name',),
    )
    distractor_dataset_name: str = Field(
        ...,
        description='Distractor dataset name. Has to be a subdirectory of SDS_ROOT_PATH, one of: "itodd", "tless", etc.',
        cli=('--distractor-dataset-name',),
    )
    phi: int = Field(
        ...,
        description=f'Model phi number. Must be: 0 <= PHI <= {ScaledParams.phi_max()}',
        required=True,
        cli=('--phi',),
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
        'val',
        description='Images source which will be used to visualize maps predictions. Can be either '
                    'path to images directory or have value "val". In latter case validation dataset '
                    'will be used for predictions and GT will be shown',
        required=False,
        cli=('--data-source',),
    )
    use_gpu: bool = Field(
        False,
        description='If set, GPU will be used. CPU is used by default',
        required=False,
        cli=('--use-gpu',),
    )


def predict_on_dataset(model: tf.keras.models.Model, ds_loader: DsLoader):
    w, h = ds_loader.img_size
    img_vis = np.zeros((2 * h, 4 * w, 3), dtype=np.uint8)
    colors = gen_colors()
    cv2.imshow('maps', img_vis)
    cv2.moveWindow('maps', 0, 0)

    for img, (noc_gt, norms_gt, seg_gt) in ds_loader.gen():
        img_in = tf.convert_to_tensor(img)[None, ...]
        img_in = tf_img_to_float(img_in)
        noc_pred, norms_pred, seg_pred = model(img_in)
        seg_pred = tf.math.argmax(seg_pred, -1)
        noc_pred, norms_pred, seg_pred = noc_pred[0].numpy(), norms_pred[0].numpy(), seg_pred[0].numpy()

        norms_gt1 = (norms_gt - 127.5) / 127.5
        norms_gt1 = np.linalg.norm(norms_gt1, axis=-1)
        norms_pred1 = np.linalg.norm(norms_pred, axis=-1)
        print(f'norms_gt. min: {norms_gt1.min()}. max: {norms_gt1.max()}. mean: {norms_gt1.mean()}')
        print(f'norms_pred. min: {norms_pred1.min()}. max: {norms_pred1.max()}. mean: {norms_pred1.mean()}')
        # norms_gt = np_float_to_img(normalize(np_img_to_float(norms_gt)))
        normalize(norms_pred, inplace=True)
        # norms_pred *= norms_gt1.max()

        img_vis[:h, :w] = img
        img_vis[:h, w:2 * w] = noc_gt
        img_vis[:h, 2 * w:3 * w] = norms_gt
        img_vis[:h, 3 * w:] = color_segmentation(colors, seg_gt)
        img_vis[h:, w:2 * w] = tf_float_to_img(noc_pred)
        img_vis[h:, 2 * w:3 * w] = tf_float_to_img(norms_pred)
        img_vis[h:, 3 * w:] = color_segmentation(colors, seg_pred)

        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        cv2.imshow('maps', img_vis)
        if cv2.waitKey() in (27, ord('q')):
            break


def predict_on_images(model: tf.keras.models.Model, img_loader: ImageLoader):
    pass


def main(cfg: Config) -> int:
    print(cfg)

    train_path, weights_path = find_train_weights_path(cfg.weights_path)
    if train_path is None or not train_path.exists():
        print(f'Cannot find weights in {cfg.weights_path}')
        return 1

    tf_set_use_device(cpu=not cfg.use_gpu)

    objs = load_objs(cfg.sds_root_path, cfg.target_dataset_name, cfg.distractor_dataset_name)
    n_classes = len(objs)

    print('Building model')
    model = build_maps_model(n_classes, cfg.phi, False)
    print(f'Loading model from {weights_path}')
    model.load_weights(weights_path.as_posix())
    model_params = ScaledParams(cfg.phi)

    ds_loader = None
    img_loader = None
    if cfg.data_source == 'val':
        index_fpath = find_index_file(train_path)
        if index_fpath is None:
            print(f'Cannot find dataset index file in {train_path}')
            return 1
        ds_index = load_cache_ds_index(index_fpath=index_fpath)
        ds_loader = DsLoader(ds_index, objs, is_training=False, img_size=model_params.input_size)
    else:
        images_path = Path(cfg.data_source)
        img_loader = ImageLoader(images_path)

    if ds_loader is not None:
        predict_on_dataset(model, ds_loader)
    else:
        predict_on_images(model, img_loader)

    return 0


if __name__ == '__main__':
    run_and_exit(Config, main, 'Run maps network inference')

