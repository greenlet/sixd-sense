import functools
import json
from datetime import datetime
from pathlib import Path
from typing import Union, Any, Optional, List, Tuple, Dict

import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R
import yaml


def read_json(file_path: Union[str, Path]) -> Any:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def write_json(data: Any, file_path: Union[str, Path]) -> Any:
    with open(file_path, 'w') as f:
        json.dump(data, f)


def read_yaml(file_path: Union[str, Path]) -> Any:
    with open(file_path, 'r') as f:
        data = yaml.load(f, yaml.FullLoader)
    return data


def write_yaml(yamldata: Any, file_path: Union[str, Path]) -> Any:
    with open(file_path, 'w') as f:
        yaml.dump(yamldata, f)


def read_ply(file_path: Union[str, Path]) -> PlyData:
    with open(file_path, 'rb') as f:
        plydata = PlyData.read(f)
        return plydata


def write_ply(plydata: PlyData, file_path: Union[str, Path]):
    with open(file_path, 'wb') as f:
        plydata.write(f)


def write_txt(txt: str, file_path: Union[str, Path]):
    with open(file_path, 'w') as f:
        f.write(txt)


def compose(*funcs):
    return functools.reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), funcs)


def datetime_str(dt: Optional[datetime] = None) -> str:
    dt = dt if dt is not None else datetime.now()
    return dt.strftime('%Y%m%d_%H%M%S')


def gen_colors(steps_per_chan: int = 10, seed: Optional[int] = None) -> List[Tuple[int, int, int]]:
    step = 256 // steps_per_chan
    chan = lambda: range(step // 2, 256, step)
    res = [(r, g, b) for r in chan() for g in chan() for b in chan()]
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(res)
    return res


def canonical_cam_mat_from_img(img_size: Tuple[int, int]) -> np.ndarray:
    w, h = img_size
    f = max(img_size)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def calc_ref_dist_from_camera(cam_mat: np.ndarray, img_size: Tuple[int, int], obj: Dict[str, Any], hist_sz: int = 0) -> float:
    img_sz = np.array(img_size, dtype=np.float64)
    f, c = cam_mat[[0, 1], [0, 1]], cam_mat[:2, 2]
    sz = np.minimum(c, img_sz - c)
    diam_px = sz * 2 * 0.9
    diam = obj['diameter']
    dist = diam * f / diam_px
    return np.max(dist)


def gen_rot_vec() -> Tuple[np.ndarray, float]:
    while True:
        r = np.random.random((3,))
        rl = np.linalg.norm(r)
        if rl > 1e-6:
            return r / rl, np.random.uniform(0, 2 * np.pi)


def make_transform(rot_vec: np.ndarray, rot_alpha: float, pos: np.ndarray) -> np.ndarray:
    rot = R.from_rotvec(rot_vec * rot_alpha)
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = pos
    return T

