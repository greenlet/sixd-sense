import functools
import json
from datetime import datetime
from pathlib import Path
from typing import Union, Any, Optional, List, Tuple, Dict

import numpy as np
from plyfile import PlyData
from scipy.spatial import KDTree
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
    r = np.random.random(2)
    ang1 = 2 * np.pi * r[0]
    cos1, sin1 = np.cos(ang1), np.sin(ang1)
    cos2 = 1 - 2 * r[1]
    sin2 = np.sqrt(1 - cos2**2)

    rvec = np.array((cos1 * sin2, sin1 * sin2, cos2))
    ang = np.random.uniform(0, 2 * np.pi)

    return rvec, ang


def make_transform(rot_vec: np.ndarray, rot_alpha: float, pos: np.ndarray) -> np.ndarray:
    rot = R.from_rotvec(rot_vec * rot_alpha)
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = pos
    return T


def calc_3d_pts_uniform(num_pts: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    phi = np.arccos(1 - 2 * indices / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    return x, y, z


def graph_from_sphere_pts(pts: np.ndarray) -> List[Tuple[int, int]]:
    n = len(pts)
    r = np.sqrt(4 * np.pi / n) * 1.5
    kdt = KDTree(pts)
    edges = set()
    for i, pt in enumerate(pts):
        inds = kdt.query_ball_point(pt, r)
        for ind in inds:
            if ind == i:
                continue
            e = (i, ind) if i < ind else (ind, i)
            edges.add(e)
    edges = list(edges)
    return edges


def calc_3d_graph_elevated(n_pts: int, n_levels: int) -> Tuple[np.ndarray, int, np.ndarray]:
    x, y, z = calc_3d_pts_uniform(n_pts)
    pts = np.stack([x, y, z], axis=1)
    levels = np.arange(n_levels)
    edges_pts = graph_from_sphere_pts(pts)
    n_edges = len(edges_pts)
    edges_pts = np.array(edges_pts)

    edges_mul = np.tile(edges_pts, (n_levels, 1))
    levels_mul = np.repeat(levels * n_pts, n_edges)
    edges = edges_mul + levels_mul[..., None]
    edges_prev = edges_mul + np.stack([levels_mul, np.roll(levels_mul, -n_edges, axis=0)], axis=1)
    edges_next = edges_mul + np.stack([levels_mul, np.roll(levels_mul, n_edges, axis=0)], axis=1)

    n_verts = n_pts * n_levels
    inds_verts = np.arange(n_verts)
    edges_up = np.stack([inds_verts, np.roll(inds_verts, -n_pts)], axis=1)
    edges_down = np.stack([inds_verts, np.roll(inds_verts, n_pts)], axis=1)

    edges = np.concatenate([edges, edges_prev, edges_next, edges_up, edges_down])

    return pts, n_verts, edges


IntOrTuple = Union[int, Tuple[int, int]]


def int_to_tuple(ti: IntOrTuple) -> Tuple[int, int]:
    return ti if type(ti) == tuple else (ti, ti)
