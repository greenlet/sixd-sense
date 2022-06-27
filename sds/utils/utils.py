import functools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pymesh
from plyfile import PlyData
from typing import Union, Any, Optional, List, Tuple, Dict
import yaml


def read_json(file_path: Union[str, Path]) -> Any:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


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


def load_objs(sds_root_path: Path, target_dataset_name: str, distractor_dataset_name: Optional[str] = None,
              models_subdir: str = 'models', load_meshes: bool = False) -> Dict[str, Dict]:
    target_models_path = sds_root_path / target_dataset_name / models_subdir
    target_objs = read_yaml(target_models_path / 'models.yaml')
    res = {}
    max_glob_num = 0
    for obj_id, obj in target_objs.items():
        obj['ds_name'] = target_dataset_name
        obj['glob_num'] = obj['id_num']
        if load_meshes:
            obj_fpath = target_models_path / f'{obj_id}.ply'
            obj['mesh'] = pymesh.load_mesh(obj_fpath.as_posix())
        max_glob_num = max(max_glob_num, obj['id_num'])
        res[f'{target_dataset_name}_{obj_id}'] = obj

    if distractor_dataset_name is not None:
        distractor_models_path = sds_root_path / distractor_dataset_name / models_subdir
        objs_dist = read_yaml(distractor_models_path / 'models.yaml')
        for obj_id, obj in objs_dist.items():
            obj['ds_name'] = distractor_dataset_name
            obj['glob_num'] = max_glob_num + obj['id_num']
            if load_meshes:
                obj_fpath = distractor_models_path / f'{obj_id}.ply'
                obj['mesh'] = pymesh.load_mesh(obj_fpath.as_posix())

            res[f'{distractor_dataset_name}_{obj_id}'] = obj

    return res

