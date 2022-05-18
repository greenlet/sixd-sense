import functools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from plyfile import PlyData
from typing import Union, Any, Optional, List, Tuple
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


