import blenderproc as bproc
from blenderproc.python.material import MaterialLoaderUtility

import argparse
import itertools
import os
from pathlib import Path
import sys
from typing import Any, Dict, Tuple, List, Optional, Union
import time

import h5py
import numpy as np
from pydantic_yaml import YamlModel

import bpy
from mathutils import Matrix


def run():
    parser = argparse.ArgumentParser(description='Fixing models\' normals by pointing them outwards')
    parser.add_argument('--models-path', type=Path, required=True,
                        help='Path to PLY files. Models will be changed in place')
    args = parser.parse_args()
    models_path: Path = args.models_path

    for fpath in models_path.iterdir():
        if fpath.suffix.lower() != '.ply':
            continue
        objs = bproc.loader.load_obj(fpath.as_posix())
        assert len(objs) == 1
        obj = objs[0]
        



if __name__ == '__main__':
    run()

