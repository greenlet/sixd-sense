import json
import sys
import time
from enum import Enum
from pathlib import Path
import shutil
from typing import Optional, Dict, List, Any, Tuple

import cv2
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLUT import *
from OpenGL.GLU import *
import h5py
import yaml

import numpy as np
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
import pymesh
from scipy.spatial.transform import Rotation as R


def ravel(*arrs: np.ndarray) -> List[np.ndarray]:
    res = []
    for arr in arrs:
        res.append(arr.ravel())
    return res


def gen_full_sphere_coords_inds(n: int) -> Tuple[np.ndarray, np.ndarray]:
    inds = np.arange(0, n)
    coords = inds / inds.mean() - 1
    ix, iy, iz = np.meshgrid(inds, inds, inds, indexing='ij')
    x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
    inds = np.stack(ravel(ix, iy, iz)).T
    coords = np.stack(ravel(x, y, z)).T
    coords_norms = np.linalg.norm(coords, axis=1)
    fsphere_mask = coords_norms <= 1 + 1e-10
    inds, coords = inds[fsphere_mask], coords[fsphere_mask]
    ii = np.arange(0, len(inds))
    np.random.shuffle(ii)
    inds, coords = inds[ii], coords[ii]
    return inds, coords * np.pi


# gen_full_sphere_coords_inds(10)
gen_full_sphere_coords_inds(128)

