from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pymesh

from sds.utils.utils import read_yaml


def load_mesh(mesh_fpath: Path) -> Tuple[pymesh.Mesh, Dict[str, np.ndarray]]:
    mesh = pymesh.load_mesh(mesh_fpath.as_posix())
    verts, faces = mesh.vertices, mesh.faces
    normals_attr_name = 'vertex_normal'
    if not mesh.has_attribute(normals_attr_name):
        mesh.add_attribute(normals_attr_name)
    normals = mesh.get_vertex_attribute(normals_attr_name)
    mesh_dict = {
        'verts': verts, 'faces': faces, 'normals': normals,
    }
    return mesh, mesh_dict


def load_objs(sds_root_path: Path, target_dataset_name: str, distractor_dataset_name: Optional[str] = None,
              models_subdir: str = 'models', load_meshes: bool = False,
              load_target_id_num: Optional[int] = None, load_target_glob_id: Optional[str] = None) -> Dict[str, Dict]:
    target_models_path = sds_root_path / target_dataset_name / models_subdir
    target_objs = read_yaml(target_models_path / 'models.yaml')
    res = {}
    max_glob_num = 0
    for obj_id, obj in target_objs.items():
        obj['ds_name'] = target_dataset_name
        obj['glob_num'] = obj['id_num']
        glob_id = f'{target_dataset_name}_{obj_id}'
        if load_meshes and (
                (load_target_id_num is None or load_target_id_num == obj['id_num']) and
                (load_target_glob_id is None or load_target_glob_id == glob_id)
        ):
            obj_fpath = target_models_path / f'{obj_id}.ply'
            _, mesh_dict = load_mesh(obj_fpath)
            obj['mesh'] = mesh_dict
        max_glob_num = max(max_glob_num, obj['id_num'])
        res[glob_id] = obj

    if distractor_dataset_name is not None:
        distractor_models_path = sds_root_path / distractor_dataset_name / models_subdir
        objs_dist = read_yaml(distractor_models_path / 'models.yaml')
        for obj_id, obj in objs_dist.items():
            obj['ds_name'] = distractor_dataset_name
            obj['glob_num'] = max_glob_num + obj['id_num']
            if load_meshes:
                obj_fpath = distractor_models_path / f'{obj_id}.ply'
                _, mesh_dict = load_mesh(obj_fpath)
                obj['mesh'] = mesh_dict

            res[f'{distractor_dataset_name}_{obj_id}'] = obj

    return res
