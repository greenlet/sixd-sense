import time
from pathlib import Path
import shutil
from typing import Optional

import yaml

import numpy as np
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
import pymesh


from sds.utils import utils


MAX_REL_DIST_DEFAULT = 0.01


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
    max_rel_dist: float = Field(
        MAX_REL_DIST_DEFAULT,
        description=f'Maximal edge size relatively to object\' diameter (default = {MAX_REL_DIST_DEFAULT})',
        required=False,
        cli=('--max-rel-dist',),
    )
    overwrite: bool = Field(
        False,
        description=f'Objects are read from "SDS_ROOT_PATH/DATASET_NAME/models" directory, processed and saved '
                    f'to SDS_ROOT_PATH/DATASET_NAME/models_dense" directory with the same names. This process happens '
                    f'in case either destination file does not exist or OVERWRITE flag is set, in which case files are '
                    f'owerwritten. Otherwise, corresponding file is not processed',
        required=False,
        cli=('--overwrite',),
    )


def densify(mesh: pymesh.Mesh, max_rel_dist: float = MAX_REL_DIST_DEFAULT, remove_self_intersection: bool = False,
            tab: str = ' ' * 4) -> pymesh.Mesh:
    p_min, p_max = mesh.bbox
    max_dist = np.linalg.norm(p_max - p_min) * max_rel_dist

    def print_cap(cap: str) -> float:
        print(f'{tab}{cap}')
        return time.time()

    def print_verts(t_start: Optional[float] = None):
        time_suffix = ''
        if t_start is not None:
            time_suffix = f'. Time spent: {time.time() - t_start:.3f}'
        print(f'{tab}Vertices: {mesh.num_vertices}{time_suffix}')

    print_verts()

    t = print_cap(f'Splitting long edges with max edge length = {max_dist:.5f}')
    mesh, _ = pymesh.split_long_edges(mesh, max_dist)
    print_verts(t)

    t = print_cap(f'Removing duplicate vertices')
    mesh, _ = pymesh.remove_duplicated_vertices(mesh)
    print_verts(t)

    t = print_cap(f'Removing degenerate triangles')
    mesh, _ = pymesh.remove_degenerated_triangles(mesh, 20)
    print_verts(t)

    abs_threshold = max_dist * 2 / 3
    t = print_cap(f'Collapsing short edges with abs threshold = {abs_threshold:.5f}')
    mesh, _ = pymesh.collapse_short_edges(mesh, abs_threshold=abs_threshold, preserve_feature=True)
    print_verts(t)

    if remove_self_intersection:
        t = print_cap(f'Removing self intersections')
        mesh = pymesh.resolve_self_intersection(mesh)
        print_verts(t)

    t = print_cap(f'Removing duplicate vertices')
    mesh, _ = pymesh.remove_duplicated_vertices(mesh)
    print_verts(t)

    t = print_cap(f'Removing duplicate faces')
    mesh, _ = pymesh.remove_duplicated_faces(mesh)
    print_verts(t)

    t = print_cap(f'Removing isolated vertices')
    mesh, _ = pymesh.remove_isolated_vertices(mesh)
    print_verts(t)

    return mesh


def main(cfg: Config) -> int:
    print(cfg)
    norm_attr_name = 'vertex_normal'
    sds_ds_path = cfg.sds_root_path / cfg.dataset_name
    sds_models_path = sds_ds_path / 'models'
    sds_models_dense_path = sds_ds_path / 'models_dense'

    sds_models_dense_path.mkdir(exist_ok=True)

    for fpath_src in sds_models_path.iterdir():
        fpath_dst = sds_models_dense_path / fpath_src.name
        if fpath_src.suffix == '.yaml':
            if cfg.overwrite or not fpath_dst.exists():
                print(f'{fpath_src} -> {fpath_dst}')
                shutil.copy(fpath_src, fpath_dst)
        else:
            if fpath_dst.exists() and not cfg.overwrite:
                print(f'{fpath_dst} already exists')
            else:
                if not fpath_src.name.startswith('obj_000017'):
                    continue
                print(f'Loading mesh from {fpath_src}')

                mesh = pymesh.load_mesh(fpath_src.as_posix())

                if mesh.has_attribute(norm_attr_name):
                    mesh.remove_attribute(norm_attr_name)
                mesh = densify(mesh, max_rel_dist=cfg.max_rel_dist)

                print(f'Writing dense mesh in {fpath_dst}\n')
                pymesh.save_mesh(fpath_dst.as_posix(), mesh, norm_attr_name)

    return 0


if __name__ == '__main__':
    run_and_exit(Config, main, "Reads objects' models ply files, make new dense and uniform meshes "
                               "and saves them in a separate directory near the source")

