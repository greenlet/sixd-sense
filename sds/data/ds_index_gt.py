import time
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from tqdm import tqdm

from sds.data.utils import read_gt_item, glob_inst_to_glob_id
from sds.utils.utils import load_objs


def make_index_gt(ds_path: Path, objs: Dict[str, Any], max_scenes: int = 0) -> pd.DataFrame:
    data_path = ds_path / 'data'
    glob_ids = sorted(list(objs.keys()))
    glob_id_to_ind = {glob_id: i for i, glob_id in enumerate(glob_ids)}
    col_names = ('scene_num', 'file_num', *glob_ids)
    n_cols = len(col_names)
    data = []
    print('Gathering Gt')
    t1 = time.time()
    scene_paths = list(sorted(data_path.iterdir()))
    if max_scenes > 0:
        scene_paths = scene_paths[:max_scenes]
    for scene_path in tqdm(scene_paths):
        scene_name = scene_path.name
        for item_path in sorted(scene_path.iterdir()):
            gt_item = read_gt_item(item_path, load_img=False, load_segmap=False)
            row = [0] * n_cols
            row[0] = int(scene_name)
            row[1] = int(item_path.with_suffix('').name)
            for glob_inst_id in gt_item.segmap_key_to_num:
                glob_id = glob_inst_to_glob_id(glob_inst_id)
                row[2 + glob_id_to_ind[glob_id]] += 1
            data.append(row)
    t2 = time.time()
    print(f'Gt gathered: {t2 - t1:.2f}')
    print(f'Building DataFrame')
    t1 = t2
    df = pd.DataFrame(data, columns=col_names)
    t2 = time.time()
    print(f'DataFrame built: {t2 - t1:.2f}')

    return df


def write_df_to_csv(df: pd.DataFrame, csv_path: Path):
    df.to_csv(csv_path.as_posix())


def read_df_from_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path.as_posix())


def load_index_gt(ds_path: Path, objs: Dict[str, Any], recalc: bool = False, max_scenes: int = 0) -> pd.DataFrame:
    index_fname = 'index_gt.csv'
    index_fpath = ds_path / index_fname
    if not recalc and index_fpath.exists():
        return read_df_from_csv(index_fpath)
    df = make_index_gt(ds_path, objs, max_scenes)
    write_df_to_csv(df, index_fpath)
    return df


def _test_load_index():
    sds_root_path = Path('/data/data/sds')
    target_ds_name, dist_ds_name = 'itodd', 'tless'
    max_scenes = 0
    recalc = True
    objs = load_objs(sds_root_path, target_ds_name, dist_ds_name)
    df = load_index_gt(sds_root_path / target_ds_name, objs, recalc=recalc, max_scenes=max_scenes)
    print(df)


if __name__ == '__main__':
    _test_load_index()

