from pathlib import Path
import shutil

import yaml
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit

from sds.utils import utils


def get_models_subdir_name(bop_ds_name: str) -> str:
    postfix = ''
    if bop_ds_name == 'tless':
        postfix = '_eval'
    return 'models' + postfix


class Config(BaseModel):
    bop_root_path: Path = Field(
        ...,
        description='Path to BOP datasets (containing datasets: ITODD, TLESS, etc.)',
        cli=('--bop-root-path',),
    )
    dataset_name: str = Field(
        ...,
        description='Dataset name. Has to be a subdirectory of BOP_ROOT_PATH, one of: "itodd", "tless", etc.',
        cli=('--dataset-name',),
    )
    sds_root_path: Path = Field(
        ...,
        description='Path to SixDSense datasets (containing datasets: ITODD, TLESS, etc.)',
        cli=('--sds-root-path',),
    )


def main(cfg: Config) -> int:
    print(cfg)
    bop_ds_path = cfg.bop_root_path / cfg.dataset_name
    sds_ds_path = cfg.sds_root_path / cfg.dataset_name
    bop_models_path = bop_ds_path / get_models_subdir_name(cfg.dataset_name)
    sds_models_path = sds_ds_path / 'models'
    print(f'Processing objects from {bop_models_path}')

    bop_models_info = utils.read_json(bop_models_path / 'models_info.json')

    # If any object's diameter is more than 10 then it cannot be in meters
    in_mm = any(obj['diameter'] > 10 for obj in bop_models_info.values())

    models_out_path = sds_models_path / 'models.yaml'
    # if models_out_dir.exists():
    #     shutil.rmtree(models_out_dir)
    sds_models_path.mkdir(parents=True, exist_ok=True)

    models_out = {}
    for obj_id, obj in bop_models_info.items():
        obj_id_num = int(obj_id)
        obj_id_pad = f'{obj_id_num:06d}'
        obj_id_full = f'obj_{obj_id_pad}'
        print(f'Object: {obj_id_full}')

        model = {
            'id': obj_id_full,
            'id_num': obj_id_num,
        }
        for key, val in obj.items():
            if in_mm and (key == 'diameter' or key.startswith('min') or key.startswith('max') or key.startswith('size')):
                val /= 1000
            model[key] = val
        models_out[obj_id_full] = model

        for fpath_src in bop_models_path.glob(f'{obj_id_full}.*'):
            fpath_dst = sds_models_path / fpath_src.name
            if fpath_dst.exists():
                continue
            print(f'Copying: {fpath_src} --> {fpath_dst}')
            if in_mm:
                ply = utils.read_ply(fpath_src)
                ply['vertex']['x'] /= 1000
                ply['vertex']['y'] /= 1000
                ply['vertex']['z'] /= 1000
                utils.write_ply(ply, fpath_dst)
            else:
                shutil.copy(fpath_src, fpath_dst)

    with open(models_out_path, 'w') as f:
        yaml.dump(models_out, f, default_flow_style=None)

    return 0


if __name__ == '__main__':
    run_and_exit(Config, main, 'Convert ply object to meters, restructure objects\' metadata')

