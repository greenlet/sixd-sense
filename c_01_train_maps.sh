#!/bin/bash

code_path=$HOME/prog
data_path=$HOME/data
sds_src_path=$code_path/sixd_sense

export PYTHONPATH=$PYTHONPATH:$sds_src_path

bop_root_path=$data_path/bop
sds_root_path=$data_path/sds
target_dataset_name=itodd
distractor_dataset_name=tless
models_subdir='models'
phi=0
freeze_bn_arg=''


python a_01_convert_ply_models.py $freeze_bn_arg \
  --sds-root-path $sds_root_path
  --target-dataset-name $target_dataset_name \
  --distractor-dataset-name $distractor_dataset_name \
  --models-subdir $models_subdir \
  --phi $phi


