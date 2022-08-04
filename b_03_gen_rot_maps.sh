#!/bin/bash

code_path=$HOME/prog
#data_path=$HOME/data
#sds_src_path=$code_path/sixd_sense
data_path=/data/data
sds_src_path=$code_path/sds

export PYTHONPATH=$PYTHONPATH:$sds_src_path
export MESA_GL_VERSION_OVERRIDE=3.3

sds_root_path=$data_path/sds
dataset_name=itodd
models_subdir='models'
model_id_num=1

python b_03_gen_rot_maps.py \
  --sds-root-path $sds_root_path \
  --dataset-name $dataset_name \
  --models-subdir $models_subdir \
  --model-id-num $model_id_num

