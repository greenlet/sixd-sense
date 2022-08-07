#!/bin/bash

code_path=$HOME/prog
#data_path=$HOME/data
#sds_src_path=$code_path/sixd_sense
data_path=/data/data
sds_src_path=$code_path/sds
train_root_path=$data_path/sds_train_pose

export PYTHONPATH=$PYTHONPATH:$sds_src_path
export CUDA_VISIBLE_DEVICES=-1
export MESA_GL_VERSION_OVERRIDE=3.3

sds_root_path=$data_path/sds
dataset_name=itodd
phi=0
#weights_subdir=20220731_211130_itodd_obj_000001
weights_subdir=20220804_215428_itodd_obj_000003
use_gpu=""
bool_args="$use_gpu"
data_source=loader

python d_02_pose_inference.py $bool_args \
  --sds-root-path $sds_root_path \
  --dataset-name $dataset_name \
  --weights-path $train_root_path/$weights_subdir \
  --data-source $data_source

