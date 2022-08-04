#!/bin/bash

code_path=$HOME/prog
#data_path=$HOME/data
#sds_src_path=$code_path/sixd_sense
data_path=/data/data
sds_src_path=$code_path/sds
train_root_path=$data_path/sds_train_maps

export PYTHONPATH=$PYTHONPATH:$sds_src_path

sds_root_path=$data_path/sds
target_dataset_name=itodd
distractor_dataset_name=tless
phi=0
weights_subdir=20220527_093824_itodd_t124814_v13868
use_gpu=""
bool_args="$use_gpu"
data_source=val

python d_01_maps_inference.py $bool_args \
  --sds-root-path $sds_root_path \
  --target-dataset-name $target_dataset_name \
  --distractor-dataset-name $distractor_dataset_name \
  --phi $phi \
  --weights-path $train_root_path/$weights_subdir \
  --data-source $data_source

