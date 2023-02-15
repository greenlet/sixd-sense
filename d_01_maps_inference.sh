#!/bin/bash

code_path=$HOME/prog
#data_path=$HOME/data
#sds_src_path=$code_path/sixd_sense
data_path=/ws/data
sds_src_path=$code_path/sds
train_root_path=$data_path/sds_train_maps

export PYTHONPATH=$PYTHONPATH:$sds_src_path
export CUDA_DIR=/ws/miniconda3/envs/sds/pkgs/cuda-toolkit
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/dri/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/ws/miniconda3/envs/sds/pkgs/cuda-toolkit

sds_root_path=$data_path/sds
target_dataset_name=itodd
distractor_dataset_name=tless
phi=0
weights_subdir=20221222_152244_itodd_t90990_v10110
#deviceid="-1"
device_id="1"
data_source="val"
#data_source="/media/mburakov/AEF2B64EF2B61B13/data/itodd_test_all/test/000001/gray/"

python d_01_maps_inference.py \
  --sds-root-path $sds_root_path \
  --target-dataset-name $target_dataset_name \
  --distractor-dataset-name $distractor_dataset_name \
  --phi $phi \
  --weights-path $train_root_path/$weights_subdir \
  --device-id $device_id \
  --data-source $data_source

