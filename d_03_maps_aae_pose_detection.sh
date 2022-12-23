#!/bin/bash

code_path=$HOME/prog
#data_path=$HOME/data
data_path=/ws/data
sds_src_path=$code_path/sds
sds_root_path=$data_path/sds
maps_train_root_path=$data_path/sds_train_maps
aae_train_root_path=$data_path/sds_train_aae

export PYTHONPATH=$PYTHONPATH:$sds_src_path
export CUDA_DIR=/ws/miniconda3/envs/sds/pkgs/cuda-toolkit
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/dri/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/ws/miniconda3/envs/sds/pkgs/cuda-toolkit


target_dataset_name=itodd
distractor_dataset_name=tless
maps_phi=0
maps_weights=last
aae_weights=last
device_id="1"
data_source="val"

python d_03_maps_aae_pose_detection.py \
  --sds-root-path $sds_root_path \
  --target-dataset-name $target_dataset_name \
  --distractor-dataset-name $distractor_dataset_name \
  --maps-train-root-path $maps_train_root_path \
  --aae-train-root-path $aae_train_root_path \
  --maps-phi $maps_phi \
  --maps-weights $maps_weights \
  --aae-weights $aae_weights \
  --device-id $device_id \
  --data-source $data_source

