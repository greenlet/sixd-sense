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

aae_img_size=128
aae_rot_vecs_num=5000
aae_rot_angs_num=180

device_id="0"
data_source="val"
obj_ids=(obj_000001)

python d_03_maps_aae_pose_detection.py \
  --sds-root-path $sds_root_path \
  --target-dataset-name $target_dataset_name \
  --distractor-dataset-name $distractor_dataset_name \
  --maps-train-root-path $maps_train_root_path \
  --aae-train-root-path $aae_train_root_path \
  --maps-phi $maps_phi \
  --maps-weights $maps_weights \
  --aae-weights $aae_weights \
  --aae-img-size $aae_img_size \
  --aae-rot-vecs-num $aae_rot_vecs_num \
  --aae-rot-angs-num $aae_rot_angs_num \
  --device-id $device_id \
  --data-source $data_source \
  --obj-ids "${obj_ids[@]}"

