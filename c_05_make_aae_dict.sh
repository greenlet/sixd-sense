#!/bin/bash

code_path=$HOME/prog
#data_path=$HOME/data
#sds_src_path=$code_path/sixd_sense
data_path=/ws/data
sds_src_path=$code_path/sds

export PYTHONPATH=$PYTHONPATH:$sds_src_path
export CUDA_DIR=/ws/miniconda3/envs/sds/pkgs/cuda-toolkit
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/dri/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/ws/miniconda3/envs/sds/pkgs/cuda-toolkit

train_root_path=$data_path/sds_train_aae
obj_id=teamug
obj_path=$data_path/sds_data/objs/teamug.stl

obj_id=obj_000001
obj_path=$data_path/sds/itodd/models

img_size=128
weights=last

rot_vecs_num=5000
rot_angs_num=180
batch_size=2500
pose_gen_workers=40

#rot_vecs_num=100
#rot_angs_num=10
#batch_size=2
#pose_gen_workers=2


python c_05_make_aae_dict.py \
  --obj-id $obj_id \
  --obj-path $obj_path \
  --train-root-path $train_root_path \
  --img-size $img_size \
  --weights $weights \
  --rot-vecs-num $rot_vecs_num \
  --rot-angs-num $rot_angs_num \
  --batch-size $batch_size \
  --pose-gen-workers $pose_gen_workers

