#!/bin/bash

code_path=$HOME/prog
#data_path=$HOME/data
#sds_src_path=$code_path/sixd_sense
data_path=/ws/data
sds_src_path=$code_path/sds

export PYTHONPATH=$PYTHONPATH:$sds_src_path
#export MESA_GL_VERSION_OVERRIDE=4.1
#export MESA_LOADER_DRIVER_OVERRIDE=i965
#export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_SAVE
export CUDA_DIR=/ws/miniconda3/envs/sds/pkgs/cuda-toolkit
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/dri/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/ws/miniconda3/envs/sds/pkgs/cuda-toolkit

train_root_path=$data_path/sds_train_aae
obj_id=teamug
obj_path=$data_path/sds_data/objs/teamug.stl

obj_id=obj_000001
obj_path=$data_path/sds/itodd/models/${obj_id}.ply

img_size=128
learning_rate=1e-3
loss_bootstrap_ratio=4

iterations=200000
batch_size=200
pose_gen_workers=40
loss_acc_interval=100
model_save_interval=1000

#iterations=20
#batch_size=2
#pose_gen_workers=2
#loss_acc_interval=5
#model_save_interval=10


python c_03_train_pose_aae.py \
  --obj-id $obj_id \
  --obj-path $obj_path \
  --train-root-path $train_root_path \
  --img-size $img_size \
  --learning-rate $learning_rate \
  --loss-bootstrap-ratio $loss_bootstrap_ratio \
  --iterations $iterations \
  --batch-size $batch_size \
  --loss-acc-interval $loss_acc_interval \
  --model-save-interval $model_save_interval \
  --pose-gen-workers $pose_gen_workers

