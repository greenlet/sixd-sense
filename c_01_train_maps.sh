#!/bin/bash

code_path=$HOME/prog
#sds_src_path=$code_path/sixd_sense
#data_path=$HOME/data
#data_path=/data/data
data_path=/ws/data
sds_src_path=$code_path/sds

export PYTHONPATH=$PYTHONPATH:$sds_src_path
export CUDA_DIR=/ws/miniconda3/envs/sds/pkgs/cuda-toolkit
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/dri/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/ws/miniconda3/envs/sds/pkgs/cuda-toolkit

sds_root_path=$data_path/sds
target_dataset_name=itodd
distractor_dataset_name=tless
ds_path=/media/mburakov/AEF2B64EF2B61B13/data/sds_itodd
models_subdir='models'
phi=0
freeze_bn_arg=''
train_root_path=$data_path/sds_train_maps
weights_to_use='none'
new_learning_subdir_arg=''
train_split_perc=90
learning_rate=1e-3
epochs=200
batch_size=3
train_steps=2000
val_steps=200
debug_arg=""
bool_args="$freeze_bn_arg $new_learning_subdir_arg $debug_arg"

#epochs=3
#train_steps=5
#val_steps=2

python c_01_train_maps.py $bool_args \
  --sds-root-path $sds_root_path \
  --target-dataset-name $target_dataset_name \
  --distractor-dataset-name $distractor_dataset_name \
  --ds-path $ds_path \
  --models-subdir $models_subdir \
  --phi $phi \
  --train-root-path $train_root_path \
  --weights-to-use $weights_to_use \
  --learning-rate $learning_rate \
  --epochs $epochs \
  --train-split-perc $train_split_perc \
  --batch-size $batch_size \
  --train-steps $train_steps \
  --val-steps $val_steps

