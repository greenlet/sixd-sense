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
#model_id_num=1
model_id_num=3
train_root_path=$data_path/sds_train_pose
learning_rate=1e-3
epochs=200
batch_size=4
train_steps=2000
val_steps=200

#epochs=3
#train_steps=5
#val_steps=2

python c_02_train_pose.py \
  --sds-root-path $sds_root_path \
  --dataset-name $dataset_name \
  --models-subdir $models_subdir \
  --model-id-num $model_id_num \
  --train-root-path $train_root_path \
  --learning-rate $learning_rate \
  --epochs $epochs \
  --batch-size $batch_size \
  --train-steps $train_steps \
  --val-steps $val_steps

