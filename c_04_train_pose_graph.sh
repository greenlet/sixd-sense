#!/bin/bash

code_path=$HOME/prog
#data_path=$HOME/data
#sds_src_path=$code_path/sixd_sense
data_path=/ws/data
sds_src_path=$code_path/sds

export PYTHONPATH=$PYTHONPATH:$sds_src_path
export MESA_GL_VERSION_OVERRIDE=3.3

sds_root_path=$data_path/sds
dataset_name=itodd
obj_models_subdir='models'
obj_id_num=1
#obj_id_num=2
#obj_id_num=3
img_size=256
graph_pts=1000
graph_levels=100
train_root_path=$data_path/sds_train_pose
learning_rate=1e-3
epochs=200
batch_size=40
train_steps=2000
val_steps=200
pose_gen_workers=20

epochs=50
batch_size=150
train_steps=100
val_steps=10
pose_gen_workers=50

batch_size=4
epochs=3
train_steps=5
val_steps=2
pose_gen_workers=0

python c_04_train_pose_graph.py \
  --sds-root-path $sds_root_path \
  --dataset-name $dataset_name \
  --obj-models-subdir $obj_models_subdir \
  --obj-id-num $obj_id_num \
  --img-size $img_size \
  --graph-pts $graph_pts \
  --graph-levels $graph_levels \
  --train-root-path $train_root_path \
  --learning-rate $learning_rate \
  --epochs $epochs \
  --batch-size $batch_size \
  --train-steps $train_steps \
  --val-steps $val_steps \
  --pose-gen-workers $pose_gen_workers"

