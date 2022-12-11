#!/bin/bash

code_path=$HOME/prog
data_path=$HOME/data
#sds_src_path=$code_path/sixd_sense
#data_path=/ws/data
sds_src_path=$code_path/sds

export PYTHONPATH=$PYTHONPATH:$sds_src_path
export MESA_GL_VERSION_OVERRIDE=3.3

train_root_path=$data_path/sds_train_aae
obj_id=teamug
obj_path=$data_path/sds_data/objs/teamug.stl

img_size=128
learning_rate=1e-3
loss_bootstrap_ratio=4

iterations=200000
batch_size=40
pose_gen_workers=10
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

