#!/bin/bash

code_path=$HOME/prog
data_path=$HOME/data
#sds_src_path=$code_path/sixd_sense
# data_path=/ws/data
sds_src_path=$code_path/sds

bop_root_path=$data_path/bop
sds_root_path=$data_path/sds
dataset_name=$1

export PYTHONPATH=$PYTHONPATH:$sds_src_path


python a_01_convert_ply_models.py \
  --bop-root-path $bop_root_path \
  --dataset-name $dataset_name \
  --sds-root-path $sds_root_path

