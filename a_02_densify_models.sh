#!/bin/bash

code_path=$HOME/prog
data_path=$HOME/data
sds_src_path=$code_path/sixd_sense

sds_root_path=$data_path/sds
dataset_name=$1

export PYTHONPATH=$PYTHONPATH:$sds_src_path


python a_02_densify_models.py \
  --sds-root-path $sds_root_path \
  --dataset-name $dataset_name

