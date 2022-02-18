#!/bin/bash

code_path=$HOME/prog
data_path=$HOME/data
sds_src_path=$code_path/sixd_sense

sds_root_path=$data_path/sds
target_dataset_name=$1
distractor_dataset_name=$2
models_subdir=$3
output_type=$4
debug_arg=$5


export PYTHONPATH=$PYTHONPATH:$sds_src_path


python b_02_gen_vec_gt.py $debug_arg \
  --sds-root-path $sds_root_path \
  --target-dataset-name $target_dataset_name \
  --distractor-dataset-name $distractor_dataset_name \
  --models-subdir $models_subdir \
  --output-type $output_type

