#!/bin/bash

code_path=$HOME/prog
data_path=$HOME/data
#sds_src_path=$code_path/sixd_sense
data_path=/ws/data
sds_src_path=$code_path/sds
ds_path=/media/mburakov/AEF2B64EF2B61B13/data/sds_itodd

sds_root_path=$data_path/sds
target_dataset_name=$1
distractor_dataset_name=$2
models_subdir=$3
output_type=$4
debug_arg=$5


export PYTHONPATH=$PYTHONPATH:$sds_src_path
#export MESA_GL_VERSION_OVERRIDE=3.3
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/dri/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

python b_02_gen_maps.py $debug_arg \
  --sds-root-path $sds_root_path \
  --target-dataset-name $target_dataset_name \
  --distractor-dataset-name $distractor_dataset_name \
  --ds-path $ds_path \
  --models-subdir $models_subdir \
  --output-type $output_type

