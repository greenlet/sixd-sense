#!/bin/bash

code_path=$HOME/prog
data_path=$HOME/data

ds_name=itodd
#ds_name=tless
models_path=$data_path/sds/$ds_name/models_dense


action=${1:-"run"}


cd $sds_src_path
blenderproc $action a_03_fix_normals.py \
  --models-path $models_path

