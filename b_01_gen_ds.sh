#!/bin/bash

code_path=$HOME/prog
data_path=$HOME/data
sds_src_path=$code_path/sixd_sense

bop_root_path=$data_path/bop
bop_dataset_name=itodd
action=${1:-"run"}

#config_path=$sds_src_path/dsgen_config_debug.yaml
config_path=$sds_src_path/dsgen_config_itodd.yaml


cd $sds_src_path
blenderproc $action b_01_gen_ds.py \
  --config-path $config_path

