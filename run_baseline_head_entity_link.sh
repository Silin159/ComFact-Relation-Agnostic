#!/bin/bash

lm="deberta-large"
portion="all"
window="nlu"
task="entity"
sony_eval_set="test_head"
# root="/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/home/sigao/runs"
root="runs"

# Prepare directories for intermediate results of each subtask
mkdir -p pred/${portion}-${lm}-${window}-${task}
visible=0

CUDA_VISIBLE_DEVICES=${visible} python baseline.py \
   --eval_only \
   --checkpoint ${root}/${portion}-${lm}-${window}-${task}/ \
   --params_file ${root}/${portion}-${lm}-${window}-${task}/params-${lm}.json \
   --eval_dataset ${sony_eval_set} \
   --dataroot data/${portion}/${task}/${window} \
   --output_file pred/${portion}-${lm}-${window}-${task}-${sony_eval_set}/predictions.json
