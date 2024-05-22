#!/bin/bash

start_seed=1
end_seed=9
exit_threshold=0.05
load_checkpoint_path="checkpoints/frozenBackbone_attention_3losses.pth"
model="deediff_uvit"
cmmd_batch_size=32
cmmd_max_count=10

python CMMD_evaluation/main.py \
    --model $model \
    --load_checkpoint_path $load_checkpoint_path \
    --exit_threshold $exit_threshold \
    --start_seed $start_seed \
    --end_seed $end_seed \
    --cmmd_batch_size $cmmd_batch_size \
    --cmmd_max_count $cmmd_max_count \