#!/bin/bash

exit_threshold=0.05
checkpoint_entry_name="frozenBackbone_attention_3losses"
cmmd_batch_size=32
cmmd_max_count=10

python CMMD_evaluation/main.py \
    --checkpoint_entry_name $checkpoint_entry_name \
    --exit_threshold $exit_threshold \
    --cmmd_batch_size $cmmd_batch_size \
    --cmmd_max_count $cmmd_max_count