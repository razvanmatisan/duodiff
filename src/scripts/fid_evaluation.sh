#!/bin/bash

exit_threshold=0.125
checkpoint_entry_name="frozenBackbone_attention_3losses"

python FID_evaluation.py \
    --checkpoint_entry_name $checkpoint_entry_name \
    --exit_threshold $exit_threshold