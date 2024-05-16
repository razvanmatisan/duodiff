#!/bin/bash

start_seed=1
end_seed=5
exit_threshold=0.125

# Loop through each sample_seed value
for seed in $(seq $start_seed $end_seed)
do
    log_path="samples/${exit_threshold}_${seed}"

    python sample.py --load_checkpoint_path logs/6257371/cifar10_deediff_uvit.pth --log_path $log_path --exit_threshold $exit_threshold --n_samples 1 --sample_seed $seed
done