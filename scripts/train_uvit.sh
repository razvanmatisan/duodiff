model=uvit
number_of_training_steps=100000
batch_size=128
config_path=configs/uvit_imagenet_class_cond.yaml

CUBLAS_WORKSPACE_CONFIG=:4096:8 python main.py \
    --model $model \
    --n_steps $number_of_training_steps \
    --batch_size $batch_size \
    --use_amp \
    --log_every_n_steps 10000 \
    --save_every_n_steps 10000 \
    --save_new_every_n_steps 10000 \
    --seed 1 \
    --config_path $config_path \
    --dataset imagenet \