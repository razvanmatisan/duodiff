batch_size=4
config_path=configs/uvit_imagenet_class_cond.yaml
checkpoint_path=checkpoints/downloaded_checkpoints/imagenet256_uvit_large.pth
output_folder=samples/imagenet/

python sampler.py \
    --parametrization predict_noise \
    --batch_size $batch_size \
    --seed 1 \
    --config_path $config_path \
    --checkpoint_path $checkpoint_path \
    --output_folder $output_folder \
    --class_id 3