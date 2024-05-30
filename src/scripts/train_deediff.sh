model=deediff_uvit
number_of_training_steps=100000
batch_size=128
classifier_type="attention_probe"

python train.py \
    --model ${deediff_uvit} \
    --n_steps ${number_of_training_steps} \
    --batch_size ${batch_size} \
    --classifier_type ${classifier_type} \
    --normalize_timesteps