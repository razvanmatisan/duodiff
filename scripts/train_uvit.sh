model=uvit
number_of_training_steps=100000
batch_size=128

python train.py \
    --model ${model} \
    --n_steps ${number_of_training_steps} \
    --batch_size ${batch_size} \