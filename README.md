# DuoDiff - Accelerating Diffusion Models with a Dual-Backbone Approach
The first step should be configuring a proper environment. On UNIX,
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
On Windows,
```powershell
python -m venv venv
Set-ExecutionPolicy Unrestricted -Scope Process
venv\Scripts\Activate
pip install -r src\requirements.txt
```

## Training
In this section, we will see how to train early-exit models and DuoDiff on the CelebA dataset. Training on other datasets is straightforward, and we recommend checking the different options in `main.py`.

<details>
<summary>The first step is to obtain a full-model backbone.</summary>

```bash
python train.py \
    --n_steps 500000 \
    --batch_size 128 \
    --log_path "${log_path}" \
    --dataset "celeba" \
    --log_every_n_steps 2500 \
    --save_every_n_steps 25000 \
    --save_new_every_n_steps 100000 \
    --sample_height 64 \
    --sample_width 64 \
    --img_size 64 \
    --patch_size 4 \
    --seed 1 \
    --model uvit \
    --normalize_timesteps \
    --use_amp \
    --parametrization "predict_noise"
```
</details> 

### Early-exit training (DeeDiff / AdaDiff)
<details>
<summary>Then, we can train an early-exit model based on the full-model backbone.</summary>

 We will assume that `load_backbone` points to the weights obtained in the previous step.
```bash
python main.py \
    --n_steps 100000 \
    --batch_size 128 \
    --log_path "${log_path}" \
    --log_every_n_steps 2500 \
    --save_every_n_steps 2500 \
    --save_new_every_n_steps 10000 \
    --seed 1 \
    --load_backbone "${load_backbone}" \
    --model "deediff_uvit" \
    --use_amp \
    --normalize_timesteps \
    --parametrization "predict_noise" \
    --freeze_backbone \
    --dataset "celeba" \
    --classifier_type "mlp_probe_per_layer" \
    --sample_height 64 \
    --sample_width 64 \
    --img_size 64 \
    --patch_size 4 \
    --config_path "configs/deediff_celeba.yaml"
```

</details>

### DuoDiff training
<details>
<summary>Our proposed model, DuoDiff, involves training a shallow model that will be used alongside the full-model during inference. </summary>

```bash
python main.py \
    --model "uvit" \
    --n_steps 500000 \
    --batch_size 128 \
    --log_path ${log_path} \
    --log_every_n_steps 2500 \
    --use_amp \
    --save_every_n_steps 25000 \
    --save_new_every_n_steps 100000 \
    --sample_height 64 \
    --sample_width 64 \
    --seed 1 \
    --normalize_timesteps \
    --config_path "configs/uvit_celeba_3.yaml" \
    --dataset "celeba" \
    --parametrization "predict_noise" \
```

</details>

## Running inference
In this section, we will see how to generate images using the models trained on the previous section.
### Early-exit sampling
Here, `checkpoint_path` points to the trained early-exit model (not the full model).
```bash
python eesampler.py \
    --seed ${seed} \
    --checkpoint_path "${checkpoint_path}" \
    --batch_size 128 \
    --output_folder "${output_folder}" \
    --threshold 0.08 \
    --config_path "configs/deediff_celeba.yaml"
```
### DuoDiff inference
Notice that we are using two different models, the full one, and the shallow one.
```bash
python sampler.py \
    --seed ${seed} \
    --checkpoint_path "${shallow_model_path}" \
    --checkpoint_path_late "${full_model_path}" \
    --batch_size 128 \
    --parametrization "predict_noise" \
    --output_folder "${output_folder}" \
    --config_path "configs/uvit_celeba_3.yaml" \
    --config_path_late "configs/uvit_celeba.yaml" \
    --t_switch 300
```
## Computing FID scores
We can easily compute the FID scores running the following script.
```bash
python fid.py \
    --dataset "celeba"
    --samples_path "${samples_path}"
```


## Dev instructions
The first time:
1. Run `pre-commit install`
2. If you use VSCode, it might be helpful to add the following to `settings.json`:
    ```json
    "[python]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.codeActionsOnSave": {
            "source.fixAll": "explicit",
            "source.organizeImports": "explicit"
        }
    },
    ```

After that, be sure that all the tests are passing before a commit. Otherwise, GitHub Actions will complain ;) You can check by running
```bash
python -m pytest tests
```
