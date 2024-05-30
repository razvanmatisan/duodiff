# Early Stopping Diffusion

[![Status](https://github.com/razvanmatisan/early-stopping-diffusion/actions/workflows/python.yml/badge.svg)](https://github.com/razvanmatisan/early-stopping-diffusion/actions/workflows/python.yml)

## Dev instructions
The first time:
1. Configure a virtual environment
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r src/requirements.txt
    ```

    On Windows:
    ```bash
    python -m venv venv
    Set-ExecutionPolicy Unrestricted -Scope Process
    venv\Scripts\Activate
    pip install -r src\requirements.txt
    ```

2. Run `pre-commit install`
3. If you use VSCode, it might be helpful to add the following to `settings.json`:
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
python -m pytest src/tests
```

## Repository structure
- `demos/`: Demos for visualising early stopping diffusion.
- `src/`: Code.
    - `CMMD_evaluation/`: Code for calculating the CMMD score of generated samples.
    - `Generated_samples/`: Directories with generated samples.
    - `datasets/`: Dataset-specific dataloaders.
    - `models/`: Model definitions.
    - `scripts/`: Scripts for training, generation, evaluation and benchmarking.
    - `snellius/`: Files for running experiments on Snellius.
    - `test/`: Unit tests.
    - `utils/`:
        - `field_utils.py` Getters for time and space embeddings.
        - `train_utils.py` Getters for models, optimizers, dataloaders, etc.
    - `FID_evaluation.py`: Code for calculating the FID score of generated samples.
    - `benchmark.py`: Code for benchmarking models.
    - `ddpm_core.py`: Code of the DDPM sampler.
    - `requirements.txt`: File with requirements for setting up the virtual environment.
    - `sample.py`: Code for sampling images.
    - `train.py`: Code for training models.
- `blogpost.md`: Blogpost about the project.

## Running experiments

### Training
Training models is done using `train.py` script.
Full specification of the script can be found with `python train.py --help` command. Below are sample commands for running training with only the essential arguments.

#### UViT backbone

Command for training the UViT backbone.
```shell
python train.py \
    --model uvit \
    --n_steps ${number_of_training_steps} \
    --batch_size ${batch_size} \
```
- `number_of_training_steps`: Number of iterations over the dataloader.
- `batch_size`: Batch size.

#### Early-exit models
Below are scripts for training early-exit diffusion model *DeeDiff*.
Each of the commands can 

Command for training DeeDiff:
```shell
python train.py \
    --model deediff_uvit \
    --n_steps ${number_of_training_steps} \
    --batch_size ${batch_size} \
    --classifier_type ${classifier_type} \
    [--load_backbone ${checkpoint_path} \]
    [--freeze_backbone \]
    [--use_unweighted_loss \]
    [--normalize_timesteps \]
```

- `number_of_training_steps`: Number of iterations over the dataloader.
- `batch_size`: Batch size.
- `classifier_type`: Type of the classifier for determining whether to early-exit. Can be one of:
    - `attention_probe`: Attention probe.
    - `mlp_probe_per_timestep`: Separate MLP probe at each timestep, shared between layers.
    - `mlp_probe_per_layer`: Separate MLP probe for each UViT layer, shared between timesteps.
    - `mlp_probe_per_layer_per_timestep`: Separate MLP probe for each UViT layer, at each time step (nothing is shared).
- `--freeze_backbone`: If present, then freeze the UViT backbone (train only the classifiers probes).
- `--use_unweighted_loss`: If present, add the unweighted loss to the remaining losses.
- `--normalize_timesteps`: If present, normalize timesteps from $[1, T]$ to $[0, 1]$.
- (optional) `checkpoint_path`: Path to the checkpoint with UViT weights. If not specified, then train DeeDiff from scratch.

### Generation
TODO
```shell
python src/sample.py \
    --load_checkpoint_path ${checkpoint_path} \
    --exit_threshold ${exit_threshold} \
    --n_samples ${n_samples} \ 
    --sample_seed ${seed}
```

### Evaluation

#### CMMD
Command for calculating the CMMD score: 
```shell
python src/CMMD_evaluation/main.py \
    --load_from_folder \
    --cmmd_batch_size ${cmmd_batch_size} \
    --cmmd_max_count ${cmmd_max_count}
```
- `cmmd_batch_size`: Batch size for embedding generation.
- `cmmd_max_count`: Maximum number of images to read from each directory.

Command for generating samples and calculating their CMMD score:
```shell
python src/CMMD_evaluation/main.py \
    --model ${model} \
    --load_checkpoint_path ${load_checkpoint_path} \
    --exit_threshold ${exit_threshold} \
    --start_seed ${start_seed} \
    --end_seed ${end_seed} \
    --cmmd_batch_size ${cmmd_batch_size} \
    --cmmd_max_count ${cmmd_max_count}
```
Optional, if the samples need to be generated:
- `model`: Model used to generate samples. 
- `load_checkpoint_path`: Path to a checkpoint with model weights.
- `exit_threshold`: Exit threshold if DeeDiff is used.
- `start_seed`: Start seed.
- `end_seed`: End seed.

We generate a batch of samples for each seed from range [`start_seed`, `end_seed`].
Before generating each batch we set the seed to make sure we can compare the samples generated by different models or sets of hyperparameters.

TODO: 
- Add an argument with the path of generated samples.
- Separate evaluation from generation.


#### FID
Command for calculating the FID score:
```shell
python src/FID_evaluation.py \
    --load_from_folder
```

Command for generating samples and calculating their CMMD score.
```shell
python src/FID_evaluation.py \
    --model $model \
    --load_checkpoint_path ${load_checkpoint_path} \
    --exit_threshold ${exit_threshold} \
    --start_seed ${start_seed} \
    --end_seed ${end_seed}
```

For description of the arguments check out section CMMD above.

### Benchmarking
TODO


## Resources
### DeeDiff

Tin's code: https://github.com/stases/EarlyDiffusion

Paper: https://arxiv.org/pdf/2309.17074

### Math Diffusion Models

Lecture Notes in Probabilistic Diffusion Models: https://arxiv.org/html/2312.10393v1

Lil'Log blogpost: https://lilianweng.github.io/posts/2021-07-11-diffusion-models

### Backbones Diffusion Models

U-Net: https://arxiv.org/pdf/1505.04597

U-ViT: https://arxiv.org/pdf/2209.12152

Diffusion Transformer (DiT): https://arxiv.org/pdf/2212.09748
