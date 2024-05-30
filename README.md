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
cd src
python -m pytest tests
```

## Repository structure
- `demos/`: Demos for visualising early stopping diffusion.
- `src/`: Code.
    - `CMMD_evaluation/`: Code for calculating the CMMD score of generated samples.
    - `samples/`: Directories with generated samples.
    - `datasets/`: Dataset-specific dataloaders.
    - `models/`: Model definitions.
    - `scripts/`: Scripts for training, generation, evaluation and benchmarking.
    - `snellius/`: Files for running experiments on Snellius.
    - `tests/`: Unit tests.
    - `utils/`:
        - `field_utils.py` Getters for time and space embeddings.
        - `train_utils.py` Getters for models, optimizers, dataloaders, etc.
    - `FID_evaluation.py`: Code for calculating the FID score of generated samples.
    - `get_flops.py`: Code for computing theoretical GFLOPs.
    - `compute_gflops_and_layer_ratio.py`: Code for computing the average layer ratio and theoretical GFLOP.
    - `ddpm_core.py`: Code of the DDPM sampler.
    - `requirements.txt`: File with requirements for setting up the virtual environment.
    - `train.py`: Code for training models.
- `blogpost.md`: Blogpost about the project.

## Running experiments
All of the experiments should be run inside `src` directory.
```
cd src
```

### Training
Training the models is done using `train.py` script.
Full specification of the script can be found with `python train.py --help` command. Below are sample commands for running training with only the essential arguments.

#### UViT backbone

Command for training the UViT backbone.
```shell
bash scripts/train_uvit.sh
```
or
```shell
python train.py \
    --model uvit \
    --n_steps 100000 \
    --batch_size 128 \
```

#### Early-exit models
Command for training the a DeeDiff model:
```shell
bash scripts/train_deediff.sh
```
or
```
python train.py \
    --model deediff_uvit \
    --n_steps 100000 \
    --batch_size 128 \
    --classifier_type attention_probe \
    --normalize_timesteps
```

Below is a specification of how to run training with other settings.
```shell
python train.py \
    --model deediff_uvit \
    --n_steps ${number_of_training_steps} \
    --batch_size ${batch_size} \
    --classifier_type ${classifier_type} \
    --normalize_timesteps \
    [--load_backbone ${checkpoint_path} \]
    [--freeze_backbone \]
    [--use_unweighted_loss \]
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
- (optional) `checkpoint_path`: Path to the checkpoint with UViT weights. If not specified, then train DeeDiff from scratch.

### Evaluation

#### CMMD
Command for generating samples and calculating the CMMD score: 
```shell
bash scripts/cmmd_evaluation.sh
```
or
```shell
python CMMD_evaluation/main.py \
    --checkpoint_entry_name frozenBackbone_attention_3losses \
    --exit_threshold 0.05 \
    --cmmd_batch_size 32 \
    --cmmd_max_count 10
```

- `cmmd_batch_size`: Batch size for embedding generation.
- `cmmd_max_count`: Maximum number of images to read from each directory.

#### FID
Command for generating samples and calculating the FID score: 
```shell
bash scripts/fid_evaluation.sh
```
or
```shell
python FID_evaluation.py \
    --checkpoint_entry_name frozenBackbone_attention_3losses \
    --exit_threshold 0.05
```

### Benchmarking
For computing the theoretical GFLOPs for the MLP probe, attention probe and output head, you can run the following script 

```shell
python get_gflops.py
```

Example script for computing the average layer ratio and theoretical GFLOPs:
```shell
python compute_gflops_and_layer_ratio.py \
    --indices_by_timestep_directory benchmarking/output/attention_frozen/indices_by_timestep
```

For computing the average layer ratio and theoretical GFLOPs for each method, one can run the following script:
```shell
python compute_gflops_and_layer_ratio.py
    --indices_by_timestep_directory ${indices_by_timestep_directory} \
```
The parameter ``indices_by_timestep_directory`` is the relative path to the folder which contains files in ``.pt`` format regarding the layers which early exit took place per timestep. These directories can be found in ``src/benchmarking/output``. Currently, we uploaded only the ``.pt`` files for the model that uses an attention probe and a frozen backbone during training. The reason why we did not include them for all methods is because the files are pretty large. If one would need the files for the other methods, please contact us.


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
