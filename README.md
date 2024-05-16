# Early Stopping Diffusion

[![Status](https://github.com/razvanmatisan/early-stopping-diffusion/actions/workflows/python.yml/badge.svg)](https://github.com/razvanmatisan/early-stopping-diffusion/actions/workflows/python.yml)

## Dev instructions
The first time:
1. Configure a virtual environment
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

    On Windows:
    ```bash
    python -m venv venv
    Set-ExecutionPolicy Unrestricted -Scope Process
    venv\Scripts\Activate
    pip install -r requirements.txt
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
python -m pytest tests
```

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
