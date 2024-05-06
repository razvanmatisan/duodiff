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

**Tin's code**: https://github.com/stases/EarlyDiffusion

**Paper**: https://arxiv.org/pdf/2309.17074

### Diffusion Models
