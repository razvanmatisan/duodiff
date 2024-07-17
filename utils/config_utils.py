from pathlib import Path
import yaml


def load_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} does not exist")

    with path.open("r") as f:
        config = yaml.safe_load(f)

    return config
