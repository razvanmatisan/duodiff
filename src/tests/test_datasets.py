from functools import wraps
from pathlib import Path

import pytest
import torch
from datasets import get_celeba_dataloader, get_cifar10_dataloader


def ignore_if_data_not_downloaded(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not Path("data/").exists():
            return

        return f(*args, **kwargs)

    return wrapper


@ignore_if_data_not_downloaded
@pytest.mark.parametrize("batch_size", [16])
def test_cifar10(batch_size):
    dataloader = get_cifar10_dataloader(batch_size=batch_size, seed=0)

    x, _ = next(iter(dataloader))
    assert x.shape == torch.Size([batch_size, 3, 32, 32])


@ignore_if_data_not_downloaded
@pytest.mark.parametrize("batch_size", [4])
def test_celeba(batch_size):
    dataloader = get_celeba_dataloader(batch_size=batch_size, seed=0)

    x, _ = next(iter(dataloader))
    assert x.shape == torch.Size([batch_size, 3, 64, 64])
