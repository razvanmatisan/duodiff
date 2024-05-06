import pytest
import torch

from datasets import get_celeba_dataloader, get_cifar10_dataloader

@pytest.mark.parametrize("batch_size", [16])
def test_cifar10(batch_size):
    dataloader = get_cifar10_dataloader(batch_size=batch_size)

    x, _ = next(iter(dataloader))
    assert x.shape == torch.Size([batch_size, 3, 32, 32])

@pytest.mark.parametrize("batch_size", [4])
def test_celebA(batch_size):
    dataloader = get_celeba_dataloader(batch_size=batch_size)

    x, _ = next(iter(dataloader))
    assert x.shape == torch.Size([batch_size, 3, 64, 64])

