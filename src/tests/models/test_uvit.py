from pathlib import Path

import pytest
import torch
from models.uvit import UViT

celeba_config = dict(
    img_size=64,
    patch_size=4,
    embed_dim=512,
    depth=12,
    num_heads=8,
    mlp_ratio=4,
    qkv_bias=False,
    mlp_time_embed=False,
    num_classes=-1,
)

cifar10_config = dict(
    img_size=32,
    patch_size=2,
    embed_dim=512,
    depth=12,
    num_heads=8,
    mlp_ratio=4,
    qkv_bias=False,
    mlp_time_embed=False,
    num_classes=-1,
)


@pytest.mark.parametrize(
    "checkpoint_path, config",
    [
        ("checkpoints/u-vit/celeba_uvit_small.pth", celeba_config),
        ("checkpoints/u-vit/cifar10_uvit_small.pth", cifar10_config),
    ],
)
def test_checkpoint_loading(checkpoint_path, config):
    if not Path(checkpoint_path).exists():
        # We are probably on GitHub
        return

    model = UViT(**config)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)


def test_shape():
    model = UViT(**cifar10_config)

    batch_size = 8
    num_channels = 3
    height = 32
    width = 32
    x = torch.zeros((batch_size, num_channels, height, width))
    t = torch.ones(batch_size)

    y = model(x, t)
    assert y.shape == x.shape


def test_backward():
    model = UViT(**cifar10_config)

    batch_size = 8
    num_channels = 3
    height = 32
    width = 32
    x = torch.zeros((batch_size, num_channels, height, width))
    t = torch.ones(batch_size)

    y = model(x, t)
    fake_loss = torch.sum(y)
    fake_loss.backward()
