import torch
from ptflops import get_model_complexity_info

from models.early_exit import EarlyExitUViT
from models.uvit import UViT


def get_gflops(model, x, t):
    batch_size = x.shape[0]

    macs, params = get_model_complexity_info(
        model,
        (x, t),
        input_constructor=lambda inputs: {"x": inputs[0], "timesteps": inputs[1]},
        as_strings=False,
        print_per_layer_stat=False,  # TODO: Do we want a detailed analysis per layer?
        verbose=False,
    )

    # From mac to GMac
    gmacs = (macs / 10**9) / batch_size

    # GFlops ~= 2 * GMacs
    gflops = gmacs * 2

    return gflops


if __name__ == "__main__":
    batch_size = 16
    num_channels = 3
    height = 32
    width = 32

    uvit = UViT(
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

    model = EarlyExitUViT(uvit=uvit, exit_threshold=-torch.inf)
    model.eval()

    x = torch.zeros((batch_size, num_channels, height, width))
    t = torch.ones(batch_size)

    gflops = get_gflops(model, x, t)
    print(gflops)
