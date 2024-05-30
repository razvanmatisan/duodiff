import torch
from ptflops import get_model_complexity_info

from models.early_exit import AttentionProbe, EarlyExitUViT, MLPProbe, OutputHead
from models.uvit import UViT


def get_gflops(model, x, t=None):
    batch_size = x.shape[0]

    if t is not None:
        macs, params = get_model_complexity_info(
            model,
            (x, t),
            input_constructor=lambda inputs: {"x": inputs[0], "timesteps": inputs[1]},
            as_strings=False,
            print_per_layer_stat=False,  # TODO: Do we want a detailed analysis per layer?
            verbose=False,
        )
    else:
        macs, params = get_model_complexity_info(
            model,
            (x.shape[1], x.shape[2]),
            as_strings=False,
            print_per_layer_stat=False,  # TODO: Do we want a detailed analysis per layer?
            verbose=False,
        )

    # From mac to GMac
    gmacs = macs / 10**9
    if t is not None:
        gmacs /= batch_size

    # GFlops ~= 2 * GMacs
    gflops = gmacs * 2

    return gflops


if __name__ == "__main__":
    # EarlyExitUViT
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
    print(f"GFlops for EarlyExitUViT: {gflops}")

    # Classification heads
    batch_size = 16
    num_patches = 257
    embedding_dim = 512

    x = torch.zeros((batch_size, num_patches, embedding_dim))
    attn_probe = AttentionProbe(embed_dim=embedding_dim)
    mlp_probe = MLPProbe(embed_dim=embedding_dim)

    gflops = get_gflops(attn_probe, x)
    print(f"GFlops for AttentionProbe: {gflops}")

    gflops = get_gflops(mlp_probe, x)
    print(f"GFlops for MLPProbe: {gflops}")

    # OutputHead
    output_head = OutputHead(embed_dim=embedding_dim, patch_dim=12, in_chans=3)
    gflops = get_gflops(output_head, x)
    print(f"GFlops for OutputHead: {gflops}")
