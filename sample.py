import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from models.early_exit import EarlyExitUViT
from models.uvit import UViT
from utils.train_utils import (
    get_noise_scheduler,
    seed_everything,
)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

betas = torch.linspace(1e-4, 0.02, 1000).to(device)
alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, dim=0)
alphas_bar_previous = torch.cat([torch.tensor([1.0], device=device), alphas_bar[:-1]])
betas_tilde = betas * (1 - alphas_bar_previous) / (1 - alphas_bar)


def get_args():
    parser = argparse.ArgumentParser(description="Sampling parameters")
    # Default parameters from https://github.com/baofff/U-ViT/blob/main/configs/cifar10_uvit_small.py

    # Training
    parser.add_argument("--seed", type=int, default=1, help="Seed")
    parser.add_argument(
        "--num_timesteps", type=int, default=1000, help="Number of timesteps"
    )
    parser.add_argument("--use_amp", action="store_true", default=False, help="Use AMP")
    parser.add_argument("--amp_dtype", type=str, default="bf16", help="AMP data type")
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Max gradient norm"
    )

    # Logging
    parser.add_argument(
        "--log_path",
        type=str,
        default="samples/threshold -inf",
        help="Directory for samples",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Number of images to sample for logging",
    )
    parser.add_argument(
        "--sample_height",
        type=int,
        default=32,
        help="Height of the images sampled for logging",
    )
    parser.add_argument(
        "--sample_width",
        type=int,
        default=32,
        help="Width of the images sampled for logging",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="Seed for sampling images for logging",
    )

    # Checkpointing
    parser.add_argument(
        "--load_checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint path for loading the training state",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="deediff_uvit",
        choices=["uvit", "deediff_uvit"],
        help="Model name",
    )

    parser.add_argument(
        "--exit_threshold",
        type=float,
        default=-torch.inf,
        help="Early exit threshold",
    )

    # Benchmarking
    parser.add_argument(
        "--benchmarking",
        action="store_true",
        help="True if we want to benchmark the sampler",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    writer = SummaryWriter(args.log_path)
    seed_everything(args.seed)

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

    model = EarlyExitUViT(uvit=uvit, exit_threshold=args.exit_threshold)

    if args.load_checkpoint_path:
        state_dict = torch.load(args.load_checkpoint_path, device)
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        print("The loaded checkpoint path is wrong or not provided!")
        exit(1)

    model = model.eval()
    model = model.to(device)

    noise_scheduler = get_noise_scheduler(args)

    samples, logging_dict = noise_scheduler.sample(
        model=model,
        num_steps=args.num_timesteps,
        data_shape=(3, args.sample_height, args.sample_width),
        num_samples=args.n_samples,
        seed=args.sample_seed,
        model_type=args.model,
        benchmarking=args.benchmarking,
    )

    if args.benchmarking:
        for time, gflops in sorted(logging_dict["benchmarking"], key=lambda x: x[0]):
            writer.add_scalar("benchmarking", gflops, time)

    classifier_outputs = logging_dict["classifier_outputs"]
    for timestep, outputs_t in enumerate(classifier_outputs):
        exit_layer = 13 if len(outputs_t) == 13 and any([outputs_t_l[-1] > args.exit_threshold for outputs_t_l in outputs_t]) else len(outputs_t)
        writer.add_scalar("early_exit_layers", exit_layer, timestep)
        for layer in range(exit_layer):
            writer.add_scalar(f"UEM Classifier output at layer {layer} wrt time", outputs_t[layer].mean(), timestep)
        if timestep % 50 == 0:
            for layer in range(exit_layer):
                writer.add_scalar(f"UEM Classifier output at timestep {timestep} wrt layer", outputs_t[layer].mean(), layer + 1)

    for i, sample in enumerate(samples):
        writer.add_image(f"Sample {i + 1}", sample)
