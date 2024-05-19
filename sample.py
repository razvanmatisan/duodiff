import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from models.early_exit import EarlyExitUViT
from models.uvit import UViT
from utils.train_utils import (
    get_noise_scheduler,
    seed_everything,
)


def get_device():
    if torch.cuda.is_available():
        return "cuda:0"

    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass

    return "cpu"


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

    parser.add_argument(
        "--keep_initial_timesteps",
        action="store_true",
        help="Using timesteps in [0, T] if True. Otherwise, normalize the timesteps in [0, 1]",
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

    # Train mode
    parser.add_argument(
        "--train_mode",
        action="store_true",
        help="True if we run the model in train mode",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    writer = SummaryWriter(args.log_path)
    seed_everything(args.seed)

    device = get_device()
    print(f"Device: {device}")

    betas = torch.linspace(1e-4, 0.02, 1000).to(device)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    alphas_bar_previous = torch.cat(
        [torch.tensor([1.0], device=device), alphas_bar[:-1]]
    )
    betas_tilde = betas * (1 - alphas_bar_previous) / (1 - alphas_bar)

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

    mode = "train_mode" if args.train_mode else "inference_mode"
    print(f"Sampling in {mode}...")
    ### Sampling
    samples, logging_dict = noise_scheduler.sample(
        model=model,
        num_steps=args.num_timesteps,
        data_shape=(3, args.sample_height, args.sample_width),
        num_samples=args.n_samples,
        seed=args.sample_seed,
        model_type=args.model,
        benchmarking=args.benchmarking,
        train_mode=args.train_mode,
        keep_initial_timesteps=args.keep_initial_timesteps,
    )

    ### Logging
    ## Denoised images
    if model.training:
        ### Denoised images (using each of the noise outputed by each of the 13 layers)
        print("Logging denoised images...")
        all_denoised_images = logging_dict["denoised_images"]

        for t, denoised_images_t in enumerate(all_denoised_images):
            if (t + 1) % 50 == 0:
                for i, batch in enumerate(denoised_images_t):
                    for k, img in enumerate(batch):
                        writer.add_image(
                            f"Sample {k}: Denoised images at timestep {t}",
                            img,
                            global_step=i,
                        )

    ## Benchmarking (Theoretical GFlops)
    if args.benchmarking:
        print("Logging benchmark...")
        for time, gflops in sorted(logging_dict["benchmarking"], key=lambda x: x[0]):
            writer.add_scalar("benchmarking", gflops, time)

    ## UEM classifier outputs per timestep wrt layer
    classifier_outputs = logging_dict["classifier_outputs"]
    print("Logging classifier outputs...")
    for timestep, outputs_t in enumerate(classifier_outputs):
        # if len(outputs_t) == 13:
        #     try:
        #         if torch.any(outputs_t[-1] > args.exit_threshold):
        #             exit_layer = 13
        #         else:
        #             exit_layer = len(outputs_t)
        #     except:
        #         pass
        # else:
        #     exit_layer = len(outputs_t)
        exit_layer = (
            13
            if len(outputs_t) == 13 and torch.any(outputs_t[-1] > args.exit_threshold)
            else len(outputs_t)
        )
        writer.add_scalar("early_exit_layers", exit_layer, timestep)
        for layer in range(exit_layer):
            writer.add_scalar(
                f"UEM Classifier output at layer {layer} wrt time",
                outputs_t[layer].mean(),
                timestep,
            )
        if timestep % 50 == 0:
            for layer in range(exit_layer):
                writer.add_scalar(
                    f"UEM Classifier output at timestep {timestep} wrt layer",
                    outputs_t[layer].mean(),
                    layer + 1,
                )

    ## Denoising over time
    samples_over_time = logging_dict["samples_over_time"]
    print("Logging samples over time...")
    for t, samples in enumerate(samples_over_time):
        for i, sample in enumerate(samples):
            writer.add_image(
                f"Sample {i} over time using threshold {args.exit_threshold}",
                sample,
                global_step=t,
            )
