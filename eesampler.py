import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from tqdm import tqdm

from models.early_exit import EarlyExitUViT
from models.utils.autoencoder import get_autoencoder
from models.uvit import UViT
from utils.config_utils import load_config
from utils.train_utils import seed_everything


def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass

    return "cpu"


device = get_device()
print(f"Using device {device}")

betas = torch.linspace(1e-4, 0.02, 1000).to(device)
alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, dim=0)
alphas_bar_previous = torch.cat([torch.tensor([1.0], device=device), alphas_bar[:-1]])
betas_tilde = betas * (1 - alphas_bar_previous) / (1 - alphas_bar)


def get_samples(
    model,
    batch_size: int,
    seed: int,
    num_channels: int,
    sample_height: int,
    sample_width: int,
    threshold: float,
    depth: int,
    y: int = None,
    autoencoder=None,
):
    seed_everything(seed)
    x = torch.randn(batch_size, num_channels, sample_height, sample_width).to(device)
    error_prediction_by_timestep = torch.zeros(1000, depth)
    indices_by_timestep = torch.zeros(1000, batch_size)

    for t in tqdm(range(999, -1, -1)):
        time_tensor = t * torch.ones(batch_size, device=device)
        with torch.no_grad():
            model_output, classifier_outputs, outputs = model(x, time_tensor, y)

        outputs = torch.stack(outputs + [model_output])
        classifier_outputs = torch.stack(
            classifier_outputs + [torch.zeros_like(classifier_outputs[0])]
        )
        # Simulate early exit with a global threshold
        indices = torch.argmax((classifier_outputs <= threshold).int(), dim=0)
        model_output = outputs[indices, torch.arange(batch_size)]

        # Log for visualization
        error_prediction_by_timestep[t] = classifier_outputs.mean(axis=1)[:depth]
        indices_by_timestep[t, :] = indices

        alpha_t = alphas[t]
        alpha_bar_t = alphas_bar[t]
        sigma_t = torch.sqrt(betas_tilde[t])

        z = torch.randn_like(x) if t > 0 else 0
        x = (
            torch.sqrt(1 / alpha_t)
            * (x - (1 - alpha_t) / (torch.sqrt(1 - alpha_bar_t)) * model_output)
        ) + sigma_t * z

    if autoencoder:
        x = autoencoder.decode(x)

    samples = (x + 1) / 2
    samples = rearrange(samples, "b c h w -> b h w c")
    return samples.cpu().numpy(), error_prediction_by_timestep, indices_by_timestep


def dump_samples(samples, output_folder: Path):
    # plt.hist(samples.flatten())
    # plt.savefig(output_folder / "histogram.png")
    # plt.clf()

    for sample_id, sample in enumerate(samples):
        sample = np.clip(sample, 0, 1)
        plt.imsave(output_folder / f"{sample_id}.png", sample)


def dump_statistics(
    elapsed_time, error_prediction_by_timestep, indices_by_timestep, output_folder: Path
):
    with open(output_folder / "statistics.txt", "w") as f:
        f.write(f"Elapsed time: {elapsed_time} s\n")

    torch.save(
        error_prediction_by_timestep, output_folder / "error_prediction_by_timestep.pt"
    )
    torch.save(indices_by_timestep, output_folder / "indices_by_timestep.pt")


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to yaml config file",
    )
    parser.add_argument(
        "--class_id",
        type=int,
        default=None,
        help="Number up to 1000 that corresponds to a class",
    )

    return parser.parse_args()


def main():
    args = get_args()

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config_path)
    base_model = UViT(
        img_size=config["model_params"]["img_size"],
        patch_size=config["model_params"]["patch_size"],
        in_chans=config["model_params"]["in_chans"],
        embed_dim=config["model_params"]["embed_dim"],
        depth=config["model_params"]["depth"],
        num_heads=config["model_params"]["num_heads"],
        mlp_ratio=config["model_params"]["mlp_ratio"],
        qkv_bias=config["model_params"]["qkv_bias"],
        mlp_time_embed=config["model_params"]["mlp_time_embed"],
        num_classes=config["model_params"]["num_classes"],
        normalize_timesteps=config["model_params"]["normalize_timesteps"],
    )
    model = EarlyExitUViT(base_model, config["model_params"]["classifier_type"])

    num_channels = config["model_params"]["in_chans"]
    sample_height = config["model_params"]["img_size"]
    sample_width = config["model_params"]["img_size"]
    depth = config["model_params"]["depth"]

    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model = model.eval().to(device)

    # y = (
    #     torch.ones(args.batch_size, dtype=torch.int).to(device) * args.class_id
    #     if args.class_id is not None
    #     else None
    # )

    seed_everything(args.seed)

    y = (
        torch.randint(1, 1001, (args.batch_size,)).to(device) 
        if args.class_id is not None
        else None
    )

    if "autoencoder" in config:
        autoencoder = get_autoencoder(
            config["autoencoder"]["autoencoder_checkpoint_path"]
        ).to(device)
    else:
        autoencoder = None

    tic = time.time()
    samples, error_prediction_by_timestep, indices_by_timestep = get_samples(
        model=model,
        batch_size=args.batch_size,
        seed=args.seed,
        num_channels=num_channels,
        sample_height=sample_height,
        sample_width=sample_width,
        threshold=args.threshold,
        depth=depth,
        y=y,
        autoencoder=autoencoder,
    )
    tac = time.time()
    dump_statistics(
        tac - tic, error_prediction_by_timestep, indices_by_timestep, output_folder
    )

    dump_samples(samples, output_folder)


if __name__ == "__main__":
    main()
