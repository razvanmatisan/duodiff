import math
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import List

from models.utils.autoencoder import get_autoencoder
from models.uvit import UViT
from utils.config_utils import load_config
from utils.train_utils import seed_everything

checkpoint_path_by_parametrization = {
    "predict_noise": "../logs/6218182/cifar10_uvit.pth",
    "predict_original": "../logs/6524899/cifar10_uvit.pth",
    "predict_previous": "../logs/6560733/cifar10_uvit.pth",
}


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


def predict_noise_postprocessing(model_output, x, t):
    alpha_t = alphas[t]
    alpha_bar_t = alphas_bar[t]
    sigma_t = torch.sqrt(betas_tilde[t])

    z = torch.randn_like(x) if t > 0 else 0
    return (
        torch.sqrt(1 / alpha_t)
        * (x - (1 - alpha_t) / (torch.sqrt(1 - alpha_bar_t)) * model_output)
    ) + sigma_t * z


def predict_original_postprocessing(model_output, x, t):
    alpha_t = alphas[t]
    alpha_bar_t = alphas_bar[t]
    alphas_bar_prev = alphas_bar_previous[t]
    sigma_t = torch.sqrt(betas_tilde[t])
    beta_t = betas[t]

    sigma_t = torch.sqrt(betas_tilde[t])
    z = torch.randn_like(x) if t > 0 else 0

    return (
        torch.sqrt(alphas_bar_prev) * beta_t * model_output / (1 - alpha_bar_t)
        + torch.sqrt(alpha_t) * (1 - alphas_bar_prev) * x / (1 - alpha_bar_t)
    ) + sigma_t * z


def predict_previous_postprocessing(model_output, x, t):
    sigma_t = torch.sqrt(betas_tilde[t])
    z = torch.randn_like(x) if t > 0 else 0

    return model_output + sigma_t * z


def get_samples(
    model,
    batch_size: int,
    postprocessing: callable,
    seed: int,
    num_channels: int,
    sample_height: int,
    sample_width: int,
    use_ddim: bool,
    ddim_steps: int,
    ddim_eta: float,
    timesteps_save: List[int],
    y: int = None,
    autoencoder=None,
    late_model=None,
    t_switch=np.inf,
):
    seed_everything(seed)
    x = torch.randn(batch_size, num_channels, sample_height, sample_width).to(device)
    intermediate_samples = []

    if use_ddim:
        timesteps = np.linspace(0, 999, ddim_steps).astype(int)[::-1]
        for t, s in zip(tqdm(timesteps[:-1]), timesteps[1:]):
            assert s < t

            time_tensor = t * torch.ones(batch_size, device=device)
            with torch.no_grad():
                model_output = model(x, time_tensor, y)

            sigma_t_squared = betas_tilde[t] * ddim_eta

            mean = torch.sqrt(alphas_bar[s] / alphas_bar[t]) * (
                x - torch.sqrt(1 - alphas_bar[t]) * model_output
            )
            mean += torch.sqrt(1 - alphas_bar[s] - sigma_t_squared) * model_output

            z = torch.randn_like(x) if s > 0 else 0
            x = mean + sigma_t_squared * z

            if t < 1000 - t_switch:
                model = late_model

            if 1000 - t in timesteps_save:
                intermediate_samples.append(x)

    else:
        for t in tqdm(range(999, -1, -1)):
            time_tensor = t * torch.ones(batch_size, device=device)
            with torch.no_grad():
                model_output = model(x, time_tensor, y)
            x = postprocessing(model_output, x, t)

            if t == 1000 - t_switch:
                model = late_model

            if 1000 - t in timesteps_save:
                intermediate_samples.append(x)


    if autoencoder:
        print("Decode the images...")
        x = autoencoder.decode(x)

    samples = (x + 1) / 2
    samples = rearrange(samples, "b c h w -> b h w c")

    for i, x in enumerate(intermediate_samples):
        if autoencoder:
            x = autoencoder.decode(x)
        x = (x + 1) / 2
        x = rearrange(x, "b c h w -> b h w c")
        intermediate_samples[i] = x.cpu().numpy()

    return samples.cpu().numpy(), intermediate_samples


def dump_samples(samples, output_folder: Path, timestep=1000):
    # plt.hist(samples.flatten())
    # plt.savefig(output_folder / "histogram.png")
    # plt.clf()

    num_samples = len(samples)
    grid_size = math.ceil(math.sqrt(num_samples))
    sample_height, sample_width = samples[0].shape[:2]

    # Create an empty array for the grid image
    grid_img = np.zeros((grid_size * sample_height, grid_size * sample_width, 3))

    for sample_id, sample in enumerate(samples):
        sample = np.clip(sample, 0, 1)
        filename = f"{sample_id}_{timestep}.png" if timestep !=1000 else f"{sample_id}.png"
        plt.imsave(output_folder / filename, sample)

        row, col = divmod(sample_id, grid_size)
        grid_img[
            row * sample_height : (row + 1) * sample_height,
            col * sample_width : (col + 1) * sample_width,
            :,
        ] = sample

    plt.imsave(output_folder / "grid_image.png", grid_img)


def dump_statistics(elapsed_time, output_folder: Path):
    with open(output_folder / "statistics.txt", "w") as f:
        f.write(f"Elapsed time: {elapsed_time} s\n")


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint of the model",
    )
    parser.add_argument(
        "--checkpoint_path_late",
        type=str,
        default=None,
        help="Path to checkpoint of the model to be used in the latest steps",
    )
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument(
        "--parametrization",
        type=str,
        choices=["predict_noise", "predict_original", "predict_previous"],
        required=True,
    )
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to yaml config file",
    )
    parser.add_argument(
        "--config_path_late",
        type=str,
        default=None,
        help="Path to yaml config file of the model to be used in the latest steps",
    )
    parser.add_argument(
        "--t_switch",
        type=int,
        default=np.inf,
        help="Sampling timestep where the model should be replaced by the late model",
    )
    parser.add_argument(
        "--class_id",
        type=int,
        default=None,
        help="Number up to 1000 that corresponds to a class",
    )
    parser.add_argument("--use_ddim", action="store_true")
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--timesteps_save",
        type=int,
        nargs="+",
        default=[]
    )

    return parser.parse_args()


def main():
    args = get_args()

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if args.parametrization == "predict_noise":
        postprocessing = predict_noise_postprocessing
    elif args.parametrization == "predict_original":
        postprocessing = predict_original_postprocessing
    elif args.parametrization == "predict_previous":
        postprocessing = predict_previous_postprocessing
    else:
        raise ValueError(f"Invalid parametrization {args.parametrization}")

    config = load_config(args.config_path)
    model = UViT(
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

    num_channels = config["model_params"]["in_chans"]
    sample_height = config["model_params"]["img_size"]
    sample_width = config["model_params"]["img_size"]

    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model = model.eval().to(device)

    if args.checkpoint_path_late:
        config = load_config(args.config_path_late)
        model_late = UViT(**config["model_params"])
        state_dict = torch.load(args.checkpoint_path_late, map_location="cpu")
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model_late.load_state_dict(state_dict)
        model_late = model_late.eval().to(device)
    else:
        model_late = None

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
    samples, intermediate_samples = get_samples(
        model=model,
        batch_size=args.batch_size,
        postprocessing=postprocessing,
        seed=args.seed,
        num_channels=num_channels,
        sample_height=sample_height,
        sample_width=sample_width,
        use_ddim=args.use_ddim,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        y=y,
        autoencoder=autoencoder,
        late_model=model_late,
        t_switch=args.t_switch,
        timesteps_save=args.timesteps_save
    )
    tac = time.time()
    dump_statistics(tac - tic, output_folder)

    dump_samples(samples, output_folder)

    if args.timesteps_save:
        for timestep, samples in zip(args.timesteps_save, intermediate_samples):
            dump_samples(samples, output_folder, timestep)


if __name__ == "__main__":
    main()
