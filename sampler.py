from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from tqdm import tqdm

from models.uvit import UViT
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

model = UViT(
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


def get_samples(batch_size: int, postprocessing: callable, seed: int):
    seed_everything(seed)
    x = torch.randn(batch_size, 3, 32, 32).to(device)

    for t in tqdm(range(999, -1, -1)):
        time_tensor = t * torch.ones(batch_size, device=device)
        with torch.no_grad():
            model_output = model(x, time_tensor)
        x = postprocessing(model_output, x, t)

    samples = (x + 1) / 2
    samples = rearrange(samples, "b c h w -> b h w c")
    return samples.cpu().numpy()


def dump_samples(samples, output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    plt.hist(samples.flatten())
    plt.savefig(output_folder / "histogram.png")
    plt.clf()

    for sample_id, sample in enumerate(samples):
        sample = np.clip(sample, 0, 1)
        plt.imsave(output_folder / f"{sample_id}.png", sample)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument(
        "--parametrization",
        type=str,
        choices=["predict_noise", "predict_original", "predict_previous"],
        required=True,
    )
    parser.add_argument("--output_folder", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.parametrization == "predict_noise":
        postprocessing = predict_noise_postprocessing
    elif args.parametrization == "predict_original":
        postprocessing = predict_original_postprocessing
    elif args.parametrization == "predict_previous":
        postprocessing = predict_previous_postprocessing
    else:
        raise ValueError(f"Invalid parametrization {args.parametrization}")

    model.load_state_dict(
        torch.load(args.checkpoint_path, map_location="cpu")["model_state_dict"]
    )
    model = model.eval().to(device)
    samples = get_samples(args.batch_size, postprocessing, args.seed)
    dump_samples(samples, args.output_folder)
