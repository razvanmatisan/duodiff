import argparse
import os

import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from datasets.cifar10 import get_cifar10_dataloader
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


def get_args():
    parser = argparse.ArgumentParser(description="Sampling parameters")
    # Default parameters from https://github.com/baofff/U-ViT/blob/main/configs/cifar10_uvit_small.py

    parser.add_argument("--start_seed", type=int, default=1, help="Start seed")
    parser.add_argument("--end_seed", type=int, default=9, help="End seed")
    parser.add_argument(
        "--num_timesteps", type=int, default=1000, help="Number of timesteps"
    )
    parser.add_argument("--use_amp", action="store_true", default=False, help="Use AMP")
    parser.add_argument("--amp_dtype", type=str, default="bf16", help="AMP data type")
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Max gradient norm"
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
        default=0.1,
        help="Early exit threshold",
    )

    parser.add_argument(
        "--load_from_folder", action="store_true", help="Load from folder"
    )

    # Benchmarking
    parser.add_argument(
        "--benchmarking",
        action="store_true",
        help="True if we want to benchmark the sampler",
    )

    return parser.parse_args()


def save_cifar10_images(directory, num_images=10):
    # Ensure the output directory exists
    os.makedirs(directory, exist_ok=True)

    dataloader = get_cifar10_dataloader(batch_size=1, seed=0)

    images = []

    for i in range(num_images):
        x, _ = next(iter(dataloader))  # x.shape = [1, 3, 32, 32]
        # File path to save the image, e.g., 'original_data/0.png'
        filename = os.path.join(directory, f"{i}.png")
        # Save image; 'save_image' expects a batch dimension, so use 'unsqueeze(0)'
        save_image(x, filename)

        images.append(x)

    images = torch.cat(images, dim=0)
    return images


def save_cifar10_sampled_images(directory):
    args = get_args()

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

    samples = []
    os.makedirs(directory, exist_ok=True)

    for seed in range(args.start_seed, args.end_seed):
        seed_everything(seed)

        sample, _ = noise_scheduler.sample(
            model=model,
            num_steps=args.num_timesteps,
            data_shape=(3, args.sample_height, args.sample_width),
            num_samples=1,
            seed=seed,
            model_type=args.model,
            benchmarking=args.benchmarking,
        )

        # Keep sample
        samples.append(sample)

        # File path to save the image, e.g., 'generated_data/0.png'
        filename = os.path.join(directory, f"{seed}.png")
        save_image(sample, filename)

    samples = torch.cat(samples, dim=0)
    return samples


def fid_evaluation(real_images, generated_images):
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)
    print(f"FID: {float(fid.compute())}")


def read_from_folder(fdir):
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    file_list = [f for f in os.listdir(fdir) if f.endswith(".png")]

    # Load each image into a tensor and collect them into a list
    tensor_list = []
    for file_name in file_list:
        img_path = os.path.join(fdir, file_name)
        img = Image.open(img_path).convert("RGB")  # Ensure images are in RGB format
        img_tensor = transform(img)
        tensor_list.append(img_tensor)

    # Stack the tensors into a single tensor along the first dimension
    stacked_tensor = torch.stack(tensor_list, dim=0)
    return stacked_tensor


if __name__ == "__main__":
    args = get_args()

    # Directory for original images
    output_dir_original = "Generated_samples/CIFAR10/original_data"

    # Directory for sampled images
    output_dir_model = "Generated_samples/CIFAR10/generated_data"

    if args.load_from_folder:
        real_images = read_from_folder(output_dir_original)
        generated_images = read_from_folder(output_dir_model)
    else:
        real_images = save_cifar10_images(output_dir_original)
        print(f"All CIFAR10 images have been saved in '{output_dir_original}'.")

        generated_images = save_cifar10_sampled_images(output_dir_model)
        print(f"All CIFAR10 images have been saved in '{output_dir_original}'.")

    fid_evaluation(real_images, generated_images)
