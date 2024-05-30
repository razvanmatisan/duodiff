import argparse
import os
import sys

import distance
import embedding
import io_util
import numpy as np
import torch
from checkpoint_entries import checkpoint_entries
from einops import rearrange
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

from datasets.cifar10 import get_cifar10_dataloader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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
    parser = argparse.ArgumentParser(description="CMMD evaluation parameters")

    # If you already have both the dataset images and the sampled images on which you want to compute the CMMD score,
    # pass these arguments: load_from_folder, samples_directory and dataset_directory.
    # ! Make sure the 2 folders contain the same amount of images.

    parser.add_argument(
        "--load_from_folder", action="store_true", help="Load from folder"
    )

    parser.add_argument(
        "--samples_directory",
        type=str,
        default=None,
        help="Path to the directory where to save the images from the model",
    )

    parser.add_argument(
        "--dataset_directory",
        type=str,
        default=None,
        help="Path to the directory where to save the images from the dataset",
    )

    # Otherwise, if you don't have the images already, for the samples images you have 2 options:
    # 1. Sample images from a checkpointed model.
    # 2. Read pt log files and save the corresponding images.
    # Note: in this case you need to pass the above arguments (samples_directory and dataset_directory) as directories where to save the sampled/original images.

    # For option 1, pass these arguments:
    parser.add_argument(
        "--checkpoint_entry_name",
        type=str,
        default=None,
        help="Checkpoint path for loading the training state",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of images to save from the dataset and/or sample from the model",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for sampling",
    )

    parser.add_argument(
        "--exit_threshold",
        type=float,
        default=0.1,
        help="Early exit threshold",
    )

    # For option 2, pass these arguments:
    parser.add_argument(
        "--samples_pt_directory",
        type=str,
        default=None,
        help="Path to the directory where to save the images from the model",
    )

    # CMMD metric specific parameters
    parser.add_argument(
        "--cmmd_batch_size",
        type=int,
        default=32,
        help="Batch size for embedding generation.",
    )

    parser.add_argument(
        "--cmmd_max_count",
        type=int,
        default=1024,
        help="Maximum number of images to read from each directory.",
    )

    return parser.parse_args()


device = get_device()
betas = torch.linspace(1e-4, 0.02, 1000).to(device)
alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, dim=0)
alphas_bar_previous = torch.cat([torch.tensor([1.0], device=device), alphas_bar[:-1]])
betas_tilde = betas * (1 - alphas_bar_previous) / (1 - alphas_bar)


def sample(threshold, model):
    args = get_args()
    torch.manual_seed(0)

    x = torch.randn(args.batch_size, 3, 32, 32).to(device)
    error_prediction_by_timestep = torch.zeros(1000, 13)
    indices_by_timestep = torch.zeros(1000, args.batch_size)
    for t in tqdm(range(1000, 0, -1)):
        with torch.no_grad():
            time_tensor = t / 1000 * torch.ones(args.batch_size, device=device)
            epsilon, classifier_outputs, outputs = model(x, time_tensor)

        outputs = torch.stack(outputs + [epsilon])
        classifier_outputs = torch.stack(
            classifier_outputs + [torch.zeros_like(classifier_outputs[0])]
        )

        # Simulate early exit with a global threshold
        indices = torch.argmax((classifier_outputs <= threshold).int(), dim=0)
        epsilon = outputs[indices, torch.arange(args.batch_size)]

        # Log for visualization
        error_prediction_by_timestep[t - 1] = classifier_outputs.mean(axis=1)[:13]
        indices_by_timestep[t - 1, :] = indices

        alpha_t = alphas[t - 1]
        alpha_bar_t = alphas_bar[t - 1]
        sigma_t = torch.sqrt(betas_tilde[t - 1])

        z = torch.randn_like(x) if t > 1 else 0
        x = (
            torch.sqrt(1 / alpha_t)
            * (x - (1 - alpha_t) / (torch.sqrt(1 - alpha_bar_t)) * epsilon)
        ) + sigma_t * z

    return x


def save_cifar10_images(directory, num_images):
    # Ensure the output directory exists
    os.makedirs(directory, exist_ok=True)

    dataloader = get_cifar10_dataloader(batch_size=1, seed=0)

    for i in range(num_images):
        x, _ = next(iter(dataloader))
        filename = os.path.join(directory, f"{i}.png")
        save_image(x, filename)


def save_cifar10_sampled_images(output_directory):
    args = get_args()
    device = get_device()

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    num_images = 0

    if args.checkpoint_entry_name:
        model = checkpoint_entries[args.checkpoint_entry_name].get_model()

        model = model.eval()
        model = model.to(device)

        for i in range(args.n_samples):
            x = sample(args.exit_threshold, model)
            x = (x + 1) / 2

            filename = os.path.join(output_directory, f"{i}.png")
            save_image(x, filename)

        num_images = args.n_samples

    elif args.samples_pt_directory:
        files = [f for f in os.listdir(args.samples_pt_directory) if f.endswith(".pt")]
        for j, file in enumerate(files):
            samples = torch.load(
                os.path.join(args.samples_pt_directory, file), map_location="cpu"
            )
            samples = (samples + 1) / 2
            samples = rearrange(samples, "b c h w -> b h w c")

            for i, s in enumerate(samples):
                img = (s.numpy() * 255).astype("uint8")
                img = Image.fromarray(img)
                img.save(os.path.join(output_directory, f"sample_{i + 128 * j}.png"))
                num_images += 1

    return num_images


def compute_cmmd(ref_dir, eval_dir, ref_embed_file=None, batch_size=32, max_count=-1):
    """Calculates the CMMD distance between reference and eval image sets.

    Args:
      ref_dir: Path to the directory containing reference images.
      eval_dir: Path to the directory containing images to be evaluated.
      ref_embed_file: Path to the pre-computed embedding file for the reference images.
      batch_size: Batch size used in the CLIP embedding calculation.
      max_count: Maximum number of images to use from each directory. A
        non-positive value reads all images available except for the images
        dropped due to batching.

    Returns:
      The CMMD value between the image sets.
    """
    if ref_dir and ref_embed_file:
        raise ValueError(
            "`ref_dir` and `ref_embed_file` both cannot be set at the same time."
        )
    embedding_model = embedding.ClipEmbeddingModel()
    if ref_embed_file is not None:
        ref_embs = np.load(ref_embed_file).astype("float32")
    else:
        ref_embs = io_util.compute_embeddings_for_dir(
            ref_dir, embedding_model, batch_size, max_count
        ).astype("float32")
    eval_embs = io_util.compute_embeddings_for_dir(
        eval_dir, embedding_model, batch_size, max_count
    ).astype("float32")
    val = distance.mmd(ref_embs, eval_embs)
    return val.numpy()


if __name__ == "__main__":
    args = get_args()

    if not args.load_from_folder:
        num_images = save_cifar10_sampled_images(args.samples_directory)
        save_cifar10_images(args.dataset_directory, num_images)

    print(
        "The CMMD value is: "
        f" {compute_cmmd(args.dataset_directory, args.samples_directory, batch_size=args.cmmd_batch_size, max_count=args.cmmd_max_count):.3f}"
    )
