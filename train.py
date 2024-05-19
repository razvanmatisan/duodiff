import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.train_utils import (
    InfiniteDataloaderIterator,
    get_dataloader,
    get_lr_scheduler,
    get_model,
    get_noise_scheduler,
    get_optimizer,
    seed_everything,
)


def get_args():
    parser = argparse.ArgumentParser(description="Training parameters")
    # Default parameters from https://github.com/baofff/U-ViT/blob/main/configs/cifar10_uvit_small.py

    # Training
    parser.add_argument("--seed", type=int, default=1, help="Seed")
    parser.add_argument("--n_steps", type=int, required=True, help="Number of steps")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of steps")
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
        "--log_path", type=str, default="logs", help="Directory for logs"
    )
    parser.add_argument(
        "--log_every_n_steps", type=int, default=None, help="Log every n steps"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=16,
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
        "--load_backbone",
        type=str,
        default=None,
        help="Checkpoint to a pretrained UViT backbone",
    )
    parser.add_argument(
        "--keep_initial_timesteps",
        action="store_true",
        help="Using timesteps in [0, T] if True. Otherwise, normalize the timesteps in [0, 1]",
    )

    parser.add_argument(
        "--save_checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint path for saving the training state (log_path/save_checkpoint_path)",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=None,
        help="Frequency of saving the checkpoint",
    )
    parser.add_argument(
        "--save_new_every_n_steps",
        type=int,
        default=None,
        help="Frequency of creating a new checkpoint (not overwriting the last checkpoint)",
    )

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw"],
        help="Optimizer name",
    )
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.03, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.99, help="Beta_1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta_2")

    # LR scheduler
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=1500,
        help="Number of lr scheduler warmup steps",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="uvit",
        choices=["uvit", "deediff_uvit"],
        help="Model name",
    )
    parser.add_argument("--img_size", type=int, default=32, help="Image size")
    parser.add_argument("--patch_size", type=int, default=2, help="Patch size")
    parser.add_argument("--embed_dim", type=int, default=512, help="Embed dim")
    parser.add_argument("--depth", type=int, default=12, help="Depth")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--mlp_ratio", type=int, default=4, help="MLP ratio")
    parser.add_argument(
        "--qkv_bias", action="store_true", default=False, help="QKV bias"
    )
    parser.add_argument(
        "--mlp_time_embed", action="store_true", default=False, help="MLP time embed"
    )
    parser.add_argument("--num_classes", type=int, default=-1, help="Number of classes")

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "celeba"],
        help="Dataset name",
    )

    return parser.parse_args()


def save_checkpoint(
    model, optimizer, train_dataloader, lr_scheduler, step, loss, overwrite=True
):
    path = Path(args.log_path)
    if args.save_checkpoint_path:
        path = path / args.save_checkpoint_path
    else:
        path = path / f"{args.dataset}_{args.model}.pth"

    if not overwrite:
        path = path.parent / (path.stem + f"_step-{step}" + path.suffix)

    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "dataloader_sampler_state": train_dataloader.sampler.get_state(),
            "loss": loss,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "random_rng_state": random.getstate(),
        },
        path,
    )


def load_checkpoint(args, model, optimizer, train_dataloader, lr_scheduler, device):
    checkpoint = torch.load(args.load_checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "lr_scheduler_state_dict" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    if "dataloader_sampler_state" in checkpoint:
        train_dataloader.sampler.set_state(checkpoint["dataloader_sampler_state"])

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])

    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])

    if "random_rng_state" in checkpoint:
        random.setstate(checkpoint["random_rng_state"])

    finished_steps = checkpoint.get("step", 0)

    return finished_steps, checkpoint


def loss_fn(model, batch, noise_scheduler, device, args):
    device = model.device
    data = batch[0].to(device)
    batch_size = data.size(0)
    clean_images = data

    timesteps = torch.randint(
        0, args.num_timesteps, (batch_size,), device=device
    ).long()

    noise, noisy_images = noise_scheduler.add_noise(clean_images, timesteps)

    if not args.keep_initial_timesteps:
        timesteps = timesteps.float() / args.num_timesteps

    if args.model == "uvit":
        predicted_noise = model(noisy_images, timesteps)
        loss = F.mse_loss(predicted_noise, noise)
    elif args.model == "deediff_uvit":
        predicted_noise, classifier_outputs, outputs = model(noisy_images, timesteps)

        # Reshape list of L elements of shape (bs, C, H, W) into tensor of shape (L, bs, C, H, W)
        classifier_outputs = torch.stack(classifier_outputs, dim=0)
        outputs = torch.stack(outputs, dim=0)

        # L_simple
        L_simple = F.mse_loss(predicted_noise, noise)

        # L_u_t
        u_t_hats = torch.stack(
            [F.tanh(torch.abs(output - noise)) for output in outputs], dim=0
        )
        u_t_hats = u_t_hats.mean(dim=(-1, -2, -3))
        L_u_t = F.mse_loss(classifier_outputs, u_t_hats, reduction="sum")

        # L_UAL_t
        L_n_t = torch.stack([(output - noise) ** 2 for output in outputs], dim=0)
        L_n_t = L_n_t.mean(dim=(-1, -2, -3))
        L_UAL_t = ((1 - classifier_outputs) * L_n_t).mean(dim=1).sum(dim=0)

        # Total loss
        loss = L_simple, L_u_t, L_UAL_t

    return loss


def train(
    model,
    optimizer,
    dataloader,
    noise_scheduler,
    lr_scheduler,
    accelerator,
    device,
    writer,
    args,
    checkpoint=None,
    finished_steps=0,
):
    dataloader_iterator = InfiniteDataloaderIterator(dataloader)

    # Need to restore torch rng state after creating the iterator
    if checkpoint is not None and "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])

    model.train()
    times = []
    log_times = []
    save_times = []
    step_times = []
    for step in tqdm(range(finished_steps + 1, args.n_steps + 1), desc="Training"):
        tic = time.time()
        step_tic = time.time()
        batch = next(dataloader_iterator)

        loss = loss_fn(model, batch, noise_scheduler, device, args)
        if isinstance(loss, tuple):
            regular_loss, classifier_loss, weighted_loss = loss
            writer.add_scalar("Regular train loss", regular_loss.item(), step)
            writer.add_scalar("Classifier train loss", classifier_loss.item(), step)
            writer.add_scalar("Weighted train loss", weighted_loss.item(), step)
            loss = regular_loss + classifier_loss + weighted_loss
        else:
            writer.add_scalar("Train loss", loss.item(), step)

        optimizer.zero_grad()
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()

        step_toc = time.time()
        step_times.append(step_toc - step_tic)

        if args.log_every_n_steps is not None and (
            step % args.log_every_n_steps == 0 or step == args.n_steps
        ):
            log_tic = time.time()
            print(f"step {step}: train loss = {loss.item()}")
            samples, logging_dict = noise_scheduler.sample(
                model=model,
                num_steps=args.num_timesteps,
                data_shape=(3, args.sample_height, args.sample_width),
                num_samples=args.n_samples,
                seed=args.sample_seed,
                model_type=args.model,
                keep_initial_timesteps=args.keep_initial_timesteps,
            )
            img_grid = (
                torchvision.utils.make_grid(
                    samples, nrow=int(math.sqrt(samples.size(0))), normalize=True
                )
                * 0.5
                + 0.5
            )

            writer.add_image("Samples", img_grid, global_step=step)
            log_toc = time.time()
            log_times.append(log_toc - log_tic)

        if args.save_every_n_steps and (
            step % args.save_every_n_steps == 0 or step == args.n_steps
        ):
            save_tic = time.time()
            save_checkpoint(
                model, optimizer, dataloader, lr_scheduler, step, loss.item()
            )

            save_toc = time.time()
            save_times.append(save_toc - save_tic)

        if (
            args.save_new_every_n_steps is not None
            and step % args.save_new_every_n_steps == 0
        ):
            save_checkpoint(
                model,
                optimizer,
                dataloader,
                lr_scheduler,
                step,
                loss.item(),
                overwrite=False,
            )

        toc = time.time()
        times.append(toc - tic)

    print(f"Average total time = {np.mean(times)}")
    print(f"Average save time = {np.mean(save_times)}")
    print(f"Average log time = {np.mean(log_times)}")
    print(f"Average step time = {np.mean(step_times)}")


if __name__ == "__main__":
    args = get_args()

    seed_everything(args.seed)

    model = get_model(args)
    train_dataloader = get_dataloader(args)
    optimizer = get_optimizer(model, args)
    noise_scheduler = get_noise_scheduler(args)
    lr_scheduler = get_lr_scheduler(optimizer, args)
    writer = SummaryWriter(args.log_path)

    accelerator = Accelerator(mixed_precision=args.amp_dtype)
    device = accelerator.device
    print(f"Training on {device}")
    model, optimizer, trainloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    noise_scheduler.set_device(model.device)

    if args.load_checkpoint_path:
        finished_steps, checkpoint = load_checkpoint(
            args, model, optimizer, train_dataloader, lr_scheduler, device
        )
    else:
        finished_steps = 0
        checkpoint = None

    train(
        model,
        optimizer,
        train_dataloader,
        noise_scheduler,
        lr_scheduler,
        accelerator,
        device,
        writer,
        args,
        finished_steps=finished_steps,
        checkpoint=checkpoint,
    )
