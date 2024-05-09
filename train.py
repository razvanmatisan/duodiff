import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter

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
    parser.add_argument("--batch_size", type=int, default=2, help="Number of steps")
    parser.add_argument(
        "--num_train_timesteps", type=int, default=1000, help="Number of timesteps"
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
        "--log_every_n_steps", type=int, default=100, help="Log every n steps"
    )

    # Checkpointing
    parser.add_argument(
        "--load_checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint path for loading the training state",
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
        "--model", type=str, default="uvit", choices=["uvit"], help="Model name"
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


def save_checkpoint(model, optimizer, train_dataloader, lr_scheduler, step, loss):
    path = Path(args.log_path)
    if args.save_checkpoint_path:
        path = path / args.save_checkpoint_path
    else:
        path = path / f"{args.dataset}_{args.model}.pth"

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
    checkpoint = None
    if args.load_checkpoint_path:
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
    data = batch[0].to(device)
    batch_size = data.size(0)
    clean_images = data

    timesteps = torch.randint(
        0, args.num_train_timesteps, (batch_size,), device=device
    ).long()
    timesteps_normalized = timesteps.float() / args.num_train_timesteps
    noise, noisy_images = noise_scheduler.add_noise(clean_images, timesteps)

    if args.model == "uvit":
        predicted_noise = model(noisy_images, timesteps_normalized)
        loss = F.mse_loss(predicted_noise, noise)
    elif args.model == "deediff_uvit":
        raise NotImplementedError

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
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])

    model.train()
    for step in range(finished_steps + 1, args.n_steps + 1):
        batch = next(dataloader_iterator)

        loss = loss_fn(model, batch, noise_scheduler, device, args)

        optimizer.zero_grad()
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()

        if args.save_every_n_steps and step % args.save_every_n_steps == 0:
            save_checkpoint(
                model, optimizer, dataloader, lr_scheduler, step, loss.item()
            )

        if step % args.log_every_n_steps == 0:
            writer.add_scalar("Loss/train", loss.item(), step)


if __name__ == "__main__":
    args = get_args()

    seed_everything(args.seed)

    model = get_model(args)
    train_dataloader = get_dataloader(args)
    optimizer = get_optimizer(model, args)
    noise_scheduler = get_noise_scheduler(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr_scheduler = get_lr_scheduler(optimizer, args)
    writer = SummaryWriter(args.log_path)
    finished_steps = 0  # Step from which to resume training (by default set to 0 but might be overwritten when loading checkpoint)

    accelerator = Accelerator(mixed_precision=args.amp_dtype)
    device = accelerator.device
    model, optimizer, trainloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.load_checkpoint_path:
        finished_steps, checkpoint = load_checkpoint(
            args, model, optimizer, train_dataloader, lr_scheduler, device
        )

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
