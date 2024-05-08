import argparse
import tqdm

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

from utils.train_utils import (
    get_model,
    get_dataloader,
    get_optimizer,
    get_lr_scheduler,
    get_noise_scheduler,
    InfiniteDataloaderIterator,
    seed_everything,
)

def get_args():
    parser = argparse.ArgumentParser(description="Training parameters")
    # Default parameters from https://github.com/baofff/U-ViT/blob/main/configs/cifar10_uvit_small.py

    # Training
    parser.add_argument("--seed", type=int, default=1, help="Seed")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    # parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--n_steps", type=int, required=True, help="Number of steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Number of steps")
    parser.add_argument(
        "--num_train_timesteps", type=int, default=1000, help="Number of timesteps"
    )
    parser.add_argument("--use_amp", action="store_true", default=False, help="Use AMP")
    parser.add_argument(
        "--amp_dtype", type=str, default="bf16", help="AMP data type"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Max gradient norm"
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


def train(
    model,
    optimizer,
    train_dataloader,
    noise_scheduler,
    lr_scheduler,
    device,
    writer,
    args,
    finished_steps=0,
):
    losses = []
    scaler = GradScaler()

    dataloader_iterator = InfiniteDataloaderIterator(train_dataloader)
    # TODO dataloader checkpointing
    for step in tqdm.tqdm(range(finished_steps, args.n_steps)):
        model.train()
        optimizer.zero_grad()

        batch = next(iter(train_dataloader))
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

        optimizer.zero_grad()
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()

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
    writer = SummaryWriter()

    accelerator = Accelerator(mixed_precision=args.amp_dtype)
    model, optimizer, trainloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)

    device = accelerator.device


    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    train(
        model, optimizer, train_dataloader, noise_scheduler, lr_scheduler, device, writer, args
    )
