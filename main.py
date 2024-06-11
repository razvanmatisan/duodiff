import argparse

import torch

from trainer import Trainer
from utils.train_utils import get_exp_name


def get_args():
    parser = argparse.ArgumentParser(description="Training parameters")
    # Default parameters from https://github.com/baofff/U-ViT/blob/main/configs/cifar10_uvit_small.py

    # Training
    parser.add_argument("--seed", type=int, default=1, help="Seed")
    parser.add_argument("--n_steps", type=int, required=True, help="Number of steps")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--num_timesteps", type=int, default=1000, help="Number of timesteps"
    )
    parser.add_argument("--use_amp", action="store_true", default=False, help="Use AMP")
    parser.add_argument(
        "--amp_dtype", type=str, default="bfloat16", help="AMP data type"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Max gradient norm"
    )

    # Logging
    parser.add_argument(
        "--log_path", type=str, default="logs", help="Directory for logs"
    )
    parser.add_argument(
        "--exp_name", type=str, default=None, help="Directory for experiment logs"
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
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument(
        "--normalize_timesteps",
        action="store_true",
        help="If true, normalize the timesteps in [0, 1] from [0, 1000]",
    )
    parser.add_argument("--use_unweighted_loss", action="store_true")
    parser.add_argument(
        "--parametrization",
        type=str,
        choices=["predict_noise", "predict_original", "predict_previous"],
        default="predict_noise",
    )

    parser.add_argument(
        "--save_checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint path for saving the training state (log_path/exp_name/save_checkpoint_path)",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="If true, resume from the last checkpoint from --log_path",
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
    parser.add_argument(
        "--classifier_type",
        type=str,
        default="attention_probe",
        choices=[
            "attention_probe",
            "mlp_probe_per_layer",
            "mlp_probe_per_timestep",
            "mlp_probe_per_layer_per_timestep",
        ],
        help="Classification head",
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
    parser.add_argument(
        "--data_path", type=str, default="data", help="Directory for datasets"
    )

    return parser.parse_args()


def main():
    args = get_args()

    if args.exp_name is None:
        args.exp_name = get_exp_name(args)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer = Trainer(args)
    logs = trainer.train()

    return logs


if __name__ == "__main__":
    main()
