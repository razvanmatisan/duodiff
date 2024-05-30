import subprocess
from dataclasses import dataclass

import torch
from models.early_exit import EarlyExitUViT, OldEarlyExitUViT
from models.uvit import UViT
from torch import nn


@dataclass
class CheckpointEntry:
    checkpoint_path: str
    model_type: EarlyExitUViT | OldEarlyExitUViT
    classifier_type: str

    def get_model(self) -> EarlyExitUViT | OldEarlyExitUViT:
        base_model = UViT(
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
        model: nn.Module = self.model_type(
            uvit=base_model,
            classifier_type=self.classifier_type,
            exit_threshold=float("-inf"),
        )

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


checkpoint_entries = {
    "mlp_per_timestep_frozen": CheckpointEntry(
        checkpoint_path="logs/6305297/cifar10_deediff_uvit.pth",
        model_type=EarlyExitUViT,
        classifier_type="mlp_probe_per_timestep",
    ),
    "mlp_per_timestep_initialized": CheckpointEntry(
        checkpoint_path="logs/6307068/cifar10_deediff_uvit.pth",
        model_type=EarlyExitUViT,
        classifier_type="mlp_probe_per_timestep",
    ),
    "mlp_per_layer_frozen": CheckpointEntry(
        checkpoint_path="logs/6305343/cifar10_deediff_uvit.pth",
        model_type=EarlyExitUViT,
        classifier_type="mlp_probe_per_layer",
    ),
    "mlp_per_layer_initialized": CheckpointEntry(
        checkpoint_path="logs/6307067/cifar10_deediff_uvit.pth",
        model_type=EarlyExitUViT,
        classifier_type="mlp_probe_per_layer",
    ),
    "attention_frozen": CheckpointEntry(
        checkpoint_path="logs/6266130/cifar10_deediff_uvit.pth",
        model_type=OldEarlyExitUViT,
        classifier_type="attention_probe",
    ),
    "attention_initialized": CheckpointEntry(
        checkpoint_path="logs/6344819/cifar10_deediff_uvit.pth",
        model_type=EarlyExitUViT,
        classifier_type="attention_probe",
    ),
    "mlp_per_time_per_layer_initialized": CheckpointEntry(
        checkpoint_path="logs/6374645/cifar10_deediff_uvit.pth",
        model_type=EarlyExitUViT,
        classifier_type="mlp_probe_per_layer_per_timestep",
    ),
}

if __name__ == "__main__":
    # Download the checkpoints locally
    for entry in checkpoint_entries.values():
        subprocess.call(
            [
                "rsync",
                "--mkpath",
                "--progress",
                f"dl2:early-stopping-diffusion/{entry.checkpoint_path}",
                entry.checkpoint_path,
            ]
        )
