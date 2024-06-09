import re
from pathlib import Path
from typing import Dict
from collections import OrderedDict

import torch


# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/wav2vec2/common/helpers.py
class Checkpointer:
    def __init__(self, args):
        self.log_path = Path(args.log_path) / args.exp_name

        if args.save_checkpoint_path:
            pattern = f"{args.save_checkpoint_path}_step-*"
            self.save_path = f"{args.save_checkpoint_path}"
        else:
            pattern = f"{args.dataset}_{args.model}_step-*.pth"
            self.save_path = f"{args.dataset}_{args.model}.pth"

        self.save_path = self.log_path / self.save_path

        checkpoint_last = self.save_path.parent / (
            self.save_path.stem + f"_last" + self.save_path.suffix
        )
        self.checkpoint_last = checkpoint_last if checkpoint_last.is_file() else None

        tracked = [
            (re.search("step-(\d+)\.pth", str(f)).group(1), f)
            for f in Path(self.log_path).rglob(pattern)
        ]

        self.tracked = self.tracked = OrderedDict(sorted(tracked, key=lambda t: t[0]))

        fpath = args.load_checkpoint_path or (
            self.last_checkpoint() if args.resume else None
        )

        if fpath is not None:
            print(f"Loading model from {fpath}")
            self.last_state = torch.load(fpath, map_location="cpu")
        else:
            self.last_state = None

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        dataloader: torch.utils.data.DataLoader,
        args,
        logs,
        step: int,
        new_checkpoint: bool,
    ):
        path = self.save_path

        if new_checkpoint:
            path = path.parent / (path.stem + f"_step-{step}" + path.suffix)
        else:
            path = path.parent / (path.stem + f"_last" + path.suffix)

        state = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "dataloader_sampler_state": dataloader.sampler.get_state(),
            "args": args,
            "train_state": {"logs": logs, "step": step},
        }

        self.last_state = state

        print(f"Saving {path}...")
        torch.save(state, path)

    def maybe_load_state(
        self,
        checkpoint_path=None,
        model: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scaler=None,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR = None,
        dataloader: torch.utils.data.DataLoader = None,
        train_state: Dict = None,
    ):
        if checkpoint_path is None and self.last_state is None:
            print(f"No checkpoint to load")
            return

        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")
        else:
            state = self.last_state

        if model is not None:
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)

        if optimizer is not None:
            if "optimizer_state_dict" in state:
                optimizer.load_state_dict(state["optimizer_state_dict"])
            else:
                raise ValueError("Optimizer state not found")

        if scaler is not None:
            if "scaler_state_dict" in state:
                scaler.load_state_dict(state["scaler_state_dict"])
            else:
                raise ValueError("Scaler state not found")

        if lr_scheduler is not None:
            if "lr_scheduler_state_dict" in state:
                lr_scheduler.load_state_dict(state["lr_scheduler_state_dict"])
            else:
                raise ValueError("LR scheduler state not found")

        if dataloader is not None:
            if "dataloader_sampler_state" in state:
                dataloader.sampler.set_state(state["dataloader_sampler_state"])
            else:
                raise ValueError("Dataloader's sampler state not found")

        if train_state is not None:
            if "train_state" in state:
                train_state.update(state["train_state"])
            else:
                raise ValueError("Train state not found")

    def last_checkpoint(self):
        tracked = list(self.tracked.values())
        if self.checkpoint_last is not None:
            tracked += [self.checkpoint_last]

        for fpath in reversed(tracked):
            try:
                torch.load(fpath, map_location="cpu")
                print(f"Checkpoint {fpath} loaded successfully.")
                return fpath
            except:
                print(f"Checkpoint {fpath} appears corrupted.")

        return None
