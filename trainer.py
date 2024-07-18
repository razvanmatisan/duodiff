import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from checkpointer import Checkpointer
from datasets.celeba import get_celeba_dataloader
from datasets.cifar10 import get_cifar10_dataloader
from datasets.imagenet import get_imagenet_dataloader
from ddpm_core import NoiseScheduler
from models.early_exit import EarlyExitUViT
from models.utils.autoencoder import get_autoencoder
from models.uvit import UViT
from utils.train_utils import (
    seed_everything,
)


class Trainer:
    def __init__(
        self,
        args,
    ):
        seed_everything(args.seed)

        self.args = args
        self._make_log_dir()
        self._init_device()
        self._init_checkpointer()
        self._init_model()
        self._init_dataloader()
        self._init_optimizer()
        self._init_noise_scheduler()
        self._init_lr_scheduler()
        self._init_writer()
        self._init_amp()
        self._save_hparams()

        self.train_state = dict()
        self.autoencoder = (
            get_autoencoder(self.args.autoencoder_checkpoint_path)
            if self.args.autoencoder_checkpoint_path is not None
            else None
        )

        if self.args.resume:
            self.checkpointer.maybe_load_state(
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                lr_scheduler=self.lr_scheduler,
                dataloader=self.dataloader,
                train_state=self.train_state,
            )
        else:
            self.checkpointer.maybe_load_state(
                model=self.model,
            )

    def _save_hparams(self):
        hparams_config_path = self.log_path / "hparams.json"
        with hparams_config_path.open("w") as f:
            json.dump(vars(self.args), f)

        # TODO: Not showing hparams in tensorboard
        self.writer.add_hparams(vars(self.args), {"dummy": 0})

    def _make_log_dir(self):
        self.log_path = Path(self.args.log_path) / self.args.exp_name
        print(f"Log directory is {self.log_path}")
        self.log_path.mkdir(parents=True, exist_ok=True)

    def _init_device(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training on {self.device}")

    def _init_checkpointer(self):
        self.checkpointer = Checkpointer(self.args)

    def _init_model(self):
        if self.args.model in ["uvit", "deediff_uvit"]:
            model = UViT(
                img_size=self.args.img_size,
                patch_size=self.args.patch_size,
                in_chans=self.args.num_channels,
                embed_dim=self.args.embed_dim,
                depth=self.args.depth,
                num_heads=self.args.num_heads,
                mlp_ratio=self.args.mlp_ratio,
                qkv_bias=self.args.qkv_bias,
                mlp_time_embed=self.args.mlp_time_embed,
                num_classes=self.args.num_classes,
                normalize_timesteps=self.args.normalize_timesteps,
            )

        if self.args.model == "deediff_uvit":
            print(f"Initializing EarlyExitUViT with {self.args.classifier_type}")

            if self.args.load_backbone:
                print(f"Loading backbone from {self.args.load_backbone}")
                self.checkpointer.maybe_load_state(self.args.load_backbone, model=model)

            if self.args.freeze_backbone:
                print("Freezing the backbone...")
                for param in model.parameters():
                    param.requires_grad = False

            model = EarlyExitUViT(model, classifier_type=self.args.classifier_type)

        self.model = model.to(self.device)

    def _init_optimizer(self):
        if self.args.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                betas=(self.args.beta1, self.args.beta2),
                weight_decay=self.args.weight_decay,
            )
        else:
            raise ValueError(f"Optimizer {self.args.optimizer} not implemented.")

    def _init_dataloader(self):
        if self.args.dataset == "cifar10":
            self.dataloader = get_cifar10_dataloader(
                batch_size=self.args.batch_size,
                seed=self.args.seed,
                data_dir=self.args.data_path,
            )
        elif self.args.dataset == "celeba":
            self.dataloader = get_celeba_dataloader(
                batch_size=self.args.batch_size,
                seed=self.args.seed,
                data_dir=self.args.data_path,
            )
        elif self.args.dataset == "imagenet":
            self.dataloader = get_imagenet_dataloader(
                batch_size=self.args.batch_size,
                seed=self.args.seed,
                data_dir=self.args.data_path,
            )
        else:
            raise ValueError(f"Dataset {self.args.dataset} not implemented.")

    def _init_noise_scheduler(self):
        self.noise_scheduler = NoiseScheduler(beta_steps=self.args.num_timesteps)
        self.noise_scheduler.set_device(self.device)

    def _init_lr_scheduler(self, last_epoch=-1):
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.n_steps,
            last_epoch=last_epoch,
        )

    def _init_writer(self):
        self.writer = SummaryWriter(self.log_path)

    def _init_amp(self):
        self.use_amp = self.args.use_amp
        self.amp_dtype = getattr(torch, self.args.amp_dtype)
        self.scaler = torch.amp.GradScaler(self.device)

    def _run_batch(self, batch):
        with torch.autocast(
            str(self.device), enabled=self.use_amp, dtype=self.amp_dtype
        ):
            loss = self._loss_fn(batch)

            logging_dict = dict()

            if self.args.model == "deediff_uvit":
                regular_loss, classifier_loss, weighted_loss, unweighted_loss = loss
                loss = regular_loss + classifier_loss + weighted_loss
                if self.args.use_unweighted_loss:
                    loss += unweighted_loss

                logging_dict.update(
                    {
                        "Regular train loss": regular_loss.item(),
                        "Classifier train loss": classifier_loss.item(),
                        "Weighted train loss": weighted_loss.item(),
                        "Unweighted loss": unweighted_loss.item(),
                    }
                )

            logging_dict["Train loss"] = loss.item()

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.lr_scheduler.step()

        return logging_dict

    def _log(self, step, logging_dict):
        for k, v in logging_dict.items():
            self.writer.add_scalar(k, v, step)

        if self.args.log_every_n_steps is not None and (
            step % self.args.log_every_n_steps == 0 or step == self.args.n_steps
        ):
            print(f"Step {step:>7} | logging_dict: {logging_dict}")
            samples, logging_dict = self.noise_scheduler.sample(
                model=self.model,
                num_steps=self.args.num_timesteps,
                data_shape=(3, self.args.sample_height, self.args.sample_width),
                num_samples=self.args.n_samples,
                seed=self.args.sample_seed,
                model_type=self.args.model,
            )
            img_grid = (
                torchvision.utils.make_grid(
                    samples, nrow=int(math.sqrt(samples.size(0))), normalize=True
                )
                * 0.5
                + 0.5
            )

            self.writer.add_image("Samples", img_grid, global_step=step)

    def train(self):
        dataloader_iterator = iter(self.dataloader)
        last_step = self.train_state.get("step", 0)
        logs = self.train_state.get("logs", [])

        print(f"Starting training from step {last_step + 1}")

        self.model.train()
        for step in trange(
            last_step + 1,
            self.args.n_steps + 1,
            initial=last_step + 1,
            total=self.args.n_steps,
            desc="Training",
        ):
            seed_everything(self.args.seed + step)

            batch = next(dataloader_iterator)

            if self.autoencoder is not None:
                batch[0] = self.autoencoder.encode(
                    batch[0]
                )  # (bs, 3, 256, 256) -> (bs, 4, 32, 32)

            logging_dict = self._run_batch(batch)
            self._log(step, logging_dict)

            logs.append(logging_dict)

            if (
                self.args.save_every_n_steps
                and step % self.args.save_every_n_steps == 0
            ) or step == self.args.n_steps:
                self._save_checkpoint(step, logs, False)
            if (
                self.args.save_new_every_n_steps is not None
                and step % self.args.save_new_every_n_steps == 0
            ):
                self._save_checkpoint(step, logs, True)

        return logs

    def _save_checkpoint(
        self,
        step,
        logs,
        new_checkpoint,
    ):
        self.checkpointer.save(
            self.model,
            self.optimizer,
            self.scaler,
            self.lr_scheduler,
            self.dataloader,
            self.args,
            logs,
            step,
            new_checkpoint,
        )

    def _loss_fn(self, batch):
        data = batch[0].to(self.device)
        batch_size = data.size(0)
        clean_images = data

        timesteps = torch.randint(
            0, self.args.num_timesteps, (batch_size,), device=self.device
        ).long()

        noise, noisy_images = self.noise_scheduler.add_noise(clean_images, timesteps)

        if self.args.model == "uvit":
            if self.args.parametrization == "predict_noise":
                predicted_noise = self.model(noisy_images, timesteps)
                loss = F.mse_loss(predicted_noise, noise)
            elif self.args.parametrization == "predict_original":
                predicted_original = self.model(noisy_images, timesteps)
                loss = F.mse_loss(predicted_original, clean_images)
            elif self.args.parametrization == "predict_previous":
                predicted_previous = self.model(noisy_images, timesteps)

                betas = torch.linspace(1e-4, 0.02, 1000).to(self.device)
                alphas = 1 - betas
                alphas_bar = torch.cumprod(alphas, dim=0)
                alphas_bar_previous = torch.cat(
                    [torch.tensor([1.0], device=self.device), alphas_bar[:-1]]
                )

                clean_image_coef = (
                    torch.sqrt(alphas_bar_previous[timesteps])
                    * betas[timesteps]
                    / (1 - alphas_bar[timesteps])
                )[:, None, None, None]

                noisy_image_coef = (
                    torch.sqrt(alphas[timesteps])
                    * (1 - alphas_bar_previous[timesteps])
                    / (1 - alphas_bar[timesteps])
                )[:, None, None, None]

                expected_previous = (
                    clean_image_coef * clean_images + noisy_image_coef * noisy_images
                )

                loss = F.mse_loss(predicted_previous, expected_previous)
            else:
                raise ValueError(
                    f"Unknown parametrization type {self.args.parametrization}"
                )

        elif self.args.model == "deediff_uvit":
            backbone_output, classifier_outputs, ee_outputs = self.model(
                noisy_images, timesteps
            )

            # Reshape list of L elements of shape (bs,) into tensor of shape (L, bs)
            classifier_outputs = torch.stack(classifier_outputs, dim=0)
            # Reshape list of L elements of shape (bs, C, H, W) into tensor of shape (L, bs, C, H, W)
            ee_outputs = torch.stack(ee_outputs, dim=0)

            if self.args.parametrization == "predict_noise":
                true_output = noise
            elif self.args.parametrization == "predict_original":
                true_output = clean_images
            else:
                raise ValueError(
                    f"Unknown parametrization type {self.args.parametrization}"
                )

            # L_simple
            L_simple = F.mse_loss(backbone_output, true_output)

            # L_u_t
            u_t_hats = torch.stack(
                [
                    F.tanh(torch.abs(ee_output - true_output))
                    for ee_output in ee_outputs
                ],
                dim=0,
            )
            u_t_hats = u_t_hats.mean(dim=(-1, -2, -3))
            L_u_t = F.mse_loss(classifier_outputs, u_t_hats, reduction="sum")

            # L_UAL_t
            L_n_t = torch.stack(
                [(ee_output - true_output) ** 2 for ee_output in ee_outputs], dim=0
            )
            L_n_t = L_n_t.mean(dim=(-1, -2, -3))

            # Trying u_t_hats, that is actually from 0 to 1 (classifier outputs is not!)
            L_UAL_t = ((1 - u_t_hats) * L_n_t).mean(dim=1).sum(dim=0)

            # New madeup loss
            # TODO: maybe weight with 1 - 1/l or something like that
            new_loss = L_n_t.mean(dim=1).sum(dim=0)

            # Total loss
            loss = L_simple, L_u_t, L_UAL_t, new_loss

        return loss
