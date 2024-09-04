from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from datasets.celeba import get_celeba_dataloader
from datasets.cifar10 import get_cifar10_dataloader
from datasets.imagenet import get_imagenet_dataloader


def read_samples(path):
    transform = transforms.Compose([transforms.ToTensor()])

    tensor_list = []
    for p in Path(path).rglob("*.png"):
        if "grid" not in str(p).split("/")[-1]:
            img = Image.open(p).convert("RGB")
            tensor_list.append(transform(img))

    stacked_tensor = torch.stack(tensor_list, dim=0)
    print(f"Read {len(stacked_tensor)} images")
    return stacked_tensor


def get_dataset_samples(dataset_name, data_path, seed, n_samples):
    if dataset_name == "cifar10":
        dataset = get_cifar10_dataloader(n_samples, seed, data_path, normalize=False)
    elif dataset_name == "celeba":
        dataset = get_celeba_dataloader(n_samples, seed, data_path, normalize=False)
    elif dataset_name == "imagenet64":
        dataset = get_imagenet_dataloader(n_samples, seed, data_path, normalize=False, resize=True)
    elif dataset_name == "imagenet256":
        dataset = get_imagenet_dataloader(n_samples, seed, data_path, normalize=False, resize=False)
    else:
        raise ValueError("Incorrect dataset name")

    return next(iter(dataset))[0]


def save_images(images, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(images):
        filename = Path(path) / f"{idx}.png"
        save_image(img, filename)
