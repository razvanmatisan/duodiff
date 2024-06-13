from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image

from datasets.cifar10 import get_cifar10_dataloader
from datasets.celeba import get_celeba_dataloader


def read_samples(path):
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    tensor_list = []
    for p in Path(path).rglob("*.png"):
        img = Image.open(p).convert("RGB")
        tensor_list.append(transform(img))

    stacked_tensor = torch.stack(tensor_list, dim=0)
    print(f"Read {len(stacked_tensor)} images")
    return stacked_tensor


def get_dataset_samples(dataset_name, data_path, seed, n_samples):
    if dataset_name == "cifar10":
        dataset = get_cifar10_dataloader(n_samples, seed, data_path)
    elif dataset_name == "celeba":
        dataset = get_celeba_dataloader(n_samples, seed, data_path)
    else:
        raise ValueError("Incorrect dataset name")

    return next(iter(dataset))[0]


def save_images(images, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(images):
        filename = Path(path) / f"{idx}.png"
        save_image(img, filename)
