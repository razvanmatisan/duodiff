from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from datasets.sampler import ResumableSeedableSampler


def get_cifar10_dataloader(batch_size, seed, data_dir, normalize: bool = True):
    """
    Builds a dataloader with all training images from the CIFAR-10 dataset.
    Args:
        data_dir: Directory where the data is stored.
        batch_size: Size of the batches

    Returns:
        DataLoader: DataLoader object containing the dataset.

    """
    if normalize:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
    else:
        transform = transforms.ToTensor()

    path = Path(data_dir) / "cifar10"

    dataset = CIFAR10(root=path, train=True, download=True, transform=transform)

    sampler = ResumableSeedableSampler(dataset, seed=seed)

    return DataLoader(
        dataset=dataset, batch_size=batch_size, drop_last=True, sampler=sampler
    )
