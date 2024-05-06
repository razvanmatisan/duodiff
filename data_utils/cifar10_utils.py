from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


def get_dataloader(data_dir="data/cifar10/", batch_size=16):
    """
    Builds a dataloader with all training images from the CIFAR-10 dataset.
    Args:
        data_dir: Directory where the data is stored.
        batch_size: Size of the batches

    Returns:
        DataLoader: DataLoader object containing the dataset.

    """

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    return DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
