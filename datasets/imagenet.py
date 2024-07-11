from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from datasets.sampler import ResumableSeedableSampler


# https://www.kaggle.com/datasets/dimensi0n/imagenet-256
def get_imagenet_dataloader(
    batch_size,
    seed,
    data_dir="./archive",
):
    """
    Builds a dataloader with all images from a 540k subset of ImageNet (with 256x256 resolution).
    Args:
        data_dir: Root directory where the data is stored.
        seed: The seed for ResumableSeedableSampler
        batch_size: Size of the batches

    Returns:
        DataLoader: DataLoader object containing the dataset.
    """

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    # All images from the dataset are 256x256 resolution
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    sampler = ResumableSeedableSampler(dataset, seed)

    return DataLoader(
        dataset=dataset, batch_size=batch_size, drop_last=True, sampler=sampler
    )
