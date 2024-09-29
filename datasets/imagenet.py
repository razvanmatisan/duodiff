from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from datasets.sampler import ResumableSeedableSampler


# https://www.kaggle.com/datasets/dimensi0n/imagenet-256
def get_imagenet_dataloader(
    batch_size,
    seed,
    data_dir,
    resize: bool,  # resizing to 64x64
    normalize: bool = True,
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

    # All images from the dataset are 256x256 resolution
    transformations = [transforms.ToTensor()]

    if normalize:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

        transformations.append(transforms.Normalize(mean, std))

    if resize:
        transformations.append(transforms.Resize((64, 64)))

    transform = transforms.Compose(transformations)

    path = Path(data_dir) / "imagenet"

    dataset = datasets.ImageFolder(root=path, transform=transform)

    sampler = ResumableSeedableSampler(dataset, seed=seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        sampler=sampler,
        num_workers=36,
    )
