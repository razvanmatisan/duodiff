from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA

from datasets.sampler import ResumableRandomSampler


def get_celeba_dataloader(batch_size, seed, data_dir="data/"):
    """
    Builds a dataloader with all images from the CelebA dataset.
    Args:
        data_dir: Directory where the data is stored.
        batch_size: Size of the batches

    Returns:
        DataLoader: DataLoader object containing the dataset.

    """
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.CenterCrop((178, 178)),
            transforms.Resize((64, 64)),
        ]
    )

    dataset = CelebA(
        root=data_dir, split="all", download=True, transform=data_transforms
    )

    sampler = ResumableRandomSampler(dataset, seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        sampler=sampler,
    )
