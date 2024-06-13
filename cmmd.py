import argparse
from pathlib import Path

from CMMD_evaluation.main import compute_cmmd
from utils.evaluation_utils import get_dataset_samples, save_images


def get_args():
    parser = argparse.ArgumentParser(description="CMMD evaluation parameters")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cifar10", "celeba"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling images from the dataset."
    )
    parser.add_argument(
        "--data_path", type=str, default="data", help="Directory for datasets"
    )
    parser.add_argument(
        "--samples_path",
        type=str,
        required=True,
        help="Path to the directory with samples.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the directory dataset samples.",
    )

    # CMMD metric specific parameters
    parser.add_argument(
        "--cmmd_batch_size",
        type=int,
        default=32,
        help="Batch size for embedding generation.",
    )

    parser.add_argument(
        "--cmmd_max_count",
        type=int,
        default=1024,
        help="Maximum number of images to read from each directory.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    n_samples = len(list(Path(args.samples_path).rglob("*.png")))
    real_images = get_dataset_samples(
        args.dataset, args.data_path, args.seed, n_samples
    )
    save_images(real_images, args.dataset_path)

    cmmd = compute_cmmd(
        args.dataset_path,
        args.samples_path,
        batch_size=args.cmmd_batch_size,
        max_count=args.cmmd_max_count,
    )
    print(f"The CMMD value is {cmmd}")


if __name__ == "__main__":
    main()
