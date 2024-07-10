import argparse

from torchmetrics.image.fid import FrechetInceptionDistance

from utils.evaluation_utils import read_samples, get_dataset_samples


def get_args():
    parser = argparse.ArgumentParser(description="FID evaluation parameters")

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

    return parser.parse_args()


def fid_evaluation(real_images, generated_images):
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)
    print("Evaluating FID")
    print(f"FID: {float(fid.compute())}")


def main():
    args = get_args()
    generated_images = read_samples(args.samples_path)
    n_samples = len(generated_images)
    real_images = get_dataset_samples(
        args.dataset, args.data_path, args.seed, n_samples
    )

    fid_evaluation(real_images, generated_images)


if __name__ == "__main__":
    main()
