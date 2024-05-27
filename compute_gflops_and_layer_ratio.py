import argparse
import os

import torch
from einops import rearrange

_FROM_EE_LAYER_TO_GFLOPS = {
    "attention_probe": [
        0.27,
        2.16,
        4.05,
        5.94,
        7.83,
        9.73,
        11.62,
        13.51,
        15.67,
        17.83,
        20,
        22.15,
        24.31,
        26.2,  # no early exit
    ],
    "mlp_probe": [
        0.0072,
        1.63,
        3.25,
        4.88,
        6.5,
        8.13,
        9.75,
        11.38,
        13.27,
        15.17,
        17.06,
        18.96,
        20.85,
        22.75,  # no early exit
    ],
}


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--indices_by_timestep_directory",
        type=str,
        default="output/attention_frozen/indices_by_timestep",
        help="Path to the directory where indices by timestep are saved",
    )

    return parser.parse_args()


def _read_files(folder_path):
    indices_by_timestep = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            assert file_path[-3:] == ".pt"
            indices_by_timestep.append(torch.load(file_path, map_location="cpu"))

    indices_by_timesteps = torch.stack(indices_by_timestep)
    indices_by_timesteps = rearrange(indices_by_timesteps, "N T bs -> T (N bs)").int()

    return indices_by_timesteps


if __name__ == "__main__":
    args = get_args()

    indices_by_timesteps = _read_files(args.indices_by_timestep_directory)

    if "attention" in args.indices_by_timestep_directory:
        classifier_type = "attention_probe"
    elif "mlp" in args.indices_by_timestep_directory:
        classifier_type = "mlp_probe"
    else:
        raise ValueError("The directory path is wrong")

    lookup_table = _FROM_EE_LAYER_TO_GFLOPS[classifier_type]
    lookup_table = torch.tensor(lookup_table)

    gflops_per_timestep_per_sample = lookup_table[indices_by_timesteps]  # (1000, 1024)

    # Average GFlops per sample
    avg_gflops_per_sample = gflops_per_timestep_per_sample.mean(axis=0)  # (1024,)

    print(
        f"Avg GFlops using {avg_gflops_per_sample.shape[0]} samples: {avg_gflops_per_sample.mean()}"
    )
    print(
        f"Std GFlops using {avg_gflops_per_sample.shape[0]} samples: {avg_gflops_per_sample.std()}"
    )

    # Average layer ratio
    average_layer_ratio_per_sample = (
        indices_by_timesteps.float().mean(axis=0) / 13.0
    )  # (1024,)

    average_layer_ratio = average_layer_ratio_per_sample.mean()
    print(
        f"Average layer ratio using {average_layer_ratio_per_sample.shape[0]} samples: {average_layer_ratio}"
    )
