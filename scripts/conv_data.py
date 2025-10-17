import argparse
import time
from pathlib import Path

import load
import numpy as np
import utils
from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description="Convert known datasets to unified format"
    )
    parser.add_argument(
        "-n",
        "--dataset-name",
        required=True,
        type=str,
        choices=["mnist", "p53", "trevi", "gist"],
        help="Name of the dataset to convert",
    )
    parser.add_argument(
        "-a",
        "--num-points-a",
        type=int,
        default=None,
        help="Number of points in set A (set B will contain remaining points)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=str,
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for Chamfer distance calculation",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Path to the raw dataset directory",
    )

    args = parser.parse_args()

    logger.info("Starting dataset to Chamfer distance conversion")
    total_start_time = time.time()

    # Load dataset
    if args.dataset_name == "mnist":
        X = load.mnist()
    elif args.dataset_name == "trevi":
        if not args.raw_dir:
            raise ValueError(
                "Path to raw dataset directory must be provided for trevi dataset"
            )
        X = load.trevi(Path(args.raw_dir))
    elif args.dataset_name == "p53":
        if not args.raw_dir:
            raise ValueError(
                "Path to raw dataset directory must be provided for p53 dataset"
            )
        X = load.p53(Path(args.raw_dir))
    elif args.dataset_name == "sift":
        if not args.raw_dir:
            raise ValueError(
                "Path to raw dataset directory must be provided for sift dataset"
            )
        X = load.sift(Path(args.raw_dir))
    elif args.dataset_name == "gist":
        if not args.raw_dir:
            raise ValueError(
                "Path to raw dataset directory must be provided for gist dataset"
            )
        X = load.gist(Path(args.raw_dir))
    else:
        logger.error(f"Unknown dataset: {args.dataset_name}")
        return

    # Split data into sets A and B
    n_total = len(X)

    # Determine number of points in A
    if args.num_points_a is not None:
        n_a = args.num_points_a
        if n_a >= n_total:
            raise ValueError(
                f"Number of points in A ({n_a}) must be less than total points ({n_total})"
            )
    else:
        # If not specified, split in half
        n_a = n_total // 2

    n_b = n_total - n_a

    # Shuffle the data first
    indices = np.random.permutation(n_total)
    X_shuffled = X[indices]

    # Split into A and B
    A = X_shuffled[:n_a]
    B = X_shuffled[n_a : n_a + n_b]

    # Process and save dataset
    output_dir = Path(args.output_dir)
    utils.process_and_save_dataset(A, B, output_dir, args.batch_size)

    total_elapsed_time = time.time() - total_start_time
    logger.info(
        f"Dataset conversion complete! (Total time: {total_elapsed_time:.2f} seconds)"
    )


if __name__ == "__main__":
    main()
