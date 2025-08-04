import argparse
import logging
import time
from pathlib import Path

import numpy as np
from chamfer_distance import chamfer_distance
from mnist import load_mnist
from p53 import load_p53
from trevi import load_trevi
from gist import load_gist
from utils import create_metadata, save_binary_data, setup_logging


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
        "-l",
        "--log-level",
        choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"],
        default="WARN",
        help="Set the logging level (default: WARN)",
    )
    parser.add_argument(
        "--p53-file",
        type=str,
        default=None,
        help="Path to the p53 dataset file",
    )
    parser.add_argument(
        "--trevi-dir",
        type=str,
        default=None,
        help="Path to the Trevi dataset directory containing .bmp files",
    )
    parser.add_argument(
        "--gist-dir",
        type=str,
        default=None,
        help="Path to the GIST dataset directory containing gist_base.fvecs file",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(logging.getLevelNamesMapping()[args.log_level])

    logging.info("Starting dataset to Chamfer distance conversion")
    total_start_time = time.time()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    # Load dataset
    if args.dataset_name == "mnist":
        X = load_mnist()
    elif args.dataset_name == "p53":
        if not args.p53_file:
            raise ValueError(
                "Path to p53 dataset file must be provided for p53 dataset"
            )
        X = load_p53(args.p53_file)
    elif args.dataset_name == "trevi":
        # For Trevi dataset, we need a directory containing .bmp files
        if not args.trevi_dir:
            raise ValueError(
                "Path to Trevi dataset directory must be provided for trevi dataset (use --trevi-dir argument)"
            )
        X = load_trevi(Path(args.trevi_dir))
    elif args.dataset_name == "gist":
        # For GIST dataset, we need a directory containing gist_base.fvecs file
        if not args.gist_dir:
            raise ValueError(
                "Path to GIST dataset directory must be provided for gist dataset (use --gist-dir argument)"
            )
        X = load_gist(Path(args.gist_dir))
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

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

    logging.info(f"""Split data into:
    Set A: {A.shape[0]} points
    Set B: {B.shape[0]} points
    Dimensions: {A.shape[1]}""")

    # Save binary files
    save_binary_data(A, output_dir / "A.bin")
    save_binary_data(B, output_dir / "B.bin")

    # Calculate Chamfer distance
    if args.batch_size is None:
        batch_size = 1024
    else:
        batch_size = args.batch_size
    chamfer_dist = chamfer_distance(A, B, batch_size)

    # Create metadata
    create_metadata(
        chamfer_dist=chamfer_dist,
        num_points_a=A.shape[0],
        num_points_b=B.shape[0],
        num_dimensions=A.shape[1],
        filepath=output_dir / "metadata.json",
    )

    total_elapsed_time = time.time() - total_start_time
    logging.info(
        f"Dataset conversion complete! (Total time: {total_elapsed_time:.2f} seconds)"
    )
    logging.info(f"""Files saved to: {output_dir}
    A.bin: {A.shape[0]} points x {A.shape[1]} dimensions
    B.bin: {B.shape[0]} points x {B.shape[1]} dimensions
    metadata.json: Chamfer distance = {chamfer_dist:.6f}""")


if __name__ == "__main__":
    main()
