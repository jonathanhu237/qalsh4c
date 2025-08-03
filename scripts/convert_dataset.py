import argparse
import logging
import time
from pathlib import Path

import numpy as np
from chamfer_distance import chamfer_distance
from sklearn.datasets import fetch_openml
from utils import create_metadata, save_binary_data, setup_logging


def load_mnist() -> np.ndarray:
    """Load MNIST dataset (70,000 images total)."""
    logging.info("Loading MNIST dataset...")
    X, _ = fetch_openml(
        "mnist_784", version=1, as_frame=False, parser="auto", return_X_y=True
    )

    # Convert to double
    X = X.astype(np.double)

    logging.info(f"Loaded {X.shape[0]} images with {X.shape[1]} features each")
    return X


def load_p53(file_path: Path) -> np.ndarray:
    """Load p53 dataset."""
    logging.info("Loading p53 dataset...")

    data = []

    with open(file_path, "r") as file:
        for line_num, line in enumerate(file, 1):
            # Skip empty lines
            if not line.strip():
                continue

            # Skip lines containing "?"
            if "?" in line:
                continue

            # Split the line by comma
            entries = line.strip().split(",")

            # Convert all entries except the last one to double
            try:
                # All entries except the last two (label and empty value)
                point = [float(entry) for entry in entries[:-2]]
                data.append(point)
            except ValueError as e:
                logging.warning(
                    f"Skipping line {line_num}: Error converting to float - {e}"
                )
                continue

    # Convert to numpy array
    if data:
        X = np.array(data, dtype=np.double)
        logging.info(f"Loaded {X.shape[0]} points with {X.shape[1]} features each")
        return X
    else:
        logging.error("No valid data found in the file")
        raise ValueError("No valid data found in the file")


def main():
    parser = argparse.ArgumentParser(
        description="Convert known datasets to unified format"
    )
    parser.add_argument(
        "-n",
        "--dataset-name",
        required=True,
        type=str,
        choices=["mnist", "p53"],
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

    args = parser.parse_args()

    # Setup logging
    setup_logging(logging.getLevelNamesMapping()[args.log_level])

    logging.info("Starting MNIST to Chamfer distance conversion")
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
