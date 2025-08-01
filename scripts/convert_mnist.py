import argparse
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from utils import chamfer_distance, create_metadata, save_binary_data, setup_logging


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


def main():
    parser = argparse.ArgumentParser(
        description="Convert MNIST to Chamfer distance format"
    )
    parser.add_argument(
        "--output-dir", default="data/mnist", help="Output directory for the dataset"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of processes for parallel computation (default: auto-detect)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    logging.info("Starting MNIST to Chamfer distance conversion")
    total_start_time = time.time()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    # Load MNIST dataset
    X = load_mnist()

    # Split data into two equal sets A and B
    n_total = len(X)
    n_half = n_total // 2

    # Shuffle the data first
    indices = np.random.permutation(n_total)
    X_shuffled = X[indices]

    # Split into A and B
    A = X_shuffled[:n_half]
    B = X_shuffled[n_half : 2 * n_half]  # Ensure equal sizes

    logging.info(f"""Split data into:
    Set A: {A.shape[0]} points
    Set B: {B.shape[0]} points
    Dimensions: {A.shape[1]}""")

    # Save binary files
    save_binary_data(A, output_dir / "A.bin")
    save_binary_data(B, output_dir / "B.bin")

    # Calculate Chamfer distance
    chamfer_dist = chamfer_distance(A, B, args.processes)

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
