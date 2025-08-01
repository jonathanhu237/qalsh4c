import argparse
import logging
import time
from pathlib import Path

import numpy as np
from utils import chamfer_distance, create_metadata, save_binary_data, setup_logging


def generate_point_set(
    num_points: int, num_dimensions: int, left_boundary: float, right_boundary: float
) -> np.ndarray:
    """Generate a point set with specified parameters using uniform distribution."""
    logging.info(
        f"Generating point set with {num_points} points and {num_dimensions} dimensions"
    )
    np.random.seed()
    points = np.random.uniform(
        left_boundary, right_boundary, (num_points, num_dimensions)
    )
    points = points.astype(np.double)
    return points


def main():
    parser = argparse.ArgumentParser(
        description="Generate a dataset for Chamfer distance estimation."
    )

    parser.add_argument(
        "-a",
        "--num_points_a",
        type=int,
        default=1000,
        help="Number of points in the first point set (A) (default: 1000)",
    )
    parser.add_argument(
        "-b",
        "--num_points_b",
        type=int,
        default=1000,
        help="Number of points in the second point set (B) (default: 1000)",
    )
    parser.add_argument(
        "-d",
        "--num_dimensions",
        type=int,
        default=256,
        help="Number of dimensions for the point sets (default: 256)",
    )
    parser.add_argument(
        "-l",
        "--left_boundary",
        type=float,
        default=-1024.0,
        help="The left boundary of uniform distribution used in dataset generation (default: -1024)",
    )
    parser.add_argument(
        "-r",
        "--right_boundary",
        type=float,
        default=1024.0,
        help="The right boundary of uniform distribution used in dataset generation (default: 1024)",
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        type=str,
        required=True,
        help="The directory to save dataset",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of processes for parallel computation (default: auto-detect)",
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARN", "ERROR"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(log_level)

    logging.info("Starting dataset generation")
    total_start_time = time.time()

    # Create output directory
    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    # Generate the point sets
    logging.info("Generating point set A...")
    A = generate_point_set(
        args.num_points_a, args.num_dimensions, args.left_boundary, args.right_boundary
    )

    logging.info("Generating point set B...")
    B = generate_point_set(
        args.num_points_b, args.num_dimensions, args.left_boundary, args.right_boundary
    )

    # Save binary files
    save_binary_data(A, output_dir / "A.bin")
    save_binary_data(B, output_dir / "B.bin")

    # Calculate the Chamfer distance between two sets
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
        f"Dataset generation complete! (Total time: {total_elapsed_time:.2f} seconds)"
    )
    logging.info(f"""Files saved to: {output_dir}
    A.bin: {A.shape[0]} points x {A.shape[1]} dimensions
    B.bin: {B.shape[0]} points x {B.shape[1]} dimensions
    metadata.json: Chamfer distance = {chamfer_dist:.6f}""")


if __name__ == "__main__":
    main()
