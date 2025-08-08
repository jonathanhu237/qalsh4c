import argparse
import logging
import time
from pathlib import Path

import numpy as np
from utils import chamfer_distance, create_metadata, save_binary_data, setup_logging


def generate_point_set(
    num_points: int,
    num_dimension: int,
    outlier_fraction: float,
    offset: float,
    spread: float,
) -> np.ndarray:
    """Generate a point set with specified parameters using two Gaussian distributions."""
    logging.info(
        f"Generating {num_points} points with 1 core component and {outlier_fraction * 100}% outliers."
    )

    # Create two center points
    # c1 should be located at the origin
    c1 = np.zeros(num_dimension)

    # c2 should be located such that its L1 distance from c1 is exactly offset
    # Generate a random direction vector, normalize to unit L1 norm, then scale by offset
    c2_direction = np.random.randn(num_dimension)
    c2_direction = np.abs(c2_direction)  # Make all components positive
    l1_norm = np.sum(c2_direction)  # Calculate L1 norm
    c2_direction = c2_direction / l1_norm  # Normalize to unit L1 norm
    c2 = c2_direction * offset  # Scale to desired L1 distance

    # Calculate number of points for each distribution
    num_core_points = int((1 - outlier_fraction) * num_points)
    num_outlier_points = num_points - num_core_points

    # Generate points sampled from a Gaussian distribution centered at c1
    core_points = np.random.normal(c1, spread, (num_core_points, num_dimension))

    # Generate points sampled from a Gaussian distribution centered at c2
    outlier_points = np.random.normal(c2, spread, (num_outlier_points, num_dimension))

    # Combine the points
    points = np.vstack([core_points, outlier_points])

    # Shuffle the points to mix core and outlier points
    np.random.shuffle(points)

    points = points.astype(np.double)
    return points


def main():
    parser = argparse.ArgumentParser(
        description="Generate a dataset for Chamfer distance estimation using Gaussian Mixture Models."
    )

    parser.add_argument(
        "-a",
        "--num-points-a",
        type=int,
        default=1000,
        help="Number of points in the first point set (A)",
    )
    parser.add_argument(
        "-b",
        "--num-points-b",
        type=int,
        default=1000,
        help="Number of points in the second point set (B)",
    )
    parser.add_argument(
        "-d",
        "--num-dimensions",
        type=int,
        default=256,
        help="Number of dimensions for the point sets",
    )
    parser.add_argument(
        "-f",
        "--outlier-fraction",
        type=float,
        default=0.1,
        help="Fraction of outliers in the dataset",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=1024.0,
        help="L1 distance between the two center points",
    )
    parser.add_argument(
        "--spread",
        type=float,
        default=16.0,
        help="Standard deviation for the Gaussian distributions",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        type=str,
        required=True,
        help="The directory to save dataset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for Chamfer distance calculation",
    )
    parser.add_argument(
        "--log-level",
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
        args.num_points_a,
        args.num_dimensions,
        args.outlier_fraction,
        args.offset,
        args.spread,
    )

    logging.info("Generating point set B...")
    B = generate_point_set(
        args.num_points_b,
        args.num_dimensions,
        args.outlier_fraction,
        args.offset,
        args.spread,
    )

    # Save binary files
    save_binary_data(A, output_dir / "A.bin")
    save_binary_data(B, output_dir / "B.bin")

    # Calculate the Chamfer distance between two sets
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
        f"Dataset generation complete! (Total time: {total_elapsed_time:.2f} seconds)"
    )
    logging.info(f"""Files saved to: {output_dir}
    A.bin: {A.shape[0]} points x {A.shape[1]} dimensions
    B.bin: {B.shape[0]} points x {B.shape[1]} dimensions
    metadata.json: Chamfer distance = {chamfer_dist:.6f}
    Core component: {100 * (1 - args.outlier_fraction):.1f}% of points
    Outlier component: {100 * args.outlier_fraction:.1f}% of points""")


if __name__ == "__main__":
    main()
