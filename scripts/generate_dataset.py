import argparse
import logging
import time
from pathlib import Path

import numpy as np
from chamfer_distance import chamfer_distance
from utils import create_metadata, save_binary_data, setup_logging
from sklearn.mixture import GaussianMixture


def generate_point_set(
    num_points: int,
    num_dimensions: int,
    outlier_fraction: float,
    outlier_scale: float,
) -> np.ndarray:
    """Generate a point set with specified parameters using Gaussian Mixture Model."""
    total_components = 2  # 1 core component + 1 outlier component
    logging.info(
        f"Generating {num_points} points with 1 core component and {outlier_fraction * 100}% outliers."
    )

    # Create a Gaussian Mixture Model
    gmm = GaussianMixture(
        n_components=total_components, random_state=np.random.randint(0, 10000)
    )

    # Generate random means for each component within a reasonable range
    means = np.random.uniform(-512, 512, (total_components, num_dimensions))

    # Generate one core component covariance
    covariances = []
    A = np.random.randn(num_dimensions, num_dimensions)
    core_cov = np.dot(A, A.T) + np.eye(num_dimensions)
    covariances.append(core_cov)

    # Create a scaled-up covariance for the outlier component
    outlier_cov = core_cov * outlier_scale
    covariances.append(outlier_cov)
    covariances = np.array(covariances)

    # Generate weights based on outlier fraction
    core_weight = 1 - outlier_fraction  # Weight for the single core component
    weights = np.array([core_weight, outlier_fraction])  # [core_weight, outlier_weight]

    # Set the GMM parameters
    gmm.means_ = means
    gmm.covariances_ = np.array(covariances)
    gmm.weights_ = weights
    gmm.precisions_cholesky_ = np.array(
        [np.linalg.cholesky(np.linalg.inv(cov)).T for cov in covariances]
    )

    # Generate samples
    points, _ = gmm.sample(num_points)
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
        help="Number of points in the first point set (A) (default: 1000)",
    )
    parser.add_argument(
        "-b",
        "--num-points-b",
        type=int,
        default=1000,
        help="Number of points in the second point set (B) (default: 1000)",
    )
    parser.add_argument(
        "-d",
        "--num-dimensions",
        type=int,
        default=256,
        help="Number of dimensions for the point sets (default: 256)",
    )
    parser.add_argument(
        "-f",
        "--outlier_fraction",
        type=float,
        default=0.1,
        help="Fraction of outliers in the dataset (default: 0.1)",
    )
    parser.add_argument(
        "-s",
        "--outlier-scale",
        type=float,
        default=4096,
        help="Scale factor for the outlier covariance (default: 4096)",
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
        args.outlier_scale,
    )

    logging.info("Generating point set B...")
    B = generate_point_set(
        args.num_points_b,
        args.num_dimensions,
        args.outlier_fraction,
        args.outlier_scale,
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
    GMM components: 1 core + 1 outlier""")


if __name__ == "__main__":
    main()
