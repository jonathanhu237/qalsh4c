import argparse
import json
import logging
import os

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def generate_point_set(
    num_points: int,
    num_dimension: int,
    left_boundary: int,
    right_boundary: int,
    file_path: str,
) -> np.ndarray:
    point_set = np.random.uniform(
        low=left_boundary, high=right_boundary, size=(num_points, num_dimension)
    )
    point_set.astype(np.float32).tofile(file_path)
    return point_set


def compute_chamfer_distance(point_set_a: np.ndarray, point_set_b: np.ndarray) -> float:
    """Compute the exact Chamfer distance between two point sets."""
    # Distance from each point in A to closest point in B
    distances_a_to_b = cdist(point_set_a, point_set_b, metric="cityblock")
    min_distances_a_to_b = np.min(distances_a_to_b, axis=1)

    # Distance from each point in B to closest point in A
    distances_b_to_a = cdist(point_set_b, point_set_a, metric="cityblock")
    min_distances_b_to_a = np.min(distances_b_to_a, axis=1)

    # Chamfer distance is the sum of both directions
    chamfer_distance = np.sum(min_distances_a_to_b) + np.sum(min_distances_b_to_a)
    return float(chamfer_distance)


def generate_dataset(
    num_points_a: int,
    num_points_b: int,
    num_dimensions: int,
    left_boundary: int,
    right_boundary: int,
    output_directory: str,
):
    """Generate a dataset with two point sets and compute their Chamfer distance."""
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Generate point sets and save to binary files
    point_set_a_path = os.path.join(output_directory, "A.bin")
    point_set_b_path = os.path.join(output_directory, "B.bin")

    logger.info(f"Generating point set A with {num_points_a} points...")
    point_set_a = generate_point_set(
        num_points_a, num_dimensions, left_boundary, right_boundary, point_set_a_path
    )

    logger.info(f"Generating point set B with {num_points_b} points...")
    point_set_b = generate_point_set(
        num_points_b, num_dimensions, left_boundary, right_boundary, point_set_b_path
    )

    # Compute exact Chamfer distance
    logger.info("Computing exact Chamfer distance...")
    chamfer_distance = compute_chamfer_distance(point_set_a, point_set_b)

    # Create metadata
    metadata = {
        "num_points_a": num_points_a,
        "num_points_b": num_points_b,
        "num_dimensions": num_dimensions,
        "chamfer_distance": chamfer_distance,
    }

    # Save metadata to JSON file
    metadata_path = os.path.join(output_directory, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dataset generated successfully in {output_directory}")
    logger.info(f"Chamfer distance: {chamfer_distance:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a pair of point sets for Chamfer distance approximation."
    )

    parser.add_argument(
        "--num_points_a",
        type=int,
        default=1000,
        help="Number of points in the first point set (A).",
    )
    parser.add_argument(
        "--num_points_b",
        type=int,
        default=1000,
        help="Number of points in the second point set (B).",
    )
    parser.add_argument(
        "--num_dimensions",
        type=int,
        default=256,
        help="Number of dimension for the point sets.",
    )
    parser.add_argument(
        "--left_boundary",
        type=float,
        default=-1024,
        help="The left boundary of uniform distribution used in dataset generation",
    )
    parser.add_argument(
        "--right_boundary",
        type=float,
        default=1024,
        help="The right boundary of uniform distribution used in dataset generation",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        required=True,
        help="The directory to save dataset.",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Generate the dataset
    generate_dataset(
        num_points_a=args.num_points_a,
        num_points_b=args.num_points_b,
        num_dimensions=args.num_dimensions,
        left_boundary=args.left_boundary,
        right_boundary=args.right_boundary,
        output_directory=args.output_directory,
    )
