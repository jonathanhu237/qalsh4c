import json
import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist


def setup_logging(level: int) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def save_binary_data(data: np.ndarray, filepath: Path) -> None:
    """Save numpy array as binary file with float32 format."""
    logging.info(f"Saving binary data to {filepath}")

    # Ensure data is double
    data = data.astype(np.double)

    # Save as binary file
    with open(filepath, "wb") as f:
        # Write the data directly as binary
        f.write(data.tobytes())

    logging.info(f"Saved {data.shape[0]} points with {data.shape[1]} dimensions")


def create_metadata(
    chamfer_dist: np.double,
    num_points_a: int,
    num_points_b: int,
    num_dimensions: int,
    filepath: Path,
) -> None:
    """Create metadata.json file."""
    metadata = {
        "chamfer_distance": chamfer_dist,
        "num_dimensions": num_dimensions,
        "num_points_a": num_points_a,
        "num_points_b": num_points_b,
    }

    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=4)

    logging.info(f"Saved metadata to {filepath}")


def compute_min_distances_chunk(args: Tuple[np.ndarray, np.ndarray, int]) -> np.ndarray:
    """
    Compute minimum distances for a chunk of points from set A to all points in set B.
    This function is designed to be used with multiprocessing.
    """
    from_chunk, to, chunk_idx = args
    logging.debug(f"Processing chunk {chunk_idx} with {len(from_chunk)} points")

    # Calculate distances from this chunk of A to all points in B
    distances = cdist(from_chunk, to, metric="cityblock")

    # Find minimum distances for each point in the chunk
    min_distances = np.min(distances, axis=1)

    logging.debug(f"Completed chunk {chunk_idx}")
    return min_distances


def chamfer_distance(
    A: np.ndarray, B: np.ndarray, n_processes: Optional[int]
) -> np.double:
    """
    Calculate Chamfer distance between two point sets using multiprocessing.
    """
    if n_processes is None:
        n_processes = mp.cpu_count()

    logging.info(f"Calculating Chamfer distance using {n_processes} processes...")
    start_time = time.time()

    # Split A into chunks for parallel processing
    chunk_size = len(A) // (n_processes)  # Create more chunks than processes
    A_chunks = [A[i : i + chunk_size] for i in range(0, len(A), chunk_size)]
    logging.info(f"Split set A into {len(A_chunks)} chunks of size {chunk_size}")

    # Calculate min distances from A to B
    logging.info("Computing minimum distances from A to B...")
    args_A_to_B = [(chunk, B, i) for i, chunk in enumerate(A_chunks)]
    with mp.Pool(processes=n_processes) as pool:
        min_dist_chunks_A_to_B = pool.map(compute_min_distances_chunk, args_A_to_B)

    # Split B into chunks for parallel processing
    chunk_size = len(B) // (n_processes)
    B_chunks = [B[i : i + chunk_size] for i in range(0, len(B), chunk_size)]
    logging.info(f"Split set B into {len(B_chunks)} chunks of size {chunk_size}")

    # Calculate min distances from B to A
    logging.info("Computing minimum distances from B to A...")
    args_B_to_A = [(chunk, A, i) for i, chunk in enumerate(B_chunks)]
    with mp.Pool(processes=n_processes) as pool:
        min_dist_chunks_B_to_A = pool.map(compute_min_distances_chunk, args_B_to_A)

    # Chamfer distance is the sum of minimum distances
    min_dist_A_to_B = np.concatenate(min_dist_chunks_A_to_B)
    min_dist_B_to_A = np.concatenate(min_dist_chunks_B_to_A)
    chamfer_dist = np.sum(min_dist_A_to_B) + np.sum(min_dist_B_to_A)

    elapsed_time = time.time() - start_time
    logging.info(
        f"Chamfer distance: {chamfer_dist:.6f} (computed in {elapsed_time:.2f} seconds)"
    )
    return np.double(chamfer_dist)
