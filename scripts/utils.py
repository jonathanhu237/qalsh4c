import json
import logging
import time
from pathlib import Path

import numpy as np
import torch


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
    num_dimensions: int,
    num_points_a: int,
    num_points_b: int,
    chamfer_distance_l1: np.double,
    chamfer_distance_l2: np.double,
    filepath: Path,
) -> None:
    """Create metadata.json file."""
    metadata = {
        "num_dimensions": num_dimensions,
        "num_points_a": num_points_a,
        "num_points_b": num_points_b,
        "chamfer_distance_l1": chamfer_distance_l1,
        "chamfer_distance_l2": chamfer_distance_l2,
    }

    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=4)

    logging.info(f"Saved metadata to {filepath}")


def chamfer_distance(
    A: np.ndarray, B: np.ndarray, batch_size: int, p: float
) -> np.double:
    """
    Calculate Chamfer distance using PyTorch on a GPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("CUDA is available. Using GPU for computation.")
    else:
        device = torch.device("cpu")
        logging.info("CUDA not available. Using CPU for computation.")

    logging.info("Calculating Chamfer distance using GPU (PyTorch)...")
    start_time = time.time()
    dtype = torch.double

    # Move data
    tensor_A = torch.from_numpy(A).to(device, dtype=dtype)
    tensor_B = torch.from_numpy(B).to(device, dtype=dtype)

    # For each point in p1, find the closest in p2
    min_dists_A_to_B = torch.zeros(len(tensor_A), device=device, dtype=dtype)
    for i in range(0, len(tensor_A), batch_size):
        p1_batch = tensor_A[i : i + batch_size]
        dists = torch.cdist(p1_batch, tensor_B, p=p)
        min_dists, _ = torch.min(dists, dim=1)
        min_dists_A_to_B[i : i + batch_size] = min_dists

    # For each point in p2, find the closest in p1
    min_dists_B_to_A = torch.zeros(len(tensor_B), device=device, dtype=dtype)
    for i in range(0, len(tensor_B), batch_size):
        p2_batch = tensor_B[i : i + batch_size]
        dists = torch.cdist(p2_batch, tensor_A, p=p)
        min_dists, _ = torch.min(dists, dim=1)
        min_dists_B_to_A[i : i + batch_size] = min_dists

    # Sum the distances
    chamfer_dist = torch.sum(min_dists_A_to_B) + torch.sum(min_dists_B_to_A)

    elapsed_time = time.time() - start_time
    logging.info(
        f"Chamfer distance: {chamfer_dist.item():.6f} (computed in {elapsed_time:.2f} seconds)"
    )

    return np.double(chamfer_dist.item())
