import json
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger


def save_binary_data(data: np.ndarray, filepath: Path) -> None:
    """Save numpy array as binary file with float32 format."""
    logger.info(f"Saving binary data to {filepath}")

    # Ensure data is double
    data = data.astype(np.double)

    # Save as binary file
    with open(filepath, "wb") as f:
        # Write the data directly as binary
        f.write(data.tobytes())

    logger.info(f"Saved {data.shape[0]} points with {data.shape[1]} dimensions")


def create_metadata(
    num_dimensions: int,
    num_points_a: int,
    num_points_b: int,
    chamfer_distance_l1_ab: np.double,
    chamfer_distance_l1_ba: np.double,
    chamfer_distance_l2_ab: np.double,
    chamfer_distance_l2_ba: np.double,
    filepath: Path,
) -> None:
    """Create metadata.json file."""
    metadata = {
        "num_dimensions": num_dimensions,
        "num_points": {
            "a": num_points_a,
            "b": num_points_b,
        },
        "chamfer_distances": {
            "l1": {
                "ab": chamfer_distance_l1_ab,
                "ba": chamfer_distance_l1_ba,
            },
            "l2": {
                "ab": chamfer_distance_l2_ab,
                "ba": chamfer_distance_l2_ba,
            },
        },
    }

    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Saved metadata to {filepath}")


def chamfer_distance(
    A: np.ndarray, B: np.ndarray, batch_size: int, p: float
) -> np.double:
    """
    Calculate Chamfer distance using PyTorch on a GPU.
    """
    chamfer_distance_ab, chamfer_distance_ba = chamfer_distance_components(
        A, B, batch_size, p
    )
    return np.double(chamfer_distance_ab + chamfer_distance_ba)


def directed_chamfer_distance(
    from_points: torch.Tensor,
    to_points: torch.Tensor,
    batch_size: int,
    p: float,
) -> torch.Tensor:
    """Calculate one directed Chamfer component."""
    min_dists = torch.zeros(len(from_points), device=from_points.device, dtype=from_points.dtype)
    for i in range(0, len(from_points), batch_size):
        from_batch = from_points[i : i + batch_size]
        batch_min_dists = torch.full(
            (from_batch.size(0),), float("inf"), device=from_points.device, dtype=from_points.dtype
        )
        for j in range(0, len(to_points), batch_size):
            to_batch = to_points[j : j + batch_size]
            dists = torch.cdist(from_batch, to_batch, p=p)
            current_min_dists, _ = torch.min(dists, dim=1)
            batch_min_dists = torch.min(batch_min_dists, current_min_dists)
        min_dists[i : i + batch_size] = batch_min_dists

    return torch.sum(min_dists)


def chamfer_distance_components(
    A: np.ndarray, B: np.ndarray, batch_size: int, p: float
) -> tuple[np.double, np.double]:
    """
    Calculate directed Chamfer distance components using PyTorch.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("CUDA is available. Using GPU for computation.")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available. Using CPU for computation.")

    logger.info("Calculating Chamfer distance using GPU (PyTorch)...")
    start_time = time.time()
    dtype = torch.double

    # Move data
    tensor_A = torch.from_numpy(A).to(device, dtype=dtype)
    tensor_B = torch.from_numpy(B).to(device, dtype=dtype)

    chamfer_dist_ab = directed_chamfer_distance(tensor_A, tensor_B, batch_size, p)
    chamfer_dist_ba = directed_chamfer_distance(tensor_B, tensor_A, batch_size, p)
    chamfer_dist = chamfer_dist_ab + chamfer_dist_ba

    elapsed_time = time.time() - start_time
    logger.info(
        f"Chamfer distance: {chamfer_dist.item():.6f} (computed in {elapsed_time:.2f} seconds)"
    )

    return np.double(chamfer_dist_ab.item()), np.double(chamfer_dist_ba.item())


def process_and_save_dataset(
    A: np.ndarray,
    B: np.ndarray,
    output_dir: Path,
    batch_size: int | None = None,
) -> None:
    """
    Process two point sets and save them with metadata.

    This function handles the common workflow of:
    1. Creating the output directory
    2. Saving binary data files for both point sets
    3. Calculating Chamfer distances (L1 and L2)
    4. Creating and saving metadata

    Args:
        A: First point set (numpy array of shape [num_points_a, num_dimensions])
        B: Second point set (numpy array of shape [num_points_b, num_dimensions])
        output_dir: Directory to save the dataset
        batch_size: Batch size for Chamfer distance calculation (default: 1024)
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Save binary files
    save_binary_data(A, output_dir / "A.bin")
    save_binary_data(B, output_dir / "B.bin")

    # Calculate the directed Chamfer distance components between two sets
    if batch_size is None:
        batch_size = 1024
    chamfer_distance_l1_ab, chamfer_distance_l1_ba = chamfer_distance_components(
        A, B, batch_size, 1
    )
    chamfer_distance_l2_ab, chamfer_distance_l2_ba = chamfer_distance_components(
        A, B, batch_size, 2
    )

    # Create metadata
    create_metadata(
        num_dimensions=A.shape[1],
        num_points_a=A.shape[0],
        num_points_b=B.shape[0],
        chamfer_distance_l1_ab=chamfer_distance_l1_ab,
        chamfer_distance_l1_ba=chamfer_distance_l1_ba,
        chamfer_distance_l2_ab=chamfer_distance_l2_ab,
        chamfer_distance_l2_ba=chamfer_distance_l2_ba,
        filepath=output_dir / "metadata.json",
    )

    logger.info(f"Files saved to: {output_dir}")
