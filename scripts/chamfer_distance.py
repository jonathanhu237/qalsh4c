import logging
import time

import numpy as np
import torch


def chamfer_distance(A: np.ndarray, B: np.ndarray) -> np.double:
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

    # Move data
    p1 = torch.from_numpy(A).to(device, dtype=torch.float32)
    p2 = torch.from_numpy(B).to(device, dtype=torch.float32)

    # Calculate pairwise distances (p=1 for Manhattan/L1 distance)
    # p1 is (N, D), p2 is (M, D) -> dists is (N, M)
    dists = torch.cdist(p1, p2, p=1)

    # For each point in p1, find the closest in p2
    min_dist_p1_to_p2, _ = torch.min(dists, dim=1)

    # For each point in p2, find the closest in p1
    min_dist_p2_to_p1, _ = torch.min(dists, dim=0)

    # Sum the distances
    chamfer_dist = torch.sum(min_dist_p1_to_p2) + torch.sum(min_dist_p2_to_p1)

    elapsed_time = time.time() - start_time
    logging.info(
        f"Chamfer distance: {chamfer_dist.item():.6f} (computed in {elapsed_time:.2f} seconds)"
    )

    return np.double(chamfer_dist.item())
