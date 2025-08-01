import logging
import time

import numpy as np
import torch


def chamfer_distance(A: np.ndarray, B: np.ndarray, batch_size: int) -> np.double:
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
        dists = torch.cdist(p1_batch, tensor_B, p=1)
        min_dists, _ = torch.min(dists, dim=1)
        min_dists_A_to_B[i : i + batch_size] = min_dists

    # For each point in p2, find the closest in p1
    min_dists_B_to_A = torch.zeros(len(tensor_B), device=device, dtype=dtype)
    for i in range(0, len(tensor_B), batch_size):
        p2_batch = tensor_B[i : i + batch_size]
        dists = torch.cdist(p2_batch, tensor_A, p=1)
        min_dists, _ = torch.min(dists, dim=1)
        min_dists_B_to_A[i : i + batch_size] = min_dists

    # Sum the distances
    chamfer_dist = torch.sum(min_dists_A_to_B) + torch.sum(min_dists_B_to_A)

    elapsed_time = time.time() - start_time
    logging.info(
        f"Chamfer distance: {chamfer_dist.item():.6f} (computed in {elapsed_time:.2f} seconds)"
    )

    return np.double(chamfer_dist.item())
