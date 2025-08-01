import logging
import multiprocessing as mp
import time
from multiprocessing import shared_memory
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.spatial.distance import cdist


def chamfer_distance_gpu(A: np.ndarray, B: np.ndarray) -> np.double:
    """
    Calculate Chamfer distance using PyTorch on a GPU.
    """
    if not torch.cuda.is_available():
        logging.error("CUDA is not available. This function requires a GPU.")
        return np.double("nan")

    logging.info("Calculating Chamfer distance using GPU (PyTorch)...")
    start_time = time.time()

    # Move data to the GPU
    device = torch.device("cuda")
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


def compute_min_distances_chunk(
    shm_from_name: str,
    from_shape: Tuple[int, int],
    shm_to_name: str,
    to_shape: Tuple[int, int],
    chunk_indices: Tuple[int, int],
    chunk_idx: int,
) -> np.ndarray:
    """
    Compute minimum distances for a chunk of points using shared memory.
    This function is designed to be used with multiprocessing.
    """

    # Attach to shared memory
    shm_from = shared_memory.SharedMemory(name=shm_from_name)
    shm_to = shared_memory.SharedMemory(name=shm_to_name)

    # Reconstruct numpy arrays from shared memory
    from_set = np.ndarray(from_shape, dtype=np.double, buffer=shm_from.buf)
    to_set = np.ndarray(to_shape, dtype=np.double, buffer=shm_to.buf)

    # Extract the chunk
    start_idx, end_idx = chunk_indices
    from_chunk = from_set[start_idx:end_idx]

    logging.debug(f"Processing chunk {chunk_idx} with {len(from_chunk)} points")

    # Calculate distances from this chunk to all points in to_set
    distances = cdist(from_chunk, to_set, metric="cityblock")

    # Find minimum distances for each point in the chunk
    min_distances = np.min(distances, axis=1)

    # Clean up shared memory references
    shm_from.close()
    shm_to.close()

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

    logging.info("Using shared memory to reduce memory usage...")

    # Ensure arrays are in the correct format
    A = A.astype(np.double)
    B = B.astype(np.double)

    # Create shared memory blocks for A and B
    shm_a = shared_memory.SharedMemory(create=True, size=A.nbytes)
    shm_b = shared_memory.SharedMemory(create=True, size=B.nbytes)

    # Create numpy arrays backed by shared memory
    shared_A = np.ndarray(A.shape, dtype=np.double, buffer=shm_a.buf)
    shared_B = np.ndarray(B.shape, dtype=np.double, buffer=shm_b.buf)

    # Copy data to shared memory
    shared_A[:] = A[:]
    shared_B[:] = B[:]

    # Split A into chunks for parallel processing
    chunk_size = len(A) // (
        n_processes * 4
    )  # Create more chunks than processes for better load balancing
    if chunk_size == 0:
        chunk_size = 1
    A_chunk_indices = [
        (i, min(i + chunk_size, len(A))) for i in range(0, len(A), chunk_size)
    ]
    logging.info(f"Split set A into {len(A_chunk_indices)} chunks of size {chunk_size}")

    # Calculate min distances from A to B
    logging.info("Computing minimum distances from A to B...")
    args_A_to_B = [
        (shm_a.name, A.shape, shm_b.name, B.shape, chunk_indices, i)
        for i, chunk_indices in enumerate(A_chunk_indices)
    ]
    with mp.Pool(processes=n_processes) as pool:
        min_dist_chunks_A_to_B = pool.starmap(
            compute_min_distances_chunk,
            args_A_to_B,
        )

    # Split B into chunks for parallel processing
    chunk_size = len(B) // (n_processes * 4)
    if chunk_size == 0:
        chunk_size = 1
    B_chunk_indices = [
        (i, min(i + chunk_size, len(B))) for i in range(0, len(B), chunk_size)
    ]
    logging.info(f"Split set B into {len(B_chunk_indices)} chunks of size {chunk_size}")

    # Calculate min distances from B to A
    logging.info("Computing minimum distances from B to A...")
    args_B_to_A = [
        (shm_b.name, B.shape, shm_a.name, A.shape, chunk_indices, i)
        for i, chunk_indices in enumerate(B_chunk_indices)
    ]
    with mp.Pool(processes=n_processes) as pool:
        min_dist_chunks_B_to_A = pool.starmap(compute_min_distances_chunk, args_B_to_A)

    # Clean up shared memory
    shm_a.close()
    shm_b.close()
    shm_a.unlink()
    shm_b.unlink()

    # Chamfer distance is the sum of minimum distances
    min_dist_A_to_B = np.concatenate(min_dist_chunks_A_to_B)
    min_dist_B_to_A = np.concatenate(min_dist_chunks_B_to_A)
    chamfer_dist = np.sum(min_dist_A_to_B) + np.sum(min_dist_B_to_A)

    elapsed_time = time.time() - start_time
    logging.info(
        f"Chamfer distance: {chamfer_dist:.6f} (computed in {elapsed_time:.2f} seconds)"
    )
    return np.double(chamfer_dist)
