import logging
from pathlib import Path

import cv2
import numpy as np


def load_trevi(dir: Path) -> np.ndarray:
    """Load Trevi dataset from .bmp files and convert to patches.

    Args:
        dir: Path to directory containing .bmp files

    Returns:
        np.ndarray: Array of flattened 64x64 patches as double points
    """
    logging.info(f"Loading Trevi dataset from {dir}")

    # Get all .bmp files in the directory
    bmp_files = list(dir.glob("*.bmp"))
    if not bmp_files:
        raise ValueError(f"No .bmp files found in directory: {dir}")

    logging.info(f"Found {len(bmp_files)} .bmp files")

    patches = []

    # Process each .bmp file
    for bmp_file in bmp_files:
        # Read image in grayscale
        img = cv2.imread(str(bmp_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logging.warning(f"Could not read image: {bmp_file}")
            continue

        # Get image dimensions
        h, w = img.shape

        # Split image into 64x64 patches
        for y in range(0, h - 63, 64):  # Step by 64 pixels
            for x in range(0, w - 63, 64):  # Step by 64 pixels
                # Extract 64x64 patch
                patch = img[y : y + 64, x : x + 64]

                # Flatten patch to 1D array and convert to double
                patch_flat = patch.flatten().astype(np.double)
                patches.append(patch_flat)

    if not patches:
        raise ValueError("No valid patches extracted from images")

    # Stack all patches into a single array
    X = np.stack(patches)

    logging.info(f"Loaded {X.shape[0]} patches with {X.shape[1]} features each")
    return X
