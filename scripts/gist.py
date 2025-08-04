import logging
from pathlib import Path
import numpy as np
import struct


def load_gist(dir: Path) -> np.ndarray:
    """Load GIST dataset from .fvecs file and convert to numpy array."""
    logging.info(f"Loading GIST dataset from {dir}")

    # Find the gist_base.fvecs file
    fvecs_file = dir / "gist_base.fvecs"
    if not fvecs_file.exists():
        raise ValueError(f"GIST dataset file not found: {fvecs_file}")

    logging.info(f"Reading .fvecs file: {fvecs_file}")

    # Read the .fvecs file
    data = []
    with open(fvecs_file, "rb") as f:
        while True:
            # Read the dimension (first 4 bytes as uint32)
            dim_bytes = f.read(4)
            if not dim_bytes:
                break

            # Unpack the dimension
            dimension = struct.unpack("I", dim_bytes)[0]

            # Read the vector data (dimension * 4 bytes as floats)
            vec_bytes = f.read(dimension * 4)
            if len(vec_bytes) != dimension * 4:
                raise ValueError("Incomplete vector data in .fvecs file")

            # Unpack the vector as floats
            vec = list(struct.unpack(f"{dimension}f", vec_bytes))
            data.append(vec)

    # Convert to numpy array with double precision
    X = np.array(data, dtype=np.double)

    logging.info(f"Loaded {X.shape[0]} vectors with {X.shape[1]} dimensions each")
    return X
