import json
import logging
import subprocess
from pathlib import Path

import numpy as np


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


def build_project():
    """Build the project using cmake."""
    # Configure the project
    subprocess.run(
        ["cmake", "--preset", "release"],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        check=True,
    )

    # Build the project
    subprocess.run(
        ["cmake", "--build", "--preset", "release-build"],
        capture_output=True,
        check=True,
    )
