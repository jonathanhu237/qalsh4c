import logging
from pathlib import Path

import numpy as np


def load_p53(file_path: Path) -> np.ndarray:
    """Load p53 dataset."""
    logging.info("Loading p53 dataset...")

    data = []

    with open(file_path, "r") as file:
        for line_num, line in enumerate(file, 1):
            # Skip empty lines
            if not line.strip():
                continue

            # Skip lines containing "?"
            if "?" in line:
                continue

            # Split the line by comma
            entries = line.strip().split(",")

            # Convert all entries except the last one to double
            try:
                # All entries except the last two (label and empty value)
                point = [float(entry) for entry in entries[:-2]]
                data.append(point)
            except ValueError as e:
                logging.warning(
                    f"Skipping line {line_num}: Error converting to float - {e}"
                )
                continue

    # Convert to numpy array
    if data:
        X = np.array(data, dtype=np.double)
        logging.info(f"Loaded {X.shape[0]} points with {X.shape[1]} features each")
        return X
    else:
        logging.error("No valid data found in the file")
        raise ValueError("No valid data found in the file")
