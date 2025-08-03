import logging

import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist() -> np.ndarray:
    """Load MNIST dataset (70,000 images total)."""
    logging.info("Loading MNIST dataset...")
    X, _ = fetch_openml(
        "mnist_784", version=1, as_frame=False, parser="auto", return_X_y=True
    )

    # Convert to double
    X = X.astype(np.double)

    logging.info(f"Loaded {X.shape[0]} images with {X.shape[1]} features each")
    return X
