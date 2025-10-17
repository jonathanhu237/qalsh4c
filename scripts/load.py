import struct
from pathlib import Path

import numpy as np


def mnist() -> np.ndarray:
    from sklearn.datasets import fetch_openml

    X, _ = fetch_openml(
        "mnist_784", version=1, as_frame=False, parser="auto", return_X_y=True
    )
    X = X.astype(np.double)

    return X


def trevi(dir: Path) -> np.ndarray:
    import cv2

    bmp_files = list(dir.glob("*.bmp"))
    if not bmp_files:
        raise ValueError(f"No .bmp files found in directory: {dir}")

    patches = []
    for bmp_file in bmp_files:
        img = cv2.imread(str(bmp_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h, w = img.shape
        for y in range(0, h - 63, 64):
            for x in range(0, w - 63, 64):
                patch = img[y : y + 64, x : x + 64]
                patch_flat = patch.flatten().astype(np.double)
                patches.append(patch_flat)

    if not patches:
        raise ValueError("No valid patches extracted from images")

    X = np.stack(patches)
    return X


def p53(dir: Path) -> np.ndarray:
    data = []
    with open(dir / "K9.data", "r") as file:
        for _, line in enumerate(file, 1):
            if not line.strip():
                continue

            if "?" in line:
                continue

            entries = line.strip().split(",")
            point = [float(entry) for entry in entries[:-2]]
            data.append(point)

    X = np.array(data, dtype=np.double)
    return X


def sift(dir: Path) -> np.ndarray:
    base = read_fvecs(dir / "sift_base.fvecs")
    learn = read_fvecs(dir / "sift_learn.fvecs")
    query = read_fvecs(dir / "sift_query.fvecs")

    X = np.vstack([base, learn, query])
    return X


def gist(dir: Path) -> np.ndarray:
    base = read_fvecs(dir / "gist_base.fvecs")
    learn = read_fvecs(dir / "gist_learn.fvecs")
    query = read_fvecs(dir / "gist_query.fvecs")

    X = np.vstack([base, learn, query])
    return X


def read_fvecs(filepath: Path) -> np.ndarray:
    if not filepath.exists():
        raise ValueError(f"file not found: {filepath}")

    data = []
    with open(filepath, "rb") as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dimension = struct.unpack("I", dim_bytes)[0]
            vec_bytes = f.read(dimension * 4)
            vec = list(struct.unpack(f"{dimension}f", vec_bytes))
            data.append(vec)

    X = np.array(data, dtype=np.double)
    return X
