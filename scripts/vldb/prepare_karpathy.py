"""Build the Flickr30K <-> COCO workload from the VGG-16 features of Karpathy and Fei-Fei.

The release (cs.stanford.edu/people/karpathy/deepimagesent) ships one 4096-d fc7 feature per image in vgg_feats.mat
(MATLAB, 4096 x n single). Both collections use the same network, so their features share one space. Each collection is
shuffled once with a fixed seed (as for the other datasets) and written in the binary's layout: A.bin and B.bin
(row-major float32) and metadata.json. Two directions are written: flickr30k -> coco and coco -> flickr30k.

Usage: python prepare_karpathy.py RAW_DIR OUT_ROOT
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import scipy.io as sio


def load(raw: str, name: str) -> np.ndarray:
    feats = sio.loadmat(os.path.join(raw, name, "vgg_feats.mat"))["feats"]
    x = np.ascontiguousarray(feats.T, dtype=np.float32)
    rng = np.random.default_rng(20261124)
    return x[rng.permutation(x.shape[0])]


def write(out: str, a: np.ndarray, b: np.ndarray) -> None:
    os.makedirs(out, exist_ok=True)
    a.tofile(os.path.join(out, "A.bin"))
    b.tofile(os.path.join(out, "B.bin"))
    meta = {"num_dimensions": int(a.shape[1]), "num_points": {"a": int(a.shape[0]), "b": int(b.shape[0])},
            "chamfer_distances": {"l1": {"ab": None, "ba": None}, "l2": {"ab": None, "ba": None}}}
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)


def main() -> None:
    raw, root = sys.argv[1], sys.argv[2]
    flickr, coco = load(raw, "flickr30k"), load(raw, "coco")
    for name, x in (("flickr30k", flickr), ("coco", coco)):
        norms = np.linalg.norm(x, axis=1)
        print(f"{name}: {x.shape}, zero rows {int((norms == 0).sum())}, mean norm {norms.mean():.2f}, "
              f"distinct rows {len(np.unique(x.view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))))}")
    write(os.path.join(root, "flickr2coco"), flickr, coco)
    write(os.path.join(root, "coco2flickr"), coco, flickr)


if __name__ == "__main__":
    main()
