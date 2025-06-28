import argparse
from genericpath import exists
import os
import numpy as np
from scipy.spatial import distance

parser = argparse.ArgumentParser(
    description="Generate a dataset for chamfer distance calculation."
)
parser.add_argument(
    "-n", "--name", type=str, default="toy", help="The name of the generated dataset"
)
parser.add_argument(
    "-d",
    "--directory",
    type=str,
    default="data",
    help="The directory to save the dataset",
)
parser.add_argument(
    "-D", "--num_dimensions", type=int, default=64, help="The number of dimensions"
)
parser.add_argument(
    "-N",
    "--num_points",
    type=int,
    default=1000,
    help="The number of points for each set",
)
parser.add_argument(
    "-l", "--left", type=float, default=-128, help="Left boundary for generated points"
)
parser.add_argument(
    "-r", "--right", type=float, default=128, help="Right boundary for generated points"
)


args = parser.parse_args()

set_a = np.random.uniform(
    low=args.left, high=args.right, size=(args.num_points, args.num_dimensions)
)
set_b = np.random.uniform(
    low=args.left, high=args.right, size=(args.num_points, args.num_dimensions)
)
chamfer_distance = np.sum(
    distance.cdist(set_a, set_b, "cityblock").min(axis=1)
) + np.sum(distance.cdist(set_b, set_a, "cityblock").min(axis=1))

os.makedirs(f"{args.directory}/{args.name}", exist_ok=True)

set_a.tofile(f"{args.directory}/{args.name}/{args.name}.a.bin")
set_b.tofile(f"{args.directory}/{args.name}/{args.name}.b.bin")

with open(f"{args.directory}/{args.name}/{args.name}.gt", "w") as f:
    f.write(f"{chamfer_distance}\n")
