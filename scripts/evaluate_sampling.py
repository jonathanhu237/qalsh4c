import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from utils import build_project


def run_qalsh_sampling(dataset_path, num_samples):
    """
    Run qalsh sampling with a specific number of samples and extract results.

    Args:
        dataset_path: Path to the dataset directory
        num_samples: Number of samples to use

    Returns:
        Dictionary with num_samples and relative_error
    """
    cmd = [
        "./build/qalsh_chamfer",
        "estimate",
        "-d",
        dataset_path,
        "--in-memory",
        "sampling",
        "--use-cache",
        "-n",
        str(num_samples),
        "qalsh",
    ]

    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
        check=True,
    )

    # Extract relative error from output
    output_lines = result.stdout.split("\n")
    relative_error = None

    for line in output_lines:
        if "Relative error:" in line:
            # Extract the percentage value
            match = re.search(r"Relative error:\s*([0-9.]+%)", line)
            if match:
                relative_error = match.group(1)
                break

        if relative_error is None:
            return None

        return {"num_samples": num_samples, "relative_error": relative_error}


def main():
    parser = argparse.ArgumentParser(
        description="Run qalsh sampling multiple times with different sample counts"
    )
    parser.add_argument(
        "-m",
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of samples (default: 100)",
    )
    parser.add_argument(
        "-d",
        "--dataset-path",
        required=True,
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        default="./logs/output.log",
        help="Path to the output log file (default: ./logs/output.log)",
    )
    parser.add_argument(
        "-r",
        "--round",
        type=int,
        default=100,
        help="Number of rounds to run for each sample count (default: 100)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the project first
    build_project()

    # Run sampling for each sample count from 1 to max_samples
    # For each sample count, run 'round' times
    results = []
    for i in range(1, args.max_samples + 1):
        for j in range(args.round):
            result = run_qalsh_sampling(args.dataset_path, i)
            if result is None:
                print(f"Error: Could not extract relative error for {i} samples")
                sys.exit(1)
            result["round"] = j + 1  # Round number starts from 1
            results.append(result)

    # Write results to output file in JSON format
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
