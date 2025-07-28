import os
import argparse
import multiprocessing
from pathlib import Path
from utils import compile_program, run_command, DATABASES, DATA_BASE_PATH, EXECUTABLE

# Use the number of databases as the number of parallel processes
NUM_PROCESSES = len(DATABASES)


def run_conversion_task(task_args):
    """
    A worker function for the multiprocessing pool.
    It runs the conversion command for a single dataset.

    Args:
        task_args (tuple): A tuple containing (db_name, num_queries).
    """
    db_name, num_queries = task_args
    print(f"Starting conversion for dataset: '{db_name}' with {num_queries} queries...")

    data_path = os.path.join(DATA_BASE_PATH, db_name)
    raw_path = os.path.join(data_path, "raw")

    # Ensure the base data directory exists for the dataset
    Path(data_path).mkdir(exist_ok=True)

    convert_cmd = [
        EXECUTABLE,
        "-m",
        "-l",
        "error",
        "generate_dataset",
        "-d",
        data_path,
        "convert",
        "-n",
        db_name,
        "-r",
        raw_path,
        "-Q",
        str(num_queries),  # Convert num_queries to a string for the command
    ]

    if not run_command(convert_cmd):
        print(f"Conversion failed for dataset: '{db_name}'.")


def main():
    """Main function to run the full dataset conversion workflow."""

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Compile and run dataset conversion in parallel."
    )
    parser.add_argument(
        "-Q",
        "--num_queries",
        type=int,
        default=100,
        help="The number of queries to generate for each dataset (default: 100).",
    )
    args = parser.parse_args()

    # --- Main Execution ---
    # Compile the program once at the start
    if not compile_program():
        print("Halting script due to compilation failure.")
        return

    # Create a list of tasks to run, including the number of queries
    tasks = [(db, args.num_queries) for db in DATABASES]

    # Run all conversion tasks in parallel using a pool of workers
    print(
        f"\n--- Converting all datasets (in parallel with {NUM_PROCESSES} processes) ---"
    )
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        pool.map(run_conversion_task, tasks)

    print("\nAll dataset conversions completed!")


if __name__ == "__main__":
    # Ensure multiprocessing works correctly on all platforms
    multiprocessing.freeze_support()
    main()
