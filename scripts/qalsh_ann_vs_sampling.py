import os
import multiprocessing
from pathlib import Path
from utils import (
    compile_program,
    run_command,
    LOGS_DIR,
    DATABASES,
    DATA_BASE_PATH,
    EXECUTABLE,
)

# --- Configuration for Parallelism ---
# Use one process for each database for indexing
NUM_INDEX_PROCESSES = len(DATABASES)
# Use a separate process for each estimation task (4 dbs * 2 methods)
NUM_ESTIMATE_PROCESSES = len(DATABASES) * 2


def run_index_task(db_name):
    """
    A worker function to run the index command for a single database.
    This is designed to be run in a multiprocessing pool.

    Args:
        db_name (str): The name of the database to index.

    Returns:
        bool: True if successful, False otherwise.
    """
    print(f"Indexing {db_name}...")
    data_path = f"{DATA_BASE_PATH}/{db_name}/"
    index_cmd = [EXECUTABLE, "-m", "-l", "error", "index", "-d", data_path, "qalsh"]

    if not run_command(index_cmd):
        print(f"Indexing failed for {db_name}.")
        return False
    return True


def run_estimation_task(task_args):
    """
    A worker function to run an estimation command.
    This is designed to be run in a multiprocessing pool.

    Args:
        task_args (tuple): A tuple containing (db_name, method).
    """
    db, method = task_args
    print(f"Running estimation for '{method}' method on {db}...")
    data_path = f"{DATA_BASE_PATH}/{db}/"

    estimate_cmd = [
        EXECUTABLE,
        "-m",
        "-l",
        "error",
        "estimate",
        "-d",
        data_path,
        method,
        "qalsh",
    ]
    log_filename = os.path.join(LOGS_DIR, f"{db}_{method}_qalsh_output.log")

    if not run_command(estimate_cmd, log_file=log_filename):
        print(f"Estimation failed for '{method}' on {db}.")


def main():
    """Main function to run the full comparison workflow in parallel."""

    # Create the logs directory if it doesn't exist
    Path(LOGS_DIR).mkdir(exist_ok=True)

    # Compile the program (must be done sequentially first)
    if not compile_program():
        print("Halting script due to compilation failure.")
        return

    # Index all databases in parallel
    print(
        f"\n--- Indexing all databases (in parallel with {NUM_INDEX_PROCESSES} processes) ---"
    )
    with multiprocessing.Pool(processes=NUM_INDEX_PROCESSES) as pool:
        index_results = pool.map(run_index_task, DATABASES)

    # Stop if any of the indexing tasks failed
    if not all(index_results):
        print("\nHalting script due to one or more indexing failures.")
        return

    # Create a list of all estimation tasks to run
    estimation_tasks = []
    for db in DATABASES:
        for method in ["ann", "sampling"]:
            estimation_tasks.append((db, method))

    # Run all estimation tasks in parallel
    print(
        f"\n--- Running all estimations (in parallel with {NUM_ESTIMATE_PROCESSES} processes) ---"
    )
    with multiprocessing.Pool(processes=NUM_ESTIMATE_PROCESSES) as pool:
        pool.map(run_estimation_task, estimation_tasks)

    print("\nAll tasks completed!")


if __name__ == "__main__":
    main()
