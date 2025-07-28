from pathlib import Path
from utils import (
    compile_program,
    run_command,
    LOGS_DIR,
    DATABASES,
    DATA_BASE_PATH,
    EXECUTABLE,
)
import os


def main():
    """Main function to run the full comparison workflow."""

    # Create the logs directory if it doesn't exist
    Path(LOGS_DIR).mkdir(exist_ok=True)

    # Compile the program
    if not compile_program():
        print("Halting script due to compilation failure.")
        return

    # Process each database
    for db in DATABASES:
        print(f"\n--- Processing: {db.upper()} ---")
        data_path = f"{DATA_BASE_PATH}/{db}/"

        # Index the database
        print(f"Indexing {db}...")
        index_cmd = [EXECUTABLE, "-m", "-l", "error", "index", "-d", data_path, "qalsh"]
        if not run_command(index_cmd):
            print(f"Halting script due to indexing failure on {db}.")
            continue  # Move to the next database

        # Run estimations for both methods
        for method in ["ann", "sampling"]:
            print(f"Running estimation for '{method}' method on {db}...")
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
                print(f"Estimation failed for {method} on {db}.")

    print("\nAll tasks completed!")


if __name__ == "__main__":
    main()
