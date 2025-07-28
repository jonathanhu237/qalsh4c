import subprocess


def run_command(command, log_file=None):
    """
    Executes a command, optionally redirecting its output to a log file.

    Args:
        command (list): The command to execute as a list of strings.
        log_file (str, optional): Path to the file to save stdout and stderr.
                                  If None, output goes to the console.

    Returns:
        bool: True if the command was successful, False otherwise.
    """
    command_str = " ".join(command)
    print(f"Running: {command_str}")

    try:
        if log_file:
            with open(log_file, "w") as f:
                # Redirect stdout and stderr to the file
                subprocess.run(command, check=True, text=True, stdout=f, stderr=f)
            print(f"Output successfully saved to {log_file}")
        else:
            # Run command and let output flow to the console
            subprocess.run(command, check=True, text=True)

        return True
    except FileNotFoundError:
        print(
            f"Error: Command not found. Is '{command[0]}' in your PATH or the correct directory?"
        )
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with return code {e.returncode}.")
        if log_file:
            print(f"   Check '{log_file}' for error details.")
        else:
            # If output wasn't redirected, stderr was already printed to the console.
            pass
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False


def compile_program():
    """Compiles the program using cmake presets."""
    print("---  Compiling ---")

    # Command to configure cmake
    cmake_preset_cmd = ["cmake", "--preset", "release"]
    if not run_command(cmake_preset_cmd):
        return False

    # Command to build the project
    cmake_build_cmd = ["cmake", "--build", "--preset", "release-build"]
    if not run_command(cmake_build_cmd):
        return False

    print("Compilation successful!")
    print("-" * 25)
    return True
