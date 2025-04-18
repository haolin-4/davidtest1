import subprocess

# List of scripts to run
scripts = [
    "python flight_controller.py",
    "python track.py",
    "python obj_inference.py",
    "python rl_navigation.py",
]

processes = []

try:
    # Start all scripts
    for script in scripts:
        process = subprocess.Popen(script, shell=True)
        processes.append(process)

    # Wait for all processes to finish
    for process in processes:
        process.wait()
except KeyboardInterrupt:
    # Terminate all processes on interrupt
    for process in processes:
        process.terminate()
    print("All processes terminated.")