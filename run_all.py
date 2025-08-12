#!/usr/bin/env python

import subprocess
import sys

# List the scripts in the desired execution order.
scripts = [
    "app.py",
    "overlay_interpolation.py",
    "plot_temperature.py"
]

def run_script(script):
    print(f"\n[INFO] Running {script} ...")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"[ERROR] {script} failed with return code {result.returncode}. Aborting further execution.")
        sys.exit(result.returncode)
    else:
        print(f"[INFO] {script} completed successfully.")

def main():
    for script in scripts:
        run_script(script)
    print("\n[INFO] All scripts executed successfully.")

if __name__ == "__main__":
    main()
