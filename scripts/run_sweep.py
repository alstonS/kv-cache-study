import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "scripts/run_baseline.py"], check=True)
