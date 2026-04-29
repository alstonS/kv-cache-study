import subprocess
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs_new/baseline_gpu.yaml",
    )
    args = parser.parse_args()

    subprocess.run(
        [sys.executable, "scripts/run_baseline.py", "--config", args.config],
        check=True
    )