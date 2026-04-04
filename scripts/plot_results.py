import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "results/raw/baseline.csv"
PLOT_DIR = "results/plots"

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Could not find {CSV_PATH}")

    os.makedirs(PLOT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    plt.figure()
    plt.plot(df["input_length"], df["peak_memory_mb"], marker="o")
    plt.xlabel("Input length (tokens)")
    plt.ylabel("Peak GPU memory (MB)")
    plt.title("Baseline: Peak GPU memory vs input length")
    plt.savefig(os.path.join(PLOT_DIR, "memory_vs_input_length.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(df["input_length"], df["tokens_per_sec"], marker="o")
    plt.xlabel("Input length (tokens)")
    plt.ylabel("Tokens/sec")
    plt.title("Baseline: Throughput vs input length")
    plt.savefig(os.path.join(PLOT_DIR, "throughput_vs_input_length.png"), bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
