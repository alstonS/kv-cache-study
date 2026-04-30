import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_SUMMARY_CSV = "results2/processed/summary.csv"
DEFAULT_PLOT_DIR = "results2/plots"

METHOD_ORDER = ["baseline_fp16", "int3", "int4", "int8", "paged_vllm"]
METHOD_LABELS = {
    "baseline_fp16": "Baseline FP16",
    "int3": "INT3 KV",
    "int4": "INT4 KV",
    "int8": "INT8 KV",
    "paged_vllm": "Paged vLLM",
}

SUMMARY_METRICS = [
    ("max_successful_input_length", "Max Successful Context", "Tokens"),
    ("memory_slope_mb_per_input_token", "Memory Slope", "MB/input token"),
    ("prefill_slope_ms_per_input_token", "Prefill Slope", "ms/input token"),
    ("median_memory_savings_pct", "Median Memory Savings", "% vs baseline"),
    ("median_throughput_loss_pct", "Median Throughput Loss", "% vs baseline"),
    ("median_extra_ms_per_token", "Median Extra Decode Latency", "ms/token vs baseline"),
]


def _prepare_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "method" not in df.columns:
        raise ValueError("summary CSV must contain a 'method' column")

    order_lookup = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    df = df.copy()
    df["_method_order"] = df["method"].map(order_lookup).fillna(len(METHOD_ORDER))
    df = df.sort_values(["_method_order", "method"]).drop(columns=["_method_order"])
    df["method_label"] = df["method"].map(METHOD_LABELS).fillna(df["method"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot summary comparison charts from results2/processed/summary.csv."
    )
    parser.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_PLOT_DIR)
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    if not summary_csv.exists():
        raise FileNotFoundError(f"Could not find {summary_csv}")

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df = _prepare_summary(pd.read_csv(summary_csv))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("KV-Cache Method Summary Comparison", fontsize=16, fontweight="bold")

    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B"]
    bar_colors = colors[: len(df)]

    for ax, (metric, title, ylabel) in zip(axes.flat, SUMMARY_METRICS):
        if metric not in df.columns:
            ax.set_visible(False)
            continue

        values = df[metric]
        plot_values = values
        plot_colors = bar_colors

        if metric == "median_memory_savings_pct":
            positive_values = values[values >= 0]
            upper = max(float(positive_values.max()) * 1.35, 1.0)
            lower = -0.5
            plot_values = values.clip(lower=lower, upper=upper)
            ax.set_ylim(lower, upper)

        bars = ax.bar(df["method_label"], plot_values, color=plot_colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=25)
        ax.axhline(0, color="black", linewidth=0.8)

        if metric == "median_memory_savings_pct":
            for bar, actual, color in zip(bars, values, plot_colors):
                if actual < 0:
                    ax.text(
                        1.02,
                        0.07,
                        f"{actual:.1f}%",
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=8,
                        color=color,
                        clip_on=False,
                    )

    fig.tight_layout(rect=[0, 0, 0.98, 0.95])
    output_path = output_dir / "summary_comparison.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote summary plot: {output_path}")


if __name__ == "__main__":
    main()
