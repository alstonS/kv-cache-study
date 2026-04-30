import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_DERIVED_CSV = "results2/processed/derived.csv"
DEFAULT_PLOT_DIR = "results2/plots"

METHOD_ORDER = ["baseline_fp16", "int3", "int4", "int8", "paged_vllm"]
METHOD_LABELS = {
    "baseline_fp16": "Baseline FP16",
    "int3": "INT3 KV",
    "int4": "INT4 KV",
    "int8": "INT8 KV",
    "paged_vllm": "Paged vLLM",
}


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _ordered_methods(df: pd.DataFrame) -> list[str]:
    available = set(df["method"].dropna().unique())
    ordered = [m for m in METHOD_ORDER if m in available]
    extras = sorted(available - set(ordered))
    return ordered + extras


def _plot_metric_by_decode_length(
    ax: plt.Axes,
    method_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    offset_overlapping_series: bool = False,
) -> None:
    token_values = sorted(method_df["max_new_tokens"].dropna().unique())
    offsets = {value: 0.0 for value in token_values}
    if offset_overlapping_series and len(token_values) > 1:
        input_lengths = sorted(method_df["input_length"].dropna().unique())
        min_gap = min(
            (b - a for a, b in zip(input_lengths, input_lengths[1:])),
            default=1,
        )
        step = min_gap * 0.04
        center = (len(token_values) - 1) / 2
        offsets = {
            value: (idx - center) * step for idx, value in enumerate(token_values)
        }

    for max_new_tokens, group in method_df.groupby("max_new_tokens"):
        group = group.sort_values("input_length")
        x_values = group["input_length"] + offsets.get(max_new_tokens, 0.0)
        ax.plot(
            x_values,
            group[metric],
            marker="o",
            linewidth=2,
            label=f"{int(max_new_tokens)} new tokens",
        )

    ax.set_title(title)
    ax.set_xlabel("Input length (tokens)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    if offset_overlapping_series:
        ax.text(
            0.01,
            0.02,
            "Markers are slightly offset because memory values overlap.",
            transform=ax.transAxes,
            fontsize=8,
            color="dimgray",
        )


def plot_method(method_df: pd.DataFrame, method: str, output_dir: Path) -> Path:
    method_df = method_df.sort_values(["max_new_tokens", "input_length"])
    label = _method_label(method)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"{label}: Derived KV-Cache Metrics", fontsize=16, fontweight="bold")

    plots = [
        ("peak_memory_mb", "Peak GPU memory (MB)", "Peak Memory vs Context"),
        ("kv_cache_memory_mb", "KV cache memory (MB)", "KV Cache Memory vs Context"),
        ("decode_tokens_per_sec", "Decode tokens/sec", "Decode Throughput vs Context"),
        ("ms_per_decode_token", "Milliseconds/token", "Decode Latency vs Context"),
    ]

    for ax, (metric, ylabel, title) in zip(axes.flat, plots):
        if metric not in method_df.columns:
            ax.set_visible(False)
            continue
        _plot_metric_by_decode_length(
            ax,
            method_df,
            metric,
            ylabel,
            title,
            offset_overlapping_series=metric in {
                "peak_memory_mb",
                "kv_cache_memory_mb",
            },
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / f"{method}_derived_metrics.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot one derived metrics dashboard per method from results2/processed/derived.csv."
    )
    parser.add_argument("--derived-csv", default=DEFAULT_DERIVED_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_PLOT_DIR)
    args = parser.parse_args()

    derived_csv = Path(args.derived_csv)
    if not derived_csv.exists():
        raise FileNotFoundError(f"Could not find {derived_csv}")

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(derived_csv)
    if "method" not in df.columns:
        raise ValueError("derived CSV must contain a 'method' column")

    written = []
    for method in _ordered_methods(df):
        method_df = df[df["method"] == method].copy()
        if method_df.empty:
            continue
        written.append(plot_method(method_df, method, output_dir))

    print("Wrote derived plots:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
