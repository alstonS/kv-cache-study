# Reads in the raw csv file
# baseline CSV, quantization CSV, paged/vLLM CSV
# use analysis.py for computation

# prints to terminal
# sasves new csv file as derived.csv, summary.csv

# to execute program do: 
# python3 scripts/analyze_results.py 
#   --baseline-csv results2/raw/final_mistral7b_baseline_grid.csv \
#   --quant-csv results2/raw/final_mistral7b_quant_grid.csv \


import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from src.analysis import run_full_pipeline
from src.analysis import load_runs, aggregate_trials, derive_metrics, summarize_by_method


def printm(title:str ) -> None:
    bar = "=" * 50
    print(f"\n{bar}\n{title}\n{bar}")


def print_df(df: pd.DataFrame, columns: list[str], round_to: int = 2) -> None:
    cols = [c for c in columns if c in df.columns]
    if not cols:
        print("no columns")
        return
    print(df[cols].round(round_to).to_string(index=False))


def main() -> None:

    # args
    parser = argparse.ArgumentParser(description="KV-Cache Analysis")

    parser.add_argument(
        "--baseline-csv", default="results2/raw/baseline_gpu.csv",
        help="path to baseline run CSV, skip if file missing",
    )
    parser.add_argument(
        "--quant-csv", default="results2/raw/quant_gpu.csv",
        help="path to quant run CSV, skip if file missing",
    )
    parser.add_argument(
        "--paged-csv", default="results2/raw/paged_gpu.csv",
        help="path to paged-attention run CSV, skip if file missing",
    )

    # A100 has 40gb
    parser.add_argument(
        "--gpu-budget-mb", type=float, default=40 * 1024,
        help="GPU memory budget for max_feasible_context_tokens"
             "Default is 40960 for A100 40GB, 81920 for A100 80GB "
             "16384 for T4",
    )
    parser.add_argument(
        "--output-dir", default="results2/processed",
        help="path to write derived.csv and summary.csv.",
    )
    args = parser.parse_args()


    # Part 1 - Loading
    print(f"Loading runs, gpu_budget = {args.gpu_budget_mb:.0f} MB")
    raw = load_runs(
        baseline_csv=args.baseline_csv if Path(args.baseline_csv).exists() else None,
        quant_csv=args.quant_csv if Path(args.quant_csv).exists() else None,
        paged_csv=args.paged_csv if Path(args.paged_csv).exists() else None,
    )

    methods_loaded = sorted(raw["method"].unique().tolist())
    print(f"  Methods loaded: {methods_loaded}")
    print(f"  Total raw rows: {len(raw)}")



    # Part 2 - aggregate trails
    # 
    # Combine trial 1, trial 2, trial 3
    # into one averaged row
    # method + input_length + max_new_tokens

    aggregated = aggregate_trials(raw)
    print(f"Cells after aggregation: {len(aggregated)}")

    # Part 3 - Derviving metrics

    derived = derive_metrics(aggregated, gpu_budget_mb=args.gpu_budget_mb)

    # Part 4 - Summary
    summary = summarize_by_method(derived)

    # Print
    printm("MEMORY (per cell, sorted by method then input length)")
    print_df(
        derived.sort_values(["method", "input_length", "max_new_tokens"]),
        [
            "method", "input_length", "max_new_tokens",
            "peak_memory_mb", "kv_cache_memory_mb",
            "memory_savings_mb", "memory_savings_pct",
            "kv_mb_per_token",
        ],
    )

    printm("THROUGHPUT  (decode tokens/sec and slowdown vs baseline)")
    print_df(
        derived.sort_values(["method", "input_length", "max_new_tokens"]),
        [
            "method", "input_length", "max_new_tokens",
            "decode_tokens_per_sec",
            "decode_slowdown_factor",
            "throughput_loss_pct",
            "extra_ms_per_token",
        ],
    )

    printm("PREFILL  (TTFT and prefill throughput)")
    print_df(
        derived.sort_values(["method", "input_length", "max_new_tokens"]),
        [
            "method", "input_length", "max_new_tokens",
            "ttft_sec", "ttft_ratio",
            "prefill_tokens_per_sec",
            "prefill_throughput_ratio",
        ],
    )

    printm("TRADEOFF  (memory saved per 1% throughput sacrificed)")
    tradeoff = derived[derived["method"] != "baseline_fp16"].copy()
    print_df(
        tradeoff.sort_values(
            "memory_saved_per_1pct_throughput_loss_mb", ascending=False
        ),
        [
            "method", "input_length", "max_new_tokens",
            "memory_savings_mb", "throughput_loss_pct",
            "memory_saved_per_1pct_throughput_loss_mb",
            "decode_tokens_per_sec_per_gb",
        ],
    )

    printm("CAPACITY  (estimated max context at GPU budget)")
    print_df(
        derived.sort_values(["method", "input_length"]),
        [
            "method", "input_length", "max_new_tokens",
            "kv_mb_per_token", "max_feasible_context_tokens",
            "success_rate",
        ],
    )

    printm("PER-METHOD SUMMARY  (the decision-table inputs)")
    print_df(
        summary,
        [
            "method",
            "max_successful_input_length",
            "memory_slope_mb_per_input_token",
            "prefill_slope_ms_per_input_token",
            "median_memory_savings_pct",
            "median_throughput_loss_pct",
            "median_extra_ms_per_token",
        ],
        round_to=3,
    )

    # writing csv files
    # derived.csv 
    # has one row per method/input_length/max_new_tokens condition
    # has all raw averaged metrics and all derived metrics

    # summary.csv
    # One row per method 

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    derived_path = out_dir / "derived.csv"
    summary_path = out_dir / "summary.csv"
    derived.to_csv(derived_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote {derived_path}  ({len(derived)} rows)")
    print(f"Wrote {summary_path}  ({len(summary)} rows)")


if __name__ == "__main__":
    main()

